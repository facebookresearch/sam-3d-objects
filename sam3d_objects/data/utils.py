# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Any, Iterable, Tuple, Union, Dict, Sequence, Mapping, Container
import optree
import torch
from collections.abc import Iterable
import inspect
import ast
from torch.utils import _pytree

# None = root, Iterable[Any] = path, Any = path of one
ChildPathType = Union[None, Iterable[Any], Any]
ArgsType = Iterable[ChildPathType]
KwargsType = Mapping[str, ChildPathType]
ArgsKwargsType = Tuple[ArgsType, KwargsType]
MappingType = Union[None, ArgsKwargsType, ArgsType, KwargsType]


def tree_transpose_level_one(
    structure,
    check_children=False,
    map_fn=None,
    is_leaf=None,
):
    _, outer_spec = optree.tree_flatten(
        structure,
        is_leaf=lambda x: x is not structure,
        none_is_leaf=True,
    )

    spec = optree.tree_structure(structure, none_is_leaf=True, is_leaf=is_leaf)
    children_spec = spec.children()
    if len(children_spec) > 0:
        inner_spec = children_spec[0]
        if check_children:
            for child_spec in children_spec[1:]:
                assert (
                    inner_spec == child_spec
                ), f"one child was found having a different tree structure ({inner_spec} != {child_spec})"

        structure = optree.tree_transpose(outer_spec, inner_spec, structure)

    if map_fn is not None:
        structure = optree.tree_map(
            map_fn,
            structure,
            is_leaf=lambda x: optree.tree_structure(
                x, is_leaf=is_leaf, none_is_leaf=True
            )
            == outer_spec,
            none_is_leaf=True,
        )

    return structure


@staticmethod
def tree_tensor_map(fn, tree, *rest):
    return optree.tree_map(
        fn,
        tree,
        *rest,
        is_leaf=lambda x: isinstance(x, torch.Tensor),
        none_is_leaf=False,
    )


def to_device(obj, device):
    """Recursively moves all tensors in obj to the specified device.

    Args:
        obj: Object to move to device - can be a tensor, list, tuple, dict or any nested combination
        device: Target device (e.g. 'cuda', 'cpu', torch.device('cuda:0') etc.)

    Returns:
        Same object structure with all contained tensors moved to specified device
    """
    to_fn = lambda x: x.to(device)
    return optree.tree_map(to_fn, obj, is_leaf=torch.is_tensor, none_is_leaf=False)


def expand_right(tensor, target_shape):
    """
    e.g. Takes tensor of (a, b, c) and returns a tensor of (a, b, c, 1, 1, ...)
    """
    current_shape = tensor.shape
    dims_to_add = len(target_shape) - len(current_shape)
    result = tensor
    for _ in range(dims_to_add):
        result = result.unsqueeze(-1)
    expand_shape = list(current_shape) + [-1] * dims_to_add
    for i in range(len(target_shape)):
        if i < len(expand_shape) and expand_shape[i] == -1:
            expand_shape[i] = target_shape[i]
    return result.expand(*expand_shape)


def expand_as_right(tensor, target):
    return expand_right(tensor, target.shape)


def as_keys(path: ChildPathType):
    if isinstance(path, Iterable) and (not isinstance(path, str)):
        return tuple(path)
    elif path is None:
        return ()
    return (path,)


def get_child(obj: Any, *keys: Iterable[Any]):
    for key in keys:
        obj = obj[key]
    return obj


def set_child(obj: Any, value: Any, *keys: Iterable[Any]):
    parent = None
    for key in keys:
        parent = obj
        obj = obj[key]
    if parent is None:
        obj = value
    else:
        parent[key] = value
    return obj


def build_args_batch_extractor(args_mapping: ArgsType):
    def extract_fn(batch):
        return tuple(get_child(batch, *as_keys(path)) for path in args_mapping)

    return extract_fn


def build_kwargs_batch_extractor(kwargs_mapping: KwargsType):
    def extract_fn(batch):
        return {
            name: get_child(batch, *as_keys(path))
            for name, path in kwargs_mapping.items()
        }

    return extract_fn


empty_mapping = object()
kwargs_identity_mapping = object()


def build_batch_extractor(mapping: MappingType):
    extract_args_fn = lambda x: ()
    extract_kwargs_fn = lambda x: {}

    if mapping is None:

        def extract_args_fn(batch):
            return (batch,)

    elif mapping is empty_mapping:
        pass
    elif mapping is kwargs_identity_mapping:
        extract_kwargs_fn = lambda x: x
    elif isinstance(mapping, Sequence) and (not isinstance(mapping, str)):
        if (
            len(mapping) == 2
            and isinstance(mapping[0], Sequence)
            and isinstance(mapping[1], Dict)
        ):
            extract_args_fn = build_args_batch_extractor(mapping[0])
            extract_kwargs_fn = build_kwargs_batch_extractor(mapping[1])
        else:
            extract_args_fn = build_args_batch_extractor(mapping)
    elif isinstance(mapping, Mapping):
        extract_kwargs_fn = build_kwargs_batch_extractor(mapping)
    else:

        def extract_args_fn(batch):
            return (get_child(batch, *as_keys(mapping)),)

    def extract_fn(batch):
        return extract_args_fn(batch), extract_kwargs_fn(batch)

    return extract_fn


# >


def right_broadcasting(arr, target):
    return arr.reshape(arr.shape + (1,) * (target.ndim - arr.ndim))


def get_stats(tensor: torch.Tensor):
    float_tensor = tensor.float()
    return {
        "shape": tuple(tensor.shape),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": float_tensor.mean().item(),
        "median": tensor.median().item(),
        "std": float_tensor.std().item(),
    }


def _get_caller_arg_name(argnum=0, parent_frame=1):
    try:
        frame = inspect.currentframe()  # current frame
        frame = inspect.getouterframes(frame)[1 + parent_frame]  # parent frame
        code = inspect.getframeinfo(frame[0]).code_context[0].strip()  # get code line

        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                args = node.args
                break  # only get the first parent call

        # get first argument string (do not handle '=')
        label = ast.unparse(args[argnum]).strip()
    except:
        # TODO(Pierre) log exception
        label = "{label}"
    return label


def print_stats(tensor, label=None):
    if label is None:
        label = _get_caller_arg_name(argnum=0)
    stats = get_stats(tensor)
    string = f"{label}:\n" + "\n".join(f"- {k}: {v}" for k, v in stats.items())
    print(string)


def tree_reduce_unique(fn, tree, ensure_unique=True, **kwargs):
    values = _pytree.tree_flatten(tree, **kwargs)[0]
    values = tuple(map(fn, values))
    first = values[0]
    if ensure_unique:
        for value in values[1:]:
            if value != first:
                raise RuntimeError(
                    f"different values found, {value} and {first} should be the same"
                )
    return first
