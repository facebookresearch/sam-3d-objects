# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Any, Callable, Dict, List, Optional, Union, Iterable
import lightning.pytorch as pl
import torch
from pathlib import Path
import os
import re
from loguru import logger
from lightning.pytorch.utilities.consolidate_checkpoint import (
    _format_checkpoint,
    _load_distributed_checkpoint,
)
from glob import glob

from sam3d_objects.data.utils import get_child, set_child


def rename_checkpoint_weights_using_suffix_matching(
    checkpoint_path_in,
    checkpoint_path_out,
    model: torch.nn.Module,
    strict: bool = True,
    keys: Optional[List[Any]] = (),
):
    # extract model names
    param_names = [n for n, _ in model.named_parameters()]
    buffer_names = [n for n, _ in model.named_buffers()]
    model_names = param_names + buffer_names

    # load stored weights
    state = torch.load(checkpoint_path_in, weights_only=False)

    model_state = get_child(state, *keys)
    model_state_names = list(model_state.keys())

    # sort reversed names (sort by suffix)
    model_names_rev = sorted([n[::-1] for n in model_names])
    model_state_names_rev = sorted([n[::-1] for n in model_state_names])

    if strict and len(model_names) != len(model_state_names):
        raise RuntimeError(
            f"model and state don't have the same number of parameters ({len(model_names)} != {len(model_state_names)}), cannot match them (set strict = False to relax constraint)"
        )

    def common_prefix_length(str_0: str, str_1: str):
        for count in range(min(len(str_0), len(str_1))):
            if str_0[count] != str_1[count]:
                break
        return count

    # attempt to match every model names to largest suffic matched weight
    name_mapping = {}
    i, j = 0, 0
    last_n = 0
    while i < len(model_names_rev):
        if j < len(model_state_names_rev):
            n = common_prefix_length(model_names_rev[i], model_state_names_rev[j])
        else:
            n = 0

        if n >= last_n:
            last_n = n
            j += 1
        else:
            last_n = 0
            name_mapping[model_names_rev[i][::-1]] = model_state_names_rev[j - 1][::-1]
            i += 1

        if not j < len(model_state_names_rev) + 1:
            break

    # not all names might have been matched
    if i < len(model_names):
        raise RuntimeError("could not suffix match parameter names")

    for k, v in name_mapping.items():
        logger.debug(f"{k} <- {v}")

    # rename weights according to matches and save to disk
    model_state_out = {k: model_state[v] for k, v in name_mapping.items()}
    set_child(state, model_state_out, *keys)
    torch.save(state, checkpoint_path_out)


def remove_prefix_state_dict_fn(prefix: str):
    n = len(prefix)

    def state_dict_fn(state_dict):
        return {
            (key[n:] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }

    return state_dict_fn


def add_prefix_state_dict_fn(prefix: str):
    def state_dict_fn(state_dict):
        return {prefix + key: value for key, value in state_dict.items()}

    return state_dict_fn


def filter_and_remove_prefix_state_dict_fn(prefix: str):
    n = len(prefix)

    def state_dict_fn(state_dict):
        return {
            key[n:]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }

    return state_dict_fn


def get_last_checkpoint(path: str):
    checkpoints = glob(os.path.join(path, "epoch=*-step=*.ckpt"))
    prog = re.compile(r"epoch=(\d+)-step=(\d+).ckpt")

    checkpoints_to_sort = []
    for checkpoint in checkpoints:
        checkpoint_name = os.path.basename(checkpoint)
        match = prog.match(checkpoint_name)
        if match is not None:
            n_epoch, n_step = prog.match(checkpoint_name).groups()
            n_epoch, n_step = int(n_epoch), int(n_step)
            checkpoints_to_sort.append((n_epoch, n_step, checkpoint))

    sorted_checkpoints = sorted(checkpoints_to_sort)
    if not len(sorted_checkpoints) > 0:
        raise RuntimeError(f"no checkpoint has been found at path : {path}")
    return sorted_checkpoints[-1][2]


def load_sharded_checkpoint(path: str, device: Optional[str]):
    if device != "cpu":
        raise RuntimeError(
            f'loading sharded weights on device "{device}" is not available, please use the "cpu" device instead'
        )
    checkpoint = _load_distributed_checkpoint(Path(path))
    checkpoint = _format_checkpoint(checkpoint)
    return checkpoint


def load_model_from_checkpoint(
    model: Union[pl.LightningModule, torch.nn.Module],
    checkpoint_path: str,
    strict: bool = True,
    device: Optional[str] = None,
    freeze: bool = False,
    eval: bool = False,
    map_name: Union[Dict[str, str], None] = None,
    remove_name: Union[List[str], None] = None,
    state_dict_key: Union[None, str, Iterable[str]] = "state_dict",
    state_dict_fn: Optional[Callable[[Any], Any]] = None,
):
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    elif os.path.isdir(checkpoint_path):  # sharded
        checkpoint = load_sharded_checkpoint(checkpoint_path, device=device)
    else:  # if neither a file nor a directory, path does not exist
        raise FileNotFoundError(checkpoint_path)

    if isinstance(model, pl.LightningModule):
        model.on_load_checkpoint(checkpoint)

    # get state dictionary
    state_dict = checkpoint
    if state_dict_key is not None:
        if isinstance(state_dict_key, str):
            state_dict_key = (state_dict_key,)
        state_dict = get_child(state_dict, *state_dict_key)

    # remove names
    if remove_name is not None:
        for name in remove_name:
            del state_dict[name]

    # remap names
    if map_name is not None:
        for src, dst in map_name.items():
            if src not in state_dict:
                continue
            state_dict[dst] = state_dict[src]
            del state_dict[src]

    # apply custom changes to dict
    if state_dict_fn is not None:
        state_dict = state_dict_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)

    if device is not None:
        model = model.to(device)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        eval = True

    if eval:
        model.eval()

    return model
