# Copyright (c) Meta Platforms, Inc. and affiliates.

from .sparse_structure_flow import SparseStructureFlowModel, SparseStructureFlowTdfyWrapper
from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
from .structured_latent_vae import SLatGaussianDecoder
from .structured_latent_flow import SLatFlowModel, SLatFlowModelTdfyWrapper

def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file

    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        path_parts = path.split("/")
        repo_id = f"{path_parts[0]}/{path_parts[1]}"
        model_name = "/".join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, "r") as f:
        config = json.load(f)
    model = __getattr__(config["name"])(**config["args"], **kwargs)
    model.load_state_dict(load_file(model_file))

    return model
