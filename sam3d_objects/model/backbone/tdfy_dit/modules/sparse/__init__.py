# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
from loguru import logger

BACKEND = "spconv"
# BACKEND = "torchsparse"
DEBUG = False
ATTN = "sdpa"


def __from_env():
    import os

    global BACKEND
    global DEBUG
    global ATTN

    env_sparse_backend = os.environ.get("SPARSE_BACKEND")
    env_sparse_debug = os.environ.get("SPARSE_DEBUG")
    env_sparse_attn = os.environ.get("SPARSE_ATTN_BACKEND")
    if env_sparse_attn is None:
        env_sparse_attn = os.environ.get("ATTN_BACKEND")

    if env_sparse_backend is not None and env_sparse_backend in [
        "spconv",
        "torchsparse",
    ]:
        BACKEND = env_sparse_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == "1"
    # env_sparse_attn = "sdpa"
    if env_sparse_attn is not None and env_sparse_attn in [
        "xformers",
        "flash_attn",
        "sdpa",
    ]:
        ATTN = env_sparse_attn

    logger.info(f"[SPARSE] Backend: {BACKEND}, Attention: {ATTN}")


__from_env()


def set_backend(backend: Literal["spconv", "torchsparse"]):
    global BACKEND
    BACKEND = backend


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


def set_attn(attn: Literal["xformers", "flash_attn"]):
    global ATTN
    ATTN = attn


from .basic import *
from .norm import *
from .nonlinearity import *
from .linear import *
from .attention import *
from .conv import *
from .spatial import *
from . import transformer
