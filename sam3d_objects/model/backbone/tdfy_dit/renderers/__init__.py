# Copyright (c) Meta Platforms, Inc. and affiliates.
from .octree_renderer import OctreeRenderer
from .gaussian_render import GaussianRenderer

# handle case when nvdiffrast is not present on the machine
try:
    from .mesh_renderer import MeshRenderer
except ImportError:
    MeshRenderer = None
