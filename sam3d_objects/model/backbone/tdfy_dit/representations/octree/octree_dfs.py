# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TRIVEC_CONFIG = {
    "dim": 8,
    "rank": 8,
}

DEFAULT_VOXEL_CONFIG = {
    "solid": False,
}

DEFAULT_DECOPOLY_CONFIG = {
    "degree": 8,
    "rank": 16,
}


class DfsOctree:
    """
    Sparse Voxel Octree (SVO) implementation for PyTorch.
    Using Depth-First Search (DFS) order to store the octree.
    DFS order suits rendering and ray tracing.

    The structure and data are separatedly stored.
    Structure is stored as a continuous array, each element is a 3*32 bits descriptor.
    |-----------------------------------------|
    |      0:3 bits      |      4:31 bits     |
    |      leaf num      |       unused       |
    |-----------------------------------------|
    |               0:31  bits                |
    |                child ptr                |
    |-----------------------------------------|
    |               0:31  bits                |
    |                data ptr                 |
    |-----------------------------------------|
    Each element represents a non-leaf node in the octree.
    The valid mask is used to indicate whether the children are valid.
    The leaf mask is used to indicate whether the children are leaf nodes.
    The child ptr is used to point to the first non-leaf child. Non-leaf children descriptors are stored continuously from the child ptr.
    The data ptr is used to point to the data of leaf children. Leaf children data are stored continuously from the data ptr.

    There are also auxiliary arrays to store the additional structural information to facilitate parallel processing.
      - Position: the position of the octree nodes.
      - Depth: the depth of the octree nodes.

    Args:
        depth (int): the depth of the octree.
    """

    def __init__(
        self,
        depth,
        aabb=[0, 0, 0, 1, 1, 1],
        sh_degree=2,
        primitive="voxel",
        primitive_config={},
        device="cuda",
    ):
        self.max_depth = depth
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.device = device
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.primitive = primitive
        self.primitive_config = primitive_config

        self.structure = torch.tensor(
            [[8, 1, 0]], dtype=torch.int32, device=self.device
        )
        self.position = torch.zeros((8, 3), dtype=torch.float32, device=self.device)
        self.depth = torch.zeros((8, 1), dtype=torch.uint8, device=self.device)
        self.position[:, 0] = torch.tensor(
            [0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75], device=self.device
        )
        self.position[:, 1] = torch.tensor(
            [0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.75, 0.75], device=self.device
        )
        self.position[:, 2] = torch.tensor(
            [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75], device=self.device
        )
        self.depth[:, 0] = 1

        self.data = ["position", "depth"]
        self.param_names = []

        if primitive == "voxel":
            self.features_dc = torch.zeros(
                (8, 1, 3), dtype=torch.float32, device=self.device
            )
            self.features_ac = torch.zeros(
                (8, (sh_degree + 1) ** 2 - 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.data += ["features_dc", "features_ac"]
            self.param_names += ["features_dc", "features_ac"]
            if not primitive_config.get("solid", False):
                self.density = torch.zeros(
                    (8, 1), dtype=torch.float32, device=self.device
                )
                self.data.append("density")
                self.param_names.append("density")
        elif primitive == "gaussian":
            self.features_dc = torch.zeros(
                (8, 1, 3), dtype=torch.float32, device=self.device
            )
            self.features_ac = torch.zeros(
                (8, (sh_degree + 1) ** 2 - 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.opacity = torch.zeros((8, 1), dtype=torch.float32, device=self.device)
            self.data += ["features_dc", "features_ac", "opacity"]
            self.param_names += ["features_dc", "features_ac", "opacity"]
        elif primitive == "trivec":
            self.trivec = torch.zeros(
                (8, primitive_config["rank"], 3, primitive_config["dim"]),
                dtype=torch.float32,
                device=self.device,
            )
            self.density = torch.zeros(
                (8, primitive_config["rank"]), dtype=torch.float32, device=self.device
            )
            self.features_dc = torch.zeros(
                (8, primitive_config["rank"], 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.features_ac = torch.zeros(
                (8, primitive_config["rank"], (sh_degree + 1) ** 2 - 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.density_shift = 0
            self.data += ["trivec", "density", "features_dc", "features_ac"]
            self.param_names += ["trivec", "density", "features_dc", "features_ac"]
        elif primitive == "decoupoly":
            self.decoupoly_V = torch.zeros(
                (8, primitive_config["rank"], 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.decoupoly_g = torch.zeros(
                (8, primitive_config["rank"], primitive_config["degree"]),
                dtype=torch.float32,
                device=self.device,
            )
            self.density = torch.zeros(
                (8, primitive_config["rank"]), dtype=torch.float32, device=self.device
            )
            self.features_dc = torch.zeros(
                (8, primitive_config["rank"], 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.features_ac = torch.zeros(
                (8, primitive_config["rank"], (sh_degree + 1) ** 2 - 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.density_shift = 0
            self.data += [
                "decoupoly_V",
                "decoupoly_g",
                "density",
                "features_dc",
                "features_ac",
            ]
            self.param_names += [
                "decoupoly_V",
                "decoupoly_g",
                "density",
                "features_dc",
                "features_ac",
            ]

        self.setup_functions()

    def setup_functions(self):
        self.density_activation = (
            (lambda x: torch.exp(x - 2))
            if self.primitive != "trivec"
            else (lambda x: x)
        )
        self.opacity_activation = lambda x: torch.sigmoid(x - 6)
        self.inverse_opacity_activation = lambda x: torch.log(x / (1 - x)) + 6
        self.color_activation = lambda x: torch.sigmoid(x)

    @property
    def num_non_leaf_nodes(self):
        return self.structure.shape[0]

    @property
    def num_leaf_nodes(self):
        return self.depth.shape[0]

    @property
    def cur_depth(self):
        return self.depth.max().item()

    @property
    def occupancy(self):
        return self.num_leaf_nodes / 8**self.cur_depth

    @property
    def get_xyz(self):
        return self.position

    @property
    def get_depth(self):
        return self.depth

    @property
    def get_density(self):
        if self.primitive == "voxel" and self.voxel_config["solid"]:
            return torch.full(
                (self.position.shape[0], 1),
                1000,
                dtype=torch.float32,
                device=self.device,
            )
        return self.density_activation(self.density)

    @property
    def get_opacity(self):
        return self.opacity_activation(self.density)

    @property
    def get_trivec(self):
        return self.trivec

    @property
    def get_decoupoly(self):
        return F.normalize(self.decoupoly_V, dim=-1), self.decoupoly_g

    @property
    def get_color(self):
        return self.color_activation(self.colors)

    @property
    def get_features(self):
        if self.sh_degree == 0:
            return self.features_dc
        return torch.cat([self.features_dc, self.features_ac], dim=-2)

    def state_dict(self):
        ret = {
            "structure": self.structure,
            "position": self.position,
            "depth": self.depth,
            "sh_degree": self.sh_degree,
            "active_sh_degree": self.active_sh_degree,
            "trivec_config": self.trivec_config,
            "voxel_config": self.voxel_config,
            "primitive": self.primitive,
        }
        if hasattr(self, "density_shift"):
            ret["density_shift"] = self.density_shift
        for data in set(self.data + self.param_names):
            if not isinstance(getattr(self, data), nn.Module):
                ret[data] = getattr(self, data)
            else:
                ret[data] = getattr(self, data).state_dict()
        return ret

    def load_state_dict(self, state_dict):
        keys = list(
            set(
                self.data
                + self.param_names
                + list(state_dict.keys())
                + ["structure", "position", "depth"]
            )
        )
        for key in keys:
            if key not in state_dict:
                print(f"Warning: key {key} not found in the state_dict.")
                continue
            try:
                if not isinstance(getattr(self, key), nn.Module):
                    setattr(self, key, state_dict[key])
                else:
                    getattr(self, key).load_state_dict(state_dict[key])
            except Exception as e:
                print(e)
                raise ValueError(f"Error loading key {key}.")

    def gather_from_leaf_children(self, data):
        """
        Gather the data from the leaf children.

        Args:
            data (torch.Tensor): the data to gather. The first dimension should be the number of leaf nodes.
        """
        leaf_cnt = self.structure[:, 0]
        leaf_cnt_masks = [leaf_cnt == i for i in range(1, 9)]
        ret = torch.zeros(
            (self.num_non_leaf_nodes,), dtype=data.dtype, device=self.device
        )
        for i in range(8):
            if leaf_cnt_masks[i].sum() == 0:
                continue
            start = self.structure[leaf_cnt_masks[i], 2]
            for j in range(i + 1):
                ret[leaf_cnt_masks[i]] += data[start + j]
        return ret

    def gather_from_non_leaf_children(self, data):
        """
        Gather the data from the non-leaf children.

        Args:
            data (torch.Tensor): the data to gather. The first dimension should be the number of leaf nodes.
        """
        non_leaf_cnt = 8 - self.structure[:, 0]
        non_leaf_cnt_masks = [non_leaf_cnt == i for i in range(1, 9)]
        ret = torch.zeros_like(data, device=self.device)
        for i in range(8):
            if non_leaf_cnt_masks[i].sum() == 0:
                continue
            start = self.structure[non_leaf_cnt_masks[i], 1]
            for j in range(i + 1):
                ret[non_leaf_cnt_masks[i]] += data[start + j]
        return ret

    def structure_control(self, mask):
        """
        Control the structure of the octree.

        Args:
            mask (torch.Tensor): the mask to control the structure. 1 for subdivide, -1 for merge, 0 for keep.
        """
        # Dont subdivide when the depth is the maximum.
        mask[self.depth.squeeze() == self.max_depth] = torch.clamp_max(
            mask[self.depth.squeeze() == self.max_depth], 0
        )
        # Dont merge when the depth is the minimum.
        mask[self.depth.squeeze() == 1] = torch.clamp_min(
            mask[self.depth.squeeze() == 1], 0
        )

        # Gather control mask
        structre_ctrl = self.gather_from_leaf_children(mask)
        structre_ctrl[structre_ctrl == -8] = -1

        new_leaf_num = self.structure[:, 0].clone()
        # Modify the leaf num.
        structre_valid = structre_ctrl >= 0
        new_leaf_num[structre_valid] -= structre_ctrl[
            structre_valid
        ]  # Add the new nodes.
        structre_delete = structre_ctrl < 0
        merged_nodes = self.gather_from_non_leaf_children(structre_delete.int())
        new_leaf_num += merged_nodes  # Delete the merged nodes.

        # Update the structure array to allocate new nodes.
        mem_offset = torch.zeros(
            (self.num_non_leaf_nodes + 1,), dtype=torch.int32, device=self.device
        )
        mem_offset.index_add_(
            0, self.structure[structre_valid, 1], structre_ctrl[structre_valid]
        )  # Add the new nodes.
        mem_offset[:-1] -= structre_delete.int()  # Delete the merged nodes.
        new_structre_idx = torch.arange(
            0, self.num_non_leaf_nodes + 1, dtype=torch.int32, device=self.device
        ) + mem_offset.cumsum(0)
        new_structure_length = new_structre_idx[-1].item()
        new_structre_idx = new_structre_idx[:-1]
        new_structure = torch.empty(
            (new_structure_length, 3), dtype=torch.int32, device=self.device
        )
        new_structure[new_structre_idx[structre_valid], 0] = new_leaf_num[
            structre_valid
        ]

        # Initialize the new nodes.
        new_node_mask = torch.ones(
            (new_structure_length,), dtype=torch.bool, device=self.device
        )
        new_node_mask[new_structre_idx[structre_valid]] = False
        new_structure[new_node_mask, 0] = 8  # Initialize to all leaf nodes.
        new_node_num = new_node_mask.sum().item()

        # Rebuild child ptr.
        non_leaf_cnt = 8 - new_structure[:, 0]
        new_child_ptr = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=self.device),
                non_leaf_cnt.cumsum(0)[:-1],
            ]
        )
        new_structure[:, 1] = new_child_ptr + 1

        # Rebuild data ptr with old data.
        leaf_cnt = torch.zeros(
            (new_structure_length,), dtype=torch.int32, device=self.device
        )
        leaf_cnt.index_add_(0, new_structre_idx, self.structure[:, 0])
        old_data_ptr = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=self.device),
                leaf_cnt.cumsum(0)[:-1],
            ]
        )

        # Update the data array
        subdivide_mask = mask == 1
        merge_mask = mask == -1
        data_valid = ~(subdivide_mask | merge_mask)
        mem_offset = torch.zeros(
            (self.num_leaf_nodes + 1,), dtype=torch.int32, device=self.device
        )
        mem_offset.index_add_(
            0,
            old_data_ptr[new_node_mask],
            torch.full((new_node_num,), 8, dtype=torch.int32, device=self.device),
        )  # Add data array for new nodes
        mem_offset[
            :-1
        ] -= subdivide_mask.int()  # Delete data elements for subdivide nodes
        mem_offset[:-1] -= merge_mask.int()  # Delete data elements for merge nodes
        mem_offset.index_add_(
            0, self.structure[structre_valid, 2], merged_nodes[structre_valid]
        )  # Add data elements for merge nodes
        new_data_idx = torch.arange(
            0, self.num_leaf_nodes + 1, dtype=torch.int32, device=self.device
        ) + mem_offset.cumsum(0)
        new_data_length = new_data_idx[-1].item()
        new_data_idx = new_data_idx[:-1]
        new_data = {
            data: torch.empty(
                (new_data_length,) + getattr(self, data).shape[1:],
                dtype=getattr(self, data).dtype,
                device=self.device,
            )
            for data in self.data
        }
        for data in self.data:
            new_data[data][new_data_idx[data_valid]] = getattr(self, data)[data_valid]

        # Rebuild data ptr
        leaf_cnt = new_structure[:, 0]
        new_data_ptr = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=self.device),
                leaf_cnt.cumsum(0)[:-1],
            ]
        )
        new_structure[:, 2] = new_data_ptr

        # Initialize the new data array
        ## For subdivide nodes
        if subdivide_mask.sum() > 0:
            subdivide_data_ptr = new_structure[new_node_mask, 2]
            for data in self.data:
                for i in range(8):
                    if data == "position":
                        offset = (
                            torch.tensor(
                                [i // 4, (i // 2) % 2, i % 2],
                                dtype=torch.float32,
                                device=self.device,
                            )
                            - 0.5
                        )
                        scale = 2 ** (-1.0 - self.depth[subdivide_mask])
                        new_data["position"][subdivide_data_ptr + i] = (
                            self.position[subdivide_mask] + offset * scale
                        )
                    elif data == "depth":
                        new_data["depth"][subdivide_data_ptr + i] = (
                            self.depth[subdivide_mask] + 1
                        )
                    elif data == "opacity":
                        new_data["opacity"][subdivide_data_ptr + i] = (
                            self.inverse_opacity_activation(
                                torch.sqrt(
                                    self.opacity_activation(
                                        self.opacity[subdivide_mask]
                                    )
                                )
                            )
                        )
                    elif data == "trivec":
                        offset = (
                            torch.tensor(
                                [i // 4, (i // 2) % 2, i % 2],
                                dtype=torch.float32,
                                device=self.device,
                            )
                            * 0.5
                        )
                        coord = (
                            torch.linspace(
                                0,
                                0.5,
                                self.trivec.shape[-1],
                                dtype=torch.float32,
                                device=self.device,
                            )[None]
                            + offset[:, None]
                        ).reshape(1, 3, self.trivec.shape[-1], 1)
                        axis = (
                            torch.linspace(
                                0, 1, 3, dtype=torch.float32, device=self.device
                            )
                            .reshape(1, 3, 1, 1)
                            .repeat(1, 1, self.trivec.shape[-1], 1)
                        )
                        coord = (
                            torch.stack([coord, axis], dim=3)
                            .reshape(1, 3, self.trivec.shape[-1], 2)
                            .expand(self.trivec[subdivide_mask].shape[0], -1, -1, -1)
                            * 2
                            - 1
                        )
                        new_data["trivec"][subdivide_data_ptr + i] = F.grid_sample(
                            self.trivec[subdivide_mask], coord, align_corners=True
                        )
                    else:
                        new_data[data][subdivide_data_ptr + i] = getattr(self, data)[
                            subdivide_mask
                        ]
        ## For merge nodes
        if merge_mask.sum() > 0:
            merge_data_ptr = torch.empty(
                (merged_nodes.sum().item(),), dtype=torch.int32, device=self.device
            )
            merge_nodes_cumsum = torch.cat(
                [
                    torch.zeros((1,), dtype=torch.int32, device=self.device),
                    merged_nodes.cumsum(0)[:-1],
                ]
            )
            for i in range(8):
                merge_data_ptr[merge_nodes_cumsum[merged_nodes > i] + i] = (
                    new_structure[new_structre_idx[merged_nodes > i], 2] + i
                )
            old_merge_data_ptr = self.structure[structre_delete, 2]
            for data in self.data:
                if data == "position":
                    scale = 2 ** (1.0 - self.depth[old_merge_data_ptr])
                    new_data["position"][merge_data_ptr] = (
                        ((self.position[old_merge_data_ptr] + 0.5) / scale).floor()
                        * scale
                        + 0.5 * scale
                        - 0.5
                    )
                elif data == "depth":
                    new_data["depth"][merge_data_ptr] = (
                        self.depth[old_merge_data_ptr] - 1
                    )
                elif data == "opacity":
                    new_data["opacity"][subdivide_data_ptr + i] = (
                        self.inverse_opacity_activation(
                            self.opacity_activation(self.opacity[subdivide_mask]) ** 2
                        )
                    )
                elif data == "trivec":
                    new_data["trivec"][merge_data_ptr] = self.trivec[old_merge_data_ptr]
                else:
                    new_data[data][merge_data_ptr] = getattr(self, data)[
                        old_merge_data_ptr
                    ]

        # Update the structure and data array
        self.structure = new_structure
        for data in self.data:
            setattr(self, data, new_data[data])

        # Save data array control temp variables
        self.data_rearrange_buffer = {
            "subdivide_mask": subdivide_mask,
            "merge_mask": merge_mask,
            "data_valid": data_valid,
            "new_data_idx": new_data_idx,
            "new_data_length": new_data_length,
            "new_data": new_data,
        }
