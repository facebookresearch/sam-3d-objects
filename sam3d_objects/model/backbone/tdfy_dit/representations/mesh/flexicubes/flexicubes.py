# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from .tables import *
from kaolin.utils.testing import check_tensor

__all__ = [
    'FlexiCubes'
]


class FlexiCubes:
    def __init__(self, device="cuda"):

        self.device = device
        self.dmc_table = torch.tensor(dmc_table, dtype=torch.long, device=device, requires_grad=False)
        self.num_vd_table = torch.tensor(num_vd_table,
                                         dtype=torch.long, device=device, requires_grad=False)
        self.check_table = torch.tensor(
            check_table,
            dtype=torch.long, device=device, requires_grad=False)

        self.tet_table = torch.tensor(tet_table, dtype=torch.long, device=device, requires_grad=False)
        self.quad_split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device, requires_grad=False)
        self.quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device, requires_grad=False)
        self.quad_split_train = torch.tensor(
            [0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long, device=device, requires_grad=False)

        self.cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [
                                         1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float, device=device)
        self.cube_corners_idx = torch.pow(2, torch.arange(8, requires_grad=False))
        self.cube_edges = torch.tensor([0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6,
                                       2, 0, 3, 1, 7, 5, 6, 4], dtype=torch.long, device=device, requires_grad=False)

        self.edge_dir_table = torch.tensor([0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1],
                                           dtype=torch.long, device=device)
        self.dir_faces_table = torch.tensor([
            [[5, 4], [3, 2], [4, 5], [2, 3]],
            [[5, 4], [1, 0], [4, 5], [0, 1]],
            [[3, 2], [1, 0], [2, 3], [0, 1]]
        ], dtype=torch.long, device=device)
        self.adj_pairs = torch.tensor([0, 1, 1, 3, 3, 2, 2, 0], dtype=torch.long, device=device)

    def __call__(self, voxelgrid_vertices, scalar_field, cube_idx, resolution, qef_reg_scale=1e-3,
                 weight_scale=0.99, beta=None, alpha=None, gamma_f=None, voxelgrid_colors=None, training=False):
        assert torch.is_tensor(voxelgrid_vertices) and \
            check_tensor(voxelgrid_vertices, (None, 3), throw=False), \
            "'voxelgrid_vertices' should be a tensor of shape (num_vertices, 3)"
        num_vertices = voxelgrid_vertices.shape[0]
        assert torch.is_tensor(scalar_field) and \
            check_tensor(scalar_field, (num_vertices,), throw=False), \
            "'scalar_field' should be a tensor of shape (num_vertices,)"
        assert torch.is_tensor(cube_idx) and \
            check_tensor(cube_idx, (None, 8), throw=False), \
            "'cube_idx' should be a tensor of shape (num_cubes, 8)"
        num_cubes = cube_idx.shape[0]
        assert beta is None or (
            torch.is_tensor(beta) and
            check_tensor(beta, (num_cubes, 12), throw=False)
        ), "'beta' should be a tensor of shape (num_cubes, 12)"
        assert alpha is None or (
            torch.is_tensor(alpha) and
            check_tensor(alpha, (num_cubes, 8), throw=False)
        ), "'alpha' should be a tensor of shape (num_cubes, 8)"
        assert gamma_f is None or (
            torch.is_tensor(gamma_f) and
            check_tensor(gamma_f, (num_cubes,), throw=False)
        ), "'gamma_f' should be a tensor of shape (num_cubes,)"

        surf_cubes, occ_fx8 = self._identify_surf_cubes(scalar_field, cube_idx)
        if surf_cubes.sum() == 0:
            return (
                torch.zeros((0, 3), device=self.device),
                torch.zeros((0, 3), dtype=torch.long, device=self.device),
                torch.zeros((0), device=self.device),
                torch.zeros((0, voxelgrid_colors.shape[-1]), device=self.device) if voxelgrid_colors is not None else None
            )
        beta, alpha, gamma_f = self._normalize_weights(
            beta, alpha, gamma_f, surf_cubes, weight_scale)
        
        if voxelgrid_colors is not None:
            voxelgrid_colors = torch.sigmoid(voxelgrid_colors)

        case_ids = self._get_case_id(occ_fx8, surf_cubes, resolution)

        surf_edges, idx_map, edge_counts, surf_edges_mask = self._identify_surf_edges(
            scalar_field, cube_idx, surf_cubes
        )

        vd, L_dev, vd_gamma, vd_idx_map, vd_color = self._compute_vd(
            voxelgrid_vertices, cube_idx[surf_cubes], surf_edges, scalar_field,
            case_ids, beta, alpha, gamma_f, idx_map, qef_reg_scale, voxelgrid_colors)
        vertices, faces, s_edges, edge_indices, vertices_color = self._triangulate(
            scalar_field, surf_edges, vd, vd_gamma, edge_counts, idx_map,
            vd_idx_map, surf_edges_mask, training, vd_color)
        return vertices, faces, L_dev, vertices_color

    def _compute_reg_loss(self, vd, ue, edge_group_to_vd, vd_num_edges):
        """
        Regularizer L_dev as in Equation 8
        """
        dist = torch.norm(ue - torch.index_select(input=vd, index=edge_group_to_vd, dim=0), dim=-1)
        mean_l2 = torch.zeros_like(vd[:, 0])
        mean_l2 = (mean_l2).index_add_(0, edge_group_to_vd, dist) / vd_num_edges.squeeze(1).float()
        mad = (dist - torch.index_select(input=mean_l2, index=edge_group_to_vd, dim=0)).abs()
        return mad

    def _normalize_weights(self, beta, alpha, gamma_f, surf_cubes, weight_scale):
        """
        Normalizes the given weights to be non-negative. If input weights are None, it creates and returns a set of weights of ones.
        """
        n_cubes = surf_cubes.shape[0]

        if beta is not None:
            beta = (torch.tanh(beta) * weight_scale + 1)
        else:
            beta = torch.ones((n_cubes, 12), dtype=torch.float, device=self.device)

        if alpha is not None:
            alpha = (torch.tanh(alpha) * weight_scale + 1)
        else:
            alpha = torch.ones((n_cubes, 8), dtype=torch.float, device=self.device)

        if gamma_f is not None:
            gamma_f = torch.sigmoid(gamma_f) * weight_scale + (1 - weight_scale) / 2
        else:
            gamma_f = torch.ones((n_cubes), dtype=torch.float, device=self.device)

        return beta[surf_cubes], alpha[surf_cubes], gamma_f[surf_cubes]

    @torch.no_grad()
    def _get_case_id(self, occ_fx8, surf_cubes, res):
        """
        Obtains the ID of topology cases based on cell corner occupancy. This function resolves the 
        ambiguity in the Dual Marching Cubes (DMC) configurations as described in Section 1.3 of the 
        supplementary material. It should be noted that this function assumes a regular grid.
        """
        case_ids = (occ_fx8[surf_cubes] * self.cube_corners_idx.to(self.device).unsqueeze(0)).sum(-1)

        problem_config = self.check_table.to(self.device)[case_ids]
        to_check = problem_config[..., 0] == 1
        problem_config = problem_config[to_check]
        if not isinstance(res, (list, tuple)):
            res = [res, res, res]

        # The 'problematic_configs' only contain configurations for surface cubes. Next, we construct a 3D array,
        # 'problem_config_full', to store configurations for all cubes (with default config for non-surface cubes).
        # This allows efficient checking on adjacent cubes.
        problem_config_full = torch.zeros(list(res) + [5], device=self.device, dtype=torch.long)
        vol_idx = torch.nonzero(problem_config_full[..., 0] == 0)  # N, 3
        vol_idx_problem = vol_idx[surf_cubes][to_check]
        problem_config_full[vol_idx_problem[..., 0], vol_idx_problem[..., 1], vol_idx_problem[..., 2]] = problem_config
        vol_idx_problem_adj = vol_idx_problem + problem_config[..., 1:4]

        within_range = (
            vol_idx_problem_adj[..., 0] >= 0) & (
            vol_idx_problem_adj[..., 0] < res[0]) & (
            vol_idx_problem_adj[..., 1] >= 0) & (
            vol_idx_problem_adj[..., 1] < res[1]) & (
            vol_idx_problem_adj[..., 2] >= 0) & (
            vol_idx_problem_adj[..., 2] < res[2])

        vol_idx_problem = vol_idx_problem[within_range]
        vol_idx_problem_adj = vol_idx_problem_adj[within_range]
        problem_config = problem_config[within_range]
        problem_config_adj = problem_config_full[vol_idx_problem_adj[..., 0],
                                                 vol_idx_problem_adj[..., 1], vol_idx_problem_adj[..., 2]]
        # If two cubes with cases C16 and C19 share an ambiguous face, both cases are inverted.
        to_invert = (problem_config_adj[..., 0] == 1)
        idx = torch.arange(case_ids.shape[0], device=self.device)[to_check][within_range][to_invert]
        case_ids.index_put_((idx,), problem_config[to_invert][..., -1])
        return case_ids

    @torch.no_grad()
    def _identify_surf_edges(self, scalar_field, cube_idx, surf_cubes):
        """
        Identifies grid edges that intersect with the underlying surface by checking for opposite signs. As each edge 
        can be shared by multiple cubes, this function also assigns a unique index to each surface-intersecting edge 
        and marks the cube edges with this index.
        """
        occ_n = scalar_field < 0
        all_edges = cube_idx[surf_cubes][:, self.cube_edges].reshape(-1, 2)
        unique_edges, _idx_map, counts = torch.unique(all_edges, dim=0, return_inverse=True, return_counts=True)

        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

        surf_edges_mask = mask_edges[_idx_map]
        counts = counts[_idx_map]

        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=cube_idx.device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), device=cube_idx.device)
        # Shaped as [number of cubes x 12 edges per cube]. This is later used to map a cube edge to the unique index
        # for a surface-intersecting edge. Non-surface-intersecting edges are marked with -1.
        idx_map = mapping[_idx_map]
        surf_edges = unique_edges[mask_edges]
        return surf_edges, idx_map, counts, surf_edges_mask

    @torch.no_grad()
    def _identify_surf_cubes(self, scalar_field, cube_idx):
        """
        Identifies grid cubes that intersect with the underlying surface by checking if the signs at 
        all corners are not identical.
        """
        occ_n = scalar_field < 0
        occ_fx8 = occ_n[cube_idx.reshape(-1)].reshape(-1, 8)
        _occ_sum = torch.sum(occ_fx8, -1)
        surf_cubes = (_occ_sum > 0) & (_occ_sum < 8)
        return surf_cubes, occ_fx8

    def _linear_interp(self, edges_weight, edges_x):
        """
        Computes the location of zero-crossings on 'edges_x' using linear interpolation with 'edges_weight'.
        """
        edge_dim = edges_weight.dim() - 2
        assert edges_weight.shape[edge_dim] == 2
        edges_weight = torch.cat([torch.index_select(input=edges_weight, index=torch.tensor(1, device=self.device), dim=edge_dim), -
                                 torch.index_select(input=edges_weight, index=torch.tensor(0, device=self.device), dim=edge_dim)]
                                 , edge_dim)
        denominator = edges_weight.sum(edge_dim)
        ue = (edges_x * edges_weight).sum(edge_dim) / denominator
        return ue

    def _solve_vd_QEF(self, p_bxnx3, norm_bxnx3, c_bx3, qef_reg_scale):
        p_bxnx3 = p_bxnx3.reshape(-1, 7, 3)
        norm_bxnx3 = norm_bxnx3.reshape(-1, 7, 3)
        c_bx3 = c_bx3.reshape(-1, 3)
        A = norm_bxnx3
        B = ((p_bxnx3) * norm_bxnx3).sum(-1, keepdims=True)

        A_reg = (torch.eye(3, device=p_bxnx3.device) * qef_reg_scale).unsqueeze(0).repeat(p_bxnx3.shape[0], 1, 1)
        B_reg = (qef_reg_scale * c_bx3).unsqueeze(-1)
        A = torch.cat([A, A_reg], 1)
        B = torch.cat([B, B_reg], 1)
        dual_verts = torch.linalg.lstsq(A, B).solution.squeeze(-1)
        return dual_verts

    def _compute_vd(self, voxelgrid_vertices, surf_cubes_fx8, surf_edges, scalar_field,
                    case_ids, beta, alpha, gamma_f, idx_map, qef_reg_scale, voxelgrid_colors):
        """
        Computes the location of dual vertices as described in Section 4.2
        """
        alpha_nx12x2 = torch.index_select(input=alpha, index=self.cube_edges, dim=1).reshape(-1, 12, 2)
        surf_edges_x = torch.index_select(input=voxelgrid_vertices, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 3)
        surf_edges_s = torch.index_select(input=scalar_field, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 1)
        zero_crossing = self._linear_interp(surf_edges_s, surf_edges_x)
        
        if voxelgrid_colors is not None:
            C = voxelgrid_colors.shape[-1]
            surf_edges_c = torch.index_select(input=voxelgrid_colors, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, C)

        idx_map = idx_map.reshape(-1, 12)
        num_vd = torch.index_select(input=self.num_vd_table, index=case_ids, dim=0)
        edge_group, edge_group_to_vd, edge_group_to_cube, vd_num_edges, vd_gamma = [], [], [], [], []
        
        # if color is not None:
        #     vd_color = []

        total_num_vd = 0
        vd_idx_map = torch.zeros((case_ids.shape[0], 12), dtype=torch.long, device=self.device, requires_grad=False)

        for num in torch.unique(num_vd):
            cur_cubes = (num_vd == num)  # consider cubes with the same numbers of vd emitted (for batching)
            curr_num_vd = cur_cubes.sum() * num
            curr_edge_group = self.dmc_table[case_ids[cur_cubes], :num].reshape(-1, num * 7)
            curr_edge_group_to_vd = torch.arange(
                curr_num_vd, device=self.device).unsqueeze(-1).repeat(1, 7) + total_num_vd
            total_num_vd += curr_num_vd
            curr_edge_group_to_cube = torch.arange(idx_map.shape[0], device=self.device)[
                cur_cubes].unsqueeze(-1).repeat(1, num * 7).reshape_as(curr_edge_group)

            curr_mask = (curr_edge_group != -1)
            edge_group.append(torch.masked_select(curr_edge_group, curr_mask))
            edge_group_to_vd.append(torch.masked_select(curr_edge_group_to_vd.reshape_as(curr_edge_group), curr_mask))
            edge_group_to_cube.append(torch.masked_select(curr_edge_group_to_cube, curr_mask))
            vd_num_edges.append(curr_mask.reshape(-1, 7).sum(-1, keepdims=True))
            vd_gamma.append(torch.masked_select(gamma_f, cur_cubes).unsqueeze(-1).repeat(1, num).reshape(-1))
            # if color is not None:
            #     vd_color.append(color[cur_cubes].unsqueeze(1).repeat(1, num, 1).reshape(-1, 3))
            
        edge_group = torch.cat(edge_group)
        edge_group_to_vd = torch.cat(edge_group_to_vd)
        edge_group_to_cube = torch.cat(edge_group_to_cube)
        vd_num_edges = torch.cat(vd_num_edges)
        vd_gamma = torch.cat(vd_gamma)
        # if color is not None:
        #     vd_color = torch.cat(vd_color)
        # else:
        #     vd_color = None

        vd = torch.zeros((total_num_vd, 3), device=self.device)
        beta_sum = torch.zeros((total_num_vd, 1), device=self.device)

        idx_group = torch.gather(input=idx_map.reshape(-1), dim=0, index=edge_group_to_cube * 12 + edge_group)

        x_group = torch.index_select(input=surf_edges_x, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, 3)
        s_group = torch.index_select(input=surf_edges_s, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, 1)
        

        zero_crossing_group = torch.index_select(
            input=zero_crossing, index=idx_group.reshape(-1), dim=0).reshape(-1, 3)

        alpha_group = torch.index_select(input=alpha_nx12x2.reshape(-1, 2), dim=0,
                                            index=edge_group_to_cube * 12 + edge_group).reshape(-1, 2, 1)
        ue_group = self._linear_interp(s_group * alpha_group, x_group)

        beta_group = torch.gather(input=beta.reshape(-1), dim=0,
                                    index=edge_group_to_cube * 12 + edge_group).reshape(-1, 1)
        beta_sum = beta_sum.index_add_(0, index=edge_group_to_vd, source=beta_group)
        vd = vd.index_add_(0, index=edge_group_to_vd, source=ue_group * beta_group) / beta_sum
        
        '''
        interpolate colors use the same method as dual vertices
        '''
        if voxelgrid_colors is not None:
            vd_color = torch.zeros((total_num_vd, C), device=self.device)
            c_group = torch.index_select(input=surf_edges_c, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, C)
            uc_group = self._linear_interp(s_group * alpha_group, c_group)
            vd_color = vd_color.index_add_(0, index=edge_group_to_vd, source=uc_group * beta_group) / beta_sum
        else:
            vd_color = None
        
        L_dev = self._compute_reg_loss(vd, zero_crossing_group, edge_group_to_vd, vd_num_edges)

        v_idx = torch.arange(vd.shape[0], device=self.device)  # + total_num_vd

        vd_idx_map = (vd_idx_map.reshape(-1)).scatter(dim=0, index=edge_group_to_cube *
                                                      12 + edge_group, src=v_idx[edge_group_to_vd])

        return vd, L_dev, vd_gamma, vd_idx_map, vd_color

    def _triangulate(self, scalar_field, surf_edges, vd, vd_gamma, edge_counts, idx_map, vd_idx_map, surf_edges_mask, training, vd_color):
        """
        Connects four neighboring dual vertices to form a quadrilateral. The quadrilaterals are then split into 
        triangles based on the gamma parameter, as described in Section 4.3.
        """
        with torch.no_grad():
            group_mask = (edge_counts == 4) & surf_edges_mask  # surface edges shared by 4 cubes.
            group = idx_map.reshape(-1)[group_mask]
            vd_idx = vd_idx_map[group_mask]
            edge_indices, indices = torch.sort(group, stable=True)
            quad_vd_idx = vd_idx[indices].reshape(-1, 4)

            # Ensure all face directions point towards the positive SDF to maintain consistent winding.
            s_edges = scalar_field[surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1)].reshape(-1, 2)
            flip_mask = s_edges[:, 0] > 0
            quad_vd_idx = torch.cat((quad_vd_idx[flip_mask][:, [0, 1, 3, 2]],
                                     quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]]))

        quad_gamma = torch.index_select(input=vd_gamma, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4)
        gamma_02 = quad_gamma[:, 0] * quad_gamma[:, 2]
        gamma_13 = quad_gamma[:, 1] * quad_gamma[:, 3]
        if not training:
            mask = (gamma_02 > gamma_13)
            faces = torch.zeros((quad_gamma.shape[0], 6), dtype=torch.long, device=quad_vd_idx.device)
            faces[mask] = quad_vd_idx[mask][:, self.quad_split_1]
            faces[~mask] = quad_vd_idx[~mask][:, self.quad_split_2]
            faces = faces.reshape(-1, 3)
        else:
            vd_quad = torch.index_select(input=vd, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4, 3)
            vd_02 = (vd_quad[:, 0] + vd_quad[:, 2]) / 2
            vd_13 = (vd_quad[:, 1] + vd_quad[:, 3]) / 2
            weight_sum = (gamma_02 + gamma_13) + 1e-8
            vd_center = (vd_02 * gamma_02.unsqueeze(-1) + vd_13 * gamma_13.unsqueeze(-1)) / weight_sum.unsqueeze(-1)
            
            if vd_color is not None:
                color_quad = torch.index_select(input=vd_color, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4, vd_color.shape[-1])
                color_02 = (color_quad[:, 0] + color_quad[:, 2]) / 2
                color_13 = (color_quad[:, 1] + color_quad[:, 3]) / 2
                color_center = (color_02 * gamma_02.unsqueeze(-1) + color_13 * gamma_13.unsqueeze(-1)) / weight_sum.unsqueeze(-1)
                vd_color = torch.cat([vd_color, color_center])
            
            
            vd_center_idx = torch.arange(vd_center.shape[0], device=self.device) + vd.shape[0]
            vd = torch.cat([vd, vd_center])
            faces = quad_vd_idx[:, self.quad_split_train].reshape(-1, 4, 2)
            faces = torch.cat([faces, vd_center_idx.reshape(-1, 1, 1).repeat(1, 4, 1)], -1).reshape(-1, 3)
        return vd, faces, s_edges, edge_indices, vd_color