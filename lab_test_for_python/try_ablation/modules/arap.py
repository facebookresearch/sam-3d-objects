import torch
from pytorch3d.transforms import quaternion_to_matrix

def weighted_arap_edge_length_loss(
    control_coord,
    control_T,
    edge_index,
    edge_weight,
    normalized=True,
    eps=1e-8
):
    """
    Weighted edge-length ARAP.

    Objective:
        ||c_i' - c_k'|| ? ||c_i - c_k||

    If normalized=True:
        loss is normalized by rest edge length squared.
        This makes ARAP scale easier to tune.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=control_coord.device)

    xyz_rest = control_coord[:, :3]
    xyz_deformed = control_coord[:, :3] + control_T[:, :3]

    i = edge_index[0]
    k = edge_index[1]

    rest_edge = xyz_rest[i] - xyz_rest[k]
    deformed_edge = xyz_deformed[i] - xyz_deformed[k]

    rest_len = torch.norm(rest_edge, dim=-1)
    deformed_len = torch.norm(deformed_edge, dim=-1)

    loss_per_edge = (deformed_len - rest_len) ** 2

    if normalized:
        rest_len_sq = (rest_len ** 2).clamp_min(eps)
        loss_per_edge = loss_per_edge / rest_len_sq

    loss = torch.sum(edge_weight * loss_per_edge) / (torch.sum(edge_weight) + eps)

    return loss


def weighted_arap_rotation_loss(
    control_coord,
    control_T,
    control_q,
    edge_index,
    edge_weight,
    normalized=True,
    eps=1e-8
):
    """
    Weighted rotation-based ARAP.

    Objective:
        (c_i' - c_k') ? R_i (c_i - c_k)

    where:
        c_i' = c_i + T_i

    If normalized=True:
        loss is normalized by rest edge length squared.
        This makes the loss represent relative local distortion.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=control_coord.device)

    xyz_rest = control_coord[:, :3]
    xyz_deformed = control_coord[:, :3] + control_T[:, :3]

    q_norm = torch.nn.functional.normalize(control_q, p=2, dim=-1)
    R = quaternion_to_matrix(q_norm)

    i = edge_index[0]
    k = edge_index[1]

    rest_edge = xyz_rest[i] - xyz_rest[k]
    deformed_edge = xyz_deformed[i] - xyz_deformed[k]

    rotated_rest_edge = torch.bmm(
        R[i],
        rest_edge.unsqueeze(-1)
    ).squeeze(-1)

    loss_per_edge = torch.sum((deformed_edge - rotated_rest_edge) ** 2, dim=-1)

    if normalized:
        rest_len_sq = torch.sum(rest_edge ** 2, dim=-1).clamp_min(eps)
        loss_per_edge = loss_per_edge / rest_len_sq

    loss = torch.sum(edge_weight * loss_per_edge) / (torch.sum(edge_weight) + eps)

    return loss



def compute_arap_loss(
    control_coord,
    control_T,
    control_q,
    edge_index,
    edge_weight,
    arap_type,
    normalized=True
):
    """
    Select ARAP loss type.
    """
    if arap_type == "length":
        return weighted_arap_edge_length_loss(
            control_coord=control_coord,
            control_T=control_T,
            edge_index=edge_index,
            edge_weight=edge_weight,
            normalized=normalized
        )

    elif arap_type == "rotation":
        return weighted_arap_rotation_loss(
            control_coord=control_coord,
            control_T=control_T,
            control_q=control_q,
            edge_index=edge_index,
            edge_weight=edge_weight,
            normalized=normalized
        )

    elif arap_type == "both":
        loss_len = weighted_arap_edge_length_loss(
            control_coord=control_coord,
            control_T=control_T,
            edge_index=edge_index,
            edge_weight=edge_weight,
            normalized=normalized
        )

        loss_rot = weighted_arap_rotation_loss(
            control_coord=control_coord,
            control_T=control_T,
            control_q=control_q,
            edge_index=edge_index,
            edge_weight=edge_weight,
            normalized=normalized
        )

        return loss_rot + 0.1 * loss_len

    else:
        raise ValueError(f"Unknown arap_type: {arap_type}")
