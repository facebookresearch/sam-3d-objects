# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn.functional as F


def block_diag_attn_mask(q_seqlens, kv_seqlens, device=None, dtype=torch.float32):
    """
    Create an additive attention mask for block-diagonal attention.
    The result is shape [sum_q, sum_kv], with 0.0 in the valid
    region(s) and -inf elsewhere.
    """
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    # Start with everything "masked out"
    attn_mask = torch.full(
        (total_q, total_kv), float("-inf"), device=device, dtype=dtype
    )

    q_start = 0
    kv_start = 0
    for q_len, kv_len in zip(q_seqlens, kv_seqlens):
        attn_mask[q_start : q_start + q_len, kv_start : kv_start + kv_len] = 0
        q_start += q_len
        kv_start += kv_len

    return attn_mask


def masked_sdpa(q, k, v, q_seqlen, kv_seqlen):
    """
    Mimic xFormers' memory_efficient_attention using PyTorch 2.0 scaled_dot_product_attention.
    """
    # Build the block-diagonal additive mask
    # shape: [sum_q_len, sum_kv_len] with 0 where allowed, -inf where masked
    attn_mask_2d = block_diag_attn_mask(
        q_seqlen, kv_seqlen, device=q.device, dtype=q.dtype
    )

    # PyTorch’s scaled_dot_product_attention expects a mask broadcastable to
    # [batch_size, n_heads, q_len, kv_len]. For a single batch, single head:
    attn_mask_4d = attn_mask_2d.unsqueeze(0).unsqueeze(0)
    q = q.permute(0, 2, 1, 3)  # [N, H, L, C]
    k = k.permute(0, 2, 1, 3)  # [N, H, L, C]
    v = v.permute(0, 2, 1, 3)  # [N, H, L, C]

    # Now call PyTorch 2.0’s built-in SDPA
    # By default, it will automatically apply the "1/sqrt(dim)" scaling internally.
    out = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask_4d,  # Additive mask
        dropout_p=0.0,  # or whatever dropout you need
        is_causal=False,  # True if you want a causal (triangular) mask
    )
    # out is shape [1, sum_q_len, dim]
    out = out.permute(0, 2, 1, 3)

    return out[0]
