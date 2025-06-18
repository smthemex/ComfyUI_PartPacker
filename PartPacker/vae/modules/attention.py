"""
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import (  # , unpad_input  # noqa
        index_first_axis,
        pad_input,
    )

    FLASH_ATTN_AVAILABLE = True
except Exception as e:
    print("[WARN] flash_attn not available, using torch/naive implementation")
    FLASH_ATTN_AVAILABLE = False


# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py#L98
# flashattn 2.7.0 changes the API, we are overriding it here
def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def attention(q, k, v, mask_q=None, mask_kv=None, dropout=0, causal=False, window_size=(-1, -1), backend="torch"):
    # q: (B, N, H, D)
    # k: (B, M, H, D)
    # v: (B, M, H, D)
    # mask_q: (B, N)
    # mask_kv: (B, M)
    # return: (B, N, H, D)

    B, N, H, D = q.shape
    M = k.shape[1]

    if causal:
        assert N == 1 or N == M, "Causal mask only supports self-attention"

    # unmasked case (usually inference)
    # will ignore window_size except flash-attn impl. Only provide the effective window!
    if mask_q is None and mask_kv is None:
        if backend == "flash-attn" and FLASH_ATTN_AVAILABLE:
            return flash_attn_func(q, k, v, dropout, causal=causal, window_size=window_size)  # [B, N, H, D]
        elif backend == "torch":  # torch implementation
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=causal)
            out = out.permute(0, 2, 1, 3).contiguous()
            return out
        else:  # naive implementation
            q = q.transpose(1, 2).reshape(B * H, N, D)
            k = k.transpose(1, 2).reshape(B * H, M, D)
            v = v.transpose(1, 2).reshape(B * H, M, D)
            w = torch.bmm(q, k.transpose(1, 2)) / (D**0.5)  # [B*H, N, M]
            if causal and N > 1:
                causal_mask = torch.full((N, M), float("-inf"), device=w.device, dtype=w.dtype)
                causal_mask = torch.triu(causal_mask, diagonal=1)
                w = w + causal_mask.unsqueeze(0)
            w = F.softmax(w, dim=-1)
            if dropout > 0:
                w = F.dropout(w, p=dropout)
            out = torch.bmm(w, v)  # [B*H, N, D]
            out = out.reshape(B, H, N, D).transpose(1, 2).contiguous()  # [B, N, H, D]
            return out

    # at least one of q or kv is masked (training)
    # only support flash-attn for now...
    if mask_q is None:
        mask_q = torch.ones(B, N, dtype=torch.bool, device=q.device)
    elif mask_kv is None:
        mask_kv = torch.ones(B, M, dtype=torch.bool, device=q.device)

    if FLASH_ATTN_AVAILABLE:
        # unpad (gather) input
        # mask_q: [B, N], first row has N1 1s, second row has N2 1s, ...
        # indices: [Ns,], Ns = N1 + N2 + ...
        # cu_seqlens_q: [B+1,], (0, N1, N1+N2, ...), cu=cumulative
        # max_len_q: scalar, max(N1, N2, ...)
        q, indices_q, cu_seqlens_q, max_len_q = unpad_input(q, mask_q)
        k, indices_kv, cu_seqlens_kv, max_len_kv = unpad_input(k, mask_kv)
        v = index_first_axis(v.reshape(-1, H, D), indices_kv)  # same indice as k

        # call varlen_func
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_len_q,
            max_seqlen_k=max_len_kv,
            dropout_p=dropout,
            causal=causal,
            window_size=window_size,
        )

        # pad (put back) output
        out = pad_input(out, indices_q, B, N)
        return out
    else:
        raise NotImplementedError("masked attention requires flash_attn!")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rnorm = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rnorm).to(dtype=self.weight.dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        input_dim=None,
        output_dim=None,
        dropout=0,
        causal=False,
        qknorm=False,
        qknorm_type="LayerNorm",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
        self.causal = causal
        self.dropout = dropout
        self.qknorm = qknorm

        self.qkv_proj = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

        if self.qknorm:
            if qknorm_type == "RMSNorm":
                self.q_norm = RMSNorm(self.hidden_dim, eps=1e-6)
                self.k_norm = RMSNorm(self.hidden_dim, eps=1e-6)
            else:
                self.q_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6, elementwise_affine=False)
                self.k_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x, mask=None):
        # x: [B, N, C]
        # mask: [B, N]
        B, N, C = x.shape
        qkv = self.qkv_proj(x)  # [B, N, C] -> [B, N, 3 * D]
        qkv = qkv.reshape(B, N, 3, -1).permute(2, 0, 1, 3)  # [3, B, N, D]
        q, k, v = qkv.chunk(3, dim=0)  # [3, B, N, D] -> 3 * [1, B, N, D]
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)
        if self.qknorm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim)
        v = v.reshape(B, N, self.num_heads, self.head_dim)
        x = attention(q, k, v, mask_q=mask, mask_kv=mask, dropout=self.dropout, causal=self.causal)  # [B, N, H, D]
        x = self.out_proj(x.reshape(B, N, -1))
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        input_dim=None,
        context_dim=None,
        output_dim=None,
        dropout=0,
        qknorm=False,
        qknorm_type="LayerNorm",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.context_dim = context_dim if context_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.qknorm = qknorm

        self.q_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

        if self.qknorm:
            if qknorm_type == "RMSNorm":
                self.q_norm = RMSNorm(self.hidden_dim, eps=1e-6)
                self.k_norm = RMSNorm(self.hidden_dim, eps=1e-6)
            else:
                self.q_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6, elementwise_affine=False)
                self.k_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x, context, mask_q=None, mask_kv=None):
        # x: [B, N, C]
        # context: [B, M, C']
        # mask_q: [B, N]
        # mask_kv: [B, M]
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        if self.qknorm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, M, self.num_heads, self.head_dim)
        v = v.reshape(B, M, self.num_heads, self.head_dim)
        x = attention(q, k, v, mask_q=mask_q, mask_kv=mask_kv, dropout=self.dropout, causal=False)  # [B, N, H, D]
        x = self.out_proj(x.reshape(B, N, -1))
        return x
