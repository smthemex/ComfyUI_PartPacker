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

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import CrossAttention, SelfAttention


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dim_context=None,
        qknorm=False,
        gradient_checkpointing=True,
        qknorm_type="LayerNorm",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_context = dim_context
        self.gradient_checkpointing = gradient_checkpointing

        self.norm_attn = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        if dim_context is not None:
            self.norm_context = nn.LayerNorm(dim_context, eps=1e-6, elementwise_affine=False)
            self.attn = CrossAttention(dim, num_heads, context_dim=dim_context, qknorm=qknorm, qknorm_type=qknorm_type)
        else:
            self.attn = SelfAttention(dim, num_heads, qknorm=qknorm, qknorm_type=qknorm_type)

        self.norm_ff = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = FeedForward(dim)

    def forward(self, x, c=None, mask=None, mask_c=None):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, c, mask, mask_c, use_reentrant=False)
        else:
            return self._forward(x, c, mask, mask_c)

    def _forward(self, x, c=None, mask=None, mask_c=None):
        # x: [B, N, C], hidden states
        # c: [B, M, C'], condition (assume normed and projected to C)
        # mask: [B, N], mask for x
        # mask_c: [B, M], mask for c
        # return: [B, N, C], updated hidden states

        if c is not None:
            x = x + self.attn(self.norm_attn(x), self.norm_context(c), mask_q=mask, mask_kv=mask_c)
        else:
            x = x + self.attn(self.norm_attn(x), mask=mask)

        x = x + self.ff(self.norm_ff(x))

        return x


# special attention block for the last cross-attn query layer
# 1. simple feed-forward (mult=1, no post ln)
# 2. no residual connection
# 3. no context ln
class FlashQueryLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dim_context,
        qknorm=False,
        gradient_checkpointing=True,
        qknorm_type="LayerNorm",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_context = dim_context
        self.gradient_checkpointing = gradient_checkpointing

        self.norm_attn = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = CrossAttention(dim, num_heads, context_dim=dim_context, qknorm=qknorm, qknorm_type=qknorm_type)
        self.ff = FeedForward(dim, mult=1)

    def forward(self, x, c=None, mask=None, mask_c=None):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, c, mask, mask_c, use_reentrant=False)
        else:
            return self._forward(x, c, mask, mask_c)

    def _forward(self, x, c, mask=None, mask_c=None):
        # x: [B, N, C], hidden states
        # c: [B, M, C'], condition (assume normed and projected to C)
        # mask: [B, N], mask for x
        # mask_c: [B, M], mask for c
        # return: [B, N, C], updated hidden states

        x = self.attn(self.norm_attn(x), c, mask_q=mask, mask_kv=mask_c)
        x = self.ff(x)

        return x
