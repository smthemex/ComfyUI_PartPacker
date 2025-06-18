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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...vae.modules.attention import CrossAttention, SelfAttention


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


# Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py#L27
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        dtype = next(self.mlp.parameters()).dtype  # need to determine on the fly...
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTLayer(nn.Module):
    def __init__(self, dim, num_heads, qknorm=False, gradient_checkpointing=True, qknorm_type="LayerNorm"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing

        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn1 = SelfAttention(dim, num_heads, qknorm=qknorm, qknorm_type=qknorm_type)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn2 = CrossAttention(dim, num_heads, context_dim=dim, qknorm=qknorm, qknorm_type=qknorm_type)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = FeedForward(dim)
        self.adaln_linear = nn.Linear(dim, dim * 6, bias=True)

    def forward(self, x, c, t_emb):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, c, t_emb, use_reentrant=False)
        else:
            return self._forward(x, c, t_emb)

    def _forward(self, x, c, t_emb):
        # x: [B, N, C], hidden states
        # c: [B, M, C], condition (assume normed and projected to C)
        # t_emb: [B, C], timestep embedding of adaln
        # return: [B, N, C], updated hidden states

        B, N, C = x.shape
        t_adaln = self.adaln_linear(F.silu(t_emb)).view(B, 6, -1)  # [B, 6, C]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t_adaln.chunk(6, dim=1)

        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn1(h)

        h = self.norm2(x)
        x = x + self.attn2(h, c)

        h = self.norm3(x)
        h = h * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.ff(h)

        return x


class DiT(nn.Module):
    def __init__(
        self,
        hidden_dim=1024,
        num_heads=16,
        latent_size=2048,
        latent_dim=8,
        num_layers=24,
        qknorm=False,
        gradient_checkpointing=True,
        qknorm_type="LayerNorm",
        use_pos_embed=False,
        use_parts=False,
        part_embed_mode="part2_only",
    ):
        super().__init__()

        # project in
        self.proj_in = nn.Linear(latent_dim, hidden_dim)

        # positional encoding (just use a learnable positional encoding)
        self.use_pos_embed = use_pos_embed
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, latent_size, hidden_dim) / hidden_dim**0.5)

        # part encoding (a must to distinguish parts!)
        self.use_parts = use_parts
        self.part_embed_mode = part_embed_mode
        if self.use_parts:
            if self.part_embed_mode == "element":
                self.part_embed = nn.Parameter(torch.randn(latent_size, hidden_dim) / hidden_dim**0.5)
            elif self.part_embed_mode == "part":
                self.part_embed = nn.Parameter(torch.randn(2, hidden_dim))
            elif self.part_embed_mode == "part2_only":
                # we only add this to the second part to distinguish from the first part
                self.part_embed = nn.Parameter(torch.randn(1, hidden_dim) / hidden_dim**0.5)

        # timestep encoding
        self.timestep_embed = TimestepEmbedder(hidden_dim)

        # transformer layers
        self.layers = nn.ModuleList(
            [DiTLayer(hidden_dim, num_heads, qknorm, gradient_checkpointing, qknorm_type) for _ in range(num_layers)]
        )

        # project out
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(hidden_dim, latent_dim)

        # init
        self.init_weight()

    def init_weight(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for layer in self.layers:
            nn.init.constant_(layer.adaln_linear.weight, 0)
            nn.init.constant_(layer.adaln_linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, c, t):
        # x: [B, N, C], hidden states
        # c: [B, M, C], condition (assume normed and projected to C)
        # t: [B,], timestep
        # return: [B, N, C], updated hidden states

        B, N, C = x.shape

        # project in
        x = self.proj_in(x)

        # positional encoding
        if self.use_pos_embed:
            x = x + self.pos_embed

        # part encoding
        if self.use_parts:
            if self.part_embed_mode == "element":
                x += self.part_embed
            elif self.part_embed_mode == "part":
                x[:, : x.shape[1] // 2, :] += self.part_embed[0]
                x[:, x.shape[1] // 2 :, :] += self.part_embed[1]
            elif self.part_embed_mode == "part2_only":
                x[:, x.shape[1] // 2 :, :] += self.part_embed[0]

        # timestep encoding
        t_emb = self.timestep_embed(t)  # [B, C]

        # transformer layers
        for layer in self.layers:
            x = layer(x, c, t_emb)

        # project out
        x = self.norm_out(x)
        x = self.proj_out(x)

        return x
