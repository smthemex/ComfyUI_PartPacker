# PartPacker DiT - torch-only reimplementation, state dict keys 1:1 with
# PartPacker_flow.pt (prefix "dit." stripped by the loader).
# Semantics replicated exactly from PartPacker-main/flow/modules/dit.py and
# vae/modules/attention.py (torch backend), including:
#   - timestep_embedding cos-before-sin ordering, max_period=10000
#   - adaLN from timestep only (adaln_linear on F.silu(t_emb))
#   - part2_only part embedding (added to the second 4096-token half)
#   - full-dim RMSNorm on q/k BEFORE head reshape (not per-head!)
#   - SDPA with default scale (1/sqrt(head_dim))

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops


class RMSNorm(nn.Module):
    """Exact port of PartPacker vae/modules/attention.py RMSNorm."""

    def __init__(self, dim, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x):
        rnorm = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rnorm).to(dtype=self.weight.dtype) * self.weight


def _attention(q, k, v):
    # q/k/v: [B, N, H, D] -> [B, N, H, D] (official "torch" backend)
    B, N, H, D = q.shape
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    return out.permute(0, 2, 1, 3).contiguous()


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, qknorm=True, qknorm_type="RMSNorm", dtype=None, device=None, operations=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = operations.Linear(hidden_dim, 3 * hidden_dim, dtype=dtype, device=device)
        self.out_proj = operations.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        if qknorm:
            if qknorm_type == "RMSNorm":
                self.q_norm = RMSNorm(hidden_dim, dtype=dtype, device=device)
                self.k_norm = RMSNorm(hidden_dim, dtype=dtype, device=device)
            else:
                self.q_norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
                self.k_norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, -1).permute(2, 0, 1, 3)
        q, k, v = qkv.chunk(3, dim=0)
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)
        if hasattr(self, "q_norm") and not isinstance(getattr(self, "q_norm", None), type(None)):
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim)
        v = v.reshape(B, N, self.num_heads, self.head_dim)
        x = _attention(q, k, v)
        return self.out_proj(x.reshape(B, N, -1))


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, context_dim=None, qknorm=True, qknorm_type="RMSNorm", dtype=None, device=None, operations=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        context_dim = context_dim if context_dim is not None else hidden_dim
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        self.q_proj = operations.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.k_proj = operations.Linear(context_dim, hidden_dim, dtype=dtype, device=device)
        self.v_proj = operations.Linear(context_dim, hidden_dim, dtype=dtype, device=device)
        self.out_proj = operations.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        if qknorm:
            if qknorm_type == "RMSNorm":
                self.q_norm = RMSNorm(hidden_dim, dtype=dtype, device=device)
                self.k_norm = RMSNorm(hidden_dim, dtype=dtype, device=device)
            else:
                self.q_norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
                self.k_norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x, context):
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        if hasattr(self, "q_norm") and not isinstance(getattr(self, "q_norm", None), type(None)):
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, M, self.num_heads, self.head_dim)
        v = v.reshape(B, M, self.num_heads, self.head_dim)
        x = _attention(q, k, v)
        return self.out_proj(x.reshape(B, N, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dtype=None, device=None, operations=None):
        super().__init__()
        self.net = nn.Sequential(
            operations.Linear(dim, dim * mult, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(dim * mult, dim, dtype=dtype, device=device),
        )

    def forward(self, x):
        return self.net(x)


class TimestepEmbedder(nn.Module):
    """Exact port: max_period=10000, freqs fp32, cos-before-sin concat."""

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        import numpy as np
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
        dtype = self.mlp[0].weight.dtype
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=dtype)
        return self.mlp(t_freq)


class DiTLayer(nn.Module):
    def __init__(self, dim, num_heads, qknorm=True, qknorm_type="RMSNorm", dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, eps=1e-6, elementwise_affine=False, dtype=dtype, device=device)
        self.attn1 = SelfAttention(dim, num_heads, qknorm=qknorm, qknorm_type=qknorm_type, dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.LayerNorm(dim, eps=1e-6, elementwise_affine=False, dtype=dtype, device=device)
        self.attn2 = CrossAttention(dim, num_heads, context_dim=dim, qknorm=qknorm, qknorm_type=qknorm_type, dtype=dtype, device=device, operations=operations)
        self.norm3 = operations.LayerNorm(dim, eps=1e-6, elementwise_affine=False, dtype=dtype, device=device)
        self.ff = FeedForward(dim, dtype=dtype, device=device, operations=operations)
        self.adaln_linear = operations.Linear(dim, dim * 6, bias=True, dtype=dtype, device=device)

    def forward(self, x, c, t_emb):
        B, N, C = x.shape
        t_adaln = self.adaln_linear(F.silu(t_emb)).view(B, 6, -1)
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


class PartPackerDiT(nn.Module):
    """Signature compatible with comfy BaseModel construction:
    unet_model(**unet_config, device=device, operations=operations)."""

    def __init__(
        self,
        *,
        hidden_dim=1536,
        num_heads=16,
        num_layers=24,
        latent_size=4096,
        latent_dim=64,
        qknorm=True,
        qknorm_type="RMSNorm",
        use_pos_embed=False,
        use_parts=True,
        part_embed_mode="part2_only",
        dtype=None,
        device=None,
        operations=None,
        **kwargs,  # absorb 'image_model' etc.
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self.dtype = dtype  # comfy BaseModel.get_dtype() reads diffusion_model.dtype

        self.proj_in = operations.Linear(latent_dim, hidden_dim, dtype=dtype, device=device)

        self.use_pos_embed = use_pos_embed
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.randn(1, latent_size, hidden_dim, device=device, dtype=dtype) / hidden_dim**0.5
            )

        self.use_parts = use_parts
        self.part_embed_mode = part_embed_mode
        if self.use_parts:
            if self.part_embed_mode == "element":
                self.part_embed = nn.Parameter(
                    torch.randn(latent_size, hidden_dim, device=device, dtype=dtype) / hidden_dim**0.5
                )
            elif self.part_embed_mode == "part":
                self.part_embed = nn.Parameter(torch.randn(2, hidden_dim, device=device, dtype=dtype))
            elif self.part_embed_mode == "part2_only":
                self.part_embed = nn.Parameter(
                    torch.randn(1, hidden_dim, device=device, dtype=dtype) / hidden_dim**0.5
                )

        self.timestep_embed = TimestepEmbedder(hidden_dim, dtype=dtype, device=device, operations=operations)

        self.layers = nn.ModuleList(
            [
                DiTLayer(hidden_dim, num_heads, qknorm=qknorm, qknorm_type=qknorm_type, dtype=dtype, device=device, operations=operations)
                for _ in range(num_layers)
            ]
        )

        self.norm_out = operations.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False, dtype=dtype, device=device)
        self.proj_out = operations.Linear(hidden_dim, latent_dim, dtype=dtype, device=device)

    def forward(self, x, t, context=None, control=None, transformer_options={}, **kwargs):
        # x: [B, N, latent_dim] (token-first latent), t: [B] in 0..1000, context: [B, M, hidden_dim]
        B, N, C = x.shape

        x = self.proj_in(x)

        if self.use_pos_embed:
            x = x + self.pos_embed

        if self.use_parts:
            if self.part_embed_mode == "element":
                x = x + self.part_embed
            elif self.part_embed_mode == "part":
                x = torch.cat([x[:, : x.shape[1] // 2] + self.part_embed[0], x[:, x.shape[1] // 2:] + self.part_embed[1]], dim=1)
            elif self.part_embed_mode == "part2_only":
                # official does in-place x[:, N//2:] += part_embed[0]
                x = torch.cat([x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2:] + self.part_embed[0]], dim=1)

        t_emb = self.timestep_embed(t)

        for layer in self.layers:
            x = layer(x, context, t_emb)

        x = self.norm_out(x)
        x = self.proj_out(x)
        return x
