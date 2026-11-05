# PartPacker VAE - torch-only port of PartPacker-main/vae/model.py (decode path only).
# Config verified against checkpoint evidence (vae.pt):
#   hidden_dim=1024, dec_hidden_dim=1024, latent_size=4096, latent_dim=64,
#   point_fourier_dim=48 (proj_input [1024,51]), fourier_version="v3" (schema default),
#   qknorm=True qknorm_type="LayerNorm" (affine=False - ckpt has NO q_norm params),
#   use_flash_query=False (ckpt has NO norm_query_context; attn_query.ff.net.0 is
#   [4096,1024] = mult 4 -> plain AttentionBlock, not FlashQueryLayer),
#   salient_attn_mode="dual" (perceiver_dorases present), num_enc_layers=0,
#   num_dec_layers=24.
# Only the decode/query path is ported: inference always decodes from the flow
# latent, encode (point cloud) is training-only.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .dit import RMSNorm, SelfAttention, CrossAttention, FeedForward


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_context=None, qknorm=True, qknorm_type="RMSNorm", dtype=None, device=None, operations=None):
        super().__init__()
        if operations is None:
            import comfy.ops
            operations = comfy.ops.manual_cast
        self.dim = dim
        self.num_heads = num_heads

        self.norm_attn = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        if dim_context is not None:
            self.norm_context = nn.LayerNorm(dim_context, eps=1e-6, elementwise_affine=False)
            self.attn = CrossAttention(dim, num_heads, context_dim=dim_context, qknorm=qknorm, qknorm_type=qknorm_type, dtype=dtype, device=device, operations=operations)
        else:
            self.attn = SelfAttention(dim, num_heads, qknorm=qknorm, qknorm_type=qknorm_type, dtype=dtype, device=device, operations=operations)

        self.norm_ff = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = FeedForward(dim, dtype=dtype, device=device, operations=operations)

    def forward(self, x, c=None):
        if c is not None:
            x = x + self.attn(self.norm_attn(x), self.norm_context(c))
        else:
            x = x + self.attn(self.norm_attn(x))
        x = x + self.ff(self.norm_ff(x))
        return x


class FlashQueryLayer(nn.Module):
    """No residual, mult=1 ff, no context norm."""

    def __init__(self, dim, num_heads, dim_context, qknorm=True, qknorm_type="RMSNorm", dtype=None, device=None, operations=None):
        super().__init__()
        if operations is None:
            import comfy.ops
            operations = comfy.ops.manual_cast
        self.norm_attn = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = CrossAttention(dim, num_heads, context_dim=dim_context, qknorm=qknorm, qknorm_type=qknorm_type, dtype=dtype, device=device, operations=operations)
        self.ff = FeedForward(dim, mult=1, dtype=dtype, device=device, operations=operations)

    def forward(self, x, c):
        x = self.attn(self.norm_attn(x), c)
        x = self.ff(x)
        return x


class PartPackerVAE(nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        cfg = dict(
            hidden_dim=1024,
            num_heads=16,
            dec_hidden_dim=1024,
            dec_num_heads=16,
            num_dec_layers=24,
            qknorm=True,
            qknorm_type="LayerNorm",
            salient_attn_mode="dual",
            fourier_version="v3",
            point_fourier_dim=48,
            query_hidden_dim=1024,
            query_num_heads=16,
            latent_size=4096,
            latent_dim=64,
        )
        self.config = type("Cfg", (), cfg)()
        self.precision = dtype

        self.proj_input = nn.Linear(3 + cfg["point_fourier_dim"], cfg["hidden_dim"])

        self.perceiver = AttentionBlock(cfg["hidden_dim"], cfg["num_heads"], dim_context=cfg["hidden_dim"], qknorm=cfg["qknorm"], qknorm_type=cfg["qknorm_type"])
        # salient_attn_mode == "dual"
        self.perceiver_dorases = AttentionBlock(cfg["hidden_dim"], cfg["num_heads"], dim_context=cfg["hidden_dim"], qknorm=cfg["qknorm"], qknorm_type=cfg["qknorm_type"])

        # num_enc_layers == 0 -> no self-attention encoder blocks

        self.norm_down = nn.LayerNorm(cfg["hidden_dim"])
        self.proj_down_mean = nn.Linear(cfg["hidden_dim"], cfg["latent_dim"])
        self.proj_down_std = nn.Linear(cfg["hidden_dim"], cfg["latent_dim"])
        self.proj_up = nn.Linear(cfg["latent_dim"], cfg["dec_hidden_dim"])

        self.decoder = nn.ModuleList(
            [
                AttentionBlock(cfg["dec_hidden_dim"], cfg["dec_num_heads"], qknorm=cfg["qknorm"], qknorm_type=cfg["qknorm_type"])
                for _ in range(cfg["num_dec_layers"])
            ]
        )

        self.proj_query = nn.Linear(3 + cfg["point_fourier_dim"], cfg["query_hidden_dim"])
        self.attn_query = AttentionBlock(cfg["query_hidden_dim"], cfg["query_num_heads"], dim_context=cfg["hidden_dim"], qknorm=cfg["qknorm"], qknorm_type=cfg["qknorm_type"])
        self.norm_out = nn.LayerNorm(cfg["query_hidden_dim"])
        self.proj_out = nn.Linear(cfg["query_hidden_dim"], 1)

        self.eval()
        self.requires_grad_(False)
        self.to(dtype=dtype)

    def fourier_encoding(self, points):
        # v3 (hunyuan3d-2 style): freq_band = 2 ** arange(F)
        F_ = self.config.point_fourier_dim // (2 * points.shape[-1])
        freq_band = 2 ** torch.arange(F_, device=points.device, dtype=torch.float32)
        spectrum = points.unsqueeze(-1) * freq_band  # [B,...,3,F]
        sin, cos = spectrum.sin(), spectrum.cos()
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,3,2,F]
        input_enc = input_enc.view(*points.shape[:-1], -1)  # [B,...,6F]
        return torch.cat([input_enc, points], dim=-1).to(dtype=self.precision)

    def decode(self, latent):
        latent = latent.to(dtype=self.precision)
        hidden_states = self.proj_up(latent)
        for block in self.decoder:
            hidden_states = block(hidden_states)
        return hidden_states

    def query(self, query_points, hidden_states):
        # query_points: [B, N, 3] float32; hidden_states already context-normed
        query_points = self.fourier_encoding(query_points)
        query_points = self.proj_query(query_points)
        query_output = self.attn_query(query_points, hidden_states)
        query_output = self.norm_out(query_output)
        pred = self.proj_out(query_output)  # [B, N, 1]
        return pred

    @torch.inference_mode()
    def forward(self, latent, resolution=384, min_resolution=64, max_samples_per_iter=8192):
        """Hierarchical occupancy query. latent: [B, 4096, 64] (one part).
        Returns grid_vals [1, res+1, res+1, res+1] with NaN outside queried region."""
        latent = latent.float()
        B = latent.shape[0]
        assert B == 1, "only batch 1 supported"

        hidden_states = self.decode(latent)

        resolutions = []
        res = resolution
        while res >= min_resolution:
            resolutions.append(res)
            res = res // 2
        resolutions.reverse()

        def chunked_query(grid_points):
            if grid_points.shape[0] <= max_samples_per_iter:
                return self.query(grid_points.unsqueeze(0), hidden_states).squeeze(-1)
            all_pred = []
            for i in range(0, grid_points.shape[0], max_samples_per_iter):
                grid_chunk = grid_points[i : i + max_samples_per_iter]
                pred_chunk = self.query(grid_chunk.unsqueeze(0), hidden_states)
                all_pred.append(pred_chunk)
            return torch.cat(all_pred, dim=1).squeeze(-1)

        res = resolutions[0]
        grid_points = _construct_grid_points(res).to(latent.device)
        grid_points = grid_points.contiguous().view(-1, 3)
        grid_vals = chunked_query(grid_points).float().view(res + 1, res + 1, res + 1)

        dilate_kernel_3 = torch.ones(1, 1, 3, 3, 3, dtype=torch.float32, device=latent.device)
        dilate_kernel_5 = torch.ones(1, 1, 5, 5, 5, dtype=torch.float32, device=latent.device)
        for i in range(1, len(resolutions)):
            res = resolutions[i]
            grid_signs = grid_vals >= 0
            mask = torch.zeros_like(grid_signs)
            mask[1:, :, :] += grid_signs[1:, :, :] != grid_signs[:-1, :, :]
            mask[:-1, :, :] += grid_signs[:-1, :, :] != grid_signs[1:, :, :]
            mask[:, 1:, :] += grid_signs[:, 1:, :] != grid_signs[:, :-1]
            mask[:, :-1, :] += grid_signs[:, :-1, :] != grid_signs[:, 1:]
            mask[:, :, 1:] += grid_signs[:, :, 1:] != grid_signs[:, :, :-1]
            mask[:, :, :-1] += grid_signs[:, :, :-1] != grid_signs[:, :, 1:]
            mask += grid_vals.abs() < 0.95
            mask = (mask > 0).float()
            if res < 512:
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = F.conv3d(mask, weight=dilate_kernel_3, padding=1)
                mask = mask.squeeze(0).squeeze(0)
            cidx_x, cidx_y, cidx_z = torch.nonzero(mask, as_tuple=True)
            mask_fine = torch.zeros(res + 1, res + 1, res + 1, dtype=torch.float32, device=latent.device)
            mask_fine[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            if res < 512:
                mask_fine = mask_fine.unsqueeze(0).unsqueeze(0)
                mask_fine = F.conv3d(mask_fine, weight=dilate_kernel_3, padding=1)
                mask_fine = mask_fine.squeeze(0).squeeze(0)
            else:
                mask_fine = mask_fine.unsqueeze(0).unsqueeze(0)
                mask_fine = F.conv3d(mask_fine, weight=dilate_kernel_5, padding=2)
                mask_fine = mask_fine.squeeze(0).squeeze(0)
            fidx_x, fidx_y, fidx_z = torch.nonzero(mask_fine, as_tuple=True)
            query_points = torch.stack([fidx_x, fidx_y, fidx_z], dim=-1)
            query_points = query_points * 2 / res - 1
            pred = chunked_query(query_points).float()
            grid_vals = torch.full((res + 1, res + 1, res + 1), -100.0, dtype=torch.float32, device=latent.device)
            grid_vals[fidx_x, fidx_y, fidx_z] = pred

        grid_vals = grid_vals.unsqueeze(0)
        grid_vals[grid_vals <= -100.0] = float("nan")
        return grid_vals


def _construct_grid_points(resolution, indexing="ij"):
    x = np.linspace(-1, 1, resolution + 1, dtype=np.float32)
    y = np.linspace(-1, 1, resolution + 1, dtype=np.float32)
    z = np.linspace(-1, 1, resolution + 1, dtype=np.float32)
    xs, ys, zs = np.meshgrid(x, y, z, indexing=indexing)
    xyzs = np.stack((xs, ys, zs), axis=-1)
    return torch.from_numpy(xyzs).float()
