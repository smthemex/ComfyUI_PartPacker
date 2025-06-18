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

from typing import Literal, Optional, Tuple

import attrs


@attrs.define(slots=False)
class ModelConfig:
    # input
    use_salient_point: bool = True

    # random cutoff during training
    cutoff_fps_point: Tuple[int, ...] = (256, 512, 512, 512, 1024, 1024, 2048)
    cutoff_fps_salient_point: Tuple[int, ...] = (0, 0, 256, 512, 512, 1024, 2048)
    cutoff_fps_prob: Tuple[float, ...] = (0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.2)  # sum to 1.0

    # backbone transformer
    num_enc_layers: int = 0
    hidden_dim: int = 1024
    num_heads: int = 16
    num_dec_layers: int = 24
    dec_hidden_dim: int = 1024
    dec_num_heads: int = 16
    qknorm: bool = True
    qknorm_type: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"  # type of qknorm
    salient_attn_mode: Literal["dual_shared", "single", "dual"] = "dual"

    # query decoder
    fourier_version: Literal["v1", "v2", "v3"] = "v3"
    point_fourier_dim: int = 48  # must be divisible by 6 (sin/cos, x/y/z)
    query_hidden_dim: int = 1024
    query_num_heads: int = 16
    use_flash_query: bool = False

    # latent code
    latent_size: int = 4096  # == num_fps_point + num_fps_salient_point
    latent_dim: int = 64

    # loss
    use_ae: bool = False  # if true, variance will be ignored, and kl_weight is used as a L2 norm weight
    kl_weight: float = 1e-3

    # init weights from a pretrained checkpoint
    pretrain_path: Optional[str] = None
