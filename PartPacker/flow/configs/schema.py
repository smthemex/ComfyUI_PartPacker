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

from typing import Literal, Optional

import attrs


@attrs.define(slots=False)
class ModelConfig:
    # vae
    vae_conf: str = "vae.configs.part_woenc"
    vae_ckpt_path: Optional[str] = None

    # learn & generate parts
    use_parts: bool = False
    part_embed_mode: Literal["element", "part", "part2_only"] = "part2_only"
    shuffle_parts: bool = False
    use_num_parts_cond: bool = False

    # flow matching hyper-params
    flow_shift: float = 1.0
    logitnorm_mean: float = 0.0
    logitnorm_std: float = 1.0

    # image encoder
    dino_model: Literal["dinov2_vitl14_reg", "dinov2_vitg14"] = "dinov2_vitg14"

    # backbone DiT
    hidden_dim: int = 1536
    num_heads: int = 16
    num_layers: int = 24
    qknorm: bool = True
    qknorm_type: Literal["LayerNorm", "RMSNorm"] = "RMSNorm"
    use_pos_embed: bool = False

    # latent code
    latent_size: Optional[int] = None  # if None, will load from vae
    latent_dim: Optional[int] = None

    # preload vae weights
    preload_vae: bool = True

    # preload dinov2 weights
    preload_dinov2: bool = True

    # init weights from a pretrained checkpoint
    pretrain_path: Optional[str] = None
