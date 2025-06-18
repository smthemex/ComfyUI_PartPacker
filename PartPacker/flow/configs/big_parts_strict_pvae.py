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

from .schema import ModelConfig


def make_config():

    model_config = ModelConfig(
        vae_conf="vae.configs.part_woenc",
        vae_ckpt_path="pretrained/vae.pt",
        qknorm=True,
        qknorm_type="RMSNorm",
        use_pos_embed=False,
        dino_model="dinov2_vitg14",
        hidden_dim=1536,
        flow_shift=3.0,
        logitnorm_mean=1.0,
        logitnorm_std=1.0,
        latent_size=4096,
        use_parts=True,
    )

    return model_config
