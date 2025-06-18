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
        use_salient_point=True,
        latent_size=4096,
        cutoff_fps_point=(256, 512, 512, 512, 1024, 1024, 2048),
        cutoff_fps_salient_point=(0, 0, 256, 512, 512, 1024, 2048),
        cutoff_fps_prob=(0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.2),
        kl_weight=1e-3,
        salient_attn_mode="dual",
        num_enc_layers=0,
        num_dec_layers=24,
    )

    return model_config
