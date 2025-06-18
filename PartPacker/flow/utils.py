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

from typing import Optional

import cv2
import numpy as np


def recenter_foreground(image, mask, border_ratio: float = 0.1):
    """recenter an image to leave some empty space at the image border.

    Args:
        image (ndarray): input image, float/uint8 [H, W, 3/4]
        mask (ndarray): alpha mask, bool [H, W]
        border_ratio (float, optional): border ratio, image will be resized to (1 - border_ratio). Defaults to 0.1.

    Returns:
        ndarray: output image, float/uint8 [H, W, 3/4]
    """

    # empty foreground: just return
    if mask.sum() == 0:
        return image

    return_int = False
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
        return_int = True

    H, W, C = image.shape
    size = max(H, W)

    # default to white bg if rgb, but use 0 if rgba
    if C == 3:
        result = np.ones((size, size, C), dtype=np.float32)
    else:
        result = np.zeros((size, size, C), dtype=np.float32)

    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2
    result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
        image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA
    )

    if return_int:
        result = (result * 255).astype(np.uint8)

    return result


def get_random_color(index: Optional[int] = None, use_float: bool = False):
    # some pleasing colors
    # matplotlib.colormaps['Set3'].colors + matplotlib.colormaps['Set2'].colors + matplotlib.colormaps['Set1'].colors
    palette = np.array(
        [
            [141, 211, 199, 255],
            [255, 255, 179, 255],
            [190, 186, 218, 255],
            [251, 128, 114, 255],
            [128, 177, 211, 255],
            [253, 180, 98, 255],
            [179, 222, 105, 255],
            [252, 205, 229, 255],
            [217, 217, 217, 255],
            [188, 128, 189, 255],
            [204, 235, 197, 255],
            [255, 237, 111, 255],
            [102, 194, 165, 255],
            [252, 141, 98, 255],
            [141, 160, 203, 255],
            [231, 138, 195, 255],
            [166, 216, 84, 255],
            [255, 217, 47, 255],
            [229, 196, 148, 255],
            [179, 179, 179, 255],
            [228, 26, 28, 255],
            [55, 126, 184, 255],
            [77, 175, 74, 255],
            [152, 78, 163, 255],
            [255, 127, 0, 255],
            [255, 255, 51, 255],
            [166, 86, 40, 255],
            [247, 129, 191, 255],
            [153, 153, 153, 255],
        ],
        dtype=np.uint8,
    )

    if index is None:
        index = np.random.randint(0, len(palette))

    if index >= len(palette):
        index = index % len(palette)

    if use_float:
        return palette[index].astype(np.float32) / 255
    else:
        return palette[index]
