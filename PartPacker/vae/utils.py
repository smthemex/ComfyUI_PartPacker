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

import os
from functools import wraps
from typing import Literal

import numpy as np
import torch
import trimesh
from kiui.mesh_utils import clean_mesh, decimate_mesh


# Adapted from https://github.com/Tencent/Hunyuan3D-2/blob/main/hy3dgen/shapegen/utils.py#L38
class sync_timer:
    """
    Synchronized timer to count the inference time of `nn.Module.forward` or else.
    set env var TIMER=1 to enable logging!

    Example as context manager:
    ```python
    with timer('name'):
        run()
    ```

    Example as decorator:
    ```python
    @timer('name')
    def run():
        pass
    ```
    """

    def __init__(self, name=None, flag_env="TIMER"):
        self.name = name
        self.flag_env = flag_env

    def __enter__(self):
        if os.environ.get(self.flag_env, "0") == "1":
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        if os.environ.get(self.flag_env, "0") == "1":
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                print(f"{self.name} takes {self.time} ms")

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper


@torch.no_grad()
def calculate_iou(pred: torch.Tensor, gt: torch.Tensor, target_value: int, thresh: float = 0) -> torch.Tensor:
    """Calculate the Intersection over Union (IoU) between two volumes.

    Args:
        pred (torch.Tensor): [*] continuous value between 0 and 1
        gt (torch.Tensor): [*] discrete value of 0 or 1
        target_value (int): The value to be considered as the target class

    Returns:
        torch.Tensor: IoU value
    """
    # Ensure volumes have the same shape
    assert pred.shape == gt.shape, "Volumes must have the same shape"

    # binarize
    pred_binary = pred > thresh
    gt = gt > thresh

    # Convert the volumes to boolean tensors for logical operations
    intersection = torch.logical_and(pred_binary == target_value, gt == target_value).sum().float()
    union = torch.logical_or(pred_binary == target_value, gt == target_value).sum().float()

    # Compute IoU
    iou = intersection / union if union != 0 else torch.tensor(0.0)
    return iou


@torch.no_grad()
def calculate_metrics(
    pred: torch.Tensor, gt: torch.Tensor, target_value: int = 1, thresh: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate Precision, Recall, and F1 between two volumes.

    Args:
        pred (torch.Tensor): [*] continuous value between 0 and 1
        gt (torch.Tensor): [*] discrete value of 0 or 1
        target_value (int): The value to be considered as the target class

    Returns:
        tuple: Precision, Recall, F1 values
    """
    assert pred.shape == gt.shape, f"Pred {pred.shape} and gt {gt.shape} must have the same shape"

    # Binarize prediction
    pred_binary = pred > thresh
    gt = gt > thresh

    # True Positive (TP): pred == target_value and gt == target_value
    true_positive = torch.logical_and(pred_binary == target_value, gt == target_value).sum().float()

    # False Positive (FP): pred == target_value and gt != target_value
    false_positive = torch.logical_and(pred_binary == target_value, gt != target_value).sum().float()

    # False Negative (FN): pred != target_value and gt == target_value
    false_negative = torch.logical_and(pred_binary != target_value, gt == target_value).sum().float()

    # Precision: TP / (TP + FP), best to detect False Positives
    precision = (
        true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else torch.tensor(0.0)
    )

    # Recall: TP / (TP + FN), best to detect False Negatives
    recall = (
        true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else torch.tensor(0.0)
    )

    # f1: 2 / (1 / precision + 1 / recall)
    f1 = 2 / (1 / precision + 1 / recall) if (precision != 0 and recall != 0) else torch.tensor(0.0)

    return precision, recall, f1


# Adapted from https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/distributions/distributions.py#L24
class DiagonalGaussianDistribution:
    """VAE latent"""

    def __init__(self, mean, logvar, deterministic=False):
        # mean, logvar: [B, L, D] x 2
        self.mean, self.logvar = mean, logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.mean.device, dtype=self.mean.dtype)

    def sample(self, weight: float = 1.0):
        sample = weight * torch.randn(self.mean.shape, device=self.mean.device, dtype=self.mean.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None, dims=[1, 2]):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=dims,
                )

    def nll(self, sample, dims=[1, 2]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.mean(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


class DummyLatent:
    def __init__(self, mean):
        self.mean = mean

    def sample(self, weight=0):
        # simply perturb the mean
        if weight > 0:
            noise = torch.randn_like(self.mean) * weight
        else:
            noise = 0
        return self.mean + noise

    def mode(self):
        return self.mean

    def kl(self):
        # just an l2 penalty
        return 0.5 * torch.mean(torch.pow(self.mean, 2))


def construct_grid_points(
    resolution: int,
    indexing: str = "ij",
):
    """Generate dense grid points in [-1, 1]^3.

    Args:
        resolution (int): resolution of the grid
        indexing (str, optional): indexing of the grid. Defaults to "ij".

    Returns:
        torch.Tensor: grid points (resolution + 1, resolution + 1, resolution + 1, 3), inside bbox.
    """
    x = np.linspace(-1, 1, resolution + 1, dtype=np.float32)
    y = np.linspace(-1, 1, resolution + 1, dtype=np.float32)
    z = np.linspace(-1, 1, resolution + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyzs = np.stack((xs, ys, zs), axis=-1)
    xyzs = torch.from_numpy(xyzs).float()
    return xyzs


_diso_session = None  # lazy session for reuse


@sync_timer("extract_mesh")
def extract_mesh(
    grid_vals: torch.Tensor,
    resolution: int,
    isosurface_level: float = 0,
    backend: Literal["mcubes", "diso"] = "mcubes",
):
    """Extract mesh from grid occupancy.

    Args:
        grid_vals (torch.Tensor): [resolution + 1, resolution + 1, resolution + 1], assume to be TSDF in [-1, 1] (inner is positive)
        resolution (int, optional): Grid resolution.
        isosurface_level (float, optional): Iso-surface level. Defaults to 0.
        backend (Literal["mcubes", "diso"], optional): Backend for mesh extraction. Defaults to "diso", which uses GPU and is faster.
    Returns:
        vertices (np.ndarray): [N, 3], float32, in [-1, 1]
        faces (np.ndarray): [M, 3], int32
    """

    grid_vals = grid_vals.view(resolution + 1, resolution + 1, resolution + 1)

    if backend == "mcubes":
        try:
            import mcubes
        except ImportError:
            os.system("pip install pymcubes")
            import mcubes
        grid_vals = grid_vals.float().cpu().numpy()
        verts, faces = mcubes.marching_cubes(grid_vals, isosurface_level)
        verts = 2 * verts / resolution - 1.0  # normalize to [-1, 1]
    elif backend == "diso":
        try:
            import diso
        except ImportError:
            os.system("pip install diso")
            import diso
        global _diso_session
        if _diso_session is None:
            _diso_session = diso.DiffDMC(dtype=torch.float32).cuda()

        grid_vals = -grid_vals.float().cuda()  # diso assumes inner is NEGATIVE!
        verts, faces = _diso_session(grid_vals, deform=None, normalize=True)  # verts in [0, 1]
        verts = verts.cpu().numpy() * 2 - 1.0  # normalize to [-1, 1]
        faces = faces.cpu().numpy()

    return verts, faces


@sync_timer("postprocess_mesh")
def postprocess_mesh(mesh: trimesh.Trimesh, decimate_target=100000):
    vertices = mesh.vertices
    triangles = mesh.faces

    if vertices.shape[0] > 0 and triangles.shape[0] > 0:
        vertices, triangles = clean_mesh(vertices, triangles, remesh=False, min_f=25, min_d=5)
    if decimate_target > 0 and triangles.shape[0] > decimate_target:
        vertices, triangles = decimate_mesh(vertices, triangles, decimate_target, optimalplacement=False)
        if vertices.shape[0] > 0 and triangles.shape[0] > 0:
            vertices, triangles = clean_mesh(vertices, triangles, remesh=False, min_f=25, min_d=5)

    mesh.vertices = vertices
    mesh.faces = triangles

    return mesh


def sphere_normalize(vertices):
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    radius = np.linalg.norm(vertices - bcenter, axis=-1).max()
    vertices = (vertices - bcenter) / radius  # to [-1, 1]
    return vertices


def box_normalize(vertices, bound=0.95):
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    vertices = bound * (vertices - bcenter) / (bmax - bmin).max()
    return vertices
