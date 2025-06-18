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

import argparse
import glob
import importlib
import os
from datetime import datetime

import fpsample
import kiui
import meshiki
import numpy as np
import torch
import trimesh

from ..model import Model
from ..utils import box_normalize, postprocess_mesh, sphere_normalize, sync_timer

# PYTHONPATH=. python vae/scripts/infer.py
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="config file path", default="vae.configs.part_woenc")
parser.add_argument(
    "--ckpt_path",
    type=str,
    help="checkpoint path",
    default="pretrained/vae.pt",
)
parser.add_argument("--input", type=str, help="input directory", default="assets/meshes/")
parser.add_argument("--output_dir", type=str, help="output directory", default="output/")
parser.add_argument("--limit", type=int, help="how many samples to test", default=-1)
parser.add_argument("--num_fps_point", type=int, help="number of fps points", default=1024)
parser.add_argument("--num_fps_salient_point", type=int, help="number of fps salient points", default=1024)
parser.add_argument("--grid_res", type=int, help="grid resolution", default=512)
parser.add_argument("--seed", type=int, help="seed", default=42)
args = parser.parse_args()


TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)

kiui.seed_everything(args.seed)


@sync_timer("prepare_input_from_mesh")
def prepare_input_from_mesh(mesh_path, use_salient_point=True, num_fps_point=1024, num_fps_salient_point=1024):
    # load mesh, assume it's already processed to be watertight.

    mesh_name = mesh_path.split("/")[-1].split(".")[0]
    vertices, faces = meshiki.load_mesh(mesh_path)

    # vertices = sphere_normalize(vertices)
    vertices = box_normalize(vertices)

    mesh = meshiki.Mesh(vertices, faces)

    uniform_surface_points = mesh.uniform_point_sample(200000)
    uniform_surface_points = meshiki.fps(uniform_surface_points, 32768)  # hardcoded...
    salient_surface_points = mesh.salient_point_sample(16384, thresh_bihedral=15)

    # save points
    # trimesh.PointCloud(vertices=uniform_surface_points).export(os.path.join(workspace, mesh_name + "_uniform.ply"))
    # trimesh.PointCloud(vertices=salient_surface_points).export(os.path.join(workspace, mesh_name + "_salient.ply"))

    sample = {}

    sample["pointcloud"] = torch.from_numpy(uniform_surface_points)

    # fps subsample
    fps_indices = fpsample.bucket_fps_kdline_sampling(uniform_surface_points, num_fps_point, h=5, start_idx=0)
    sample["fps_indices"] = torch.from_numpy(fps_indices).long()  # [num_fps_point,]

    if use_salient_point:
        sample["pointcloud_dorases"] = torch.from_numpy(salient_surface_points)  # [N', 3]

        # fps subsample
        fps_indices_dorases = fpsample.bucket_fps_kdline_sampling(
            salient_surface_points, num_fps_salient_point, h=5, start_idx=0
        )
        sample["fps_indices_dorases"] = torch.from_numpy(fps_indices_dorases).long()  # [num_fps_point,]

    return sample


print(f"Loading checkpoint from {args.ckpt_path}")
ckpt_dict = torch.load(args.ckpt_path, weights_only=True)

# delete all keys other than model
if "model" in ckpt_dict:
    ckpt_dict = ckpt_dict["model"]

# instantiate model
print(f"Instantiating model from {args.config}")
model_config = importlib.import_module(args.config).make_config()
model = Model(model_config).eval().cuda().bfloat16()

# load weight
print(f"Loading weights from {args.ckpt_path}")
model.load_state_dict(ckpt_dict, strict=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
workspace = os.path.join(args.output_dir, "vae_" + args.config.split(".")[-1] + "_" + timestamp)
if not os.path.exists(workspace):
    os.makedirs(workspace)
else:
    os.system(f"rm {workspace}/*")
print(f"Output directory: {workspace}")

# load dataset
mesh_list = glob.glob(os.path.join(args.input, "*"))
mesh_list = mesh_list[: args.limit] if args.limit > 0 else mesh_list

for i, mesh_path in enumerate(mesh_list):
    print(f"Processing {i}/{len(mesh_list)}: {mesh_path}")

    mesh_name = mesh_path.split("/")[-1].split(".")[0]

    sample = prepare_input_from_mesh(
        mesh_path, num_fps_point=args.num_fps_point, num_fps_salient_point=args.num_fps_salient_point
    )
    for k in sample:
        sample[k] = sample[k].unsqueeze(0).cuda()

    # call vae
    with torch.inference_mode():
        output = model(sample, resolution=args.grid_res)

    latent = output["latent"]
    vertices, faces = output["meshes"][0]

    mesh = trimesh.Trimesh(vertices, faces)
    mesh = postprocess_mesh(mesh, 5e5)

    mesh.export(f"{workspace}/{mesh_name}.glb")
