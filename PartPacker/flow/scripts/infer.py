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

import cv2
import kiui
import numpy as np
import rembg
import torch
import trimesh

from ..model import Model
from ..utils import get_random_color, recenter_foreground
from ...vae.utils import postprocess_mesh

# PYTHONPATH=. python flow/scripts/infer.py
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="config file path",
    default="flow.configs.big_parts_strict_pvae",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    help="checkpoint path",
    default="pretrained/flow.pt",
)
parser.add_argument("--input", type=str, help="input directory", default="assets/images/")
parser.add_argument("--limit", type=int, help="limit number of images", default=-1)
parser.add_argument("--output_dir", type=str, help="output directory", default="output/")
parser.add_argument("--grid_res", type=int, help="grid resolution", default=384)
parser.add_argument("--num_steps", type=int, help="number of cfg steps", default=50)
parser.add_argument("--cfg_scale", type=float, help="cfg scale", default=7.0)
parser.add_argument("--num_repeats", type=int, help="number of repeats per image", default=1)
parser.add_argument("--num_faces", type=int, help="target number of faces for decimation", default=-1)
parser.add_argument("--seed", type=int, help="seed", default=42)
args = parser.parse_args()

TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)

bg_remover = rembg.new_session()


def preprocess_image(path):
    input_image = kiui.read_image(path, mode="uint8", order="RGBA")

    # bg removal if there is no alpha channel
    if input_image.shape[-1] == 3:
        input_image = rembg.remove(input_image, session=bg_remover)  # [H, W, 4]

    mask = input_image[..., -1] > 0
    image = recenter_foreground(input_image, mask, border_ratio=0.1)
    image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])  # white background
    return image


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

# output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
workspace = os.path.join(args.output_dir, "flow_" + args.config.split(".")[-1] + "_" + timestamp)
if not os.path.exists(workspace):
    os.makedirs(workspace)
else:
    os.system(f"rm {workspace}/*")
print(f"Output directory: {workspace}")

# load test images
if os.path.isdir(args.input):
    paths = glob.glob(os.path.join(args.input, "*"))
    paths = sorted(paths)
    if args.limit > 0:
        paths = paths[: args.limit]
else:  # single file
    paths = [args.input]

for path in paths:
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"Processing {name}")

    image = preprocess_image(path)

    kiui.write_image(os.path.join(workspace, name + ".jpg"), image)
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float().cuda()

    # run model
    data = {"cond_images": image}

    for i in range(args.num_repeats):

        kiui.seed_everything(args.seed + i)

        with torch.inference_mode():
            results = model(data, num_steps=args.num_steps, cfg_scale=args.cfg_scale)

        latent = results["latent"]
        # kiui.lo(latent)

        # query mesh
        if model.config.use_parts:
            data_part0 = {"latent": latent[:, : model.config.latent_size, :]}
            data_part1 = {"latent": latent[:, model.config.latent_size :, :]}

            with torch.inference_mode():
                results_part0 = model.vae(data_part0, resolution=args.grid_res)
                results_part1 = model.vae(data_part1, resolution=args.grid_res)

            vertices, faces = results_part0["meshes"][0]
            mesh_part0 = trimesh.Trimesh(vertices, faces)
            mesh_part0.vertices = mesh_part0.vertices @ TRIMESH_GLB_EXPORT.T
            mesh_part0 = postprocess_mesh(mesh_part0, args.num_faces)
            parts = mesh_part0.split(only_watertight=False)

            vertices, faces = results_part1["meshes"][0]
            mesh_part1 = trimesh.Trimesh(vertices, faces)
            mesh_part1.vertices = mesh_part1.vertices @ TRIMESH_GLB_EXPORT.T
            mesh_part1 = postprocess_mesh(mesh_part1, args.num_faces)
            parts.extend(mesh_part1.split(only_watertight=False))

            # some parts only have 1 face, seems a problem of trimesh.split.
            parts = [part for part in parts if len(part.faces) > 10]

            # split connected components and assign different colors
            for j, part in enumerate(parts):
                # each component uses a random color
                part.visual.vertex_colors = get_random_color(j, use_float=True)

            mesh = trimesh.Scene(parts)
            # export the whole mesh
            mesh.export(os.path.join(workspace, name + "_" + str(i) + ".glb"))

            # export each part
            for j, part in enumerate(parts):
                part.export(os.path.join(workspace, name + "_" + str(i) + "_part" + str(j) + ".glb"))

            # export dual volumes
            mesh_part0.export(os.path.join(workspace, name + "_" + str(i) + "_vol0.glb"))
            mesh_part1.export(os.path.join(workspace, name + "_" + str(i) + "_vol1.glb"))

        else:
            data = {"latent": latent}

            with torch.inference_mode():
                results = model.vae(data, resolution=args.grid_res)

            vertices, faces = results["meshes"][0]
            mesh = trimesh.Trimesh(vertices, faces)
            mesh = postprocess_mesh(mesh, args.num_faces)

            # kiui.lo(mesh.vertices, mesh.faces)
            mesh.vertices = mesh.vertices @ TRIMESH_GLB_EXPORT.T
            mesh.export(os.path.join(workspace, name + "_" + str(i) + ".glb"))
