import os
from datetime import datetime

import cv2
import kiui
import numpy as np
import rembg
import torch
import trimesh

from .flow.utils import get_random_color, recenter_foreground
from .vae.utils import postprocess_mesh


MAX_SEED = np.iinfo(np.int32).max


# get random seed
def get_random_seed(randomize_seed, seed):
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    return seed


# # process image
# @spaces.GPU(duration=10)
def process_image(image,bg_remover,mask_): #cv2  RGB

    # bg removal if there is no alpha channel
    if mask_ is None:
        image = rembg.remove(image, session=bg_remover)  # [H, W, 4]
        mask = image[..., -1] > 0
        image = recenter_foreground(image, mask, border_ratio=0.1)
    else:
        image=mask_
    image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_AREA)
    return image


# # process generation
# @spaces.GPU(duration=90)
def process_3d(model,bg_remover,input_image,TRIMESH_GLB_EXPORT,mask,output_dir, num_steps=50, cfg_scale=7, grid_res=384, seed=42, simplify_mesh=False, target_num_faces=100000):

    # seed
    kiui.seed_everything(seed)
    input_image=process_image(input_image,bg_remover,mask)
    # output path
    # os.makedirs("output", exist_ok=True)
    output_glb_path=os.path.join(output_dir, f"partpacker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.glb")
    
    # input image (assume processed to RGBA uint8)
    # img_cv2=cv2.cvtColor(input_image,cv2.COLOR_RGB2BGR)
    # cv2.imwrite("output_image.png", img_cv2)
    
    image = input_image.astype(np.float32) / 255.0
    if mask is  None:
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])  # white background
   
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float().cuda()

    data = {"cond_images": image_tensor}

    with torch.inference_mode():
        results = model(data, num_steps=num_steps, cfg_scale=cfg_scale)

    latent = results["latent"]

    # query mesh

    data_part0 = {"latent": latent[:, : model.config.latent_size, :]}
    data_part1 = {"latent": latent[:, model.config.latent_size :, :]}

    with torch.inference_mode():
        results_part0 = model.vae(data_part0, resolution=grid_res)
        results_part1 = model.vae(data_part1, resolution=grid_res)

    if not simplify_mesh:
        target_num_faces = -1

    vertices, faces = results_part0["meshes"][0]
    mesh_part0 = trimesh.Trimesh(vertices, faces)
    mesh_part0.vertices = mesh_part0.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part0 = postprocess_mesh(mesh_part0, target_num_faces)
    parts = mesh_part0.split(only_watertight=False)

    vertices, faces = results_part1["meshes"][0]
    mesh_part1 = trimesh.Trimesh(vertices, faces)
    mesh_part1.vertices = mesh_part1.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part1 = postprocess_mesh(mesh_part1, target_num_faces)
    parts.extend(mesh_part1.split(only_watertight=False))

    # some parts only have 1 face, seems a problem of trimesh.split.
    parts = [part for part in parts if len(part.faces) > 10]

    # split connected components and assign different colors
    for j, part in enumerate(parts):
        # each component uses a random color
        part.visual.vertex_colors = get_random_color(j, use_float=True)

    mesh = trimesh.Scene(parts)
    # export the whole mesh
    mesh.export(output_glb_path)
    new_mesh=mesh.to_mesh() #"trimesh.Trimesh"

    return new_mesh,output_glb_path


