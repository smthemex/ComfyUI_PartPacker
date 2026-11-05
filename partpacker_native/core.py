# PartPacker native core: comfy BaseModel wiring + mesh extraction.
#
# Sampling semantics (verified against PartPacker-main/flow/model.py + flow_matching.py):
#   official sigmas: shift*linspace(1,0,N+1)/(1+(shift-1)*linspace(1,0,N+1)), shift=3.0
#                    euler step  x -= (sigma - sigma_prev) * v,  t = 1000*sigma
#   comfy:  ModelSamplingDiscreteFlow(shift=3.0, multiplier=1000)  -> timestep(sigma)=sigma*1000
#           sigma(t) = time_snr_shift(3.0, t/1000)                 -> identical transform
#           ModelType.FLOW (CONST): calculate_denoised = x - sigma*v
#                                   noise_scaling = sigma*noise + (1-sigma)*latent
#           euler sampler: x += (sigma_next - sigma)*v             -> identical step
#   scheduler "normal" + sigma_min=0 override gives
#           timesteps = linspace(1000, 0, steps+1) -> exactly the official grid.

import logging

import numpy as np
import torch

import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.ops
import comfy.supported_models_base
import comfy.utils

from .dit import PartPackerDiT

LATENT_SIZE = 4096  # tokens per part
LATENT_DIM = 64


class LatentPartPacker(comfy.latent_formats.LatentFormat):
    # comfy's fix_empty_latent_channels treats dim 1 as "channels": our latent is
    # token-first [B, 8192, 64], so latent_channels must equal 8192 (not the
    # per-token feature dim 64) or an all-zeros empty latent gets reshaped.
    # latent_dimensions=1 avoids the 3D unsqueeze path. scale 1.0 (raw latents).
    latent_channels = LATENT_SIZE * 2
    latent_dimensions = 1
    scale_factor = 1.0


class ModelSamplingPartPacker(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    """(ModelSamplingDiscreteFlow, CONST) twin, mirroring comfy model_base's
    FLOW factory `class ModelSampling(s, c): pass`. shift=3.0 / multiplier=1000
    with sigma_min pinned to 0.0 so comfy's `normal` scheduler builds
    linspace(1000, 0, steps+1) -- exactly the official sigma grid."""

    def __init__(self, model_config=None):
        super().__init__()
        sampling_settings = model_config.sampling_settings if model_config is not None else {}
        self.set_noise_scale(sampling_settings.get("noise_scale", 1.0))
        self.set_parameters(
            shift=sampling_settings.get("shift", 3.0),
            timesteps=sampling_settings.get("timesteps", 1000),
            multiplier=sampling_settings.get("multiplier", 1000),
        )

    @property
    def sigma_min(self):
        return 0.0

    @property
    def sigma_max(self):
        return 1.0


class PartPackerFlowModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, device=None):
        super().__init__(
            model_config,
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
            unet_model=PartPackerDiT,
        )
        self.model_sampling = ModelSamplingPartPacker(model_config)

    def process_timestep(self, timestep, **kwargs):
        return timestep


class PartPackerModelConfig(comfy.supported_models_base.BASE):
    unet_config = {
        "hidden_dim": 1536,
        "num_heads": 16,
        "num_layers": 24,
        "latent_size": LATENT_SIZE,
        "latent_dim": LATENT_DIM,
        "qknorm": True,
        "qknorm_type": "RMSNorm",
        "use_pos_embed": False,
        "use_parts": True,
        "part_embed_mode": "part2_only",
    }
    unet_extra_config = {}
    sampling_settings = {"shift": 3.0, "multiplier": 1000.0}
    latent_format = LatentPartPacker
    supported_inference_dtypes = [torch.bfloat16]
    memory_usage_factor = 2.0
    vae_key_prefix = []
    clip_vision_prefix = None

    def get_model(self, state_dict, prefix="", device=None):
        return PartPackerFlowModel(self, device=device)


def load_flow_model(model_path):
    """Load PartPacker_flow.pt into a ModelPatcher over the native BaseModel.
    strict=True self-check per methodology."""
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    # strip the "dit." checkpoint prefix
    sd = {k[len("dit."):]: v for k, v in sd.items() if k.startswith("dit.")}

    model_config = PartPackerModelConfig(dict(PartPackerModelConfig.unet_config))
    parameters = comfy.utils.calculate_parameters(sd)
    load_device = comfy.model_management.get_torch_device()
    unet_dtype = comfy.model_management.unet_dtype(
        model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes
    )
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)

    model = model_config.get_model({})
    m, u = model.diffusion_model.load_state_dict(sd, strict=True)
    logging.info("[PartPacker] flow model loaded strict=True ok (%d tensors)", len(sd))

    patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
        size=parameters * comfy.model_management.dtype_size(unet_dtype),
    )
    return patcher


# ---------------------------------------------------------------------------
# mesh extraction (from PartPacker-main/vae/utils.py extract_mesh, mcubes path;
# diso is broken on this machine - DLL load failure - so it stays opt-in)
# ---------------------------------------------------------------------------

def extract_mesh(grid_vals, resolution, isosurface_level=0.0, backend="mcubes"):
    if backend == "diso":
        import diso  # may fail with DLL error on this machine
        global _diso_session
        if _diso_session is None:
            _diso_session = diso.DiffDMC(dtype=torch.float32).cuda()
        grid = -grid_vals.float().cuda()  # diso assumes inner is NEGATIVE
        verts, faces = _diso_session(grid, deform=None, normalize=True)
        verts = verts.cpu().numpy() * 2 - 1.0
        faces = faces.cpu().numpy()
        return verts, faces

    # mcubes (default)
    import mcubes
    grid = grid_vals.view(resolution + 1, resolution + 1, resolution + 1).float().cpu().numpy()
    verts, faces = mcubes.marching_cubes(grid, isosurface_level)
    verts = 2 * verts / resolution - 1.0
    return verts, faces


_diso_session = None

# official glTF axis swap: (x, y, z) -> (y, z, x)
TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)

# official get_random_color palette (30 colors, RGB uint8)
_PALETTE = np.array(
    [
        [141, 211, 199], [255, 255, 179], [190, 186, 218], [251, 128, 114],
        [128, 177, 211], [253, 180, 98], [179, 222, 105], [252, 205, 229],
        [217, 217, 217], [188, 128, 189], [204, 235, 197], [255, 237, 111],
        [102, 194, 165], [252, 141, 98], [141, 160, 203], [231, 138, 195],
        [166, 216, 84], [255, 217, 47], [229, 196, 148], [179, 179, 179],
        [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
        [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
        [153, 153, 153],
    ],
    dtype=np.float32,
) / 255.0


def decode_latent_to_mesh(vae, latent, grid_res=384, backend="mcubes", min_faces=10):
    """latent: [1, 8192, 64] from KSampler. Returns a trimesh.Trimesh with
    per-part vertex colors (connected components of part0 + part1)."""
    import trimesh

    part0 = latent[:, :LATENT_SIZE, :]
    part1 = latent[:, LATENT_SIZE:, :]

    all_meshes = []
    for part_latent in (part0, part1):
        grid_vals = vae(part_latent, resolution=grid_res)
        verts, faces = extract_mesh(grid_vals, grid_res, backend=backend)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        all_meshes.append(mesh)

    parts = []
    for mesh in all_meshes:
        parts.extend(mesh.split(only_watertight=False))
    parts = [p for p in parts if len(p.faces) > min_faces]

    for j, part in enumerate(parts):
        part.visual.vertex_colors = _PALETTE[j % len(_PALETTE)]

    merged = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    merged.vertices = merged.vertices @ TRIMESH_GLB_EXPORT.T
    return merged


def mesh_to_comfy_mesh(mesh):
    """trimesh.Trimesh (vertex-colored) -> comfy_api MESH (batch 1)."""
    from comfy_api.latest._util import MESH

    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32).unsqueeze(0)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.long).unsqueeze(0)
    colors = None
    vc = mesh.visual.vertex_colors if mesh.visual.kind == "vertex" else None
    if vc is not None:
        vc = np.asarray(vc)
        if vc.dtype == np.uint8:
            vc = vc.astype(np.float32) / 255.0
        if vc.shape[-1] == 4:  # drop alpha, gltf vertex colors are rgb here
            vc = vc[..., :3]
        colors = torch.tensor(vc, dtype=torch.float32).unsqueeze(0)
    return MESH(vertices=vertices, faces=faces, vertex_colors=colors)
