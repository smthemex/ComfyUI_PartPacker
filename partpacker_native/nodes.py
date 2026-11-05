# PartPacker native nodes (V1 style, MESH via comfy_api ComfyType bridge).
#
# Native chain:
#   LoadImage -> LoadBackgroundRemovalModel -> RemoveBackground
#     -> PartPackerImagePreprocess -> PartPackerConditioning (CLIPVisionLoader dinov2-giant)
#   PartPackerLoader -> PartPackerEmptyLatent -> KSampler (euler / normal, cfg 7.0)
#     -> PartPackerVAELoader -> PartPackerMeshDecode -> SaveGLB (native)
#
# CFG semantics identical to the official script: uncond = zero DINO tokens
# (same trick as comfy's native Hunyuan3Dv2Conditioning).

import os

import torch

import folder_paths
import comfy.model_management
import comfy.model_patcher
import comfy.utils

from . import core, conditioning

MAX_SEED = 2**32 - 1


class PartPackerLoader:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in folder_paths.get_filename_list("diffusion_models") if "partpacker" in f.lower()]
        return {
            "required": {
                "model": (files or folder_paths.get_filename_list("diffusion_models"),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "PartPacker"

    def load(self, model):
        path = folder_paths.get_full_path("diffusion_models", model)
        if path is None:
            raise ValueError(f"PartPacker flow model not found: {model}")
        patcher = core.load_flow_model(path)
        return (patcher,)


class PartPackerVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in folder_paths.get_filename_list("vae") if "partpacker" in f.lower() or f == "vae.pt"]
        return {
            "required": {
                "vae": (files or folder_paths.get_filename_list("vae"),),
            }
        }

    RETURN_TYPES = ("PARTPACKER_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "PartPacker"

    def load(self, vae):
        path = folder_paths.get_full_path("vae", vae)
        if path is None:
            raise ValueError(f"PartPacker vae not found: {vae}")
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        model = _build_vae()
        m, u = model.load_state_dict(sd, strict=True)
        load_device = comfy.model_management.get_torch_device()
        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=load_device,
            offload_device=torch.device("cpu"),
            size=sum(p.numel() * p.element_size() for p in model.parameters()),
        )
        return ({"patcher": patcher, "backend": "mcubes"},)


def _build_vae():
    from .vae_model import PartPackerVAE

    return PartPackerVAE(dtype=torch.bfloat16)


class PartPackerEmptyLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "PartPacker"

    def generate(self, batch_size):
        latent = torch.zeros(
            [batch_size, core.LATENT_SIZE * 2, core.LATENT_DIM],
            device=comfy.model_management.intermediate_device(),
        )
        return ({"samples": latent},)


class PartPackerImagePreprocess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # 1 = foreground. Feed the output of native RemoveBackground here.
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cond_image",)
    FUNCTION = "process"
    CATEGORY = "PartPacker"

    def process(self, image, mask=None):
        cond = conditioning.preprocess_cond_image(image, mask)
        return (cond,)


class PartPackerConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "cond_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "PartPacker"

    def encode(self, clip_vision, cond_image):
        tokens = conditioning.dino_tokens(clip_vision, cond_image)
        positive = [[tokens, {}]]
        negative = [[torch.zeros_like(tokens), {}]]
        return (positive, negative)


class PartPackerMeshDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("PARTPACKER_VAE",),
                "grid_res": ("INT", {"default": 384, "min": 64, "max": 512, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
            },
            "optional": {
                "backend": (["mcubes", "diso"],),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "decode"
    CATEGORY = "PartPacker"
    OUTPUT_NODE = False

    def decode(self, latent, vae, grid_res, seed, backend="mcubes"):
        samples = latent["samples"]
        if samples.shape[1] != core.LATENT_SIZE * 2:
            raise ValueError(
                f"PartPacker expects latent [B, {core.LATENT_SIZE * 2}, {core.LATENT_DIM}], got {tuple(samples.shape)}"
            )

        patcher = vae["patcher"]
        comfy.model_management.load_models_gpu([patcher])
        model = patcher.model

        # comfy returns sampler output on CPU; move to the VAE load device
        samples = samples.to(patcher.load_device)

        # official reseeds everything per repeat (kiui.seed_everything)
        torch.manual_seed(seed)

        mesh = core.decode_latent_to_mesh(
            model,
            samples,
            grid_res=grid_res,
            backend=backend,
        )
        comfy_mesh = core.mesh_to_comfy_mesh(mesh)
        return (comfy_mesh,)


NODE_CLASS_MAPPINGS = {
    "PartPackerLoader": PartPackerLoader,
    "PartPackerVAELoader": PartPackerVAELoader,
    "PartPackerEmptyLatent": PartPackerEmptyLatent,
    "PartPackerImagePreprocess": PartPackerImagePreprocess,
    "PartPackerConditioning": PartPackerConditioning,
    "PartPackerMeshDecode": PartPackerMeshDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PartPackerLoader": "PartPacker Model Loader (native)",
    "PartPackerVAELoader": "PartPacker VAE Loader",
    "PartPackerEmptyLatent": "PartPacker Empty Latent",
    "PartPackerImagePreprocess": "PartPacker Image Preprocess",
    "PartPackerConditioning": "PartPacker DINO Conditioning",
    "PartPackerMeshDecode": "PartPacker Mesh Decode",
}
