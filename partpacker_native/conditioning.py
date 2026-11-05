# DINOv2 conditioning for PartPacker.
#
# The official flow model hacks `dino.layernorm = nn.Identity()` and then applies
# an affine-free F.layer_norm on the raw encoder output. comfy's Dinov2Model
# applies its affine layernorm inside forward(), so we cannot use the output of
# the native CLIPVisionEncode node directly. Instead we load the model through
# the NATIVE CLIPVisionLoader (the original facebook/dinov2-giant HF-format
# safetensors loads as-is: comfy's Dino2AttentionBlock nests BertAttention so
# the keys are attention.attention.query.*, and the giant's SwiGLU FFN matches
# comfy's SwiGLUFFN weights_in/weights_out naming; verified strict=True) and
# call the submodules directly to obtain raw pre-layernorm tokens, then
# replicate the official norm.

import torch
import torch.nn.functional as F

import comfy.model_management

# official preprocess_cond_image: Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406])
_DINO_STD = torch.tensor([0.229, 0.224, 0.225])


@torch.inference_mode()
def dino_tokens(clip_vision, image):
    """clip_vision: native CLIPVisionLoader output (ClipVisionModel).
    image: comfy IMAGE [B, H, W, 3] in [0,1], already white-composited at 518x518.
    Returns cond tokens [B, L, 1536] in model dtype (official semantics)."""
    comfy.model_management.load_model_gpu(clip_vision.patcher)
    model = clip_vision.model
    load_device = clip_vision.load_device

    pixel = image.to(load_device).movedim(-1, 1).float()  # [B,3,H,W]
    pixel = (pixel - _DINO_MEAN.view(1, 3, 1, 1).to(load_device)) / _DINO_STD.view(1, 3, 1, 1).to(load_device)

    dtype = model.layernorm.weight.dtype
    x = model.embeddings(pixel.to(dtype))
    x, _ = model.encoder(x, intermediate_output=None)  # raw output, layernorm skipped
    # official: F.layer_norm(cond.float(), cond.shape[-1:]) -> affine-free, eps=1e-5
    cond = F.layer_norm(x.float(), (x.shape[-1],)).to(dtype)
    return cond


def recenter_foreground(image_rgba, mask, border_ratio=0.1):
    """torch port of PartPacker flow/utils.py recenter_foreground.
    image_rgba: [H, W, 4] float in [0,1]; mask: [H, W] bool/float (1=foreground).
    Returns [size, size, 4] float."""
    if mask.sum() == 0:
        return image_rgba

    H, W, C = image_rgba.shape
    size = max(H, W)
    # rgba -> zero background canvas (official uses zeros for rgba)
    result = torch.zeros((size, size, C), dtype=image_rgba.dtype, device=image_rgba.device)

    coords = torch.nonzero(mask, as_tuple=False)
    x_min, x_max = int(coords[:, 0].min()), int(coords[:, 0].max())
    y_min, y_max = int(coords[:, 1].min()), int(coords[:, 1].max())
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    y2_min = (size - w2) // 2

    crop = image_rgba[x_min:x_max, y_min:y_max].permute(2, 0, 1).unsqueeze(0)  # [1,4,h,w]
    crop = F.interpolate(crop, size=(h2, w2), mode="area").squeeze(0).permute(1, 2, 0)
    result[x2_min : x2_min + h2, y2_min : y2_min + w2] = crop
    return result


def preprocess_cond_image(image, mask=None, resolution=518):
    """Official infer.py preprocess:
    RGBA -> recenter (border 0.1) -> resize 518 -> composite on white.
    image: comfy IMAGE [B,H,W,3] (or 4ch) float [0,1]; mask: [B,H,W]/[H,W] 1=fg.
    Returns IMAGE [1, 518, 518, 3]."""
    img = image[0].float()
    if img.shape[-1] == 4:
        rgba = img
        alpha = rgba[..., 3:4]
    else:
        rgba = img
        if mask is None:
            alpha = torch.ones(img.shape[0], img.shape[1], 1, device=img.device)
        else:
            m = mask
            if m.ndim == 3:
                m = m[0]
            alpha = m.unsqueeze(-1)
        rgba = torch.cat([rgba, alpha], dim=-1)

    rgba = recenter_foreground(rgba, alpha[..., 0] > 0, border_ratio=0.1)
    rgba = rgba.permute(2, 0, 1).unsqueeze(0)  # [1,4,size,size]
    rgba = F.interpolate(rgba, size=(resolution, resolution), mode="bilinear", align_corners=False)
    rgba = rgba.squeeze(0).permute(1, 2, 0)  # [size,size,4]

    rgb = rgba[..., :3] * rgba[..., 3:4] + (1 - rgba[..., 3:4])  # white background
    return rgb.unsqueeze(0).clamp(0, 1)
