"""Unified wrappers for the three monocular depth estimation models.

Every wrapper returns a float32 depth map (H, W) aligned to the *input image*
resolution, with the convention  larger value = farther from camera.

    MiDaS 3.0 DPT-Hybrid : predicts inverse depth (disparity) -> depth = 1 / d
    Depth Anything V2    : predicts inverse depth (disparity) -> depth = 1 / d
    Depth Anything 3     : predicts depth directly
"""
import os
import sys
import types

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOS = os.path.join(ROOT, "repos")
WEIGHTS = os.path.join(ROOT, "weights")

DA2_CFG = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _resize_to(depth, hw):
    h, w = hw
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth


def _disp_to_depth(disp):
    """Inverse depth -> depth.  The epsilon is relative to the median disparity,
    so bicubic overshoot (slightly negative values at image borders) cannot
    produce spikes that would corrupt the scale estimate."""
    disp = disp.astype(np.float32)
    pos = disp[np.isfinite(disp) & (disp > 0)]
    eps = max(1e-6, 1e-4 * float(np.median(pos))) if pos.size else 1e-6
    return 1.0 / np.clip(disp, eps, None)


class MiDaS:
    """MiDaS v3.0 DPT-Hybrid (BEiT/ViT-hybrid encoder + DPT decoder)."""

    name = "MiDaS-DPT-Hybrid"
    short = "midas"
    input_size = 384
    backbone = "vitb_rn50_384"

    def __init__(self, device="cuda"):
        sys.path.insert(0, os.path.join(REPOS, "MiDaS"))
        from midas.dpt_depth import DPTDepthModel
        from midas.transforms import NormalizeImage, PrepareForNet, Resize
        from torchvision.transforms import Compose

        self.device = device
        ckpt = os.path.join(WEIGHTS, "dpt_hybrid_384.pt")
        self.model = DPTDepthModel(
            path=None,
            backbone=self.backbone,
            non_negative=True,
            invert=False,     # checkpoint outputs disparity; inverted below
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.transform = Compose(
            [
                Resize(
                    self.input_size,
                    self.input_size,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=mean, std=std),
                PrepareForNet(),
            ]
        )

    @torch.no_grad()
    def predict(self, bgr):
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = self.transform({"image": img})["image"][None]
        x = torch.from_numpy(x).to(self.device)
        disp = self.model(x)
        disp = torch.nn.functional.interpolate(
            disp[:, None], size=bgr.shape[:2], mode="bicubic", align_corners=False
        )[0, 0]
        return _disp_to_depth(disp.cpu().numpy())


class DepthAnythingV2:
    """Depth Anything V2, relative-depth models (DINOv2 encoder + DPT head)."""

    name = "Depth-Anything-V2-Base"
    short = "da2"
    input_size = 518

    def __init__(self, device="cuda", encoder="vitb", half=False, input_size=None):
        sys.path.insert(0, os.path.join(REPOS, "Depth-Anything-V2"))
        from depth_anything_v2.dpt import DepthAnythingV2 as _Net

        self.device = device
        self.half = half
        if input_size:
            self.input_size = input_size
        self.model = _Net(**DA2_CFG[encoder])
        ckpt = os.path.join(WEIGHTS, "depth_anything_v2_%s.pth" % encoder)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, bgr):
        if self.half:
            with torch.autocast("cuda", dtype=torch.float16):
                disp = self.model.infer_image(bgr, self.input_size)
        else:
            disp = self.model.infer_image(bgr, self.input_size)
        return _disp_to_depth(_resize_to(disp.astype(np.float32), bgr.shape[:2]))


def _stub_optional_modules():
    """DA3 imports 3D-export modules at package import time (pycolmap / gsplat / evo).

    Only the depth head is exercised here, so those optional back-ends are stubbed.
    """
    for mod, attrs in {
        "pycolmap": ["Reconstruction", "Image", "Camera", "Point3D"],
        "gsplat": ["rasterization"],
        "evo": [],
        "evo.core": [],
        "evo.core.trajectory": ["PosePath3D"],
    }.items():
        if mod in sys.modules:
            continue
        try:
            __import__(mod)
        except Exception:
            m = types.ModuleType(mod)
            for a in attrs:
                setattr(m, a, type(a, (object,), {}))
            sys.modules[mod] = m


class DepthAnythingV3:
    """Depth Anything 3 (unified depth-ray representation, any-view geometry)."""

    name = "Depth-Anything-V3-Base"
    short = "da3"

    def __init__(self, device="cuda", repo="da3-base"):
        _stub_optional_modules()
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        sys.path.insert(0, os.path.join(REPOS, "Depth-Anything-3", "src"))
        from depth_anything_3.api import DepthAnything3 as _Net

        self.device = device
        self.model = _Net.from_pretrained("depth-anything/%s" % repo).to(device).eval()

    @torch.no_grad()
    def predict(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = self.model.inference([rgb], process_res=504)
        depth = np.asarray(pred.depth)[0].astype(np.float32)
        return _resize_to(depth, bgr.shape[:2])


class DepthAnythingV2Small(DepthAnythingV2):
    """Same family, one size down: the capacity control."""

    name = "Depth-Anything-V2-Small"
    short = "da2s"

    def __init__(self, device="cuda"):
        super().__init__(device=device, encoder="vits")


class DepthAnythingV2Large(DepthAnythingV2):
    """Same family, one size up: used as the capacity ablation."""

    name = "Depth-Anything-V2-Large"
    short = "da2l"

    def __init__(self, device="cuda"):
        super().__init__(device=device, encoder="vitl", half=True)


REGISTRY = {
    "midas": MiDaS,
    "da2": DepthAnythingV2,
    "da2s": DepthAnythingV2Small,
    "da2l": DepthAnythingV2Large,
    "da3": DepthAnythingV3,
}


def build(key, device="cuda"):
    return REGISTRY[key](device=device)
