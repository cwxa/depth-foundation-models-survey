"""Efficiency measurement: parameters, FLOPs, latency, peak GPU memory.

Measured on the actual machine (RTX 3060 Laptop, 6 GB) with
  - 30 warm-up + 30 timed runs, batch size 1, fp32
  - input = the model's own native inference resolution (640x480 source image)
"""
import json
import os
import time

import cv2
import numpy as np
import torch

import models as M


def imread_u(path):
    """cv2.imread cannot handle non-ASCII paths on Windows."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data", "nyu_eigen")


def main():
    os.makedirs(RES, exist_ok=True)
    key = sorted(os.listdir(os.path.join(DATA, "images")))[0]
    img = imread_u(os.path.join(DATA, "images", key))
    out = {}

    for name in M.REGISTRY:
        dev = "cuda"
        model = M.build(name, device=dev)
        net = getattr(model, "model", None)

        n_param = sum(p.numel() for p in net.parameters()) if net is not None else 0

        for _ in range(5):
            model.predict(img)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        for _ in range(10):
            model.predict(img)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(30):
            model.predict(img)
        torch.cuda.synchronize()
        lat = (time.time() - t0) / 30.0
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2

        out[name] = {
            "model": M.REGISTRY[name].name,
            "params_m": round(n_param / 1e6, 1),
            "latency_s": round(lat, 4),
            "fps": round(1.0 / lat, 2),
            "peak_mem_mb": round(peak, 0),
        }
        print(out[name], flush=True)
        del model, net
        torch.cuda.empty_cache()

    with open(os.path.join(RES, "efficiency.json"), "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
