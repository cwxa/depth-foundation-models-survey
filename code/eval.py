"""Evaluation on NYU-Depth V2 (Eigen test split, 654 images).

Protocol
--------
* prediction resized to the 640x480 ground-truth resolution
* Eigen crop  rows 45:471, cols 41:601
* valid mask : 1e-3 < gt < 10 m, gt finite, pred finite & positive
* scale alignment : two protocols, both reported
    P1  median scaling, s = median(gt) / median(pred)          -- one DoF
    P2  least-squares affine fit in inverse-depth space        -- two DoF
  All models are affine-invariant, so P2 (scale + shift) is the protocol that
  matches the official evaluation scripts; P1 is kept as a sensitivity check.

Outputs
-------
results/per_image_<model>.csv    per-image metrics
results/summary.json             aggregated table
results/pred/<model>/<key>.npy   cached predictions (float32 depth, 640x480)
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch

import models as M
from metrics import CROP, MAX_D, MIN_D, eigen_metrics


def imread_u(path):
    """cv2.imread cannot handle non-ASCII paths on Windows."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "nyu_eigen")
RES = os.path.join(ROOT, "results")

H, W = 480, 640


def load_index():
    with open(os.path.join(DATA, "index.txt"), encoding="utf8") as f:
        return [x.strip() for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(M.REGISTRY))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    pred_dir = os.path.join(RES, "pred", args.model)
    os.makedirs(pred_dir, exist_ok=True)

    keys = load_index()
    if args.limit:
        keys = keys[: args.limit]

    t0 = time.time()
    model = M.build(args.model, device=args.device)
    load_s = time.time() - t0
    print("[%s] loaded in %.1fs, %d images" % (args.model, load_s, len(keys)), flush=True)

    rows = []
    for i, k in enumerate(keys, 1):
        img = imread_u(os.path.join(DATA, "images", k + ".png"))
        gt = np.load(os.path.join(DATA, "depths", k + ".npy")).astype(np.float32)
        t = time.time()
        pred = model.predict(img)
        dt = time.time() - t
        np.save(os.path.join(pred_dir, k + ".npy"), pred.astype(np.float32))
        pred_full = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
        m, n = eigen_metrics(pred_full, gt)
        if m is None:
            print("skip", k, "valid", n)
            continue
        row = {"key": k, "n_valid": n, "time_s": dt, "scale": m["p2"]["scale"]}
        for tag in ("p1", "p2"):
            for mk, mv in m[tag].items():
                row["%s_%s" % (tag, mk)] = mv
        rows.append(row)
        if i % 50 == 0:
            print("  %d/%d  absrel_p2=%.4f  absrel_p1=%.4f  %.2fs/img"
                  % (i, len(keys), m["p2"]["absrel"], m["p1"]["absrel"], dt), flush=True)

    if not rows:
        raise SystemExit("no valid samples")

    base = ["absrel", "sqrel", "rmse", "rmse_log", "log10", "d1", "d2", "d3"]
    keyset = ["p1_" + k for k in base] + ["p2_" + k for k in base]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keyset}
    summary["n"] = len(rows)
    summary["time_s_per_image"] = float(np.mean([r["time_s"] for r in rows]))
    summary["model"] = M.REGISTRY[args.model].name

    with open(os.path.join(RES, "per_image_%s.csv" % args.model), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["key"] + keyset + ["n_valid", "time_s", "scale", "p1_scale", "p2_scale"])
        w.writeheader()
        w.writerows(rows)

    path = os.path.join(RES, "summary.json")
    allsum = {}
    if os.path.exists(path):
        with open(path, encoding="utf8") as f:
            allsum = json.load(f)
    allsum[args.model] = summary
    with open(path, "w", encoding="utf8") as f:
        json.dump(allsum, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
