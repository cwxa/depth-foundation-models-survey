"""Figures: qualitative comparison grid + accuracy/efficiency trade-off plot."""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def imread_u(path):
    """cv2.imread cannot handle non-ASCII paths on Windows."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "nyu_eigen")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})
CROP = (45, 471, 41, 601)


def colorize(d, vmin, vmax):
    d = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return (plt.cm.inferno(1.0 - d)[..., :3] * 255).astype(np.uint8)


def qualitative(keys, models):
    n = len(keys)
    fig, axes = plt.subplots(n, len(models) + 2, figsize=(7.0, 1.35 * n))
    for i, k in enumerate(keys):
        img = cv2.cvtColor(imread_u(os.path.join(DATA, "images", k + ".png")), cv2.COLOR_BGR2RGB)
        gt = np.load(os.path.join(DATA, "depths", k + ".npy"))
        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel("scene %d" % (i + 1), fontsize=8)
        h1, h2, w1, w2 = CROP
        g = gt[h1:h2, w1:w2]
        m = np.isfinite(g) & (g > 1e-3) & (g < 10)
        vmin, vmax = np.percentile(g[m], 2), np.percentile(g[m], 98)
        axes[i, 1].imshow(colorize(gt, vmin, vmax))
        for j, name in enumerate(models):
            p = np.load(os.path.join(RES, "pred", name, k + ".npy"))
            p = cv2.resize(p, (gt.shape[1], gt.shape[0]))
            s = np.median(g[m]) / np.median(p[h1:h2, w1:w2][m])
            axes[i, j + 2].imshow(colorize(p * s, vmin, vmax))
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    titles = ["RGB", "GT"] + [m.upper() for m in models]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=8, pad=2)
    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "qualitative.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


def tradeoff(models):
    with open(os.path.join(RES, "summary.json"), encoding="utf8") as f:
        summ = json.load(f)
    with open(os.path.join(RES, "efficiency.json"), encoding="utf8") as f:
        eff = json.load(f)

    fig, ax1 = plt.subplots(figsize=(3.4, 2.7))
    labels = ["MiDaS", "DA-V2", "DA-V3"]
    xs = np.arange(len(models))
    absrel = [summ[m]["p2_absrel"] for m in models]
    rmse = [summ[m]["p2_rmse"] for m in models]
    ax1.bar(xs - 0.18, absrel, width=0.36, label="AbsRel", color="#c0392b")
    ax1.bar(xs + 0.18, rmse, width=0.36, label="RMSE", color="#2c7fb8")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels[: len(models)])
    ax1.set_xlabel("model")
    ax1.set_ylabel("error (lower is better)")
    ax1.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax2 = ax1.twinx()
    fps = [eff[m]["fps"] for m in models]
    ax2.plot(xs, fps, "o--", color="#27ae60", ms=4, lw=1, label="FPS")
    ax2.set_ylabel("FPS @ RTX 3060")
    ax2.tick_params(labelsize=8)
    ax1.tick_params(labelsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, "tradeoff.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    names = ["midas", "da2", "da3"]
    keys = ["00001", "00015", "00045", "00105", "00200", "00300"]
    keys = [k for k in keys if os.path.exists(os.path.join(RES, "pred", "da3", k + ".npy"))]
    if keys:
        qualitative(keys, names)
    tradeoff(names)
