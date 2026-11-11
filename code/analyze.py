"""对齐自由度分解 + 误差分布 + 失败案例。

从 results/pred/<model>/*.npy 的缓存预测出发，做三项零成本分析：

1. 对齐自由度分解：S0 无对齐 / S1 仅尺度 / S2 仅偏移 / S3 尺度+偏移
   —— 量化每个自由度对指标的贡献
2. 逐图 AbsRel 分布（箱线图）
3. 失败案例：误差最大的 3 个场景（RGB / GT / 各模型预测）
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "nyu_eigen")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

H, W = 480, 640
CROP = (45, 471, 41, 601)
MIN_D, MAX_D = 1e-3, 10.0
NAMES = {"midas": "MiDaS", "da2": "DA-V2", "da3": "DA-V3",
         "da2l": "DA-V2-L"}


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def metrics(p, g):
    th = np.maximum(g / p, p / g)
    return {
        "absrel": float(np.mean(np.abs(g - p) / g)),
        "rmse": float(np.sqrt(np.mean((g - p) ** 2))),
        "d1": float(np.mean(th < 1.25)),
    }


def align(p, g, mode):
    """p: 预测深度; g: 真值. 返回对齐后的深度."""
    q = 1.0 / np.clip(p, 1e-6, None)      # 视差
    t = 1.0 / g
    if mode == "S0":
        out = p
    elif mode == "S1":
        out = p * (np.median(g) / np.median(p))
    elif mode == "S2":
        b = float(np.mean(t - q))          # 仅偏移
        out = 1.0 / np.clip(q + b, 1e-6, None)
    else:                                   # S3 尺度 + 偏移
        A = np.stack([q, np.ones_like(q)], axis=1)
        sol, *_ = np.linalg.lstsq(A, t, rcond=None)
        inv = sol[0] * q + sol[1]
        eps = 1e-4 * float(np.median(inv[inv > 0])) if np.any(inv > 0) else 1e-6
        out = 1.0 / np.clip(inv, eps, None)
    return np.clip(out, MIN_D, MAX_D)


def load_pairs(keys, model):
    out = []
    for k in keys:
        f = os.path.join(RES, "pred", model, k + ".npy")
        if not os.path.exists(f):
            continue
        p = cv2.resize(np.load(f), (W, H), interpolation=cv2.INTER_LINEAR)
        g = np.load(os.path.join(DATA, "depths", k + ".npy")).astype(np.float32)
        h1, h2, w1, w2 = CROP
        pp, gg = p[h1:h2, w1:w2], g[h1:h2, w1:w2]
        m = np.isfinite(gg) & (gg > MIN_D) & (gg < MAX_D) & np.isfinite(pp) & (pp > 0)
        if m.sum() < 100:
            continue
        out.append((k, pp[m].astype(np.float64), gg[m].astype(np.float64)))
    return out


def decompose(keys, models):
    rows = []
    for m in models:
        pairs = load_pairs(keys, m)
        if not pairs:
            continue
        for mode in ("S0", "S1", "S2", "S3"):
            vals = []
            for k, pp, gg in pairs:
                vals.append(metrics(align(pp, gg, mode), gg))
            rows.append({
                "model": m,
                "mode": mode,
                "absrel": float(np.mean([v["absrel"] for v in vals])),
                "rmse": float(np.mean([v["rmse"] for v in vals])),
                "d1": float(np.mean([v["d1"] for v in vals])),
                "n": len(vals),
            })
    with open(os.path.join(RES, "alignment_decomp.json"), "w", encoding="utf8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("decomp done")
    for r in rows:
        print("%-8s %-3s absrel=%.3f rmse=%.3f d1=%.3f" %
              (r["model"], r["mode"], r["absrel"], r["rmse"], r["d1"]))
    return rows


def boxplot(keys, models):
    data, labels = [], []
    for m in models:
        pairs = load_pairs(keys, m)
        if not pairs:
            continue
        vals = [metrics(align(pp, gg, "S3"), gg)["absrel"] for k, pp, gg in pairs]
        data.append(vals)
        labels.append(NAMES.get(m, m))
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    bp = ax.boxplot(data, tick_labels=labels, widths=0.55, showfliers=False,
                    patch_artist=True, medianprops=dict(color="#c0392b", lw=1.2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#d6e4f0")
        patch.set_edgecolor("#34597f")
        patch.set_linewidth(0.7)
    ax.set_ylabel("per-image AbsRel")
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for i, v in enumerate(data):
        ax.text(i + 1, np.median(v), "%.3f" % np.median(v), ha="center",
                va="bottom", fontsize=6, color="#c0392b")
    fig.tight_layout()
    out = os.path.join(FIG, "boxplot.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


def failures(keys, models, topn=3):
    """挑主模型误差最大的场景，画 RGB/GT/预测。"""
    base = "da3" if "da3" in models else models[0]
    pairs = load_pairs(keys, base)
    scored = []
    for k, pp, gg in pairs:
        scored.append((metrics(align(pp, gg, "S3"), gg)["absrel"], k))
    scored.sort(reverse=True)
    worst = [k for _, k in scored[:topn]]

    n = len(worst)
    fig, axes = plt.subplots(n, len(models) + 2, figsize=(7.2, 1.5 * n))
    if n == 1:
        axes = axes[None, :]
    for i, k in enumerate(worst):
        img = cv2.cvtColor(imread_u(os.path.join(DATA, "images", k + ".png")), cv2.COLOR_BGR2RGB)
        gt = np.load(os.path.join(DATA, "depths", k + ".npy"))
        h1, h2, w1, w2 = CROP
        g = gt[h1:h2, w1:w2]
        m = np.isfinite(g) & (g > MIN_D) & (g < MAX_D)
        vmin, vmax = np.percentile(g[m], 2), np.percentile(g[m], 98)
        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel("case %d" % (i + 1), fontsize=8)
        axes[i, 1].imshow(plt.cm.inferno(1 - np.clip((g - vmin) / (vmax - vmin), 0, 1)))
        for j, mm in enumerate(models):
            f = os.path.join(RES, "pred", mm, k + ".npy")
            if not os.path.exists(f):
                continue
            p = cv2.resize(np.load(f), (W, H), interpolation=cv2.INTER_LINEAR)
            pp = p[h1:h2, w1:w2][m]
            q = align(pp, g[m], "S3")
            full = np.zeros_like(g)
            full[m] = q
            axes[i, j + 2].imshow(plt.cm.inferno(1 - np.clip((full - vmin) / (vmax - vmin), 0, 1)))
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    titles = ["RGB", "GT"] + [NAMES.get(m, m) for m in models]
    for j, t in enumerate(titles[: len(models) + 2]):
        axes[0, j].set_title(t, fontsize=8, pad=2)
    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "failure.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out, "cases:", worst)


if __name__ == "__main__":
    with open(os.path.join(DATA, "index.txt"), encoding="utf8") as f:
        keys = [x.strip() for x in f if x.strip()]
    models = [m for m in ["midas", "da2", "da3", "da2l"]
              if os.path.isdir(os.path.join(RES, "pred", m))]
    print("models:", models, "keys:", len(keys))
    decompose(keys, models)
    boxplot(keys, models)
    failures(keys, models)
