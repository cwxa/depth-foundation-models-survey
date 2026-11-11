"""在同一批图像上比较 DA-V2-Small 与 DA-V2-Base（容量对照）。

利用已缓存的预测，不需要重新推理：
- pred/da2s 只有前 306 张（原评测被中止）
- pred/da2 有全量 654 张
取两者共有的 key，在完全相同的图像子集上比较，保证受控。
"""
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "nyu_eigen")
RES = os.path.join(ROOT, "results")
H, W = 480, 640
CROP = (45, 471, 41, 601)
MIN_D, MAX_D = 1e-3, 10.0


def metrics(p, g):
    th = np.maximum(g / p, p / g)
    return (float(np.mean(np.abs(g - p) / g)),
            float(np.sqrt(np.mean((g - p) ** 2))),
            float(np.mean(th < 1.25)))


def eval_subset(model, keys):
    vals = []
    for k in keys:
        f = os.path.join(RES, "pred", model, k + ".npy")
        p = cv2.resize(np.load(f), (W, H), interpolation=cv2.INTER_LINEAR)
        g = np.load(os.path.join(DATA, "depths", k + ".npy")).astype(np.float32)
        h1, h2, w1, w2 = CROP
        pp, gg = p[h1:h2, w1:w2], g[h1:h2, w1:w2]
        m = np.isfinite(gg) & (gg > MIN_D) & (gg < MAX_D) & np.isfinite(pp) & (pp > 0)
        pp, gg = pp[m].astype(np.float64), gg[m].astype(np.float64)
        q = 1.0 / np.clip(pp, 1e-6, None)
        t = 1.0 / gg
        A = np.stack([q, np.ones_like(q)], axis=1)
        sol, *_ = np.linalg.lstsq(A, t, rcond=None)
        inv = sol[0] * q + sol[1]
        eps = 1e-4 * float(np.median(inv[inv > 0])) if np.any(inv > 0) else 1e-6
        depth = np.clip(1.0 / np.clip(inv, eps, None), MIN_D, MAX_D)
        vals.append(metrics(depth, gg))
    return tuple(float(np.mean([v[i] for v in vals])) for i in range(3))


def main():
    small = set(os.path.splitext(f)[0] for f in os.listdir(os.path.join(RES, "pred", "da2s")))
    keys = sorted(small)
    print("da2s 缓存:", len(keys), "张")

    out = {}
    for m in ["da2s", "da2", "midas", "da3"]:
        d = os.path.join(RES, "pred", m)
        if not os.path.isdir(d):
            continue
        keys_m = [k for k in keys if os.path.exists(os.path.join(d, k + ".npy"))]
        if len(keys_m) < len(keys):
            print("  跳过", m, "缺", len(keys) - len(keys_m))
            continue
        a, r, d1 = eval_subset(m, keys)
        out[m] = {"n": len(keys), "absrel": a, "rmse": r, "d1": d1}
        print("%-6s n=%d  AbsRel=%.4f  RMSE=%.4f  d1=%.4f" % (m, len(keys), a, r, d1))

    with open(os.path.join(RES, "subset_capacity.json"), "w", encoding="utf8") as f:
        json.dump({"n": len(keys), "subset_first": keys[0], "subset_last": keys[-1],
                   "results": out}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
