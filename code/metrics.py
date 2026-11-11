"""深度评测的纯数值部分：裁剪、掩码、两种对齐协议、八项指标。

只依赖 numpy —— 不导入 torch / cv2，因此可以在任何环境（含 CI、无 GPU 机器）
中运行单元测试。`eval.py` 从这里导入，避免实现与测试各写一份。
"""
import numpy as np

# Eigen 评测区域（行 45:471，列 41:601）与有效深度区间（米）
CROP = (45, 471, 41, 601)
MIN_D, MAX_D = 1e-3, 10.0

METRIC_KEYS = ("absrel", "sqrel", "rmse", "rmse_log", "log10", "d1", "d2", "d3")


def align_median(p, g):
    """协议 P1：单一全局尺度，由中位数估计。

    s = median(g) / median(p)，返回 (s * p, s)。
    """
    s = np.median(g) / np.median(p)
    return p * s, s


def align_affine(p, g):
    """协议 P2：逆深度空间的最小二乘仿射对齐。

    三个模型都是仿射不变的，只有视差图有意义，故拟合
        a * (1/p) + b ≈ 1/g
    再取倒数还原深度。返回 (depth, a)。

    截断下界取相对量（中位数的 1e-4），避免双三次插值过冲产生的
    近零视差在取倒数后变成千米级离群值、主导均值类指标。
    """
    q = 1.0 / np.clip(p, 1e-6, None)
    t = 1.0 / g
    A = np.stack([q, np.ones_like(q)], axis=1)
    sol, *_ = np.linalg.lstsq(A, t, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    pred_inv = a * q + b
    eps = 1e-4 * float(np.median(pred_inv[pred_inv > 0])) if np.any(pred_inv > 0) else 1e-6
    return 1.0 / np.clip(pred_inv, eps, None), a


def metrics(p, g):
    """八项标准指标。p, g 为一维等长数组，均已对齐且在同一量纲。"""
    thresh = np.maximum(g / p, p / g)
    return {
        "absrel": float(np.mean(np.abs(g - p) / g)),
        "sqrel": float(np.mean(((g - p) ** 2) / g)),
        "rmse": float(np.sqrt(np.mean((g - p) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(g) - np.log(p)) ** 2))),
        "log10": float(np.mean(np.abs(np.log10(g) - np.log10(p)))),
        "d1": float(np.mean(thresh < 1.25)),
        "d2": float(np.mean(thresh < 1.25 ** 2)),
        "d3": float(np.mean(thresh < 1.25 ** 3)),
    }


def eigen_metrics(pred, gt, crop=CROP, min_d=MIN_D, max_d=MAX_D):
    """按 Eigen 口径裁剪取掩码，返回 (P1/P2 两套指标, 有效像素数)。

    pred, gt 为同一尺寸的二维数组；有效像素数不足 100 时返回 (None, n)。
    """
    h1, h2, w1, w2 = crop
    p = pred[h1:h2, w1:w2]
    g = gt[h1:h2, w1:w2]
    mask = np.isfinite(g) & (g > min_d) & (g < max_d) & np.isfinite(p) & (p > 0)
    n_valid = int(mask.sum())
    if n_valid < 100:
        return None, n_valid

    p, g = p[mask].astype(np.float64), g[mask].astype(np.float64)
    out = {}
    for tag, fn in (("p1", align_median), ("p2", align_affine)):
        q, s = fn(p, g)
        q = np.clip(q, min_d, max_d)  # 截断到评测区间（BTS / AdaBins 惯例）
        m = metrics(q, g)
        m["scale"] = float(s)
        out[tag] = m
    return out, n_valid
