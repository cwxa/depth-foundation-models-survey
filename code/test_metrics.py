"""指标与对齐协议的单元测试（仅需 numpy，无需 GPU / cv2 / 数据集）。

运行：python -m unittest code/test_metrics.py -v
或：  python code/test_metrics.py
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import CROP, MAX_D, MIN_D, align_affine, align_median, eigen_metrics, metrics  # noqa: E402

H, W = 480, 640


def make_pair(seed=0, n=None):
    """构造一对 (pred, gt)：在 criop 区域内 gt 为 [1,5] 的随机深度，
    pred 由已知的仿射视差变换生成，用于检验对齐能否还原。"""
    rng = np.random.default_rng(seed)
    gt = np.full((H, W), np.nan, dtype=np.float64)
    h1, h2, w1, w2 = CROP
    inner = rng.uniform(1.0, 5.0, size=(h2 - h1, w2 - w1))
    gt[h1:h2, w1:w2] = inner
    return gt, inner


class TestMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        g = np.array([1.0, 2.0, 3.0, 4.0])
        m = metrics(g.copy(), g)
        self.assertAlmostEqual(m["absrel"], 0.0, places=12)
        self.assertAlmostEqual(m["rmse"], 0.0, places=12)
        self.assertAlmostEqual(m["d1"], 1.0, places=12)

    def test_known_error_value(self):
        # g=[2], p=[3] -> |2-3|/2 = 0.5；rmse=1；thresh=max(1.5,0.667)=1.5 -> d1=0
        g = np.array([2.0])
        p = np.array([3.0])
        m = metrics(p, g)
        self.assertAlmostEqual(m["absrel"], 0.5, places=12)
        self.assertAlmostEqual(m["rmse"], 1.0, places=12)
        self.assertEqual(m["d1"], 0.0)

    def test_delta_threshold_boundary(self):
        # ratio 恰好 1.25 -> 不算命中（严格小于）
        g = np.array([1.0])
        p = np.array([1.25])
        self.assertEqual(metrics(p, g)["d1"], 0.0)
        p = np.array([1.249])
        self.assertEqual(metrics(p, g)["d1"], 1.0)

    def test_median_alignment_is_scale_invariant(self):
        rng = np.random.default_rng(1)
        g = rng.uniform(1, 5, 2000)
        p = g * 7.3  # 纯尺度扰动
        q, s = align_median(p, g)
        self.assertAlmostEqual(s, 1 / 7.3, places=6)
        self.assertLess(metrics(q, g)["absrel"], 1e-9)

    def test_affine_recovers_planted_transform(self):
        """若 pred 由 a*(1/g)+b 精确生成，P2 应几乎完全还原。"""
        gt, inner = make_pair(seed=2)
        a_true, b_true = 3.0, 0.2
        disp = a_true / inner + b_true
        pred = np.full((H, W), np.nan)
        h1, h2, w1, w2 = CROP
        pred[h1:h2, w1:w2] = 1.0 / disp

        out, n = eigen_metrics(pred, gt)
        self.assertIsNotNone(out)
        self.assertGreater(n, 100)
        self.assertLess(out["p2"]["absrel"], 1e-6)
        # 注意约定：align_affine 拟合的是 1/g ≈ a*(1/p) + b（论文式 2），
        # 因此返回的 a 是"视差→逆深度"的斜率，即 1/a_true。
        self.assertAlmostEqual(out["p2"]["scale"], 1.0 / a_true, places=4)

    def test_affine_beats_median_on_affine_data(self):
        """核心论断的可测版本：真值为仿射变换时，P1 明显劣于 P2。"""
        gt, inner = make_pair(seed=3)
        disp = 3.0 / inner + 5.0  # 带较大偏移，模拟视差型模型
        pred = np.full((H, W), np.nan)
        h1, h2, w1, w2 = CROP
        pred[h1:h2, w1:w2] = 1.0 / disp

        out, _ = eigen_metrics(pred, gt)
        self.assertGreater(out["p1"]["absrel"], 5 * out["p2"]["absrel"])

    def test_clipping_to_eval_range(self):
        gt, inner = make_pair(seed=4)
        pred = np.full((H, W), np.nan)
        h1, h2, w1, w2 = CROP
        pred[h1:h2, w1:w2] = inner * 1e6  # 远超上限
        out, _ = eigen_metrics(pred, gt)
        self.assertLessEqual(out["p2"]["rmse"], MAX_D - MIN_D + 1e-6)

    def test_mask_excludes_out_of_range_gt(self):
        gt, inner = make_pair(seed=5)
        pred = np.full((H, W), np.nan)
        h1, h2, w1, w2 = CROP
        pred[h1:h2, w1:w2] = inner
        # 让 gt 大部分超界，有效像素应显著减少
        gt2 = gt.copy()
        gt2[h1:h2, w1:w2] = np.where(np.arange(inner.size).reshape(inner.shape) % 10 == 0,
                                     inner, 20.0)
        _, n_valid = eigen_metrics(pred, gt2)
        self.assertAlmostEqual(n_valid, inner.size // 10, delta=inner.shape[1] + 1)

    def test_too_few_valid_pixels_returns_none(self):
        gt = np.full((H, W), np.nan)
        pred = np.full((H, W), np.nan)
        out, n = eigen_metrics(pred, gt)
        self.assertIsNone(out)
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
