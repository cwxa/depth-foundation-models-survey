# depth-foundation-models-survey

[![tests](https://github.com/cwxa/depth-foundation-models-survey/actions/workflows/tests.yml/badge.svg)](https://github.com/cwxa/depth-foundation-models-survey/actions/workflows/tests.yml)

单目深度估计基础模型的**受控基准评测**代码与结果。

本项目回答一个具体问题：在**完全相同的评测口径**下，跨越三代技术路线的开源深度模型，精度与效率差距到底有多大？

## 主要发现

1. **监督信号比模型容量更关键**。同一子集上，参数量扩大 4 倍（24.8 M → 97.5 M）仅使 AbsRel 改善 6.5%；而把监督范式从混合监督换成自训练，参数量反而减少 21%，AbsRel 下降 43%。
2. **对齐协议能反转模型排序**。相对深度模型输出的是仿射不变的逆深度，若只用中位数估计尺度而不估计偏移，同一模型的 AbsRel 会从 0.057 恶化到 0.370。
3. **最新一代在单图基准上未必最优**。统一几何表示的 DA-V3 在 NYU 单目基准上未超过 DA-V2，其优势应在多视图设定下评估。

## 结果（NYU-Depth V2 Eigen，654 张，协议 P2）

| 模型 | 参数量 | AbsRel↓ | RMSE↓ | δ₁↑ | 单帧时延 |
|---|---|---|---|---|---|
| MiDaS-DPT-Hybrid | 123.1 M | 0.100 | 0.418 | 0.901 | 0.092 s |
| **Depth-Anything-V2-Base** | 97.5 M | **0.057** | **0.283** | **0.959** | 0.183 s |
| Depth-Anything-V3-Base | 135.4 M | 0.073 | 0.330 | 0.936 | 0.102 s |

对齐自由度分解（AbsRel / δ₁，654 张）：

| 对齐方式 | MiDaS | DA-V2 | DA-V3 |
|---|---|---|---|
| 无对齐 | 0.999 / 0.000 | 0.849 / 0.004 | 0.604 / 0.011 |
| 仅尺度 | 0.347 / 0.492 | 0.370 / 0.463 | 0.083 / 0.919 |
| 尺度与偏移 | 0.100 / 0.901 | 0.057 / 0.959 | 0.073 / 0.936 |

容量对照（同一批 306 张）：DA-V2-Small 24.8 M / 0.0641，DA-V2-Base 97.5 M / 0.0599，MiDaS 123.1 M / 0.0990，DA-V3 135.4 M / 0.0744。

## 目录结构

```
code/
  prepare_nyu.py       # NYU-Depth V2 Eigen 划分解包与预处理
  metrics.py           # 裁剪/掩码/两种对齐协议/八项指标（纯 numpy，可独立测试）
  test_metrics.py      # 指标实现的单元测试（无需 GPU 与数据集）
  models.py            # 四个模型的统一封装（含视差→深度转换）
  eval.py              # 评测主程序：双对齐协议 + 8 项指标
  efficiency.py        # 参数量 / 时延 / 吞吐 / 峰值显存
  analyze.py           # 对齐自由度分解 + 误差分布 + 失败案例
  subset_capacity.py   # 同子集容量对照（纯计算，复用缓存预测）
  figures.py           # 定性对比图与效率图
  gen_tables.py        # 结果表
  gen_tables2.py       # 补充分析表
  check_paper.py       # 论文合规性自检（字数/编号/引用率）
results/               # 评测结果（JSON / 逐图 CSV）
paper/figures/         # 实验图表
```

## 先验证指标，再跑实验

指标实现是最容易出错、也最影响结论的部分，因此它被抽成 `code/metrics.py`（只依赖 numpy），并有独立测试：

```bash
python code/test_metrics.py      # 9 个用例，约 0.2 s，无需 GPU / 数据集
```

覆盖：完美预测、手算已知误差、δ 阈值边界（恰为 1.25 时不计入）、中位数对齐的尺度不变性、仿射变换的精确还原、**P1 明显劣于 P2**（即本文核心论断的可测版本）、超界截断、掩码剔除、有效像素不足。


## 复现

环境：Python 3.13 + PyTorch 2.6.0 (CUDA 12.6)，实测于 NVIDIA RTX 3060 Laptop（6 GB）。

```bash
# 1. 数据：NYU-Depth V2 Eigen 划分（654 张），放入 data/nyu/*.tar
python code/prepare_nyu.py

# 2. 权重放入 weights/（见下表），文件名需与脚本一致

# 3. 评测
python code/eval.py --model midas
python code/eval.py --model da2
python code/eval.py --model da3
python code/eval.py --model da2s        # 容量对照

# 4. 效率与分析
python code/efficiency.py
python code/analyze.py
python code/subset_capacity.py
python code/gen_tables.py && python code/gen_tables2.py
```

权重（自行下载，不随仓库提供）：

| 模型 | 文件名 | 大小 |
|---|---|---|
| MiDaS v3.0 DPT-Hybrid | `weights/dpt_hybrid_384.pt` | 470 MB |
| Depth-Anything-V2-Small | `weights/depth_anything_v2_vits.pth` | 95 MB |
| Depth-Anything-V2-Base | `weights/depth_anything_v2_vitb.pth` | 372 MB |
| Depth-Anything-V3-Base | 由官方仓库自动下载 | 1.3 GB |

## 评测口径说明

- 预测双线性插值到 640×480 后按 Eigen 区域裁剪（行 45:471，列 41:601）
- 有效像素：真值 ∈ [1e-3, 10] 米且预测为正有限值
- 尺度对齐后截断到同一区间（BTS / AdaBins 惯例），避免近零视差像素产生千米级离群值
- P2 协议：在逆深度空间做最小二乘仿射拟合，与官方评测脚本一致

## 已知限制

- DA-V2-Large（335 M）未纳入：在 6 GB 显卡上实测单帧 78.6 s（FP16 亦需 38 s），为 kernel 回退而非显存不足所致，全量评测不具可行性
- 仅在 NYU-Depth V2 室内集上评测，跨域结论待验证

## 许可

本仓库为课程作业产出，暂未附加开源许可证。如需在 MIT / Apache-2.0 下复用，请先联系作者。
