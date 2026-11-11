"""生成补充分析的两张表：容量对照（同子集）、对齐自由度分解。"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = r"E:/cv_paper/tables"

NAME = {"da2s": "DA-V2-Small", "da2": "DA-V2-Base", "midas": "MiDaS-DPT-Hybrid",
        "da3": "DA-V3-Base"}
PARAM = {"da2s": 24.8, "da2": 97.5, "midas": 123.1, "da3": 135.4}
MODE = {"S0": "无对齐（S0）", "S1": "仅尺度（S1）", "S3": "尺度与偏移（S3）"}


def f3(v):
    return "%.3f" % v


def capacity_table():
    with open(os.path.join(RES, "subset_capacity.json"), encoding="utf8") as f:
        d = json.load(f)
    res, n = d["results"], d["n"]
    order = [m for m in ["da2s", "da2", "midas", "da3"] if m in res]
    best = min(res[m]["absrel"] for m in order)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{容量对照：同一批 %d 张图像上的参数量与精度}" % n,
        r"\label{tab:scale}",
        r"\zihao{-5}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"模型 & 参数量/M & AbsRel$\downarrow$ & $\delta_1\uparrow$ \\",
        r"\midrule",
    ]
    for m in order:
        r = res[m]
        a = "%.4f" % r["absrel"]
        if abs(r["absrel"] - best) < 1e-9:
            a = r"\textbf{%s}" % a
        lines.append("%s & %.1f & %s & %.4f \\\\" % (NAME[m], PARAM[m], a, r["d1"]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{-0.6em}", r"\end{table}", ""]
    with open(os.path.join(OUT, "scale.tex"), "w", encoding="utf8") as fh:
        fh.write("\n".join(lines))
    print("wrote scale (n=%d)" % n)


def decomp_table():
    with open(os.path.join(RES, "alignment_decomp.json"), encoding="utf8") as f:
        rows = json.load(f)
    by = {}
    for r in rows:
        by.setdefault(r["mode"], {})[r["model"]] = r
    models = [m for m in ["midas", "da2", "da3"] if m in by.get("S3", {})]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{对齐自由度分解：各档对齐下的 AbsRel / $\delta_1$（654 张）}",
        r"\label{tab:decomp}",
        r"\zihao{-5}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}l" + "c" * len(models) + r"@{}}",
        r"\toprule",
        "对齐方式 & " + " & ".join(NAME[m] for m in models) + r" \\",
        r"\midrule",
    ]
    for mode in ("S0", "S1", "S3"):
        cells = []
        for m in models:
            r = by.get(mode, {}).get(m)
            cells.append("%s / %s" % (f3(r["absrel"]), f3(r["d1"])) if r else "--")
        lines.append("%s & %s \\\\" % (MODE[mode], " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\vspace{-0.6em}", r"\end{table}", ""]
    with open(os.path.join(OUT, "decomp.tex"), "w", encoding="utf8") as fh:
        fh.write("\n".join(lines))
    print("wrote decomp")


if __name__ == "__main__":
    capacity_table()
    decomp_table()
