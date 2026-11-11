"""Generate LaTeX tables for the paper from results/*.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = r"E:/cv_paper/tables"
os.makedirs(OUT, exist_ok=True)

LABEL = {"midas": "MiDaS-DPT-Hybrid", "da2": "DA-V2-Base", "da3": "DA-V3-Base"}
ORDER = ["midas", "da2", "da3"]
COLS = ["absrel", "rmse", "rmse_log", "d1", "d2", "d3"]
HEAD = [r"AbsRel$\downarrow$", r"RMSE$\downarrow$", r"RMSE$_{\log}\downarrow$",
        r"$\delta_1\uparrow$", r"$\delta_2\uparrow$", r"$\delta_3\uparrow$"]


def table(summary, tag, caption, label, ref):
    best = {}
    for c in COLS:
        vals = [summary[m]["%s_%s" % (tag, c)] for m in ORDER if m in summary]
        best[c] = min(vals) if c not in ("d1", "d2", "d3") else max(vals)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{%s}" % caption,
        r"\label{%s}" % ref,
        r"\zihao{-5}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}l" + "c" * len(COLS) + r"@{}}",
        r"\toprule",
        "方法 & " + " & ".join(HEAD) + r" \\",
        r"\midrule",
    ]
    for m in ORDER:
        if m not in summary:
            continue
        cells = []
        for c in COLS:
            v = summary[m]["%s_%s" % (tag, c)]
            s = "%.3f" % v
            if abs(v - best[c]) < 1e-9:
                s = r"\textbf{%s}" % s
            cells.append(s)
        lines.append("%s & %s \\\\" % (LABEL[m], " & ".join(cells)))
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\vspace{-0.6em}",
        r"\end{table}",
        "",
    ]
    with open(os.path.join(OUT, label + ".tex"), "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    print("wrote", label)


def eff_table(eff):
    cols = [("params_m", "参数量/M"), ("latency_s", "单帧时延/s"),
            ("fps", "吞吐/FPS"), ("peak_mem_mb", "峰值显存/MB")]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{效率对比（RTX 3060 Laptop 6\,GB，FP32，批大小 1，30 次平均）}",
        r"\label{tab:eff}",
        r"\zihao{-5}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}l" + "c" * len(cols) + r"@{}}",
        r"\toprule",
        "方法 & " + " & ".join(c[1] for c in cols) + r" \\",
        r"\midrule",
    ]
    for m in ORDER:
        if m not in eff:
            continue
        vals = []
        for k, _ in cols:
            v = eff[m][k]
            vals.append("%.1f" % v if k != "latency_s" else "%.3f" % v)
        lines.append("%s & %s \\\\" % (LABEL[m], " & ".join(vals)))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\vspace{-0.6em}", r"\end{table}", ""]
    with open(os.path.join(OUT, "eff.tex"), "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    print("wrote eff")


def main():
    with open(os.path.join(RES, "summary.json"), encoding="utf8") as f:
        summ = json.load(f)
    table(summ, "p2",
          "NYU-Depth V2 Eigen 测试集上的定量结果（协议 P2：逆深度空间最小二乘尺度与偏移对齐，654 张）",
          "acc_p2", "tab:acc")
    table(summ, "p1",
          "同一批预测在协议 P1（仅中位数尺度对齐）下的结果，用于对齐敏感性分析",
          "acc_p1", "tab:acc_p1")
    eff_path = os.path.join(RES, "efficiency.json")
    if os.path.exists(eff_path):
        with open(eff_path, encoding="utf8") as f:
            eff_table(json.load(f))


if __name__ == "__main__":
    main()
