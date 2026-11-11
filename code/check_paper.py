# -*- coding: utf-8 -*-
"""论文合规性自动检查：图表编号、公式、引用、字数、封面/目录/页码。"""
import re

import pymupdf

TEX = r"E:/cv_paper/main.tex"
PDF = r"E:/cv_paper/main.pdf"

src = open(TEX, encoding="utf8").read()
doc = pymupdf.open(PDF)
pages = [p.get_text() for p in doc]
full = "".join(pages)

print("=" * 60)
print("一、结构与字数")
print("=" * 60)
print("总页数            :", doc.page_count)
body_end = full.find("参考文献")
cjk_body = len(re.findall(r"[\u4e00-\u9fff]", full[:body_end]))
print("正文汉字数        :", cjk_body, "(要求 >= 6000)", "✓" if cjk_body >= 6000 else "✗")
i_ab = full.find("摘")
i_kw = full.find("关键词")
ab = full[i_ab:i_kw]
print("摘要汉字数        :", len(re.findall(r"[\u4e00-\u9fff]", ab)), "(要求 200-300)")
title = "单目深度估计基础模型：方法演进与基准评测"
print("题目字数          :", len(title), "(要求 <= 20)", "✓" if len(title) <= 20 else "✗")

# 署名（姓名/学号/专业/研究方向）与页码
sign_ok = all(k in pages[0] for k in ["陈纬昕", "1026040838", "计算机科学与技术", "计算机视觉"])
print("署名四要素        :", "✓" if sign_ok else "✗")
numbers = [p for p in pages if re.search(r"\n\d+\s*$", p)]
print("带页码的页        :", "%d/%d" % (len(numbers), len(pages)))

print()
print("=" * 60)
print("二、图表与公式")
print("=" * 60)
for kind, pat in (("表", r"表\s*(\d+)"), ("图", r"图\s*(\d+)")):
    caps = sorted(set(int(x) for x in re.findall(r"(?m)^" + kind + r"\s*(\d+)", full)))
    refs = sorted(set(int(x) for x in re.findall(kind + r"\s*(\d+)", full)))
    print("%s 题注编号        : %s" % (kind, caps))
    print("%s 正文引用编号    : %s" % (kind, refs))
    missing = [x for x in caps if x not in refs]
    print("  未被正文引用    :", missing if missing else "无 ✓")

eqs = sorted(set(int(x) for x in re.findall(r"\((\d)\)\s*\n", full)))
print("公式编号          :", eqs)

print()
print("=" * 60)
print("三、参考文献与引用")
print("=" * 60)
bibs = re.findall(r"\\bibitem\{([^}]+)\}", src)
print("文献条数          :", len(bibs), "(要求 >= 50)", "✓" if len(bibs) >= 50 else "✗")
idx = {k: i + 1 for i, k in enumerate(bibs)}
cites = re.findall(r"\\cite\{([^}]+)\}", src)
used = set()
bad_order = []
for c in cites:
    ks = [x.strip() for x in c.split(",")]
    for k in ks:
        used.add(k)
        if k not in idx:
            print("  !! 未定义引用:", k)
    nums = [idx[k] for k in ks if k in idx]
    if nums != sorted(nums):
        bad_order.append((ks, nums))
print("被引用文献数      :", len(used), "/", len(bibs))
print("引用编号乱序      :", bad_order if bad_order else "无 ✓")
years = [int(y) for y in re.findall(r"\b(20\d\d)\b", "\n".join(
    re.findall(r"\\bibitem\{[^}]+\}(.*)", src)))]
recent = sum(1 for y in years if y >= 2023)
print("近三年(>=2023)    : %d / %d = %.1f%% (要求 >= 33.3%%)" %
      (recent, len(bibs), 100.0 * recent / len(bibs)),
      "✓" if recent * 3 >= len(bibs) else "✗")

print()
print("=" * 60)
print("四、编译质量")
print("=" * 60)
log = open(r"E:/cv_paper/main.log", encoding="utf8", errors="ignore").read()
print("错误 (!)          :", len(re.findall(r"^!", log, re.M)))
print("未定义引用        :", log.lower().count("undefined"))
print("Overfull hbox     :", log.count("Overfull"))
