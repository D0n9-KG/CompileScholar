# -*- coding: utf-8 -*-
"""Architecture figure for 项目文档: four-layer pipeline + knowledge hypergraph.
Renders a PNG (high DPI, designed width ~ textwidth) to embed via graphicx."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Register SimHei for Chinese (report-writing convention: addfont)
FP = "C:/Windows/Fonts/simhei.ttf"
font_manager.fontManager.addfont(FP)
matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10.2, 5.6), dpi=200)
ax.set_xlim(0, 10.2); ax.set_ylim(0, 5.6); ax.axis("off")

GRAY = "#eef1f6"; ORANGE = "#fdeedc"; EDGE = "#555a66"; INK = "#1a1d24"
SUB = "#41464f"; ACC = "#6366f1"

def box(x, y, w, h, title, sub, fc, tc=INK, fs=11, sfs=8.6, edge=EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.09",
                                linewidth=1.1, edgecolor=edge, facecolor=fc, zorder=2))
    if sub:
        ax.text(x + w/2, y + h*0.64, title, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", zorder=3)
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center", fontsize=sfs,
                color=SUB, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", zorder=3)

# left x0, width
LX, LW = 0.55, 7.0
rows = [
    ("L0 查询理解层", "意图解析 → 子查询分解 → 三类改写变体　·　超图位① 概念变体扩展", 4.55),
    ("L1 多源召回层", "S2 + Crossref + OpenAlex 并行召回 · embedding 语义重排", 3.45),
    ("L2 综合排序层", "分块判分 → 分档 → 语义-权威混合排序　·　超图位② 意图边路由", 2.35),
    ("L3 归纳输出层", "超图位③ 方法演化链 · 方法族分组 · 证据溯源", 1.25),
]
BOX_H = 0.92
cy = {}
for title, sub, y in rows:
    box(LX, y, LW, BOX_H, title, sub, GRAY)
    cy[title] = y + BOX_H/2

# solid arrows between layers
for ytop, ybot in [(4.55, 3.45+BOX_H), (3.45, 2.35+BOX_H), (2.35, 1.25+BOX_H)]:
    ax.add_patch(FancyArrowPatch((LX+LW/2, ytop-0.02), (LX+LW/2, ybot+0.02),
                                 arrowstyle="-|>", mutation_scale=15, lw=1.3,
                                 color=EDGE, zorder=1))

# knowledge hypergraph box on the right
KX, KW = 8.15, 1.7
box(KX, 1.25, KW, 4.22, "知识超图", "", ORANGE)
ax.text(KX+KW/2, 1.25+4.22*0.62, "知识超图", ha="center", va="center",
        fontsize=12, color="#7c4a12", fontweight="bold", zorder=3)
ax.text(KX+KW/2, 1.25+4.22*0.34, "增量生长：每查询对\n高相关论文自动建图",
        ha="center", va="center", fontsize=8.2, color="#8a5a1e", zorder=3)

# dashed arrows: hypergraph -> L0 / L2 / L3
for tgt_y, lbl in [(cy["L0 查询理解层"], "①"), (cy["L2 综合排序层"], "②"),
                   (cy["L3 归纳输出层"], "③")]:
    ax.add_patch(FancyArrowPatch((KX, tgt_y), (LX+LW+0.02, tgt_y),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.1,
                                 color=ACC, linestyle=(0, (4, 3)), zorder=1))

fig.savefig("figs/architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved figs/architecture.png")
