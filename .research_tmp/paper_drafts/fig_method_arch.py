# -*- coding: utf-8 -*-
"""图3-1 总体架构图。设计尺寸=版心宽16cm，600DPI，PNG+SVG 双输出。
data manifest: 无数据（示意图）；输出 fig_method_arch.png/.svg。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

FIG_W_CM, FIG_H_CM = 16.0, 9.0
fig, ax = plt.subplots(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54))
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

C_BUILD = "#eef2fb"; C_EDGE = "#3b4a6b"; C_GOV = "#fdf3e3"; C_GOV_E = "#a06a1f"
C_ACC = "#eaf6ee"; C_ACC_E = "#2e6b45"; C_OUT = "#f6eefb"; C_OUT_E = "#6b3a86"

def box(x, y, w, h, title, body, fc, ec, ts=7.6, bs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=ts,
            color=ec, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top", fontsize=bs,
                color="#222222", linespacing=1.5)

def arrow(x1, y1, x2, y2, ec=C_EDGE, style="-|>", lw=1.0, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=9, color=ec, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3))

# ---- row 1: construction pipeline (y=36..50) ----
box(1, 36, 13, 14, "论文语料", "PDF/解析全文", C_BUILD, C_EDGE)
box(17, 36, 22, 14, "抽取管线", "切片注入的大模型抽取\n确定性后检链\n金丝雀度量", C_BUILD, C_EDGE)
box(42, 36, 17, 14, "类型化记录层", "7 类记录\n逐字引文+条件维度\n认识状态分轨", C_BUILD, C_EDGE)
arrow(14.4, 43, 16.6, 43)
arrow(39.4, 43, 41.6, 43)
ax.text(15.5, 44.4, "解析", fontsize=6, ha="center", color="#555555")

# ---- governance loop (below record layer) ----
box(38, 14, 25, 15, "治理环（三时钟）", "实体注册表 / 维度词表\n批次生长 → 聚类 → 人工仲裁\n版本化冻结 · 用量审计减法", C_GOV, C_GOV_E)
arrow(50.5, 35.6, 50.5, 29.6, ec=C_GOV_E, style="-|>")
ax.text(51.4, 32.4, "实体候选/残差队列", fontsize=6, ha="left", color=C_GOV_E)
arrow(56.5, 29.4, 56.5, 35.6, ec=C_GOV_E, style="-|>")
ax.text(57.4, 32.4, "冻结版本 vN", fontsize=6, ha="left", color=C_GOV_E)

# ---- row: views ----
box(66, 36, 16, 14, "编译视图", "对比矩阵(分带)\n谱系 · 覆盖地图\n方法卡档案", C_ACC, C_ACC_E)
arrow(59.4, 43, 65.6, 43, ec=C_ACC_E)
ax.text(62.5, 44.4, "确定性编译", fontsize=6, ha="center", color="#555555")

# ---- access layer ----
box(62, 14, 24, 15, "类型化访问层", "9 工具（本地确定性执行）\n七招动作手册\n（每招指认消费的表示语义）", C_ACC, C_ACC_E)
arrow(74, 35.6, 74, 29.6, ec=C_ACC_E)

box(86, 20, 13, 24, "ReAct 循环", "笔记全量重写\n(回指+锚点)\n四道验证闸\n反循环组套\n步数预算", "#ffffff", C_OUT_E, ts=7.4, bs=6.2)
arrow(86.4, 24, 84.4, 24, ec=C_OUT_E, style="-|>")
arrow(84.4, 30, 86.4, 30, ec=C_OUT_E, style="-|>")
ax.text(85.4, 27, "动作/观察", fontsize=5.8, ha="center", va="center", color=C_OUT_E, rotation=90)

ax.text(50, 52.6, "构建（批次间演化）", fontsize=7, ha="center", color=C_EDGE)
ax.text(50, 8.6, "访问（运行时冻结）", fontsize=7, ha="center", color=C_ACC_E)
ax.plot([1, 99], [11.2, 11.2], lw=0.6, ls=(0, (4, 3)), color="#999999")
ax.text(99, 3.2, "带溯源答案", fontsize=7.4, ha="right", va="center", color=C_OUT_E,
        fontweight="bold")
arrow(92.5, 19.6, 92.5, 6.0, ec=C_OUT_E, lw=1.1)

out_png = "C:/Users/D0n9/Desktop/LogicKG/.research_tmp/paper_drafts/fig_method_arch.png"
out_svg = "C:/Users/D0n9/Desktop/LogicKG/.research_tmp/paper_drafts/fig_method_arch.svg"
fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
print("figure saved:", out_png, out_svg)
