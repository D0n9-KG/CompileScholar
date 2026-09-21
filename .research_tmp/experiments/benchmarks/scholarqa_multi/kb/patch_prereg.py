# -*- coding: utf-8 -*-
"""One-shot: MULTI-PREREG v0.1 -> v0.2 (final corpus numbers + LightRAG)."""
import io

p = r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/scholarqa_multi/MULTI-PREREG.md"
s = io.open(p, encoding="utf-8").read()
n0 = len(s)

REPL = [
("> 状态：**v0.1 草稿**——待语料解析完成（corpus_map 全量）后填 TBD 数字并交用户审，批准后冻结 v1.0。",
 "> 状态：**v0.2 草稿**（09-21：语料终数已填齐 + 基线阵容按 09-21 终局矩阵更新）——待用户审 4 项待裁后冻结 v1.0。"),

("**检验逻辑**：闭卷固定语料（441 篇 union 语料编译库）上，同题同判分对比：我方编译栈 vs 闭卷协议自跑的检索式代表系统（OpenScholar-8B / PaperQA2）+ 已发表数字引用行（协议不同不混表，分层呈现）。",
 "**检验逻辑**：闭卷固定语料（430 篇 union 语料编译库）上，同题同判分对比：我方编译栈 vs 闭卷协议自跑的检索式代表系统（PaperQA2 / LightRAG，09-21 终局矩阵裁定）+ 已发表数字引用行（协议不同不混表，分层呈现）。"),

("- **语料 = 108 题全部 ctxs 的去重并集**：`TBD` 篇（解析中，探样估计 441±5；155 篇被多题复用）。每篇：题名→OpenAlex/S2 身份解析→全文获取（arXiv 优先 ~40%+OA PDF ~40%+本地 DOI 档案/Crossref 备援）→mineru 统一解析（含公式 LaTeX）。**获取失败篇入披露清单，不静默替换。**",
 "- **语料 = 108 题全部 ctxs 的去重并集**（终数，09-21 收口）：439 题名 → 436 身份解析（398 DOI + 80 arXiv，3 无身份）→ 432 获取成功（99.1%）→ **430 篇唯一全文就位**（三组语料条目各共享一个文件；4 篇不可达入披露）。渠道分布：Sciverse 167 / arXiv 91 / OpenAlex OA 43 / S2 OA 42 / Crossref link 39 / 本地 DOI 库 36 / 手动+浏览器 18。解析=mineru VLM（PDF）+ Sciverse .md 直拷；LSST Science Book（300 页/40MB）单独长超时解析（1.93M 字符）。**获取失败篇入披露清单，不静默替换。**\n"
 "- **语料披露清单（v0.2 定稿）**：①4 篇不可达（Judging the Judges 无 DOI / Dispersion Compensation IEEE 2024 付费墙且同题不同文 / \u201c210983829\u201d 脏数据条目 / Electrochemical biosensors CRC 书章节）——影响约 4 题各 1 篇引用；②2 篇获取事故已修复（Speckle-OCT 与 On-chip nanoparticle 初次抓到的是 Erratum/Corrigendum 页——身份解析阶段拿到的 DOI 本身就是勘误条目的 DOI；经修正 DOI 重抓真身并解析，证据留存 corpus/short_md_quarantine/；获取链教训=题名验证防不住勘误页，因为其标题包含原题名）；③1 篇题名不一致沿承上程人工判定（基准题名 Strategies to enhance... EPR effects vs 实抓 Izci Chem Rev 2021，内容主题相符）；④三组语料条目共享同一文件（islet 四基因敲除 / Label-free nanoparticle / Laser cooling——判分 id_mapping 按 1文本→多ctx 处理）。"),

("- **干扰池（待裁，两案）**：A 案（推荐）=不添加人工干扰，441 篇即全库（协议最干净、与题集自然对齐；检索挑战=每题从 441 选对 2-10 篇）；B 案=A+同域干扰池（PaperScope 经验复制，检验 findability 稳健性）。**推荐 A 案为主跑、B 案作可选稳健性臂**——待用户裁。",
 "- **干扰池（待裁，两案）**：A 案（推荐）=不添加人工干扰，430 篇即全库（协议最干净、与题集自然对齐；检索挑战=每题从 430 选对 2-10 篇）；B 案=A+同域干扰池（PaperScope 经验复制，检验 findability 稳健性）。**推荐 A 案为主跑、B 案作可选稳健性臂**——待用户裁。"),

("| OpenScholar-8B 自跑 | 闭卷指定语料 | 对照 |\n| PaperQA2 自跑 | 闭卷指定语料 | 对照 |",
 "| PaperQA2 自跑 | 闭卷指定语料 | 对照（检索式代表，09-21 矩阵） |\n| LightRAG 自跑 | 闭卷指定语料 | 对照（通用结构化/KG-RAG 代表，09-21 用户裁定恢复：正面对比支撑类型化>通用图主张） |"),

("- **主判据**：Citation F1 配对差 ours vs 自跑对照（OpenScholar-8B）CI 下界 > 0 且点估计优势成立；rubric 分同向为辅证。两轨背离时以确定性 Citation F1 为主（rubric 偏差披露）。",
 "- **主判据**：Citation F1 配对差 ours vs 自跑对照（PaperQA2、LightRAG 各自配对）CI 下界 > 0 且点估计优势成立；rubric 分同向为辅证。两轨背离时以确定性 Citation F1 为主（rubric 偏差披露）。"),

("  - 自跑行：闭卷协议下的 OpenScholar-8B / PaperQA2（语料指定为本 union 语料）",
 "  - 自跑行：闭卷协议下的 PaperQA2 / LightRAG（语料指定为本 union 语料）"),

("| S3 | search-only 消融臂+对照自跑（OpenScholar-8B/PaperQA2 环境搭建，可并行） | S2 后 |",
 "| S3 | search-only 消融臂+对照自跑（PaperQA2/LightRAG 环境搭建，可并行） | S2 后 |"),

("- 对照表=引用行+自跑行分层，LightRAG/商业系除名（09-20）",
 "- 对照表=引用行+自跑行分层（09-20）；**LightRAG 恢复为自跑对照**（09-21 用户裁定：KG-RAG 是通用结构化的代表作，正面对比支撑核心主张；商业系仍除名）"),
]

for old, new in REPL:
    if old not in s:
        print("MISS:", old[:60])
    else:
        s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print(f"prereg updated: {n0} -> {len(s)} chars")
