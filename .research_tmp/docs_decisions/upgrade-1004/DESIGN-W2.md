# W2 跨论文骨架：细化设计与预注册（10-04，待用户确认后开工）

> 依据 DESIGN-CROSSPAPER.md §3.1–3.3、§5.2、§7.1。本档只补三样东西：实测可行性数字、落盘格式与代码位置、闸门（E1–E3）的预注册判读。
> 所有新代码进 `src/compilescholar/compile/skeleton/`，产物进 `data/state/`（gitignore，MANIFEST 钉哈希）。

## 0. 可行性实测（10-04，scratch/bib_probe.py，只读）

- 136 篇综述全文，135 篇有参考文献节；书目三种形态：数字式 68、作者-年份式 60、裸数字式 7。
- 确定性切分得 21,153 条书目；正文标记 40,993 个，**94.9% 对上书目条目**（每篇中位 99.6%）。
- 书目条目自带 arXiv id / DOI 的 18.6%；其余 81% 要靠标题解析（OAI 快照 → KB → OpenAlex → Crossref）。
- 解析率 < 0.8 的 18 篇：12 篇作者-年份式（多为 "Name and et al" 等非标准写法或正文无可识别标记）、6 篇裸数字式（上标渲染成行内数字，无法与正文普通数字区分）。
  这 18 篇改走 arXiv HTML 的 LaTeXML `<cite>` → `#bib.bibN` 锚点（18 × 15.5 s ≈ 5 分钟，遵守 robots）；仍不行的整篇标为"标记不可解析"，不进计数。

## 1. 组件与产物

| 步 | 代码 | 产物（`data/state/`） | 关键规则 |
|---|---|---|---|
| P0-1 书目 + 标记 | `skeleton/bib.py` | `bib_entries.jsonl`（survey, key, raw, ids, title, authors, year）、`markers.jsonl`（survey, record_id / chunk, offset, key） | 三形态确定性切分；标记区间展开上限 20；作者-年份按"首作者姓 + 年份（含 a/b）"唯一匹配，歧义不猜 |
| P0-2 条目 → 论文 | `skeleton/resolve.py` | `cites.jsonl`（entry → paper_id / stub_id, method, confidence）、`papers_stub.jsonl` | 顺序：显式 id → OAI 快照标题索引（前 40 字符相等 + 年份 ±1 + 首作者姓在条目中）→ KB 标题 → OpenAlex 批量（6 题/10 credits）→ Crossref；全不中 = 名字级桩，不进任何计数 |
| P0-3 提出者 | `skeleton/proposes.py` | `methods.jsonl`（method, aliases, proposed_by, evidence quote） | 自述：LLM 一遍、必须给原文句；他述：综述句"X [n] proposes/introduces M" 经 P0-2 落地；通用缩写（"Transformer"、"GNN"）标 generic，不作提出 |
| P0-4 谱系落地 | `skeleton/lineage.py` | `lineage.jsonl`（from, rel, to, 端点各自的 paper_id 或 None, asserted_by[]） | 标记—端点同分句最近前置对齐；歧义交 LLM 选择题（A/B/none）；时间违例打标不删；同边多综述断言合并计数 |

## 2. 闸门（预注册；结果出来前写死，写入 PREREG 修订条目）

| 闸门 | 抽样 | 通过线 | 不过线时 |
|---|---|---|---|
| E1 标记 → 论文 | 分层 200（数字式 100 / 作者-年份式 100），我按书面规范标、用户盲审其中 20% | 精度 ≥ 0.95；覆盖：数字式 ≥ 85%、作者-年份式 ≥ 60% 落到论文 id | 停在 P0-2 修，不进 P0-3 |
| E2 提出者 | 随机 100 篇非综述论文 | 精度 ≥ 0.90、召回 ≥ 0.75 | 停在 P0-3 修 |
| E3 谱系落地 | 已落地边随机 100 | 精度 ≥ 0.85、时间违例 ≤ 5%；≥ 35% 谱系记录至少一端落地 | 停在 P0-4 修 |

- 抽样用固定种子（seed 1004），抽样清单在标注前入库并打哈希。
- 用户盲审 20% 与我的标注不一致率 > 10% 时，整批按用户口径重标。

## 3. 额度与时间

- OpenAlex：P0-2 预计 ~5k credits（半天额度），在没有其他引文扩展任务时跑。
- arXiv HTML：18 篇综述（~5 分钟）+ KB 非综述论文中缺 referenced_works 的（P1 才需要，本阶段不抓）。
- LLM：P0-3 自述一遍 ~1,126 篇（本地 27B），P0-4 选择题 ~300 次。
- 预计 5–6 天（含抽检）。

## 4. 需要用户确认

1. 本档的闸门线与"不过线就停"规则。
2. E1 的盲审方式（我先标、你盲审 20%，约 40 条，~30 分钟）。
