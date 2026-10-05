# 抽取层适用性审查（FITNESS-EXTRACTION，2026-10-05）

> 审查对象包括旧深抽（`kb_compiler.records.*`）和新写的抽取层（`compilescholar.extract.*`、`documents.*`、`citations.markdown`）。审查标准是 INTEGRATED-SYSTEM-1005 §1–§4 规定的新系统硬要求。
>
> 做法分三步：逐个读代码；取 3 篇非 CS 论文，经 MinerU 9605 解析后实测；调用本地 Qwen3.8-27B 共 30 次。没有改仓库代码，临时文件已删除。
>
> 证据格式统一为 `file:line`、实测数字或输出片段。旧指标凡是在 CS 语料、旧 schema 或非本地模型下测得的，都标了 **[旧条件]**。

## 0. 结论先行

- **旧深抽 v1.4 是现有唯一能从非 CS 全文里抽出 config / 数值 / 明确 absence / notation 的实现。** 3 篇实测的原句定位率为 88% / 92% / 93%。新 T2 在同样 3 篇上一条 config 都没抽到，旧深抽抽到 81 条。
  - 但旧深抽不能原样进新系统，原因有五：分块对 MinerU 原始输出失效（base64 图片把块撑到 169k 字符）；会把参考文献也送进 LLM；中文提示词导致输出混入中文；`not_reported` absence 实际上是在猜论文缺了什么；schema 以 method 为中心，偏 CS。
  - 结论：保留它的"概念 + 分块 + 先抄原句"三件，提示词和 schema 改写后并入 T2，判 **需重构**。
- **新抽取层在非 CS 论文上主路径基本断开**：
  - 3 篇的引用句一条都没抽到。
  - T2 的方法/实验那一遍（deep call）从未触发，因为 `method_and_experiments` 依赖 "Introduction" 标题，这 3 篇都没有，返回 0 字符。
  - T2 实际只读了前 7k 字符，占正文 15–30%，其中还包括作者和单位行。
  - 化学论文里唯一一张表没有产出结果单元，也没有记入 residue。
- **最先要修的是文档层**（documents / citations），抽取提示词排在后面。MinerU Markdown 中 base64 占 90–98%；浮动体（图注、PACS 号）会插在句子中间把句子截断；首字下沉的字母会丢失；还有字形错码（`¼` 代 `=`、`þ` 代 `+`、`B` 代 `~`）。这些问题同时影响原句子串检查、引用句切分和分块。

## 1. 实测（3 篇非 CS，MinerU 9605 standard 档）

| 代号 | DOI | 领域 / 形态 | 页数 | MinerU 耗时 | Markdown 总长 | 其中 base64 | 正文实长 |
|---|---|---|---|---|---|---|---|
| phys | 10.1103/PhysRevLett.110.010403 | 物理，PRL letter，无节标题，`[n]` 引用 | 5 | 9.1 s | 265,881 | 238,244（90%） | 27,637 |
| bio | 10.1038/ncomms10002 | 神经科学，Nat Commun，`<sup>n</sup>` 引用 | 12 | 27.4 s | 2,560,152 | 2,501,368（98%） | 58,784 |
| chem | 10.1021/ja400020e | 有机化学，JACS communication，ACS 上标 + 字母子引用 | 4 | 9.1 s | 808,406 | 784,500（97%） | 23,906 |

PDF 来自本地 Sci-Hub 库，查询方式是 `items JOIN archives`，再从 zip 里读出。挑选时剔除了 `ncomms10000`（勘误）和 `jacs.5b00004`（ASAP 稿）。3 篇 MinerU 产物里都没有 `<table>`，chem 的 Table 1 被渲染成 pipe 表。

### 1a. documents/units 与 citations/markdown

| 指标 | phys | bio | chem |
|---|---|---|---|
| units（原始 MinerU） | section 1 / para 73 | section 4 / para 109 | section 9 / para 80 / caption 1 |
| 含 base64 的 unit 数 / 最长 unit | 4 / 89,268 | 62 / 192,048 | 15 / 185,752 |
| `abstract_and_intro` 中 base64 占比（原始） | 2,286/7,000（33%） | 0 | 3,795/7,000（54%） |
| `abstract_and_intro` 覆盖正文（去 base64 后） | 26% | 15% | 30% |
| `method_and_experiments` | **0 字符** | **0 字符** | **0 字符** |
| caption unit（实际图注 / 表注数） | 0（3 个 `FIG. n`） | 0（8 个 `Figure n \|`） | 1（6 个 Scheme + 1 个 Table） |
| table unit | 0 | 0 | 0（实际有 1 张 pipe 表） |
| `find_refs` | None（31 条 `[n]` 条目，无标题） | 找到，53 条 | None（标题为 `## ■ REFERENCES`，条目文本被扭曲） |
| bib markers / 引用句对 | 0 / 0 | 0 / 0 | 0 / 0 |

逐项原因（已核实）：

- **base64 没剥离**：`units.from_markdown` 不处理图片（units.py:41-80）。
- **两个取文函数都靠 "Introduction" 标题定位**：`abstract_and_intro` 在没有该标题时，会一路收集到 7k 字符，开头就是作者和单位行。实测 AI head：`T. Fink<sup>1,2</sup> and H. Bluhm<sup>1,2</sup> 1<sub>2nd</sub> Institute of Physics C…`（units.py:129-144）。`method_and_experiments` 要求先出现 introduction 标题，所以这里永远返回空（units.py:156-163）。
- **图注识别失败**：`CAPTION` 正则写死成 `(table|figure|fig\.)\s*\d+[.:]`（units.py:24），所以 Nature 的 `Figure 1 |`、PRL 的 `FIG. 1 (color online).`、化学的 `Scheme n.` 都识别不出来。
- **pipe 表不进 table unit**：`TABLE_BLOCK` 只认 `<table>`（units.py:23）。即使把 pipe 表直接交给 `result_pass.run`，也是 `{'result_units': 0, 'tables_with_residue': 0}`，因为首列是 `entry` 序号，整张表被静默丢弃（子审查在 `documents/tables.py` 上独立复现了同一现象）。
- **上标引用全部失效**：bio 有 45 处 `<sup>` 数字引用，chem 有 43 处 `<sup>`（含 `4j,k`、`5k−m` 和一半在上标、一半是普通文本的 `5a,<sup>j</sup>`）。把 `<sup>n</sup>` 模拟替换成 `[n]` 后，bio 也只剩 1 个 marker。原因是 `prev.isalnum()` 守卫：上标紧贴前一个单词，被当成 LaTeX 下标误标而丢弃（citations/markdown.py:101）。
- **没有 References 标题就找不到参考文献**：PRL 有 31 条 `[n]` 条目但没有标题。手工插入标题后得到 31 条条目、45 个 marker、44 个引用对（22 句），说明句子切分本身没问题，问题出在参考文献块的检测（markdown.py:23-24, 51-64）。
- **DOI 带上了尾部 `)`**：PRL 的 30 个 DOI 全部带尾部 `)`，例如 `10.1103/PhysRevLett.105.216803)`。原因是 Markdown 链接 `(http://dx.doi.org/…)` 的右括号被 `bib.DOI` 正则吃进去了（compile/skeleton/bib.py:31）。PRL 条目没有论文标题，DOI 是唯一的解析键，所以这个 bug 会让解析整体落空。bio 的 53 条条目都不带 DOI，只能靠标题解析。
- **chem 参考文献块被扭曲**：MinerU 的输出形如 `u, J. Q.; , . J. - c va on; pr nger: er n, . (1) Y Shi Z C H A ti ti S i B li 2010 (2) For recent…`。ACS 的 `(n) … (a) … (b)` 行内子条目没有对应的解析模式。

文档层还有几个会破坏原句检查的问题：

- 浮动体插进句子中间。例如 PRL `may be increased to 100 Hz when a single-shot readout is\n\nPACS numbers: …\n\napplied`，以及 `experiments of\n\n![img]\n\nFIG. 1 …`。
- 首字下沉的字母丢失：`he central canal…`（bio）、`n the past few decades… I interest`（chem）。
- 字形错码：bio 有 34 处 ` ¼ `（实为 `=`），`B20 ms`（实为 `~20 ms`），`B þ 25 mV`（实为 `~+25 mV`）；phys 有 13 个 C0 控制字符代替希腊字母，例如 `As \x07 is reduced`。
- MinerU 的页脚 HTML（`<small><span class="docvortex-page-footnote" …>`）混在正文段落里。

MinerU 9605 的 job 响应里有 `middle_json` / `structured_content` 两个输出槽，本次只请求了 markdown，没有测试。content_list 和页码 / bbox 能否拿到，需要阶段 B 再测一次。

### 1b. 同一篇上：新 self_pass（T1 / T2 / deep）对比旧 deep_extract 提示词

**设置**：

- 输入一律用去掉 base64 后的 Markdown。如果用原始输出，旧 `chunk_text` 产出的块里含 base64：chem 有 13 块，最大一块 169,000 字符；bio 有 14 块，每块 24–48k 字符。在 500k 上限下，旧深抽只能覆盖 chem 47%、bio 25% 的真实正文。
- 旧深抽不注入 registry / vocab（新系统也不搬这部分），手工跳过了参考文献、致谢等块，共 20 次调用（17 块 + 3 次 absence）。
- 新侧共 9 次调用：T1 / T2 原样跑；T2 的 deep call 因为输入为 0 没有触发，所以另外用正文第 7k–16k 字符喂 `DEEP_PROMPT` 模拟。
- other_pass 跑了 1 次（PRL，插入标题后得到前 24 对）。
- 总计 30 次调用，全部成功。

**原句通过率说明**：旧侧用旧 postcheck 门（NFKC、去空白、去 `$`、LaTeX 折叠），新侧用 `in_text`（只做空白归一）。旧记录先按 `finalize_paper` 的做法把 `method` 等字段映射成 `*_ref`，之后 `missing_field` 违规归零。

| | phys | bio | chem |
|---|---|---|---|
| **旧** 记录数（按 kind） | 42：finding 26 / config 7 / notation 4 / absence 3 / lineage 2 | 150：config 72 / finding 68 / result 8 / absence 2 | 35：finding 19 / result 8 / absence 6 / config 2 |
| 旧 原句定位（postcheck） | 37/42（88%） | 138/150（92%） | 27/29 有 quote 的（93%）；全门通过 33/35 |
| 旧 quote 中不同数值个数 | 22 | 69 | 28 |
| **新 T1** 保留 / 产出 | 6/6 | 6/6 | 4/5（proposes 被正则拒绝） |
| **新 T2** 保留 / 产出（实际 = T1 + 前 7k） | 7/8 | 6/6 | 6/7 |
| 新 deep（模拟） in_text | 12/13 | 10/14 | 14/15 |
| 新 全部 quote 中不同数值个数 | 6 | 16 | 12 |
| 新 config / absence / notation | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

**旧深抽：值得保留的抽取能力（原始输出片段）**：

- bio config：`{"method":"fluid pulse stimulation","item":"pulse pressure","value":"20 p.s.i.","role":"used_in_experiment"}`，Q: `action potentials were evoked with somewhat higher pulse pressure (20 p.s.i.; Fig. 1g)`
- chem result：`{"metric":"isolated yield","value":"85%",…,"dims":{"subject":"cis-3,4-dihydroisoquinolinone 4aa"}}`，Q: `providing cis-3,4-dihydroisoquinolinone 4aa in 85% isolated yield (Scheme 2)`
- phys / bio 的明确 absence 全部是句级、有原句：`A specific contribution of ASIC3 channels to the pH-sensitivity of CSF-c neurons has not been tested in mammals.`；`available data [12,25] are insufficient to quantitatively characterize the high-frequency behavior.`
- phys notation：`{"symbol":"$H$","quantity":"qubit Hamiltonian","definition":"$\frac {\hbar}{2} [ \Omega + \beta (t) ] \hat {\sigma} _ {z}$"}`
- 条件：chem finding 的 `condition: {"setup": ["high concentration"]}`；含 condition / scope / dims 的记录在 phys / bio / chem 分别占 11/42、35/150、21/35。

**旧深抽：跨领域问题**：

- **中文提示词导致输出混语**：chem 的 6 条 absence 字段全是中文。
- **`not_reported` 是在猜**：chem 的 6 条 absence 全是 `not_reported`，quote 为空，例如 `"missing":"未报告产物 4aa 和 5aa 的 ee 值","evidence":"…暗示产物为外消旋体但未明确声明。"`。这个反应是非对映选择性的，产物是外消旋体，要求 ee 值没有意义。这类记录违反"明确写出的 absence"这条要求。
- **direction 被强行套用**：bio 的 result 带了 `direction:"lower_better"`（response latency）和 `"higher_better"`（net depolarization）。这是给"性能指标"设计的字段，用在生理量上属于语义编造。
- **method 槽偏 CS**：phys 的 `method` 被填成一整句描述（`alternative method to extract the spectrum from…`）；bio 的 `method` 填的是实验技术（`patch recordings`）。
- **字形错码带进数值**：`"value":"B20 ms"`，实为 `~20 ms`。postcheck 数值门照样通过。
- **不过滤参考文献块**：路由规则不匹配就退回全槽（deep_extract.py:250-255）。chem 的前言 + 摘要块（c0，3,315 字符）返回 `{"records": [], "overflow": []}`，摘要层的贡献完全漏掉。
- **原句失败的类型**：
  - LLM 跨浮动体把被截断的句子拼回去，例如 phys `Reilly et al. [13], where…`、`averag ing`；
  - 用省略号 `...` 删减原句（phys 2 条）；
  - MinerU 字形 / `\mathsf` 宏没被折叠（bio 十余条）；
  - 去掉 `<sup>` 引用号。去 tag 加去引用号这一通道能救回 bio 11 条中的 10 条（postcheck 的 lenient 通道）。

**新 self_pass：问题**：

- **T2 输入太窄**：等于 T1 加前 7k 字符，抽不出 config 和数值。3 篇共得到 6 / 16 / 12 个数值，旧深抽是 22 / 69 / 28。
- **`proposes.validate` 的正则误判**：
  - 误拒：chem T1 `A rhenium-magnesium cocatalyzed [4+2] annulation … is described.`；chem T2 `we herein disclose the first redox-neutral [4+2] annulation…`（`disclose` 不在词表里）。
  - 误收：phys T1 把**论文标题** `Noise Spectroscopy Using Correlations of Single-Shot Qubit Readout` 当成方法名收下了，因为"名字必须出现在 quote 里"这条没有实现（proposes.py:40-48, 58-71；子审查另外构造了 6 个误收 / 误拒用例）。
- **`in_text` 只做空白归一**：bio deep 的 14 条里有 4 条失败，原因是 LLM 抄原句时自然去掉了 `<sup>22,23</sup>` 这类引用号。phys T2 有 1 条失败，原文是控制字符 `\x01-pulses`，LLM 写成了 `\pi-pulses`。
- **setting 语句的 quote 不是原句**：quote 取的是 `(abstract or text)[:300]`，即摘要前 300 字符，没有做子串检查（self_pass.py:147）。setting 的键是 CS 式的 `tasks / datasets / metrics / baselines`，于是出现 `datasets: GaAs spin qubits`、`datasets: Lamprey spinal cord CSF-c neurons` 这样的结果。
- **deep 模拟的质量可以接受**：42 条里 36 条通过 in_text。chem 的底物范围失败被正确归为 limitation，例如 `only C-H alkenylated product 3ak was formed when 4-octyne was subjected…`。也有误标：bio 把对照组阴性结果 `motor neurons … did not show a response` 标成了 limitation。

**other_pass（PRL，24 对，1 次调用）**：

- 24 对全部解析成功，`dropped_about` 为 0。function 分布：background 15 / data 8 / basis 2 / contrast 1。
- `data` 用错了：`Direct measurement of the state allows for detection of low-frequency spectral content &1 Hz [8,9]` 是背景陈述，不是数据引用。
- **重复**：合引句 `This assumption is adequate for many experiments on electron spins [13,15–17] and for superconducting…` 被拆成 8 对，`text` 几乎相同，提示词里同一句也重复了 8 次（other_pass.py:77-79）。
- 成本：每对约 124 个输出 token，24 对共输入 4,084 / 输出 2,965 token，耗时 92 s。

## 2. token 成本与耗时

**测量口径**：调用台账 `prompt_tokens / completion_tokens`。字符与 token 之比为提示词 2.98 字符/token、输出 3.45 字符/token。29 次调用、42k 输出 token 在并发 8–12 下用时 217 s，单流约 25 tok/s，与任务说明中的 30 tok/s 一致。

**换算假设**：耗时按"输出 token ÷ 720 tok/s"计算。提示词预填（prefill）没有测聚合吞吐，但单请求观察到的速度约 3k tok/s（bio absence 遍：17.9k 输入、260 输出、用时 15 s），所以表中耗时是**下界**。

| 方案 | 单篇输入 tok | 单篇输出 tok | 500 篇 | 5,000 篇 | 依据 |
|---|---|---|---|---|---|
| 旧 deep_extract（实测） | phys 27.7k / bio 46.8k / chem 24.6k | 6.6k / 21.5k / 6.4k | — | — | 3 篇 20 次调用；chunk 输出均值 1.95k，最大 5.1k |
| 旧 deep_extract，换算到 5 万字符一篇 | ≈ 7 块 × (1.3–1.5k 开销 + 2.7k 文本) + absence (0.9k + 全文至多 110k 字符 ≈ 37k) ≈ 45k | ≈ 2.9k/万字符 × 5 + 0.5k ≈ 15k | **2.9 h** | **29 h** | 输出密度取 3 篇实测 2.4k / 3.6k / 2.7k 每万字符的均值 |
| 同上，按 max_tokens 上限 | 同上 | 7 × 9,000 + 6,000 = 69k | 13 h | 133 h | deep_extract.py:370, 374 |
| 同上，走 compilescholar 客户端默认设置 | — | — | × 约 4 | × 约 4 | `max_tokens ≥ 8000` 会进入大调用通道，宽度 = max(2, 48 // 8) = **6**（llm/client.py:148-157, 185） |
| 现 T2（实测均值） | 5.7k | 2.6k | 0.5 h | 5 h | 只覆盖正文 15–30%，非 CS 论文上 deep call 为 0 |
| 推荐 T2（§4，估计） | ≈ 30k | ≈ 11k | ≈ 2.1 h | ≈ 21 h | 分块同旧；用句号编码替代整句回抄，quote 字符占旧块输出的 27%；去掉全文 absence 遍 |
| 现 T1（实测） | 470 | 520 | — | 30 万篇 ≈ 60 h；100 万篇 ≈ 8.4 天 | quote 加 text 占 T1 输出字符的 66% |
| 推荐 T1（估计） | ≈ 600 | ≈ 250 | — | 30 万篇 ≈ 29 h | 用句号编码；`text` 可选，缺省等于原句 |
| other_pass（实测） | 170/对 | 124/对 | — | — | PRL 每句平均 2.0 对，按句分组约可减半 |

## 3. 模块判定表

工作量单位：人日（d），含测试。

| 模块 / 能力 | 判定 | 证据 | 具体改法 | 工作量 |
|---|---|---|---|---|
| **deep_extract 分块**（≤8k 字符、按节、超长节按 `\n\n` 再切、小块合并、上限 500k） | 需优化 | 原始 MinerU 输出下，base64 行里没有 `\n\n`，块切不开（chem 一块 169k 字符）；500k 上限只覆盖 25–47% 的真实正文（deep_extract.py:198, 217-227）。去掉 base64 后块数正常：phys 4 / bio 13 / chem 5 | 改为在 units 上分块（已剥 base64、浮动体已分离），按字符预算打包；参考文献、致谢、作者、页脚这类结构性 unit 直接排除；上限按真实文本算 | 1 d |
| deep_extract 路由（`ROUTE_RULES` 正则加 card 标签） | 退役 | PRL 没有节标题，chem 只有 Scheme 被当成标题，结果全部退回全槽；正则按标题判断节的语义（deep_extract.py:131-142），而 card 遍不再存在 | 统一全槽；"块里有没有表格 / 公式"这种结构信号只用来开关可选子遍 | 0 |
| 七类记录的概念 | 并入（改形态） | 3 篇实测：config 81、明确 absence 5、notation 4、条件 67 条有值 | config → 新 facet `config` + `meta.config{item, value, unit, applies_to}`；absence 只保留 explicitly_stated → facet `absence`；finding 的 strength / epistemic 改为 Statement 一级字段 `epistemic ∈ {demonstrated, stated, hypothesized}`，cited 的走 kind=other；`condition`（原文限定词）提升为一级字段；notation → facet `definition`（可选）；lineage → role，target 由引用号确定；shift → 退役（3 篇 0 条）；result 的 `direction`、`aggregation` 改为可选，默认 null | schema 1 d（需用户确认，见 §5） |
| quote-first 规则（QUOTE_FIRST_RULES 1–1e、6、8） | 可复用（改写） | 旧原句定位 88–93%；规则 1c 要求逐字抄公式，实测 phys 的 notation 都能定位 | 翻成英文，删除与注册表、词表有关的规则 4；新增"禁止省略号、禁止跨浮动体拼句"；改为按句号编码引用（§4） | 0.5 d |
| 单独的 absence 遍（读全文至多 110k 字符） | 退役 | 每篇多花 10–18k 输入 token，只换来 2–6 条；明确的 absence 都是句级的（phys / bio 5/5）；`not_reported` 6/6 是猜测且输出中文（deep_extract.py:108-123, 358-363） | 明确 absence 并入块提示词 | −1 次调用/篇 |
| 注册表 / 词表注入（`build_injection`、dims 词表、`entity_queue`、F16） | 退役 | 新系统不搬实体注册表；本次实测不注入，原句率仍在 88–93% | 名字只保留表面名，由 Identity 模块对齐 | 0 |
| 中文提示词（CHUNK / ABSENCE / REPAIR） | 替换 | chem 有 6 条中文字段；源文全是英文 | 改为英文提示词，输出语言固定为英文 | 含在上面 |
| `salvage_json_records` 截断救援 | 直接可用 | 本次 0 次触发；旧事故有记录（common.py:45-92） | 迁到 `llm/jsonparse` | 0.2 d |
| **postcheck 五道门** | 需重构 | 原句门（NFKC、去空白、去 `$`、LaTeX 折叠）把旧原句定位提到 88–93%，新 `in_text` 只有 86%；但 fuzzy 0.85 通道接受非子串（postcheck.py:267-286），违反"原句必须是来源子串"；dims 门依赖词表（:330-359）；修复提示词是中文（:365-375）；LaTeX 映射缺 `\mathsf`、`\textsf`、`\circ`、`\mathrm`（:120-132） | 做成统一终检（§4）：保留门 1（必填字段）、门 2（枚举）、门 3（子串定位）、门 4（数值逐字）；门 3 定位用"声明过的投影视图"，见 §4，fuzzy 只做定位辅助，命中就拒收；删门 5；修复调用留一次，改英文，给出所在 unit 的原文 | 1.5 d |
| canary / canary_granular | 需重构 | 哨兵内容有用（7 个事实 + 6 个陷阱），但打分器读旧 record 字段；没有测试，只有旧入口（子审查：canary.py:107-285） | 打分器改为判定 Statement；补一个 MinerU 形态的化学或医学变体（上标引用、无参考文献标题、base64、LaTeX、科学计数法表）；加两类陷阱：quote 非子串、数值改写 | 2 d |
| table_semantic（F35） | 需重构 | **[旧条件：CS 语料 AirQA/PS]** subject 填充率从 0 提到 43%；提示词写死 ML/NLP/CV，blocklist 全是 CS 数据集，注入 120 个注册表名（子审查：table_semantic.py:61-75, 174-201） | 保留"LLM 提议、结构门回写"的做法；输入改为网格加 caption；轴角色改成跨领域的：对象 / 样品 / 化合物 / 组别，条件，测量量和单位 | 2 d |
| documents/tables.py | 需优化 | 实测：chem 的 pipe 表因首列是序号被静默丢弃。子审查实测：`39<sup>12</sup>` 被拼成 3912；`1.2×10^{-3}` 只剩 1.2；`<0.001`、`wt%` 被丢弃；表头里有数值（"300 K"）时整表跳过 | 先剥 sup / sub 再折叠；值保留科学计数法指数；序号列与下一文本列合并为行头；单位从表头括号里解析；所有跳过的表计入 residue；删掉 direction 规则；补化学、医学、物理 fixture | 2–3 d |
| notation_harvest | 需优化 | **[旧条件：PS-53，CS]** 93% 的论文有公式，约 34 个/篇；G2 只检查"公式 ⊂ quote"，不检查"quote ⊂ 原文"；批次失败后不重试 | 补"定义句是原文子串"的检查；输出 facet=definition；失败进重试队列；只在含公式的块上触发 | 1 d |
| figure_channel | 退役（冻结） | 数值靠 VLM 估读，违反"数值逐字"；caption 缺失时会造一个 quote；没有调用方 | 以后做的话，单开 facet，标 `meta.estimated=true` | 0 |
| coarse_extract、compile/state/coarse.py | 退役 | 没有原句检查；未知 kind 被静默改成 finding；CLI 报 NameError（子审查） | 先把 field_state 改为读 statements，再删除 | 0.5 d |
| survey_extract | 需重构（缓） | 用正则按标题做语义回退；chunk 失败后不重试；不允许引用作者名作方法名，与医学和生物的作者年份引用写法冲突 | 等主线稳定后，按 Statement 和引用号重写；第三方关系需要先扩展 schema（§5） | 2–3 d |
| compile/skeleton/proposes.validate | 需重构 | 实测 chem 误拒 2/2，phys 把标题误收；`PROPOSAL`、`RELIANCE` 正则是在做语义判断（proposes.py:40-48）；`GENERIC` 是 CS 词表（:49） | 只保留两条结构检查：证据是原文子串、名字或别名出现在 quote 里；"是不是本文提出的"交给 LLM 字段 `relation: proposes\|uses`；generic 交给 Identity | 0.5–1 d |
| arbitration_list、chunk_retry | 退役 | 都绑在注册表上；chunk_retry 的教训是 Paratera 重试 3 次后仍有 5–8% 失败 **[旧条件]**，要落实到新系统的重试队列 | — | 0 |
| **新 self_pass（T1）** | 需优化 | 单篇输出 520 token，66% 是 quote 加 text；proposes 正则误判 | 摘要句编号，LLM 回传句号；`text` 可选；去掉正则判定 | 0.5 d |
| **新 self_pass（T2）** | 需重构 | 只读引言 ≤7k 加方法实验 ≤9k；非 CS 论文上 deep call 为 0；没有 config / absence / 条件 / epistemic；setting 的 quote 不是原句 | 改成全文分块的 T2，用一套英文提示词，覆盖全部 facet（§4） | 2–3 d |
| result_pass 顺序与 own_methods | 需优化 | build.py:113-115 先跑 R.run 再跑 S.run，而且不传 own_methods；子审查实测所有行都被判成 baseline_comparison | 先跑 T2，把它的 proposes 名字加别名作为 own_methods 传进去；非 CS 论文用"本文研究对象"代替"本文方法" | 0.3 d |
| `in_text`（只做空白归一） | 替换 | bio deep 4/14 失败，因为去掉了引用号；phys 1 条失败，因为控制字符 | 并入统一终检 | 含在终检 |
| done_self / done_pairs 不论成败都写 | 需重构 | build.py:119 在 `parse_failed` 时也写 done_self；build.py:153 解析失败也写 done_pairs | 改成工作项状态表（§4） | 1 d |
| other_pass 词面支撑 0.6 | 需优化 | 本批 `dropped_about` 为 0，暂无误删证据；但这是规则在判语义（other_pass.py:29, 69-74） | 降为提示信号：不删，只标 `meta.low_overlap`；忠实度交给 LLM 批量判断（§4） | 0.5 d |
| other_pass 批处理形态 | 需优化 | 合引句重复 8 次（other_pass.py:77-79）；`function=data` 误用 | 按句分组：每句一项，带 cited keys 列表，输出逐 key 的 role / function；quote 就是这个句子 | 0.5 d |
| **documents/units** | 需重构 | 不剥 base64（54% / 33% 的窗口被占）；图注识别 0/3、0/8、1/7；pipe 表不进 table unit；不支持 MinerU content_list；没有页码和 bbox；节定位依赖标题 | 先剥 base64：图片存为资产，unit 里留占位符加资产 id；把浮动体（图注、Scheme、页脚、PACS）分离为独立 unit；被浮动体截断的段落按结构规则拼回（上一段没有句末标点，且下一段小写开头）；优先读 content_list，带页码和 bbox；生成句级 id `<pid>#u12.s3`；不再提供 "intro / method" 取文函数 | 2–3 d |
| **citations/markdown** | 需重构 | 3 篇的引用句对都是 0；上标被 `isalnum` 守卫杀掉；没有标题时找不到参考文献块；`■` 挡住标题匹配；DOI 带尾 `)` | 新增 superscript 风格：`<sup>` 内是数字、范围、字母后缀时整体当作 marker，不走 LaTeX 守卫；参考文献块改为从文末向前找连续的条目形行，标题可有可无，允许 `■` 等装饰符；支持 ACS 行内 `(n) … (a) …`；DOI 去掉尾部 `)`、`.`、`]`；条目带 DOI 时直接建只有元数据的论文 | 2–3 d |
| statements 表的溯源字段 | 需优化 | DDL 没有 loc、schema_version、run_id 列（schema.py:92-94）；确定性的 result 语句也写了 `S.MODEL`（build.py:98） | 加 `unit_id`、`sent_id`、`char_start/end`、`schema_version`、`run_id`、`pass`；确定性产物的 model 记为 `deterministic:<代码哈希>` | 0.3 d |

合计约 22–27 人日，不含 survey_extract。

## 4. 推荐的新抽取层结构

```
documents (确定性)                       citations (确定性)
  MinerU content_list 优先 / Markdown      参考文献块：文末条目形行，标题可无
  剥 base64 → 资产；浮动体独立 unit         marker：[n] / 上标 / 作者-年份 / 字母标签
  断句拼回；句级 id；页码+bbox              引用句 = 句级 id；条目 DOI/标题
  匹配视图（只用于检查，不改存储）
        │                                         │
        ▼                                         ▼
 T1 自述（摘要）   T2 自述（全文分块）  ─┬─►  结果遍（表）       他述（引用句，按句分组）
  每篇 1 次        每块 1 次，全 facet   │    确定性网格+LLM    每批 ≤24 句
  回传句号         回传句号+span         │    轴角色；own_methods
                   可选子遍：notation    │    来自 T2
                   （仅含公式的块）      │
        └────────────────┬───────────────┴────────────┬───────────────┘
                         ▼                            ▼
                统一终检（确定性） ──失败──► 修复 1 次（LLM，给所在 unit）──仍失败──► 丢弃+记录
                         │
                         ▼
          statements（带 loc / model / prompt_sha / schema_version / run_id）
          LLM 批量核对（可选，抽样）：他述忠实度、epistemic 判定
```

**每一遍读什么**：

- **T1**：标题加编号后的摘要句。输出按句号给出：`facet`、`role`、可选 `text`、`proposes[{name, aliases, artefact, relation: proposes|uses}]`。quote 直接取该句，天然是子串。
- **T2**：在 units 上打包，每块至多约 8k 字符，内容为编号后的句子。排除参考文献、致谢、作者、页脚这类结构性 unit。
  - 提示词只有一套，用英文，facet 包括 contribution、method、result（文字）、limitation、absence（只收明确写出的）、config{item, value, unit, applies_to}、setting（研究对象 / 体系、测量量、条件、技术）、categorization。
  - 每条带 `epistemic`、`condition`（原文限定词 span）、`sent_id`。需要子句精度时附 `span`，span 必须是该句的子串。
  - 写到他人工作的句子不在 T2 里抽，交给他述遍处理（该句有引用号）。没有引用号却在转述他人的句子，标 `epistemic=cited`，about 留给编译层解析。
- **结果遍**：表格网格（tables.py 修好之后）加 caption 和上下文句。LLM 只提议轴角色和测量量名称，结构门决定是否写入。数值只从单元格里取。
- **他述遍**：一句一项，项内是 cited keys 列表，quote 就是这个句子。原来的 0.6 词面阈值降为提示信号，不再用来删除。

**哪些规则改由 LLM 判断**：

- "是不是本文提出的"：去掉 proposes.py 里的 PROPOSAL / RELIANCE 正则，改由 LLM 输出 `relation` 字段。
- 节语义路由：去掉 ROUTE_RULES 和 SKIP_SEC。
- 他述是否忠实：0.6 阈值降为信号，用 LLM 批量判断，作为抽样核对或在入库前执行。
- 数值的 `direction`：只有表格或上下文里明确写了"越高越好"才填，不再按 higher/lower 词规则判定。

规则只保留结构性的工作：子串定位、数值逐字、枚举、marker 与条目对应、DOI 识别、浮动体分离、段落拼回。

**统一终检**：只有一处实现，三遍抽取的产物都经过它。

1. 定位：把 quote 和来源都投影到同一个**匹配视图**上做子串查找。视图依次做这些处理：NFKC、空白去除、去 `$`、LaTeX 折叠（补齐 `\mathsf / \textsf / \mathrm / \circ`）、去 HTML tag、去引用号（`<sup>…</sup>`、`[n]`）、MinerU 字形映射表（`¼ → =` 等）。视图只用于匹配，存储的 quote 保持原文。命中后记录原文的 unit 和字符区间。fuzzy 只用来报告"差在哪"，不放行。
2. 数值逐字：沿用 postcheck 门 4，即 `_num_forms`、ws 双通道、公式通道。
3. 枚举和必填字段：沿用门 1、门 2，改为按新 schema 检查。
4. 不过关的：调用一次修复，给所在 unit 的原文；再检查一次，仍不过关就丢弃，并写入 `dropped`，带违规码。
5. 每条通过的语句写入 `unit_id / sent_id / char_start / char_end / model / prompt_sha / schema_version / run_id / pass`。

**失败重试与断点续跑**：

- 派生库 `extract.sqlite` 新建 `work(paper_id, pass, item_id, prompt_sha, model, status ∈ {pending, ok, failed, gave_up}, attempts, last_error, updated_at)`，item_id 指块 id、句组 id 或表 id。
- 只有 `ok` 才算完成。`failed` 在下次运行时重试，超过 N 次（建议 3 次）后转为 `gave_up`，并在报告里显示。删掉 `done_self` 和 `done_pairs`。
- 某篇论文的某一遍算完成，当且仅当它的所有 item 都是 ok。
- 模型输出解析失败、截断、超时都记为 failed，不写 statements。修复后丢弃的语句不影响 item 的状态。
- **与 registry 的关系**：
  - `paper_compile_states(kb_name='extract')`：T1 全部 ok 时置为 `level=shallow`；T2、结果遍、他述遍全部 ok 时置为 `level=deep`。有 gave_up 时 `status=partial`，`notes_json` 记录 `{run_id, schema_version, counts, gave_up_items}`。这张表取代派生库里的 done_self。
  - `extraction_runs`：每篇论文每次运行写一行，`llm_config_hash = sha(model + 各遍 prompt_sha + schema_version)`，`status` 取 `ok / partial / failed`。
  - prompt_sha 或 schema_version 变化时，manifest 判为过期，只重跑受影响的那一遍。
- **并发**：块调用的 `max_tokens` 建议设为 6,000（实测最大输出 5.1k），这样不会进入宽度只有 6 的大调用通道；也可以显式设置 `LOCAL_LARGE_MAX_CONCURRENT=48`。

## 5. 之前被默认可用、实际不满足要求的点

1. **"旧深抽已验证（95.9% 逐字、config 91.9%）"**。**[旧条件：21 篇 CS hub 论文、DeepSeek-V4-Flash、注入了注册表和词表、输入是干净文本]** 在 MinerU 原始输出上，分块被 base64 撑破，500k 上限只覆盖 25–47% 的正文；在非 CS 论文上，会出现中文字段、猜测型 absence 和编造的 direction。本次去掉 base64 后的原句定位率是 88–93%，低于旧指标。
2. **"新 T2 读全文的引言和方法"**。在 3 篇非 CS 论文上，方法实验窗口都是 0 字符，因为依赖 "Introduction" 标题；引言窗口其实是前 7k 字符，开头是作者和单位。
3. **"citations/markdown 能处理 MinerU"**。3 篇的引用句都是 0。上标引用会被 LaTeX 守卫误杀；没有标题、或标题前有 `■` 时，找不到参考文献块；DOI 一律带尾 `)`，PRL 恰好只能靠 DOI 解析。
4. **"documents/tables 能处理 pipe 和 HTML 表"**。化学论文唯一一张表产出 0 条结果，也没有 residue；units 不把 pipe 表当作 table unit。
5. **"proposes.validate 只做结构校验"**。实际上它用正则判断"是不是本文提出的"，在化学论文上误拒 2/2，在物理论文上把论文标题收成了方法名。
6. **"self_pass 的原句都检查过"**。setting 语句的 quote 是摘要前 300 字符，没有做子串检查（self_pass.py:147）。
7. **"postcheck 保证子串"**。它有一个 fuzzy 0.85 通道接受非子串（postcheck.py:267-286），lenient 通道只记位置为 None。新的硬要求是"原句必须是子串"，这两个通道都需要改。
8. **"本地 48 并发下旧块调用能跑满"**。compilescholar 客户端在 `max_tokens ≥ 8000` 时进入宽度为 6 的通道，旧块的 `max_tokens=9000` 会把吞吐降到大约四分之一。
9. **"MinerU 输出可以直接当原文"**。有字形错码（`¼`、`þ`、`B`、控制字符）、首字下沉丢字母、浮动体截断句子、页脚 HTML 混入正文。原句检查和数值逐字检查都要建立在"匹配视图 + 原文存储"这套分离机制上；同时 documents 层要能拿到 content_list，这一点本次未测。

**需要用户裁定的 schema 变更**（本审查只给出建议，没有执行）：

1. FACETS 增加 `config`、`absence`、`definition`。
2. Statement 增加一级字段 `epistemic`、`condition`、`loc{unit_id, sent_id, start, end}`、`schema_version`。
3. 是否允许 `kind=other` 的语句带 `target`，用来表达综述或他述中"A extends B"这类第三方关系（子审查提出）。
4. 旧的 `shift` 类型是否正式退役（3 篇实测为 0 条）。
