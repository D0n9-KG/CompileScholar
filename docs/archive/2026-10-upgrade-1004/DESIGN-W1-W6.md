# W1 评测修正 + W6 项目规范化：细化设计与预注册（10-04，待用户确认后开工）

> 依据：UPGRADE-PLAN.md v1（用户 10-04 "都按建议"）、AUDIT-VENUE-READINESS §4、AUDIT-CODE-ARCHITECTURE §5–§7、AUDIT-WORKSPACE §5–§7。
> 原则：W6 先搬家（行为零变化，有等价测试守着），W1 的行为修复在新包里逐条单独提交；每条修复都写明"改了什么行为、怎么验证、
> 对已报数字有没有影响"。冻结 v9b 的 test 结果不重算、不重解读（预注册规则 1），只在附录标注"若用修正后的脚本重算是否变化"。

---

## 0. 现在先定死的事实（开工前测过）

- W0 已完成：NAS 备份校验通过；凭据已移出代码；未推送历史已重写并推送（main=origin/main=d5db519b，.git 226 MB）。
- 冻结 v9b 的 14 个文件哈希在工作区与 NAS 上都一致；重写后 git 哈希映射在 rewrite/commit-map-20261004.tsv。
- CS2 test v9b 终数已入库（paired_stats_test100_v9b.txt）。此后任何对 CS2 汇总口径的修改，都要在 v9b 文件上复算并报告"变 / 不变"。
- GPTR 臂公平性（代码审查 H8）初查：GPTR test 100 题全部有引用（中位 11 条），0 题无片段引用，first-pass 日志里检索器报错 0 次、
  熔断关键词 0 次。**判断：没有证据表明 GPTR 的检索被熔断压低**；但 GPTR 的检索器每题拿到的结果数没有逐次记录，
  W1-8 会补记（不重跑 GPTR，除非补记显示空检索率 >10%）。

---

## 1. W6 项目规范化（先做）

### 1.1 目标布局（采用代码审查 §5.1，工作区审查的 bench/ 并入 experiments/）

```
src/compilescholar/
  core/      config.py  paths.py  secrets.py  cutoff.py  manifest.py
  llm/       client.py  embedding.py  jsonparse.py
  sources/   http.py（节流+内容寻址缓存+录制回放）  sciverse.py  openalex.py  s2.py  crossref.py  arxiv.py  refgraph.py  search.py
  kb/        store.py（KB 目录+MANIFEST 哈希校验）  index.py（原 hybrid.py）
  compile/   records/  registry/  views/  state/（build_state、merge_state、field_state）  pipelines/
  answer/    pipeline.py  stages/{probe,plan,gather,screen,write,assemble}.py  prompts/*.txt  formats/{cs2,dsb,multi}.py
  eval/      cs2/{judge,scoring,adapters}.py  dsb/{judge,summary}.py  multi108/  field/  stats.py
  baselines/ harness/{runner,mcp_server,compat_proxy}.py  gptr.py  storm.py  paperqa.py  memorized.py
  cli.py
configs/     base.yaml  local.example.yaml  kb/*.yaml  bench/*.yaml  ablations/*.yaml
experiments/<bench>/README.md + run.sh
third_party/ asta-bench（git submodule @9bf087a）  deepscholar-bench（submodule @c95413b）  patches/*.patch
tests/       unit/  golden/  fixtures/
docs/        README 状态行  ARCHITECTURE.md  EXPERIMENTS.md  RESULTS.md  DECISIONS.md  paper/  archive/
data/ artifacts/ runs/ cache/ scratch/   （全部 gitignore，只跟踪 MANIFEST）
legacy/      旧代码 + INDEX.md（不打包、不进 CI）
```

### 1.2 迁移步骤（每步一个提交，每步都跑等价验证）

| 步 | 内容 | 等价验证（必须通过才进下一步） |
|---|---|---|
| S1 | 建包骨架 + pyproject（单一包、依赖补全、锁版本）+ conftest；`pip install -e .` | `python -c "import compilescholar"`；现有 146 个测试仍全过 |
| S2 | **先写表征测试锁住现行为**（在旧代码上跑通）：cutoff.allowed 判定表；hybrid 在小 KB 上的排名；_interleave；assemble 用 test r1 的 trace 重放、断言 sections 逐字节一致；refgraph 用缓存里的真实响应做离线夹具；cs2_scoring.summarize 对 7 个 test 分数文件复算一致；paired_stats 复现 dev +0.094 与 test 主表 | 全部在旧代码上通过 |
| S3 | 把活代码复制进新包（不删旧文件），修导入，去 sys.path / 绝对路径 / import 时副作用；prompt 抽成 prompts/*.txt | S2 的表征测试改为指向新包后全过；**每个 prompt 渲染结果与旧 answer_pipeline 逐字节相同**（不调 LLM 即可验证） |
| S4 | configs/*.yaml + Settings；库代码不读环境变量，只有 cli 读一次；进程环境优先于 .env | 用 configs/bench/cs2_test_v9b.yaml 解析出的配置，与 FREEZE v9b 记录的开关逐项一致 |
| S5 | CLI：`compilescholar answer/judge/score/verify/field-eval/kb build`；runs/<run_id>/ 自动写 manifest（git sha + dirty、加载模块的 LF 归一化 blob 哈希、prompt 哈希、KB MANIFEST 哈希、第三方 commit+patch 哈希、pip freeze 哈希、非密钥环境快照） | `compilescholar verify` 能在本机核验 v9b 的 14 个文件；dev 2 题冒烟（开回放缓存）与旧 runner 输出逐字段一致 |
| S6 | third_party 子模块 + patches（direct_judge 的 4 处补丁集中成 eval/cs2/judge_adapter.py，每处带开关和注释；nuggetizer base_url 补丁成 .patch） | 用新判分入口重判 v9b test r1 的 5 题，与已存分数在判分噪声内（同 judge 的重判 |Δ| 均值 0.043 为参照）——**只核验管线等价，不更新任何报出数字** |
| S7 | 旧文件 `git mv` 进 legacy/，写 legacy/INDEX.md（每个文件支撑过哪些已报结果）；冗余判分脚本归档；score_compare.py 删除（口径错误） | 全部测试过；experiments/<bench>/run.sh 能跑通 |
| S8 | 工作区：.research_tmp 根下散文件按主题进 scratch/；退役实验（paperscope 归档 19 GB 等）移 NAS 冷存储并写 RETIRED.md + MANIFEST；venv311 移出仓库；补跟踪 paper_drafts / phase0 / docs_decisions / test 答案（>5 MB 先 gzip） | 每移一批对 MANIFEST 校验 sha256；移出前 NAS 已有副本 |
| S9 | 文档最小集：README 重写（v8 主张 + 状态行）、ARCHITECTURE、EXPERIMENTS（每基准数据来源、冻结清单、复现命令、已知坑）、RESULTS（只收冻结数字，每行带 run_id/哈希/CI）、DECISIONS（ADR 一行一条，旧 134 份只建索引）；旧 DIRECTION/ASSET-STATE 进 docs/archive/ | — |
| S10 | 防复发：pre-commit（check-added-large-files 5 MB、gitleaks 锁版本、拒绝 C:\Users 绝对路径）；CI 跑单测 + golden 回放 | 故意提交一个 6 MB 文件被拦 |

约束：
- 搬迁阶段（S1–S7）**行为零变化**；任何会改输出的修复放到 W1，单独提交。
- 不改写已推送历史；不碰冻结 v9b 的工作区文件（新包是复制，旧文件 S7 才 git mv，v9b 复现用 tag `freeze-cs2-v9b` → 8b0d69ff）。
- base_kb 当冻结数据制品（用户决策 9）：base_kb/ 与 base_kb_v2/ 打 MANIFEST.sha256 发布，建库脚本进 legacy/ 并在 EXPERIMENTS 写清顺序；不重跑带 LLM 的建库步骤。

### 1.3 规模控制
单人研究项目：配置用 dataclass + pyyaml（沿用 kb_compiler/config.py 的分层合并），不引 Hydra/DVC；数据发布 = 目录 + sha256 清单 + 下载脚本。

---

## 2. W1 评测修正（在新包里，逐条提交）

每条格式：问题 → 修法 → 验证 → 对已报数字的影响。

| # | 问题（出处） | 修法 | 验证 | 对已报数字 |
|---|---|---|---|---|
| W1-1 | cs2_scoring：任一 scorer 报错整题四项记 0，判分器故障与"没答出"混为一类（审查 §4 #1）；分数文件不存在静默全 0（#2） | 区分三态：answered / no_answer（计 0）/ judge_error（**拒绝汇总**，必须重判到成功）；文件不存在直接报错 | 单测覆盖三态；对 7 个 test 分数文件复算 | v9b test 7 臂 judge 错误行均为 0 → **不变**（复算确认后写入附录） |
| W1-2 | 判分行不记 judge 名 / rubric 哈希 / 时间（#8）；test 模式逐题打印分数（#9） | 每行写 judge、rubric sha256、split、时间、重试次数、idx 平移是否触发；`--quiet-scores` 在 split=test 时强制开启 | 单测；新判分文件字段齐全 | 无 |
| W1-3 | 配对统计无入库脚本（#10）；留出检验 families 无 CI（#11）；百分位 bootstrap 在小 n 覆盖率偏低 | eval/stats.py：配对 BCa bootstrap（B=10,000）+ 符号翻转置换 + Holm；所有表的 CI 只能由它产出 | 复现 dev +0.094、test 主表；BCa 与百分位并列报一次，差异写进附录 | test 主表 CI 可能变宽或变窄几个千分点，**点估计不变** |
| W1-4 | 留出检验：判分只看前 250 条候选（#12）；JSON 失败静默记未命中（#15）；direct 摘要截 600 字（#14）；state/state_v2 是同一系统两次运行（#13） | 候选分块判分取并集；失败块重判到成功，否则剔除并报出；direct 用全文摘要；每臂（state/direct/flat/memory + 同预算 map-reduce）跑 3 次，报均值 ± SD | 修后在已有 18 篇上重跑（这是 M2，属于 W5，但脚本修复在 W1 做完） | §6.5 现行数字标注"协议修正前"，修正版出来后替换 |
| W1-5 | DSB 汇总缺答剔除，CS2 缺答记 0（代码审查 M5） | DSB 统一缺答记 0；judge_nuggets 失败写错误行、不 continue | 用 harness_v2（3 题失败）复算：已按记 0 报 0.241，脚本化后应一致 | 不变（之前手算已按记 0） |
| W1-6 | 引用格式红利未排除（审查 R2/B7） | 格式对齐重判，三变体：(i) 我们的片段截到与 harness 中位同长（~240 字符，取与句子最相关的 1–2 句，用确定性的句子重叠打分选句，不用 LLM）；(ii) harness 的引用补整段摘要；(iii) 两边都只给标题 | **只重判已有 test 答案，不重新生成**；所有变体用同一 judge | 这是新增分析，不替换主表；结果进 §6.2 读法与 §7.2 |
| W1-7 | 长度混杂（B6）；DSB 长度差 3× | CS2：score ~ system + log(words) 回归（题级固定效应）；DSB：我们的答案按 harness 中位词数截断后重判（截断取前 N 词，段落边界对齐） | 回归系数与 CI；截断版 nugget | 新增分析 |
| W1-8 | GPTR 检索器结果数无记录（H8） | 在 baselines/gptr.py 的检索器里记每次 search 的返回条数与异常 | 下一次跑 GPTR 时生效；**不为此重跑**，除非用户要求 | 无 |
| W1-9 | refgraph 暂时失败被永久缓存成空（M2，现有 7 条） | 缓存三态：正结果 / 确定的负结果（404+原因+时间）/ 暂时失败（不缓存）；清掉现有 7 条 source=none 的路由缓存 | 单测：超时、额度跳过、网络错都不写缓存 | 只影响之后的运行 |
| W1-10 | LLM 静默降级不计数（M1） | 每题 trace 带 degradation 计数（plan_fallback / screen_parse_fail / write_empty / cite_timeout / cite_http_fail / ext_errors），运行级 health.json | 单测 + 对 v9b r1/r2 trace 回填可得的计数 | 无 |
| W1-11 | Multi runner 重跑必崩（H9） | KB 统一加载；无状态层时 state_search 返回 [] | 冒烟 2 题 | 无（Multi 已报数字来自崩溃前的运行） |
| W1-12 | KB 两个检索通道不按截止过滤（DSB 公平性，DESIGN-CROSSPAPER P0-0） | Cutoff 对象显式传入 KB 检索，按论文年月过滤；trace 记被过滤条数 | 单测；DSB 48 题统计"若当时开启会过滤掉多少被引 KB 证据" | 若 DSB 已报的 0.310 里有超截止的 KB 证据被引用，**如实报告条数**；是否重跑由数量决定（>0 即重跑 DSB 我们的臂，作为修正版） |
| W1-13 | 同一论文两通道两个引用编号（M16） | 证据表按 DOI / arXiv id / 规范化标题合并论文身份 | 单测；对 v9b r1 统计受影响节数（已知 11/355） | 行为修复，只对升级版生效 |
| W1-14 | 判分器效度（B1–B3）〔10-04 修订：官方 judge 地区受限不可用，用户裁定主口径 = DeepSeek-V4.1-Flash（Paratera），见 PREREG 修订 1〕 | (a) 官方 judge `google/gemini-3-flash-preview` 经 OpenRouter 全量重判 CS2 test 7 臂（用户已批费用；去掉 4 处补丁的原样 scorer，只保留 Connection: close 这类不改语义的网络修复）；(b) DeepSeek 作为第二 judge 已有；报两 judge 排名 Kendall τ 与每个配对差的方向一致性；(c) 人工 60 题一致性 | 费用估算见 §4 | 主表增加"官方 judge"列；**主口径按预注册改为官方 judge**，DeepSeek 列作次口径 |

---

## 3. 预注册（写进 PREREG.md，commit 后才开始跑任何新的 dev/test）

1. **v9b test 是升级前记录**：数字照报，不做逐题失分分析；W1 的口径修正在 v9b 文件上复算，只报告"变 / 不变"。
2. **主口径**：CS2 主表以官方 judge（gemini-3-flash-preview，官方 scorer 原样）为主口径，DeepSeek-V4.1-Flash 为次口径；两者结论冲突时两者都报，不挑。
3. **格式对齐判读**（W1-6，结果出来前写死）：若变体 (i) 下我们对 harness 的 CR+CP 优势仍 > 0 且 CI 下界 > 0，论文可以说"逐句证据引用在格式对齐后仍提升引用质量"；否则只能说"在官方格式下引用分更高，其中格式贡献占 X"，并删除"更扎实"的表述。
4. **长度判读**（W1-7）：报回归系数；不据此调整主表。
5. **作废规则只看机制指标**：cite=0 的题 >20%、外检报错 >10%、judge 错误行 >0、进程被外部中断——任一触发整轮重跑；否则结果再差也不重跑。
6. **dev 使用**：offset 10–24 的 15 题已被选择污染，只用于回归测试（行为零变化的等价验证），不再当证据；升级版的确认集 = dev offset 25–99（75 题），开发期最多看 2 次。
7. **test 只再跑一次**：升级版冻结后跑 CS2 test 一次（2 次运行算一次），与 v9b 同表；比 v9b 差也照报。
8. 本文件与 PREREG.md 的 commit hash 写进论文附录。

---

## 4. 成本与时间估算

- 官方 judge 全量重判：CS2 test 7 臂 × 100 题。每题判分输入（答案 + rubric）中位 ~13.5k token，三个 scorer 加上 citation 逐段调用，
  估每题 ~60–100k 输入 token、~5k 输出 token → 700 题 ≈ 42–70M 输入 + 3.5M 输出。gemini-3-flash-preview 现价 $0.5/M 输入、$3/M 输出
  （OpenRouter 10-04 实查）→ **约 $32–46**；用 batch 价约半价。实际值先跑 5 题实测后再报给用户。
  OpenRouter key 已核（10-04）：有效、付费账户、未设额度上限，历史用量 $352.5。
- W6 S1–S7：约 5–7 天；S8–S10：约 2 天。
- W1：约 3–4 天（不含官方 judge 跑批的墙钟时间）。

---

## 5. 用户确认（10-04：四项全部同意；官方 judge 费用不设上限，无需再问）

1. 本设计的顺序（W6 搬家先于 W1 修复）与"搬家阶段行为零变化"的约束。
2. 官方 judge 走 OpenRouter 的 gemini-3-flash-preview；费用不设上限（用户 10-04），先跑 5 题只为核对管线。
3. W1-12 的规则：DSB 若发现超截止的 KB 证据被引用（>0 条）就重跑 DSB 我们的臂，作为修正版（旧数字保留为"修正前"）。
4. 退役实验数据（约 28 GB）移到 NAS 冷存储后从本机删除（NAS 上先有副本并校验）——这是 S8 里唯一的删除动作。
