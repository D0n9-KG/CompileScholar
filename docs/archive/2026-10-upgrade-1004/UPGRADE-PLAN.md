# 系统升级总计划（UPGRADE-PLAN，10-04，v1 待用户裁定）

> 输入：AUDIT-VENUE-READINESS.md（顶会就绪度）、DESIGN-CROSSPAPER.md（跨论文层设计）、
> AUDIT-CODE-ARCHITECTURE.md（代码架构）、AUDIT-WORKSPACE.md（工作区治理）。
> 用户裁定（10-04）：不管时间；按顶会要求彻底升级（跨论文层为重点）+ 项目规范化；必要时联网调研。
> 状态：四份已全部合并；交用户裁 §6 的决策项后开工。

## 0. 一句话

现在的系统赢在"检索 + 逐句引用原文写作"，跨论文层只有架子；要上顶会，得把状态层做成真的
（有骨架、有成员、有支持集、有状态），让答题端结构化地用它，再用一个"必须跨论文聚合"的题集证明它有用，
同时把评测修到经得起复现。

## 1. 工作流（按依赖排序）

### W0 安全线（先做，零风险）
- CS2 test（冻结 v9b）跑完前不改冻结路径（_shared/、cs2/、src/、review_1002/p6）；之后的改动在 git 分支 `upgrade/1004` 上做。
- v9b test 结果 = "升级前系统"记录：不做逐题失分分析、不用它指导升级（预注册规则 1）。
- 预注册文档 PREREG.md 在任何升级版 dev/test 结果出来之前入库。

### W1 评测修正（M1/M7/B11，1.5–2 天）——不修评测，后面所有数字都不可信
- cs2_scoring：区分"没答出（计 0）"与"判分器报错（拒绝汇总，必须重判）"；文件不存在直接报错。
- 每行判分记录 judge 名、rubric 哈希、时间戳；test 模式不打印逐题分数。
- 新增 cs2/paired_stats.py（配对 BCa bootstrap + 置换检验 + Holm），草稿里每个 CI 必须由它重算。
- summarize_field_eval 补 families 的 CI；留出检验判分分块（去掉 250 条截断）、JSON 失败计数不静默。
- 引用格式对齐对照（B7）：片段截到 ~250 字符 / harness 补整段摘要 / 两边只给标题，三变体重判 dev。
- 长度回归（B6）。

### W2 跨论文骨架（DESIGN-CROSSPAPER P0，约 6.5 天，带人工抽检闸门）
- P0-0 KB 两个检索通道接截止过滤（DSB 公平性）。
- P0-1 综述书目解析 + 句中标记 → 书目条目（数字式确定性 99.2%；作者-年份式用 arXiv HTML cite 锚点）。
- P0-2 书目条目 → 论文 id（显式 id → arXiv OAI 快照标题索引 → KB → OpenAlex 批量 → Crossref）；KB 外 = 桩节点。
- P0-3 "本文提出的方法"抽取（自述 LLM + 他述经 P0-2 落地）。
- P0-4 谱系边两端落到论文 + 时间一致性检查。
- 闸门：E1 解析精度 ≥0.95 / E2 提出者精度 ≥0.90 召回 ≥0.75 / E3 谱系落地精度 ≥0.85。不过线不往下走。

### W3 状态层真编译 + 答题端消费（P1 + 审查 S1/S4/S5，约 9–11 天）
- 结构族（综述章节内引用 → 成员，零 LLM）+ 归纳族（分区后全库跑 field_state）→ 按成员重合对齐。
- 留出 20 篇综述及其记录从所有编译输入剔除（硬约束）。
- 答题端：状态卡（族级事实 + 跨论文支持集，装配时展开为多篇论文的引用）、从状态图规划、成员展开检索、
  写作前跨论文"一致/分歧/条件"整理步；按题型路由（具体问题不走状态通道）；snippet 去掉系统前缀。
- 方差控制：plan/screen/write 温度默认 0；外部检索按题落盘可离线回放。

### W4 状态分层与时间（P2 + 审查 S3，约 5.5 天）
- 族级事实立场判定（supports/contradicts/qualifies，限定必须给出条件原文）+ 独立性计数 + 预注册状态规则
  （consensus / contested / qualified / single-source / open）。
- as-of 切片（状态 = 支持集的纯函数）、综述比较表单元格、族的评测清单。不做通用结果矩阵。

### W5 决定性实验与评测严谨性（审查 D1–D5、B1–B10，约 15–20 天 + GPU ~80–100h）
- 留出综述检验修正版（M2）：全部按现版重编、每臂 3 次、direct 用全文摘要、同预算 map-reduce 臂、报精度；
  族成员按 TaxoBench 的 ARI/V-Measure 对标；人工校准判分。
- FieldQA（D1，主实验）：≥150 题、≥40 篇截止后综述、人工核可答性；臂 = Full / −state / 现场编译 / −KB /
  综述 taxonomy 作状态（上界）/ 外部系统；主假设 H1 写死。
- CS2 机制分层（D3）、状态忠实度人工审计（D4）、去记忆（D5）、规模曲线（D2）。
- judge：官方 judge（gemini-3-flash-preview）重判 + DeepSeek 第二 judge + 人工 60 题一致性。
- 基线补齐：PaperQA2、OpenScholar、STORM、Asta SQA solver（同 27B）、AutoSurvey/SurveyForge（留出与 FieldQA）。
- 成本与时延表。

### W6 项目规范化（代码架构 + 工作区治理，待两份审查回来细化）
- 把活代码从 .research_tmp 收进可安装的包，实验只剩薄 runner + 配置；去硬编码路径与 sys.path 注入；
  配置单一来源；prompt 集中；测试覆盖活路径。
- 工作区：代码 / 配置 / 数据产物（带校验和）/ 运行输出 / 缓存 / 日志 / 草稿 / 决策档 分区；git 跟踪规则；
  文档最小集（README / ARCHITECTURE / REPRODUCE / RESULTS / DECISIONS）；旧文档归档。
- 迁移在 CS2 test 跑完之后、用 git mv 保留历史；不删除，只移动到归档并留清单。

## 2. 顺序与并行

W0（test 跑完 + 备份 + 推送解堵：工作区 Phase 1–3）→ W1 ‖ W6（规范化先做，后续新代码直接写进新结构）→ W2 → W3 → W4 → W5。
推送解堵排在最前：215 个提交只在一块硬盘上，是当前最大的单点风险。
W5 里 FieldQA 题集构建（人工标注）可以和 W2–W3 并行（题集只依赖留出综述，不依赖系统）。

## 3. 预注册规则（摘自审查 §5，开工前写进 PREREG.md）
1. test 被碰过的历史全部披露；v9b 结果照实报，不做逐题分析。
2. 升级版只在 dev 上开发；dev offset 10–24 已被选择污染，确认集用 dev 25–99（开发期最多看 2 次）。
3. 升级版冻结后 test 只跑一次（3 次运行算一次），与 v9b 同表；更差也照报。
4. 每个基准一个主假设；次假设族内 Holm；官方 judge 为主口径。
5. 作废只看机制指标（cite=0 >20%、外检报错 >10%、judge 错误行、进程中断），不看分数。
6. FieldQA 题目、金标、可答性标注在任何系统输出之前完成并打哈希。

## 4. 代码与工作区（AUDIT-CODE-ARCHITECTURE 已回；AUDIT-WORKSPACE 待回）

### 4.1 代码架构审查要点
- 答题核心（answer_pipeline 584 + refgraph 439 + cutoff 72 + hybrid 132 行）在 .research_tmp 实验目录、靠 sys.path 注入，不在包里；
  一次导入拉进 ~9.9k 行（registry.py 3,806 行只为 normalize_doi；8 个 _see_* 垫片带进 openai/dotenv）。src/ 30 个文件 ~7k 行不可达。
- 硬编码：sys.path 注入 742 处、C:/Users 绝对路径 254 处、局域网 IP 32 处；活路径上的判分/对照臂脚本换机即坏。
- 测试 146 个全过但活路径零覆盖（answer_pipeline / refgraph / cutoff / hybrid / field_state / cs2_scoring / direct_judge）。
- 冻结清单哈希按 CRLF 工作区字节算、漏 llm.py/embedding/判分器源码/rubric/缓存；KB v2 输入只匹配工作区不匹配 git，就地改写脚本顺序无记录。
- 发布阻断：17 个 >100 MB blob、6.1 GB 数据被跟踪、promote_to_deep.py 明文 token（本地未推送，需轮换）、llm.py 全局关闭 TLS 校验。
- 判分器：astabench 是未跟踪 scratch 目录的 editable 安装 + direct_judge 运行时 4 处补丁；DSB nuggetizer 本地改过、不受版本控制。
- 正确性/公平性：GPTR 臂经 SearchService（15 s 档期 + 600 s 熔断）访问 Sciverse，我们的臂直连排队（未证实影响，先统计空检索率）；
  Multi runner 重跑必崩（MultiKB 不调父类 __init__）；DSB 汇总缺答剔除（CS2 缺答记 0，口径相反）；refgraph 暂时失败被永久缓存成空（7 条）；
  Sciverse 结果不缓存不可重放；LLM 静默降级不计数；同一论文经 KB/外部两通道拿到两个引用编号（r1 355 节中 11 节）。
- 目标架构：单一包 src/compilescholar/{core,llm,sources,kb,compile,answer,eval,baselines} + configs/*.yaml + experiments/<bench>/run.sh +
  third_party 子模块与 patches + runs/<run_id>/（自动 manifest）+ data/ 与 cache/ 出 git；统一检索门面（所有臂共用）、依赖注入、
  LLM/HTTP 录制回放、每题降级计数、缓存三态、截止显式化、manifest 自动生成并可在任意机器 verify。
- 迁移：test 跑完前冻结闭包一律不动、主工作区不做 checkout/stash/gc；重构在 sparse worktree（refactor/package）里做，
  先写表征测试锁住行为、prompt 逐字节等价，再逐条提交行为修复；test 跑完打 tag 后合回 main，旧文件 git mv 进 legacy/。
  对外发布用新的干净仓库，不改写研究仓库历史。

### 4.2 并入工作流
- W1 增：DSB 汇总缺答记 0；GPTR 臂空检索/熔断率统计（决定是否统一检索后重跑对照臂）；Multi runner 崩溃修复；refgraph 缓存三态。
- W6 = 代码审查 §5–§6 的阶段 1–3（阶段 4 改写历史需用户批准）；W6 先于 W2 落地，W2 起新代码直接写进新包。
- 安全项立即处理：轮换 MINERU_TOKEN（用户操作）、llm.py 恢复 TLS 校验（在新包里改，冻结闭包不动）。

### 4.3 代码审查提出的待拍板项（并入 §6 第 7–12 条）

### 4.4 工作区治理审查要点（AUDIT-WORKSPACE.md）
- 盘点：项目约 65 GB；.git 14 GB（有用 pack 237 MB，散对象 10.2 GiB，中断操作残留 tmp 文件 2.94 GiB + 死掉的 gc.pid）；
  .research_tmp ≈49.6 GB，其中已退役实验 ≈28.5 GB（paperscope 归档 19.2 GB）、可重建缓存 ≈12.5 GB。
- **最急：main 推不上 GitHub。** 09-20 后 215 个提交只在这块硬盘上；17 个 >100 MB 文件（13.79 GB，主要是 09-23 的 LightRAG 向量库）。
  已推送历史干净（最大 35 MB），所以只需重写 origin/main..main 这一段，push 仍是 fast-forward。
- HEAD 树 5.78 GB，83% 是 16 个向量/嵌入缓存；两个 .bin 处于已修改状态（**不能 git commit -a**，否则再塞 ~0.9 GB）。
- ignore 规则与跟踪状态脱节：1,039 个文件是强制 add 的；反过来 RESULTS-LEDGER、paper_drafts、phase0、104 份决策档、test100 答案都没入库（只有一份）。
  deepscholar/venv311 没被 ignore，git status 里 3.3 万条噪声。
- 第二处密钥：GPUStack EMBEDDING_API_KEY 写在 _shared/tools/multi_closedbook_recall.py:36 和 STORM 的 run_config.json 产出里（未推送）。
- 冻结 v9b 14 个哈希重算全部一致；但 direct_judge 依赖未跟踪的 scratch/asta-bench；record_vecs.f32（453 MB）只此一份。
- .git/config 的 branch.main.merge 有两个值（一个指向远端不存在的分支），直接 git pull 可能出错。
- 根目录文档（README/DIRECTION/ASSET-STATE）停在 09-20～09-25；当前口径分散在 5 处，memory 里多条"最新"互相矛盾。
- 目标布局与治理规则（与代码审查 §5 合并，以代码审查的包结构为准）：conf/ + src/compilescholar/ + bench（= experiments）/<bench>/ +
  third_party/ + tests/ + docs/{README 状态行, ARCHITECTURE, EXPERIMENTS, RESULTS, DECISIONS, paper/, archive/} +
  data/、artifacts/（ignore，只跟踪 MANIFEST.tsv）+ results/（只放冻结 run，单文件 <5 MB）+ runs/、cache/、scratch/（ignore）；
  单文件 >5 MB 不进 git（pre-commit 拦）、禁止 git add -f、gitleaks、run_id 命名 <bench>-<split>-<arm>-<yyyymmdd>-<tag>、每天 push。
- 迁移阶段（与代码审查 §6 合并）：
  - Phase 1（test 跑完后）：整仓备份到外部盘 + git bundle，校验 FREEZE 哈希。
  - Phase 2（低风险）：删 tmp_* 与 gc.pid；修 .git/config；两处密钥改读 env、STORM 落盘抹 key；补 .gitignore；
    向量缓存 git rm --cached（文件留盘，先登记 MANIFEST）；补跟踪 paper_drafts/phase0/docs_decisions/test 答案。
  - Phase 3（高风险，需批准）：git-filter-repo 只重写 origin/main..main（去 >20 MB blob 与两个密钥），先在副本上 dry-run，保存 commit-map，推新分支再 fast-forward。
  - Phase 4（论文 test 数字定稿后）：目录迁移 + 包重构（即 W6），每步冒烟比对。
  - Phase 5：冷存储释放 ~27 GB。Phase 6：pre-commit + manifest + 每日 push 防复发。

## 5. 风险
- KB 只有 1,262 篇，题目落不进族 → 下游消融仍 ≈0：按"状态覆盖率"分层报；退路 = 领域层直接检验 + 分析论文。
- 局限/开放问题仍 ≈ flat：如实报。
- 状态卡伤引用精度：S1-b 变体（卡只用于规划不作引用）对照。
- 外部额度（OpenAlex 10k/日）与跑批争用：骨架解析排在 test 跑完之后，先用本地 arXiv OAI 快照。

## 6. 需要用户拍板
1. 不做通用跨论文数值结果矩阵，改做"综述比较表单元格 + 族评测清单"。
2. KB 外、被 ≥2 篇库内论文引用的论文作为桩节点进入状态层（改变 KB 构成，论文里披露）。
3. 升级版上不上 CS2 test（建议：上，作为新臂、预注册在先、只跑一次，与 v9b 同表）。
4. 人工标注分工（骨架抽检 ~750 条 + FieldQA 可答性 + judge 一致性 60 题）：我按书面规范先标、你盲审 20%，还是关键项你亲自标。
5. 官方 judge（gemini-3-flash-preview）全量重判要 API 费用，是否批。
6. 叙事走向按 D1 结果二选一（方法论文 / 基准+分析论文），判读规则写进 PREREG，你认可吗。
7. 发布形态：新建干净发布仓库（推荐），还是在镜像上 filter-repo 改写研究仓库历史后推送。两者都要先轮换 MINERU_TOKEN。
8. 对照臂检索统一：先统计 GPTR 的空检索/熔断率，影响可忽略则冻结旧结果并披露路径差异（推荐先做）；或统一后重跑对照臂。
9. 旧建库链：做成可重跑的 DAG，还是把 base_kb 当冻结数据制品（带哈希）发布、脚本归档并写明顺序（推荐后者）。
10. 重构阶段行为零变化、有等价测试守着；行为修复逐条单独提交、只在 dev 评估（推荐）。
11. 判分适配层（重试次数、编号平移、Connection: close、DeepSeek 判分模型）在论文里逐条披露；官方 judge 重判用去掉补丁的原样 scorer。
12. 数据制品托管位置（Zenodo / HF dataset / 校内）与许可：KB 含论文逐字片段，发布前确认可再分发范围。
13. 历史重写（工作区审查 Phase 3）：只重写未推送的 215 个提交、去大文件与密钥，提交哈希会变（保存对照表）；
    或备选：从 origin/main 开新分支放一个瘦身快照提交，细粒度历史只留在 bundle 里。两者都需要你明确同意。
14. 外部盘备份：Phase 1 要整仓（~65 GB）或关键子集复制到外部盘，需要你提供盘符/位置。
15. 修 .git/config 重复项、临时设 gc.auto 0（防止后台 auto-gc 去打包 10 GB 散对象）——改 git 配置需你同意。
16. 轮换 MinerU token 与 GPUStack embedding key（需你在对应服务上操作）。
