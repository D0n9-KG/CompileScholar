# CompileScholar 全量修复与规范化计划（2026-09-23 定稿）

> 用户裁定：仔细扫描全流程每个细节，合并既有 33 项修复清单，覆盖设计漏洞、
> 效率、工程规范化（编排/测试/日志），防屎山。本文件为唯一权威执行清单。
> 原则：P0 正确性 → P1 效率/编排 → P2 规范化 → P3 能力扩展（跑后议）。
> 每项完成打勾并附 commit hash。

---

## P0 正确性（修完重判分，我方臂成绩才有效）

- [x] **P0-1 笔记回引解析器修复**（新发现，最高优先）
  bug：`_has_valid_id` 先按 `[|,;/\s]+` 拆词再逐词匹配——`[entity: X]`
  和含空格的实体名永远失配，即使名字在 registry 里完全合法。
  实测后果：2,898 次假拒绝（最狠一题 1,097 次）→ 85/108 触发防循环
  硬停 → 强制编译兜底 → 引用丢失 → Citation F1 归零。
  修法：匹配顺序改为 整段内容 → `entity: X` 后缀 → 拆词兜底。
  验证：对 108 题的既有轨迹重放解析器，假拒绝率应从 ~100% 降到个位数。

- [x] **P0-2 编译兜底路径引用保真**（新发现）
  bug：85/108 走了 fallback 编译（LLM 从笔记重写答案），编译输出丢失
  笔记里的 `[paper_stem]` 引用标记（34 题自造 `[Wan#5400]` 格式、74 题
  全裸）。官方桥翻译不了 → 0 引用。
  修法：①编译 prompt 强制"逐字保留笔记行首的 [stem] 标记"（已有但
  不够，需加示例+反例）；②编译后确定性回查：笔记含 N 个不同 stem、
  答案里一个都没有 → 拒收重编一次 → 仍失败则机械回填（把 stem 插到
  对应段落末尾——保真优先于文采）。
  验证：重跑 108 题的编译层（不重新答题，笔记都在），引用保留率
  应 >95%。

- [x] **P0-3 直接答案路径引用核查**
  同样核查非编译路径（23 题）的答案引用是否存活，同修法回查。

- [x] **P0-4 判分-数据版本锁定 + 全量重判分**
  教训：append-only 判分 + 可变输入文件 + 旧代码常驻进程 = 缝合分数
  （0.5165 作废）。
  修法：①判分行记录输入文件 sha256；②重判时输入哈希不一致自动作废；
  ③postpass 改为答题结束后一次性运行（删除常驻循环——数据冻结后
  周期重写纯属可变性自造）。
  然后全量重判 ours 三轨（F1/AutoAIS/rubric），出真实成绩。

- [x] **P0-5 工作纪律入档**：对共享数据写路径的任何代码变更，必须清点
  并重启全部相关常驻进程（今天 13:19 postpass 带旧代码跑到晚上的根因）。
  写入 POSTRUN-FIXES 教训节 + 长期记忆。

## P1 效率/编排（用户点名：真实科研场景的可用性）

**答题侧**（现状：median 11 步/题 × 步间串行 × 30k 预填充 ≈ 23 分钟/题；
  85/108 是硬停烧满预算的——P0-1 修复后此数会大幅下降，先修再测）

- [~] **P1-1 拒绝螺旋根治后的效率复测**：20 题分层样本（seed 20260924，
  按 subject 配额 6/6/4/2/1/1）已开跑（resample20 tag，registry_v3+全部
  G1 修复生效）。三指标=步数/硬停率/墙钟。
- [x] **P1-2 笔记膨胀治理**：83/108 超 5k 协议上限（median 7.2k、max
  18k）→ 预填充随步数滚雪球。落实真正的截断纪律：超限时机械截断
  最旧低优先级行（非提示词恳求）。
- [ ] **P1-3 步内并行 action 推广**：协议允许一步多 action（成本算一
  步），模型用得少。yanyu 案例一步 4 个 card 并行是对的——在工具
  目录提示里加"实体扫描类查询应合并为一步多 action"的显式引导。
- [ ] **P1-4 比较题批查模式**：多实体比较题第一步就把全部实体的 card
  并行扫掉（现在是逐个查浪费 3-5 步）。
- [ ] **P1-5 动态步数下限校准**：当前公式偏保守，小题浪费预算。

**建库侧**（现状：全链 ~14h，其中 slot 马拉松 ~8h、postcheck 2.4h、
  registry_growth 38min、views 1min）

- [x] **P1-6 postcheck O(n²) 写放大**（旧 #1）：514MB × 430 次全量重写。
  改 append-only 分片（per-paper jsonl + 末次合并）。
- [x] **P1-7 build.py 依赖组真并行**（审计项过期——并行调度已实装，dry-run 验证）（旧 #4）：PARALLEL_GROUPS 已声明，
  驱动仍线性。Deep_extract+table_extract 并行、postcheck+notation 并行，
  预计全链 14h→9h。
- [x] **P1-8 deep_extract WAL 分片**（旧 #5）：**核销——A2 的 chunk 级
  WAL（records_slot.json.progress.jsonl，(pid,cid) 键、ok=false 行不算
  完成标记、断尾行容忍）已满足诉求**；单文件 8MB/430 篇，10× 语料
  重放也是一次性分钟级，无按 paper 分片必要（09-24 复核）。
- [x] **P1-9 GPU 调度分级**（旧 #21）：大 max_tokens 调用（280s 级）与
  短调用混跑撞墙。kb_infra 信号量分级：大调用独立低并发闸。
- [~] **P1-10 LightRAG 轮询→回调**（旧 #3）+ PaperQA 索引增量（旧 #2）：
  **PaperQA 半核销**——索引按目录持久+按 doc 断点续跑（历史失败全在
  配置哈希与探针 bug，#30/31 已修）；**LightRAG 并行 ingest 推迟到 QASA
  接入时**——Multi 入库已完成无验证目标，串行+核实标记设计是打过仗的
  （430/430），现在改纯属无验证机会的 churn（09-24 判断）。

**基线侧**
- [x] **P1-11 PaperQA 原生工具调用验证轮**（F1 0.4784 取优）：服务器模型稳定后重跑
  （模拟版 F1=0.4493 已有；原生版取更高者报告）。今天 53 题 404 垃圾
  已清理。

## P2 工程规范化（用户点名：防屎山）

**目录与包结构**
- [ ] **P2-1 实验工具收编**：_shared/tools/ 下 17 个脚本 49 处
  sys.path hack → 全部走 editable install 的包内 import
  （kb_benchmarks 包：arms/ judges/ rubrics/ 三子模块）。
  **09-24 调整：推迟到 resample20 + G2 A/B 跑完之后**——中途重构会
  造成 A/B 两臂间代码漂移（除 grounding 开关外必须零差异），且 GPU
  占用时无法做端到端导入冒烟。纯结构无行为变更的活，留到有完整
  验证窗口时做。
- [ ] **P2-2 入口统一**：所有可执行入口收进 `kb` CLI（现有 build.py +
  kb CLI 扩展 answer/judge/stats 子命令），README 对照表更新。
  （随 P2-1 一起推迟）

**日志系统**（现状：kb_compiler 123 处 print、0 处 logging）
- [x] **P2-3 统一 logging 框架**：runlog 落地（runs/<stage>-<ts>/
  events.jsonl + summary.json），**09-24 收口：子进程 stdout 捕获进
  run 目录（stdout.log）**——124 处 print 保留为人类可读流随 run 目录
  落盘；全量 print→event 转化判定为得不偿失（stage 以子进程跑，
  父进程事件层已够）。
- [x] **P2-4 LLM 调用日志已是 JSONL 账本**（现有），补：每账本行加
  caller（哪个阶段哪条臂）+ prompt 哈希（脱敏），判分账本独立命名
  已做（judge/ledger_judge.jsonl）。**09-24 落地：caller 字段 +
  prompt_hash（sha256-12）+ 嵌入调用入账（cst-embed/local-embed/
  paratera-embed 行；audit #12 一并收）。**
- [x] **P2-5 看门狗升级**：加"连续 N 次 ok=False"探针（今天 GPUStack
  模型掉线 40 分钟才发现——进程活着但全 404 的形态旧探针盲区）；
  探针全部单调化（今天的两起假警报教训）。

**测试**
- [ ] **P2-6 现有 106 测试迁移归位**：tests/ 与新包结构对齐。
  （随 P2-1 一起推迟；当前 132 测试全绿）
- [x] **P2-7 补关键回归测试**：①回引解析器（P0-1 的 108 题轨迹重放
  做成永久回归）；②编译引用保真（P0-2）；③判分版本锁定（P0-4）；
  ④postpass 幂等性。**09-24 增：registry dedup 5 例 + card 重定向
  3 例 + compare 词汇提示 4 例 + 账本字段 2 例 + 纯度闸 4 例。**
- [x] **P2-8 冒烟隔离纪律**（旧 #29）：**09-24 落地：kb/smoke/ 隔离区**
  （cards/records/slot 冒烟产物 + postcheck_smoke_dry 移出生产 kb 根）。

**数据管理**
- [x] **P2-9 判分输入冻结**：**09-24 落地：multi_judge_freeze.py**
  （三臂答案+题面快照进 judge/snapshots/<ts>/ + sha256 MANIFEST；
  judge --snapshot 读冻结副本）。
- [x] **P2-10 产物清单**：**09-24 落地：write_run_manifest 共享助手**
  （答题 run 产出 MANIFEST-<tag>.json：答案/轨迹/账本哈希+git sha），
  与 build.py 的 runs/manifest-*.json 对齐。
- [x] **P2-12 纯度断言最小调用量闸**：check_arm_purity min_calls（空账本
  假 PURE 盲区，PaperQA 索引死亡案例）；两基线 runner 已接
  min_calls=len(todo_qs)。

## P3 能力扩展（本轮跑完重议，旧 #16-18 + #22）

- [ ] 跨篇符号消歧（notation symbol-namespace）
- [ ] 多步公式推理（符号依赖图）
- [ ] 图表通道集成（figure_channel，VLM 预算 ~22M 待批）
- [ ] registry_growth 证据随行重设计（设计层，用户升级过）
- [ ] postcheck 汇流终检（通道对称性，旧 #19）
- [ ] Linux 迁移评估（旧 #20，Windows 文件语义三案之后）

## 执行顺序与依赖

```
P0-1 → P0-2/3 → P0-4（重判分出真实成绩）→ P1-1（复测定效率基线）
     → P2-3（日志先行，后续改动才有观测）→ P2-1/2/6/7（结构+测试）
     → P1-2..10（效率逐项）→ P2 其余 → P3 议后
```

预计工作量：P0 半天、P2 一天、P1 一到两天（P1-6/7/9 是大头）。


## 09-23/24 深夜执行记录（P0 全清 + N5-N13 + P1-C/D + 效率项）

- P0 全部六项完成（N1 answer_raw 桥接 = 引用灾难真根因；N3/N4 判词器；
  P0-1 匹配器；N2/N7 数值门；N11 原子写；P0-4 版本锁）
- N5-N13 九项全部完成（F18 死代码复活等）
- **P1-C 引用修复**（新发现：编译路径重编号/越界/伪造 hex）：
  ours F1 0.5183→**0.6503**，双基线配对显著（+0.172/+0.136，CI 下界>0）
- **P1-D 防复发**：借 PaperQA types.py:511 的幻觉引用剥离设计 +
  编译 prompt 正反例；LightRAG 的 References 段设计入备选
- P1-2 笔记机械截断（分层优先级）落地
- P1-6 postcheck 分片写 + 分片恢复（写放大 430×500MB→单次合并）
- P1-9 LLM 信号量分级（大调用独立低并发道）
- 基线核查：PaperQA/LightRAG 无同类引用格式病（机制差异分析入档）
- AutoAIS 轨按用户裁定：不重跑，已有数字留档披露节
- commits: d21ecbd2, dba24b69, a37820ca, P1-C, P1-D, P1-2, P1-6, P1-9


## 附录：POSTRUN-FIXES 35 项与主计划的完整对账（防遗漏，2026-09-24）

| # | 问题 | 状态 |
|---|---|---|
| 1 | postcheck O(n²) 写放大 | ✅ P1-6 分片写 |
| 2 | PaperQA 索引无增量断点 | ⏳ P1-10（QASA 开工前） |
| 3 | LightRAG 入库轮询 | ⏳ P1-10（QASA 开工前） |
| 4 | build.py 并行未实装 | ✅ 核销（审计项过期，已实装） |
| 5 | deep_extract WAL 全量重放 | ⏳ P1-8（QASA 开工前） |
| 6 | record_count=0 论文清单 | ⏳ 小项，下轮建库时进 manifest |
| 7 | result 记录 role/scope 弱 | ⏳ P3 域（影响检索质量，下轮建库观察） |
| 8 | 表格综述全进 overflow | ⏳ P3 域（类型系统扩容议题） |
| 9 | fallback 路由 31% | ⏳ P3 域（cards sections 覆盖） |
| 10 | c23 永久失败 chunk | ✅ 已披露不追（9k 预算超限） |
| 11 | 纯度断言空账本盲区 | ⏳ P2-12 待加最小调用量闸 |
| 12 | litellm 嵌入不进账本 | ⏳ P2-4 |
| 13 | 账本多进程写竞态 | ⏳ 低风险观察项 |
| 14 | 拆点器/裁判同模型张力 | ⏳ goldcov 重启时改机械规范 |
| 15 | monitor 误报教训 | ✅ 已吸收进看门狗设计（物理探针+单调化） |
| 16-18 | 符号消歧/公式推理/图表 | ⏳ P3（能力扩展，跑完重议） |
| 19 | postcheck 通道不对称 | ⏳ P3（汇流终检） |
| 20 | Linux 迁移 | ⏳ P3（Windows 文件语义三案后评估） |
| 21 | GPU 混合负载调度 | ✅ P1-9 信号量分级 |
| 22 | registry_growth 上下文薄 | ⏳ P3 设计层（证据随行方案） |
| 23 | 嵌入超时级联 | ✅ 当晚修复（参数回退） |
| 24 | registry run5 翻案 | ✅ salvage 泛化+WAL |
| 25 | LightRAG WinError-5 | ✅ atomic_write 重试补丁 |
| 26 | 看门狗 run_id 脆弱 | ✅ 当晚修复 |
| 27 | PaperQA 账本全损 | ✅ 异步双列表挂载 |
| 28 | 编译引用形态断裂 | ✅ P0-2/P1-C 全链修复 |
| 29 | 冒烟残留污染 | ⏳ P2-8 |
| 30 | PaperQA 三层死因链 | ✅ 全修（shim/模拟层/凭证） |
| 31 | PaperQA 索引侵蚀 | ✅ use_absolute=False 根修 |
| 32 | 队首写盘阻塞 | ✅ as_completed |
| 33 | 答题效率 | ✅ P1-2 落地+P1-1 复测待跑 |
| 34 | 常驻进程代码漂移 | ✅ 教训入档+一次性 postpass |
| 35 | 模型掉线盲区 | ✅ P2-5 FAILING 态 |

**对账结论**：35 项中 18 项已修、1 项核销、16 项在册未失联
（P1 待做 4 项 / P2 待做 6 项 / P3 域 6 项）。无遗漏。

## 五、系统级差距清单（2026-09-24 全景复盘新增，答题行为数据驱动）

> 数据来源：108 题真实轨迹的 tool-usage 审计 + registry/views 产物交叉分析。
> 这些不是 bug——是"系统建成后第一次拿真实负载照 X 光"照出的设计差距。
> **09-24 夜间更新：G1 三件套已落地（B1/B2/B3），G3 已结案（见下）。**

### G1 编译视图在答题端严重未兑现（最重要的战略发现）
- 实测：compare 全场仅 27 次调用（我们建了 5,079 张矩阵表）；
  lineage 18 次；find_gap 14 次；而 **search_text（全文裸搜兜底）351 次**
  ——模型用兜底通道的频率是 typed 工具主力的 13 倍
- 根因三层：①compare 的 metric 参数用自由文本（"accumulation in
  brain"）对撞编译好的 metric 词表（"psf||psf contrast"式）→ 67% 空结果
  ②card 32% 空返回（44% 的 registry 实体是 out_of_corpus——模型从
  grounding 列表里挑了无语料记录的实体）③矩阵表的组织方式
  （subject||metric 双键）没有暴露给模型可发现的结构
- 修法方向（P4-B，答题端改造）：**全部落地 09-24**
  - [x] B1 compare 参数词汇提示：空结果返回 vocab_hint（近似编译键
    top-10 + 实体在/不在矩阵诊断 + 轴混淆提示"X 是矩阵实体不是
    subject/metric"——QLoRA/NV-Embed 实测形态）。**重放验证：
    "accumulation in brain"空调用现在直接给出 "icr mice||brain
    accumulation" 等 4 个正确键。**
  - [x] B2 grounding 标注：数据修正——153 次空 card 中 146 次是
    out_of_corpus 实体、7 次自由文本、**0 次带星实体**（`*` 标记本身
    有效）；修法改为 card() miss 返回 nearest_in_corpus 重定向
    top-3 + prompt 强化"未标记实体 card() 恒空"。
  - [x] B3 比较题引导：playbook 第 9 条 compare-first（数字先入笔记，
    findings/card 只做解释；空结果跟 vocab_hint 重定向）。
  - [ ] 验收（G1 修复后 compare/card 空返回率 <10%）随 resample20 出数。

### G2 registry 未仲裁实体直接上线答题
- 7,759 个 round2_growth 实体（74% of registry）从未人工审过就在
  答题端活着——audit #22 的"上下文薄"问题的另一面。至少 4543 个
  out_of_corpus 是潜在噪声源
- 修法：仲裁队列处理（抽检+批量规则），或答题端 grounding 只暴露
  仲裁通过的子集（保守模式 A/B 测试）
- **09-24：GROUNDING_CONSERVATIVE=1 开关已落地（星标-only 实体列表），
  resample20 跑完即以同题集跑 B 臂。**

### G3 views 的 cards 投影损耗 —— **已结案（09-24，非 views bug）**
- 95 张缺口全分解：**87 = 综述/practice 论文**（method_identity.
  canonical_name 为空，卡片层正确判定无方法身份；内容经 dossier pass
  流入被提及实体卡）；**8 = 多论文同实体合并**（IL-B4 跨篇卷宗设计：
  DLCZ protocol×2、PCEM×2 等 7 例合法；1 例真错并 = classifier-free
  guidance 被并进 ADM 别名）
- **顺藤挖出真 bug：registry_v2 有 1,450 行重复实体**（923 个 entity_id
  ×2-11）——growth 的 action=new 路径不查重，LLM 用不同大小写提已存
  canonical 就 md5 撞 id 追加；views.cards 按 entity_id 建字典被静默覆盖、
  grounding 列表重复污染。**修法三件套**：源头守卫（dup_guard_folds）+
  registry_dedup.py 确定性修复（v2→v3：10,418→8,968）+ CFG/ADM 拆分
  patch（CFG 卡现归 Ho&Salimans 原论文）。views 已从 v3 重建；build.py
  加 registry_dedup 站。

### G4 引用链探索零实现
- 我们的差异化主张"沿引用网络游走"目前只有 fetch_chunk（53 次调用）
  一个近亲——真正的 citation-graph 工具（"这篇引了谁/被谁引/方法谱系
  上下游"）不存在。lineage 18 次调用是唯一谱系工具但只覆盖 971 条边
- 与 P4-A 检索模块合并设计：库内引用图（manifest 有 refs 数据）+
  外部引用图（OpenAlex/S2 的 citation API）统一成一个
  citation_graph 工具

### G5 效率结构性账本（真实场景可用性）
- 答题：23 分钟/题（修复后待复测，P1-1 resample20 在跑；注意服务器
  单流速率约首跑一半——聚合吞吐持平，单题墙钟会偏长，判读三指标时
  以步数/硬停率为准，墙钟除以并发修正）
- 建库：~14h/430 篇（P1-7 并行已实装但未全链复测）
- 检索延迟：P4-A 目标快路径 <2s
- 三者都达标才构成"真实科研场景可用"的论文叙事
