# 外部检索模块设计基线分析（2026-09-24，用户四点指示的落实）

> 用户裁定：sci-evo-extract 现状≠成熟检索系统。四点问题：①免费号限流
> ②arXiv 检索通道缺席疑云 ③初始召回质量（关键词搜索太浅）④真实场景
> 延迟。目标=测试集问题/真实用户查询/科学 agent 自主科研查询三种输入
> 都能低延迟高准确返回文献+解答。

## 一、sci-evo-extract 现状盘点（代码级核验，非记忆）

### 已有的（比担心的好一点）
- **获取链已分层递进**（acquisition_chain.py）：arXiv PDF（有 arxiv_id
  时最优）→ OA PDF（OpenAlex/S2/Crossref 元数据里的开放获取 URL）→
  Sciverse → 本地 DOI 库，每级身份验证（题名匹配防错抓）
- **限流处理已有基础**：OpenAlex polite pool（mailto 注册，~10x 预算）
  + 429 指数退避；S2 有 x-api-key 通道（无 key 走匿名池，429 高发）
  + Retry-After 荣誉；Crossref 客户端在位
- **三源元数据发现**（acquisition.py discover_by_query）：OpenAlex +
  Crossref + Sciverse 并行搜索（不是串行递进——是全发+合并）

### 确认的缺陷（用户记忆全部属实）
1. **arXiv 作为"搜索源"缺席**：sources.py 里 arXiv 只出现在 S2 查询
   构造器（`ArXiv:` 前缀传给 S2），**没有独立的 arXiv API 搜索客户端**。
   arXiv 只在"已知 arxiv_id"时当下载通道。代码注释自证："arXiv search
   API itself is rate-limited to death under sustained load (429 bursts)
   and must NOT be the discovery path"——所以当初故意不做的，但这话只对
   sustained load 成立，**单查询场景应该有它**（对 CS 域它是最准的源）
2. **限流是"重试等待"不是"递进降级"**：OpenAlex 429 时退避重试同一源，
   **不会切换到 Crossref/S2**。用户设想的"OpenAlex 限流→转 Crossref"
   递进逻辑不存在。我们的 Multi 语料战役实际用外层脚本（resolve_corpus
   .py 手写 429 冷却+attempt 循环）打补丁解决的，不在库里
3. **查询理解层为零**：discover_by_query 是裸关键词直传三个 API——
   没有问题分解、没有查询改写、没有意图识别（query_type 只分
   doi/title/topic 三类做路由，不做语义增强）
4. **无延迟意识**：三源并行全发（延迟=最慢源），没有按需分级（快源
   先回先出）、没有缓存层、没有并发预算管理

## 二、成熟系统怎么做（调研要点，PaSa/SPAR/OpenScholar/学术检索经典）

| 能力 | 成熟做法 | 我们的差距 |
|---|---|---|
| 查询意图增强 | 问题分解（复杂问题→多个子查询）+ 关键词抽取 + 同义改写（LLM 一次调用产 3-5 个变体查询） | 零 |
| 源选择 | 按域/年份/文献类型路由（CS 新文献→arXiv 优先，生医→PubMed/EuropePMC） | 无路由，三源盲发 |
| 限流韧性 | 每源健康度追踪（熔断器模式：连续 429→该源冷却 N 分钟→自动降级到次源） | 单源退避重试，无降级 |
| 结果融合 | RRF/加权合并多源候选+去重（DOI 归一） | candidates 简单拼接 |
| 重排 | 嵌入模型对 query×候选摘要打分（cross-encoder 或 bi-encoder） | 无（title 字面匹配） |
| 缓存 | 查询→结果 LRU 缓存 + 元数据 DOI 缓存 | 无 |
| 延迟分级 | 快路径（缓存命中/单源快回）<2s；慢路径（全源+重排）<10s | 无 |

## 三、模块设计骨架（讨论稿，裁点等你）

```
查询输入（问题/自然语言/agent 查询）
  ↓
[1] 查询理解层（LLM，一次调用，~2s）
    → 意图分类（找论文/找事实/综述扫描/追踪方法谱系）
    → 分解 2-4 个子查询 + 关键词组 + 域提示（CS/bio/physics）
    → 每个子查询标注源偏好
  ↓
[2] 源路由+熔断层（确定性，0s）
    → 按域偏好排序源（CS: arXiv→S2→OpenAlex；bio: OpenAlex→Crossref）
    → 熔断器：各源滑动窗口 429 计数，超阈→冷却 10min→从本轮路由摘除
    → 递进策略：首源限流/空结果→自动降级次源（用户的层级递进设想）
  ↓
[3] 获取层（复用 sci-evo 获取链，加分片缓存）
    → 元数据候选（带 DOI 归一去重）
    → 按需取全文（首 K 篇走获取链，其余留元数据）
  ↓
[4] 重排层（本地嵌入，~1s）
    → query 嵌入 × 候选（title+abstract）嵌入 → 余弦重排
    → 与我们 KB 的 typed tools 对齐：命中已入库论文直接回 KB 记录
  ↓
[5] 融合输出
    → 文献列表（排序分+DOI+OA链接）+ 延迟预算报告
```

延迟目标：快路径（KB 命中+缓存）<2s；全路径 <15s（查询理解 2s + 获取
5-8s + 重排 1s + 余量）。这对真实场景可用。

## 四、与修复计划的关系

新清单项（插入 MASTER-FIX-PLAN）：
- **P4-A 外部检索模块**（原"SQA2 前置"升级为独立模块工程）：
  A1 源熔断+递进降级（治限流，确定性改造，~0.5 天）
  A2 arXiv 搜索客户端（单查询场景，礼貌限速 1req/3s，~0.5 天）
  A3 查询理解层（LLM 分解+改写+源偏好，~1 天）
  A4 嵌入重排+缓存（~1 天）
  A5 延迟分级+仪表（快慢路径，~0.5 天）
  A6 初步验证=对 Multi 语料 108 题做"闭卷模拟开放检索"（把 430 篇语料
     从 KB 里藏起来，用检索模块从零召回，量 Recall@K——不依赖 SQA2）
- SQA2 排在 P4-A 全部完成+验证之后（用户裁定，已修正推进顺序）

## 五、落地记录（2026-09-24 深夜，A1-A5+G4 全部建成）

全部在 sci-evo-extract 仓库（commits: A1+A2 / A3+A4 / A5+deadlock 修复 /
G4 / 单测 7 例），每个模块都有现场验证：

- **A1 熔断+递进降级**（circuit.py + discover_tiered）：SourceCircuit
  滑动窗口（3 失败/120s→OPEN 600s→HALF_OPEN 单探针）；discover_tiered
  按 tier 顺序查，circuit-open 跳过、错误/超时（默认 20s，防 S2 匿名池
  Retry-After 分钟级挂死）降级+计数、诚实空结果降级不熔断、DOI 归一去
  重、达标早停。现场验证：arXiv 406 限流→超时降级→S2 429→超时降级→
  OpenAlex 1.3s 接住且目标论文第一。
- **A2 arXiv 搜索客户端**（sources.py ArxivClient）：Atom 解析、
  1req/3s 跨实例礼貌间隔、退避重试。两个实测坑：search_query 里
  `:` 必须字面（%3A→HTTP 406）；406 也是 arXiv 的限流信号（同一 URL
  200 后连发数分钟内变 406）→长退避重试。现场：0.3s 命中目标论文。
- **A3 查询理解层**（query_understanding.py）：一次小 LLM 调用→
  intent+domain+2-4 关键词式子查询+源偏好；可注入 chat callable
  （CompileScholar 侧接 kb_infra 入账）；LLM 失败确定性回退原查询。
  现场（GPU 满载）：LNP 题→bio/openalex+3 子查询（含 DLS/NMR/HPLC
  特征化词表）；CFG 题→cs/arxiv+s2。
- **A4 嵌入重排+两级缓存**（rerank.py）：candidate_text 重建（OpenAlex
  倒排索引摘要/arXiv summary/S2 abstract）+余弦重排；QueryResultCache
  （进程内 LRU+TTL）+DoiMetaCache（文件 JSONL 原子写）。现场：目标论文
  第 4→第 1（0.886），异域 diffusion 沉底，0.4s/批。
- **A5 延迟分级**（search_service.py）：fast（缓存命中/单层 5s 帽，无
  LLM 无重排）/full（A3→分层发现→去重→A4）/auto（fast 薄结果升级）；
  每结果带分级延迟明细。现场：fast 1.7s、缓存 0ms、full 14.6s（GPU
  满载；idle 时 understand ~2s）。**修复两枚真 bug**：SourceCircuit
  .snapshot() 持非重入锁自死锁；executor shutdown 阻塞调用方退出。
- **G4 引用图服务**（citation_graph.py）：设计勘误——manifest 无 refs
  字段（430 条实测零），库内引用结构=genealogy（已有 lineage 工具）；
  缺的是外部游走。S2 双向（OpenAlex 只出向，作出向兜底）+DOI 键磁盘
  缓存（7 天 TTL）+语料回指标注。现场：CFG 论文 52 边 2.0s，ADM 论文
  正确标 in_corpus。
- **A6 闭卷验证**在跑（multi_closedbook_recall.py，108 题全量，
  Recall@10 ≥0.7 过门）。初判风险：gold 是特定论文选择（同一主题下
  检索返回的 top-10 全切题但可能不含 gold 本尊）——若 FAIL，改进方向
  =子查询多样性/每子查询单独配额/S2 key（匿名池 429 是当前主要降级源）。

## 六、S2 替代源判决表（2026-09-25，用户指示：S2 key 申请未回，评估国内源）

> 一手实测（每条都是真调用验证，非文档转述）。

| 能力 | Sciverse（上海AI Lab，已有token） | AMiner（清华） | PubScholar（中科院） |
|---|---|---|---|
| 论文检索 | ✅ meta-search + **agentic-search（语义检索全文 chunk）** | ✅ 论文搜索 API（"限免"） | ✅ 检索 API 存在（`/hky/open/resources/api/v1/articles`） |
| 摘要字段 | ✅ **922-1037 字符全文 abstract**（meta-search 返回） | ？（文档页需登录才见字段） | ✅ abstracts 全文 |
| **词汇鸿沟能力**（A6 主诉求） | ✅ **实测 agentic-search 直接命中 gold**：glycosylation 查询 1.4s 拿到 "sweet side of protein corona" 正文 chunk——OpenAlex 关键词永远搜不到的那篇 | 未测（需 key） | 未测（需过签名墙） |
| 引文数据 | ⚠️ meta-search 带 citation_count/influential_citation_count 字段，**无引用图 API**（引文游走仍靠 S2/OpenAlex） | ✅ 论文引用 API（0.1元/次，出向） | ❓ 未确认 |
| 中文文献 | ✅ 实测中文查询返回中文结果（钙钛矿稳定性研究） | ✅（本土优势） | ✅（本土优势） |
| 限流 | ✅ 8/8 连发零失败（0.5s 间隔） | 未测 | 有签名墙（x-xsrf-token+signature+nonce+timestamp 指纹校验） |
| 接入成本 | **零**（客户端已在库里，只差进 tier 路由） | 需注册+充值（按次计费 0.01-1 元，论文搜索"限免"） | 高（WAF 反爬签名，浏览器才能过；无公开 API 文档） |
| 数据规模 | ？ | 论文 3.0 亿+ | 期刊论文 1.08 亿 |

### 判决

1. **Sciverse 立即顶上 S2 的检索位**——这不再是妥协方案，是升级：
   - agentic-search 是**语义检索**（检索全文 chunk），正好补 A6 诊断出的词汇
     鸿沟（gold 靠特定发现连题、题面无 gold 词汇）——单发关键词检索的地板
     直接被抬起来
   - meta-search 已带全 abstract + 引文计数，SQA2 broker 的证据形态（裁点
     1：abstract 即证据文本）完全满足
   - 零接入成本：客户端已在 sci-evo 库里，缺的只是把它加进 discover_tiered
     的 tier 路由 + agentic-search 包一个通道
   - 唯一缺口：**引用图**（双向引文游走）Sciverse 没有 API——G4 的
     CitationGraphService 保持 S2（匿名池凑合）+OpenAlex（出向）现状，
     S2 key 批下来是锦上添花
2. **PubScholar 出局**（当前形态）：接口藏在浏览器后面（x-xsrf-token +
   signature + fingerprint 指纹校验），无公开 API 文档，程序化接入=持续
   对抗其 WAF，脆弱且不适合论文系统；除非他们将来出正式开放 API
3. **AMiner 观望**：论文搜索"限免"+按次计费（组合接口 0.2 元/次），
   注册充值成本之外的增量能力（相对 Sciverse）只有出向引用 API——
   不值得为它多一条维护线；若未来 Sciverse 限流再评估
4. **S2 key 申请继续挂着**：批下来只用于引用图（G4 的双向游走是 S2 独占
   能力），检索位已由 Sciverse 补齐

## 七、知识模型驱动检索落地（2026-09-25 下午，用户方向纠偏后）

> 用户裁决："光用外部检索工具和引用链别人都做过没创新性"——检索的差异性
> 必须来自我们系统的独特知识（缺口/谱系/空格坐标）。引擎层（Sciverse 语义
> 通道+熔断+重排）保留为执行底座，新增的是**知识模型驱动的检索原语**。

### 三个新原语（全部 live 验证，sci-evo 仓库 commits）

| 原语 | 机制（别人没有的部分） | live 验证 |
|---|---|---|
| **gap_search** | 题面嵌入匹配 3,953 条缺口坐标（absence+derived hole）→ 合成查询携带**题面没有的词汇**（缺口记录注入实体名+缺失内容词）→ 共享引擎执行 → fill 标注回配 | 签名场景：题面零 SAMMR → 缺口注入 SAMMR 局限原文 → 返回 OCT 色散补偿真补缺文献+fill 标注 |
| **lineage_walk** | 库内**类型化**谱系 BFS（extends/improves/replaces 边，区别于无差别引用 BFS）→ 边界检测（终节点+深度截断有后继者）→ 外部续走合成后继查询 → 后继标注 | LLaVA-1.5→LLaVA→InstructBLIP 库内链→边界→外部返回 TG-LLaVA/LLaVA-CoT 真后继 |
| **Tier 1 粗抽器** | 摘要→同 schema 子集轻记录（finding/method/limitation + mentions 谱系钩子），coarse: id 前缀防碰撞，provenance=coarse 区分 | CFG 摘要→4 记录；sweet-side 摘要→limitation 记录（缺口信号） |

### 投资阶梯（用户定的粗/深分层设计）

```
Tier 0 元数据（免费）→ Tier 1 粗抽 11-16s/篇 → Tier 2 全流程 ~2min/篇
选择权在答题循环的模型手里：初召回 Tier 0 → 核心论文 Tier 1（extract_paper
工具）→ 答题必需的 1-3 篇 Tier 2。延迟结构=快路径粗证据初答+深抽异步升级。
```

### Broker 接线（external_tools.py，已 live 验证）

五个外部工具注入答题循环（fork 侧，冻结模块不动）：search_papers /
gap_search / lineage_walk_ext / citation_graph / extract_paper。观测=
title+abstract（官方 asta 工具同构形态）；引用回指门放宽到 title/doi；
prompt 目录第 13-17 条（何时用驱动式 vs 盲搜的引导）。

### 检索=多步任务（用户定位："深入检索才是根本问题"）

ReAct 循环的检索移动集：①词表精炼（实测有效）②引用跟随（业界主流）
③**缺口定向**（我们独有）④**谱系外推**（我们独有）⑤实体枢纽。差异
性=③④+①②⑤的移动选择受知识模型指导。两轮协议最小验证已通过（round-1
池摘要见 "selective" → round-2 精炼 → gold 排第 3）。

### 存储衔接（用户裁定）

sci-evo 的 SQLite registry（papers 主键+processing_jobs+extraction_runs）
直接用：粗/深两级状态挂 processing_jobs，抽取账目进 extraction_runs。
两项目后续合并成一个完整项目（SQA2 之后）。
