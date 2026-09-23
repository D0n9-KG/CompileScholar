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
