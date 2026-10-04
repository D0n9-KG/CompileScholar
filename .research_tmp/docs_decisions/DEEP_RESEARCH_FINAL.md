# 深度调研最终结论 (2026-08-14, 读6个工作全文/摘要)

## 调研覆盖
本地全文: DIAL-KG, SCION, Hyper-KGGen, Intern-Atlas, AutoSchemaKG(新), Adaptive Schema-aware EE(新)
OpenAlex title搜索: 14个query, 覆盖 schema evolution/adaptive/n-ary/method evolution/ontology learning等

## 两个新工作的澄清(读全文后)

### AutoSchemaKG (ACL 2026, Yangqiu Song组)
- **不是自演化schema** — 是"事后schema conceptualization"(抽完triple再LLM生成概念短语)
- 无schema演化操作(无add/split/merge/retire), 无收敛控制, 无pattern拓扑
- binary triple + event, web通用域, 50M文档
- 评测: triple F1 + schema语义对齐(92%) + multi-hop QA

### Adaptive Schema-aware Event Extraction with RAG (EMNLP 2025 findings)
- **不是schema演化** — 是"检索选固定schema"(从几百固定schema检索匹配)
- schema是固定的, 只做retrieval+matching, 无演化操作
- event(binary), 无拓扑, 无跨论文
- 评测: schema matching + extraction accuracy

## 6个工作都没做的(我们的真差异化)

| 能力 | DIAL-KG | SCION | Hyper-KGGen | Intern-Atlas | AutoSchemaKG | Adaptive-EE | 我们 |
|------|---------|-------|-------------|--------------|--------------|------------|------|
| schema演化(多操作) | add+merge | single-shot | static skill | citation-causal | 事后抽象 | 固定 | **五操作** |
| 收敛控制 | parsimony(定性) | N/A | N/A | N/A | 无 | N/A | **coherence gate(量化)** |
| n-ary超图 | binary | binary | n-ary | binary | binary | event | **n-ary** |
| pattern级拓扑 | 无 | inter-event | 无 | citation-causal(binary) | 无 | 无 | **dep/con/comp** |
| 跨论文方法演化 | 增量 | N/A | 无 | citation-anchored | 无 | 无 | **共享meta跨论文** |
| 科学域 | 通用 | 通用 | 通用 | AI域 | web通用 | 事件 | **颗粒流** |
| 综述gold评测 | 无 | 无 | HyperDocRED | NMR/ERR | triple F1 | schema match | **综述gold** |

## 真正的创新点(确认)

**三要素组合仍成立, 且更清晰:**
1. **自演化schema(五操作)+量化收敛控制** — 6个工作都没做(都有schema但无演化操作+收敛)
   - AutoSchemaKG事后抽象, Adaptive-EE固定schema, DIAL-KG有merge但无量化收敛
   - 我们的coherence gate是唯一的量化收敛控制
2. **n-ary超图+pattern级拓扑** — 只有Hyper-KGGen有n-ary但无拓扑, 其他都binary
3. **跨论文方法演化(共享meta)** — Intern-Atlas用citation-anchored(binary), 我们用共享meta演化

## 诚实警示
- 收敛控制(coherence gate)是我们独有, 但"收敛"要可量化证明(否则和DIAL-KG parsimony一样定性)
- pattern拓扑(dep/con/comp)是我们独有, 但当前是弱语义(共现推断), LOO测出无用 — 要强化或降级
- n-ary和Hyper-KGGen重叠(它也n-ary), 但+拓扑+演化组合独有

## 推荐方向(不变)
A. 收敛的自演化schema(攻Gap1, 我们独有coherence gate)
B. 下游任务驱动schema评测(攻Gap2, 验证演化价值)
**B优先于A** — 先验证演化有价值(下游), 再投入收敛控制。

## 和之前判断的变化
之前担心AutoSchemaKG/Adaptive-EE撞我们核心 — 读全文后确认都没撞:
- AutoSchemaKG是事后抽象(非演化)
- Adaptive-EE是固定schema检索(非演化)
我们的"自演化schema(五操作)+收敛"仍是真差异化。
