# 新发现工作对创新定位的冲击 (2026-08-14)

## 两个直接撞我们的新工作

### 1. AutoSchemaKG (ACL 2026, Yangqiu Song 组)
- 全名: "Autonomous KG Construction through Dynamic Schema Induction from Web-Scale Corpora"
- 核心: "fully autonomous KG, eliminates predefined schemas, LLM simultaneously extract triples AND induce schemas directly from text"
- "conceptualization to organize instances into semantic categories"
- 规模: 50M文档, 900M节点, 5.9B边 (ATLAS)
- **撞我们**: "同时抽取+诱导schema" = 我们的 extract+schema演化; "conceptualization组织实例" = 我们的 pattern归类
- 差异: 它是 binary triple + 大规模web; 我们是 n-ary超图 + 颗粒流科学域 + 综述评测
- **但核心机制(同时抽取+诱导schema)撞了**

### 2. Adaptive Schema-aware Event Extraction with RAG (EMNLP 2025 findings)
- 核心: "selecting appropriate schemas from hundreds of candidates" + 检索增强
- 自承两个gap: (1) rigid schema fixation (2) absence of benchmarks for evaluating joint schema matching and extraction
- **撞我们**: 检索式schema注入(我们刚做) + schema评测(我们想做)
- 差异: 它是事件抽取(binary event); 我们是n-ary超图
- **但检索式选schema + schema评测它都做了**

## 对我们创新定位的冲击

之前4个候选创新点重新评估:
- A. 收敛的自演化schema: AutoSchemaKG也做dynamic schema induction, 我们的"收敛"要和它差异化(它没提收敛控制)
- B. 下游任务驱动schema评测: Adaptive Schema-aware EE已做"joint schema matching and extraction benchmark"
- C. 篇内n-ary↔跨论文: AutoSchemaKG大规模但binary, 我们n-ary+跨论文方法演化可能仍差异
- D. 强语义拓扑: 仍无人做(n-ary+pattern级拓扑)

## 读AutoSchemaKG全文后的修正(2026-08-14)

**AutoSchemaKG的schema induction机制(读全文确认):**
- 事后批量conceptualization: 先抽triple, 再对每个元素(entity/event/relation)用LLM生成≥3个概念短语(不同抽象层级)
- **不是动态演化**: "抽完再抽象", 不是"边抽边演化schema"
- **无split/merge/retire/收敛控制**: 只批量生成概念短语, 无schema演化操作
- 评测: triple F1(vs OpenIE) + schema语义对齐(92% with human) + multi-hop QA + LLM factuality

**关键区别(它没撞我们核心):**
| | AutoSchemaKG | 我们 |
|--|-------------|------|
| schema生成 | 事后批量conceptualize | 边抽边演化(extract+演化闭环) |
| schema操作 | 无(只生成概念短语) | 五操作(add/split/merge/retire/rename) |
| 收敛控制 | 无 | 有(coherence gate) |
| instance | binary triple+event | n-ary超图 |
| pattern拓扑 | 无 | dep/con/comp |
| 域 | web通用 | 颗粒流科学 |

**修正结论: AutoSchemaKG不是"自演化schema"——是"事后schema conceptualization"。**
它没有schema演化操作、收敛控制、pattern拓扑。我们的"自演化schema(五操作)+收敛控制"和它本质不同。
"同时抽取+诱导schema"是宣传语, 实际机制不同(它事后抽象, 我们边抽边演化)。

**但Adaptive Schema-aware EE仍撞检索式注入+schema评测**——那个要再读确认。
论文里要讲清: 我们是"边抽边演化的自进化schema(有操作+收敛)", 不是"事后schema抽象"。
