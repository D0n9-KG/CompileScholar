# Plan: 跨论文对齐 agent 详设（builder 群第 3 个）

## 形态：EDC pipeline + MELD 5-outcome + HASTE 跨域不混
对标 EDC(Extract→Define→Canonicalize) + MELD(5-outcome claim裁决) + HASTE(跨域分层不混) + DecentMem(centralized塌缩多样性)。

## 现有局限
align_concepts 单pass LLM批判同义; surface为主+LLM补判据弱; 无跨域隔离; 无5-outcome(只merge/not); 无增量(全量重); 无对齐质量度量。

## pipeline
1. **Define**(LLM, ≠抽取模型): 每新concept(METHOD/PHENOMENON/PARAMETER)生成自然语言定义(EDC Define)——不只surface有语义描述供对齐判
2. **Canonicalize**(embedding+LLM judge): embedding(GLM-Embedding-2现有)找同域内最近候选; LLM judge判是否对齐(EDC式); 走**5-outcome**(MELD): insert(新)/merge(合并surface+provenance)/relate(连独立)/conflict(同名矛盾进冲突检测)/reject(噪声丢); claim-key identity(surface+symbol)+embedding+NLI(LLM)判定
3. **增量对齐**: 每篇ingest后只对齐本篇新concept+候选,不全量重跑(EDC增量)
4. **跨域不混**(HASTE): 只在同domain namespace找候选+对齐,跨域不对齐

## 跟底座接
role=builder.aligner, write contract=align_merge/add_concept/add_concept_relation/conflict_mark(只A-box,不改T-box)。Mutation→transactional commit(5-outcome裁决在底座kernel层做,aligner调)。

## 创新点作用+ablation(守顶会)
跨论文对齐支撑"n-ary超图A-box跨论文累积"——创新点"n-ary+方法演化关系一等公民"A-box靠对齐跨论文成图。ablation:
- embedding+LLM judge vs surface-only(现有) vs LLM-only
- 5-outcome vs 二元merge/not(现有)
- 增量 vs 全量
- 跨域隔离 vs 不隔离(看混域污染)

## 现有结构改(增量,不破坏)
- Concept加definition字段(EDC Define,现有canonical_name/symbol,加definition供对齐判)
- align_concepts改EDC pipeline(Define→Canonicalize embedding+judge→5-outcome)
- 跨域namespace(底座分层,aligner只查同域)
- 增量对齐接口(ingest后只对齐新concept)

## 验证
对齐质量(同概念合并对不对judge≠抽取模型/误合并/漏合并); 5-outcome分流正确率; 跨域不混(颗粒不跟ML合并); 增量vs全量; ablation(embedding+judge/surface-only/LLM-only/5-outcome vs二元/隔离vs不隔离)。

## 不做
不跨域对齐(塌缩多样性); 不二元merge/not(5-outcome是有控核心不降级); 不全量重(增量但设计完整非简化)。
