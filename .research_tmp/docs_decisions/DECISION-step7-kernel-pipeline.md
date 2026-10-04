# DECISION: Step7 新管线(process_paper_via_kernel) vs 旧管线(process_paper_hypergraph)并存

## 背景
Step7 把 Step1-5 builder群接进 agent.py. 设计原意(GOAL_MODE_BOOTSTRAP行61/72): 现有代码重组为"底座读写者"非搬非第二条并行管线. 但完全替换 process_paper_hypergraph 风险高(评测脚本/P0 baseline依赖).

## 决策
**加新方法 process_paper_via_kernel 走KB底座, 保留旧 process_paper_hypergraph 不动(ablation对照+评测连续性). KB复用agent的meta_hg/concept_graph同对象(transactional). 两条管线并存但同KB对象, 不是第二条独立管线.**

## 理由
1. surgical: 不破坏能跑通的旧管线(P0/SCION baseline/评测全靠它).
2. 设计"重组为读写者": 新方法是读写者版(走ledger/validate/replay), 旧方法保留作对照.
3. ablation: 新vs旧可直接对比(KB transactional vs 直接改meta), 是Step8 ablation的天然对照.

## LangGraph定位(子代理确认)
设计行99"框架LangGraph StateGraph+Command+Store指向KB内核(编排脚手架)". Step7 用纯Python串行循环对——LangGraph是脚手架非核心, 单builder链串行不需要StateGraph. consumer群(Step6未做)+HITL才需LangGraph interrupts. **Step7不引LangGraph, 推迟到consumer/HITL阶段(MINOR).**

## MA1: evolver failure feed 合成假trigger(子代理抓的MAJOR)
process_paper_via_kernel 初版合成 Hyperedge(pattern_type="_unknown", node_ids=["a","b"], evidence_span=sec_text[:120]) 喂evolver. 真跑实测 n_proposed=0(schema并进空跑, version没bump), 旧管线split=1. 这是合成trigger被cross_node gate拦(cross=1<2)的正确行为, 但效果是演化没发生.
**修法**: extraction_agent.verify/fix 返回真 dropped_edges(带verifier reason), process_paper_via_kernel 喂真失败边给evolver.propose_validate_failures, 不再合成. 跨section累积cross_node>=2才触发演化(设计要求的recurring crystallize).
诚实: 单section单paper的cross_node=1被拦是设计正确(recurring才演化), 不是bug. 但合成trigger是假信号, 必须换真失败边.

## MA2: _kernel_llm_extract的deepseek分支死代码
if llm=="deepseek": call_llm(model="deepseek-chat") 永不触发(llms默认"DeepSeek-V4-Flash"). 删分支或改显式provider参数.

## 不做(诚实范围)
- HITL: 预留接口(evolver标needs_hitl), 真HITL由consumer/HITL阶段接
- maintainer prune在batch end跑(非per-paper)避免误retire, 单篇调用者自调
- consumer SAGE反馈接口预留(Step6接真consumer)
