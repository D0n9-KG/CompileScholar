# DECISION: corpus_driver.py — dual-track blocks + paper_id-keyed instance store (Side B2)

Date: 2026-08-14
Goal step: 阶段顺序 step 2

## 现状调研（动手前必须看清）
1. LogicKG 现有 `load_paper_blocks(paper_id)`（structure_mapper.py:80）直接读本地
   `MINERU_BASE = science_evo/data/upstream/remote_mineru/mineru_2355/papers/{pid}/content_list.json`
   （2355 篇 granular 语料）。
2. sci-evo-extract 库 `data/library/papers/`（219 篇，含 mineru markdown/content_list artifacts + references）。
3. **关键发现：两边 paper_id 形态一致（OpenAlex PPR_xxx）但 0 重叠**——2355 篇和 219 篇是不同论文集。
   → "双轨验证"不能在同篇论文上比 local vs API；改为分别验证两条路径各自产出同格式 blocks。
4. sci-evo `mineru_content_list` artifact 是合法 JSON 数组（{type:text,text:...}），
   与本地 content_list.json 结构一致 → 可复用 load_paper_blocks 的解析逻辑。
5. InstanceCorpus（hypergraph_schema.py:113）已按 paper_id 内存存实例，但无持久化。
   → step2 的"超图按paper_id存"= 加持久化层（JSON 文件 keyed by paper_id），step6 再迁入 sci-evo artifact。

## 决策（最小、可验证）
新建 `src/granular_agent/corpus_driver.py`，`CorpusDriver`：
- `load_blocks(paper_id)`：local-first（MINERU_BASE/{pid}/content_list.json）→ 缺则 API fallback
  （client.get_content_list → mineru_content_list artifact JSON）。统一复用 _blocks_from_content_list
  解析（与 load_paper_blocks 同格式：{index,char_start,char_end,text}，跳引用块）。
  → 这是"双轨"：2355 走 local，sci-evo 219 走 API，统一入口。
- `load_references(paper_id, fetch=False)`：委托 client.get_references（step3 结构信号数据源）。
- `save_instance(paper_id, inst)` / `load_instance(paper_id)`：实例超图 JSON 持久化到
  instance_root（默认 .research_tmp/instances/{pid}.json），按 paper_id 键。
  InstanceHypergraph 已有 to_dict/as_dict 风格 → 用现有序列化（见 hypergraph_schema InstanceHypergraph.to_dict）。
- `has_instance(paper_id)`：避免重复抽取（持久化缓存）。

## 验证标准（纪律4，单篇迭代）
smoke_corpus_driver.py：
1. local 路径不回归：取一个 2355 篇 paper_id，load_blocks 非空（与旧 load_paper_blocks 同结果）。
2. API 路径产出同格式：取一个 sci-evo 篇（PPR_D3F75E896E34），load_blocks（local 缺→API fallback）
   非空，block 字段齐全 {index,char_start,char_end,text}。
3. references：对 PPR_6AB55969EFBC load_references ≥1 条。
4. instance 持久化 round-trip：对一个已抽取的 InstanceHypergraph save→load，节点/超边数一致。

## 不做（防过度）
- 不改 agent.py 的 process_paper_hypergraph 主流程接 client（step3+ 才动 lift_corpus；
  agent 接入可留到机制验证后）。
- 不迁 2355 进 sci-evo（重，非必要）。
- 不在 sci-evo 侧加超图 artifact 存储（step6）。
- 不做 embedding 对齐/合并（InstanceCorpus 已有，不碰）。

## 纪律
- 临时文件（DECISION/smoke/instances）进 .research_tmp，不 commit；只 commit src/corpus_driver.py。
- 不引入新依赖（client 已是 stdlib urllib）。
