# 阶段 0.2 判决：报告适配器（叙事编译层）建成（2026-09-27）

## 交付物

`experiments/benchmarks/_shared/tools/report_adapter.py`（与答题栈同层，
槽 3/4/5 共用）：

| 层 | 实现 | 验证 |
|---|---|---|
| notes 解析（确定性） | F31V2 N/X 行 → 主张+回指；X 行剔除 | 单测+真实 31/118/9 条 notes |
| 叙事编译（LLM） | 主张+证据 quote → 分节行文+[Ck] 标记；matched 27B（thinking 必须显式关，坑见下） | 三题+模板题全过 |
| 装配（确定性） | [Ck]→[n] 重编号；**按论文合并引用条目**（同源多主张共用一号，官方同款）；snippets=KB verbatim quote（1.0 档）；相邻同号去重；残留清理 | 双向 id 校验 0 错 |
| 渲染器 | CS2 JSON + Markdown（槽 4/5，含 References 表） | 产物实读 |
| 校验器 | 对照官方消费路径（extract_json_from_response 语义/同节无重复 id/id 逐字在文/1.0 档占比） | 全部 valid=True |

## 冒烟结果（真实 Multi-108 notes，本地 27B）

| 题 | notes | 词数 | 节 | 引用 | 1.0 档 | 违规 |
|---|---|---|---|---|---|---|
| norman_bio_1 | 31 | 641 | 5 | 4 | 4/4 | 0 |
| yanyu_photonics_5 | 118 | 963 | 5 | 6 | 6/6 | 0 |
| jacqueline_cs_5 | 9 | 589 | 5 | 4 | 3/4（1 条 paper 级=0.5 档，如实降档） | 0 |
| yanyu_photonics_5（四节模板） | 118 | 662 | Landscape/Comparison/Open Gaps/Dynamics | 6 | 5/6 | **1 例 snippet 落文被抓** |

- LLM 编造的 C 标记（超界引用）被剥离并计数（jacqueline_cs_5: 4 个 orphan）
- 官方契约全部一手核验后落地：types/sqa.py 的 section/citation schema、
  task.py 的 all_at_once 判分路径、filter_citation 的 alpha 规约
  （snippet 复现在正文 → 降 0.5 档——适配器有检测器，冒烟实测能抓）

## 工程坑（记录在案）

1. **call_local 必须 `enable_thinking=False`**：否则 4000 max_tokens 全被
   思考吃掉，content 空——GLM-5.3 教训在本地 Qwen 同样成立
2. **call_local 的模型名是服务端名**（`Qwen3.8-27B`），不是 harness 的
   `local:` 前缀路由写法——30ms 即拒，ledger 一眼可辨
3. LLM 会用 `[C5, C7]` 逗号复合格式，正则需兼容；同论文多主张应合并
   引用条目（首版一主张一号导致 11 个号同篇的丑形态，已修）

## 对设计档的回写

- §八-1 落地清单"输出适配器"完成：**KB 真实 quote 作 snippets = 1.0 档
  citation recall 路径打通**（官方不验 snippets 真实性 → 我们验且真）
- 叙事编译层=硬依赖（审计档原话）已就位；prompt 质量迭代归阶段 1
  dev20 形态税实验（本次 4 题中 1 例 snippet 落文=检测器在岗的实证）
- 槽 4/5 节模板参数已验（FieldState 四节/Landscape 形态直接可用）
