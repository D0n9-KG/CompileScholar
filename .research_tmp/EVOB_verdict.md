# 演化收益对照实验判决（EVOB，2026-08-27）

设计：同 5 篇 granular 验收语料 × 同 API 双 judge（GLM+gpt-oss-120b），双臂唯一
差变量 = 起始 schema：SEED 臂（种子 12 patterns，零演化起点）vs ML 臂（加载
ml_schema_v1.json=ML 域 6 篇演化的 29 patterns）。两臂 arm=full（演化同开）。
runner：run_evo_benefit.py。单 run 对照（未多 seed）。

## 结果

| 臂 | 边数 | both-pass | either-pass |
|---|---|---|---|
| SEED (12 pat) | 808 | 318/808 = 39.4% | 62.9% |
| ML v1 (29 pat) | 1001 | 407/1001 = 40.7% | 63.2% |

逐篇 both 率（ML vs SEED）：DQN 50.0/40.9、理论 34.5/23.9、实验 36.5/39.7、
建模 41.4/42.0、短文 44.3/43.6。

## 判决

1. **质量 +1.3pt（不显著）+ 召回 +24%（显著，超 ±13% 方差带）**——演化 schema
   的价值形态=更丰富词汇抽到种子 schema 结构性抽不出的边（边多而 both 率不掉
   =新增边质量与存量持平）。检索场景下游要覆盖，这是直接利好。
2. **跨域迁移成立**：ML 演化 schema 用于 granular 语料无衰退反增益；领先集中
   在方法演化叙述最密的文体（DQN/理论），DRL 演化 pattern 的用武之地。
3. **架构承重证据**：离线演化→在线 frozen 复用的路线拿到首个对照证据。

## 边界

- 单 run；质量差在噪声带内（+1.3pt）——报告引用需标注"方向性证据"
- ML 臂加载的 schema 含 DRL 特化 pattern，对 granular 理论篇的增益可能与
  pattern 数量（29 vs 12）本身有关，未拆分"域特异 vs 纯数量"贡献
