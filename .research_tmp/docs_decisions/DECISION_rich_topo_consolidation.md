# DECISION: 富拓扑确定性 consolidation（深修方向）

日期: 2026-08-14
状态: 已确认问题，选定先走确定性 consolidation，不够再升级两遍 LLM

## 确认的问题（5 篇 ARFM instance 实测）

在落盘 instance 上跑 `infer_rich_topology_direct` + 手写 `clean_method_labels` 复核：

| paper | METHOD 原→清 | m_param | m_phen | m_reg | nary | comp | law_par |
|---|---|---|---|---|---|---|---|
| Pouliquen_1999 | 16→16 | 5 | 1 | 2 | 5 | 6 | 6 |
| Bouzid_2013 | 17→17 | 2 | 1 | 0 | 2 | 0 | 10 |
| Bouzid_2015 | 25→22 | 4 | 3 | 0 | 0 | 1 | 7 |
| Kamrin_2015 | 13→13 | 1 | 5 | 1 | 1 | 0 | 1 |
| Midi_2004 | 16→6 | 0 | 0 | 2 | 0 | 4 | 30 |

### 4 类系统性问题
1. **METHOD 标签垃圾（clean regex 漏抓）**：论文标题、`Eq. (1)` 公式引用、设备（rheometer/laser sheet/image processing/front tracking）、泛词（Theoretical approaches/empirical fit）
2. **重复方法 surface**：Kamrin 的 NGF 一个方法 5 个 surface（nonlocal granular fluidity model / NGF model / nonlocal fluidity model / nonlocal fluidity / nonlocal granular fluidity）；inertial rheology×3
3. **孤儿方法节点**：Midi 的 μ(I) rheology 连 0 边（律超边只含参数，方法被抽成孤儿没接线）
4. **composition 全噪声**："dry grains ⊃ continuous shear"、"分成两族"当组成、几何列举当组成

### 关键正面发现
**真 method-bearing 边存在，只是被埋住**：Kamrin 的 5 条 method_phenomenon 是真 captures（NGF captures nonlocal effects/bistable hysteresis/Reynolds dilation/silo jamming），能对齐 gold captures_edges。Midi 是最差个例（综述体结构散），不能代表全部。

## 选定修法：先扩确定性 consolidation

### 为什么不直接两遍 LLM
- 真边被垃圾标签+重复 surface 埋住，确定性规则能捞回一大部分（Kamrin 5 captures 只需去重 NGF surface 就能对齐）
- 两遍 LLM 翻倍 deepseek 调用（慢+不稳+贵），是确定性不够再升级的后手
- 确定性可验证、可逆，符合 [[no-simplification-avoid-rework]] 后处理路线（和 infer_rich_topology_direct 同类）

### consolidation 三步（在 proto 脚本快循环验证，不重跑 deepseek）
1. **扩 clean_method_labels**：加 Eq./equation 引用、论文标题（matches paper_id 模式）、设备（rheometer/laser sheet/image processing system/front tracking/X-ray）、泛词（Theoretical approaches/empirical fit/Theoretical model）→ 重标 PROPERTY 或弃 METHOD
2. **dedup_method_surfaces**：同篇内 METHOD 节点 surface 模糊匹配（子串/缩写展开 NGF/nonlocal fluidity 等已知同义），合并为单一节点，把所有引用该节点 id 的超边指向合并后节点
3. **wire_orphan_method_into_law**：对孤儿 METHOD（0 边），若其 surface 是命名律（μ(I)/local rheology），找 evidence 含该律符号的 constitutive_law 超边，注入 METHOD 节点（grounded：律符号在律 evidence 里，非共现猜）

### 验证标准
5 篇上 method-bearing 边（去重后）：质量提升（人工抽检 10 条 method_phenomenon 真比例↑）、composition 噪声↓、METHOD 节点数降到每篇 3-8（合理方法数）。达标则折入 hypergraph_evolution.py 真实管线；不达标升级两遍 LLM。

### 不做的
- 不退回 co-occurrence 猜关系（铁律）
- 不为评测而 post-process 出假边（评测独立不循环）
- composition 噪声若清理后仍无真值，诚实报 0 不凑数
