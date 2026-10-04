# Schema v1.3 仲裁材料包（用户拍板项）— 2026-09-11 深夜备妥

## 议题一：finding 的 cited 语义位缺口（本轮新证据加固）

**问题**：转述他篇研究内容的 finding（相关工作里的机制描述、对他法结论的批评/转引）没有合法的认知状态标记。result 类型有 epistemic 字段（cited/demonstrated/stated），finding 只有 strength（证据强度轴）——两个维度不同轴。

**实测证据（PSFIX 收集，schema_v13_evidence.json）**：主重建+验证跑共 **53 条记录因模型自发写 strength='cited' 被枚举门丢弃**（enum:strength 违规共 31 条为最大枚举违规类）。抽样人工实读（REBUILD-MANUAL-READ B2/B3 两例）判语一致："模型想标'转述他篇方法的机制'——语义直觉正确，冻结 schema 无合法枚举承载→按门丢弃正确，缺口在 schema"。典型 claim："Huang et al. (2022) propose an approach that makes use of rank constraints…"——这显然不是本篇 demonstrated 的结论。

**选项**：

- **A（推荐）：finding 增加 epistemic 字段**，枚举沿用 result 的 (cited/demonstrated/stated)。维度正交（强度×来源各一轴）；result 已有先例；1d 教练已在教"strength 与 epistemic 是两个不同字段"——模型行为证明它需要的正是 epistemic 位。成本：schema v1.2→v1.3 升版、KIND_SLICES 注入更新、旧记录字段缺失=None 向后兼容、下游读字段处兼容。**无需全量重建**（增量字段，旧库不重抽也可用）。
- **B：strength 枚举加 reported_elsewhere**。改动最小，但把"证据强度"和"知识来源"混进单轴——后续任何强度×来源组合都要造复合枚举值，语义债。
- **C：不改，继续丢弃**。损失一类真实有价值的内容（他篇机制转述=gap/trend 类问题的原料；跨论文综述场景核心材料）。53 条/17 篇的丢弃率说明这不是长尾。

**连带小项**：figure kind（图通道记录类型）同属 v1.3 议题，但依赖图通道规格评审（FIGURE-CHANNEL-SPEC-OUTLINE.md），建议拆开——本轮只仲裁 finding-cited，figure kind 随图通道规格一起定。

## 议题二：937+ 新实体仲裁（材料包另备，约整块时间）

registry round2 治理产出 937 个新实体（1232 总量），战役建库前的人工仲裁义务。已实测的污染形态（PSFIX F16 诊断顺带）：3 个 pid 字面实体（3mnWvUZIXt/WyEdX2R4er/ez7w0Ss4g9 注册为 out_of_corpus）+ 描述串实体（"methods for obtaining statistical guarantees…"等 9 个 >60 字符 canonical）+ 引用列表串实体（"Athiwaratkun et al., 2022; Cassano et al., 2023;…"）+ 公式串实体（LaTeX 整式当实体名）+ 通用词实体（"activation"）。FX-B 修复已阻止新增 pid/标题形态入册；存量清污=仲裁内容。预分桶材料包见 psfix_2026-09-11/arb937_pack/（自动分桶：明确保留/形态可疑/重复可疑，可疑桶逐条过、保留桶抽读）。

**注意**：FX-E 重抽会产生增量表面名进 entity_queue（预注册已披露），仲裁应在重抽并入后一次做完，避免两轮。
