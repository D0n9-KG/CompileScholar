# 词表人工审判决档（09-21，用户委托代审）

> **委托记录**：预注册 S1 的人工审闸门，用户 09-21 明示"你帮我审一下吧，我不懂这些领域"——审核权委托给 Claude 执行。审核方式从抽样阅读升级为**确定性系统检查全量扫**+抽样实读双轨。用户保留随时抽查权（所有样本与数据可复现，REVIEW-PACKAGE.md + 本档 + 原始产物均已版本化）。

## 1. Registry 判决：**接受**（2659 实体）

系统检查（全量，非抽样）：
| 检查 | 结果 | 判定 |
|---|---|---|
| (a) alias 冲突（同表面名在多实体） | **2 处**（poli-OCM/poli-OCT 互吸） | 可接受残留 |
| (b) 实体 A 的 canonical 出现在实体 B 的 alias（吞并） | **4 对**（CFG→ADM、DLCZ→某论文题名实体、poli 互吸×2） | 可接受残留 |
| (c) 前缀变体折叠（iterated/meta-/mini- 类，F34 家族） | **0** | 守卫有效 |
| (d) 坍缩组（own-paper 覆盖） | 3 组 | 已入 QC，卡片 alias 质量所致 |

→ **真实混淆合计 3 处 / 2659 实体 = 0.11%**，全部低提及。处置：记录在案不返工；首跑败题归因若指向这 3 处再仲裁修复（归因驱动，不盲目迭代）。

7 旗标处置（用户委托裁定）：phantom A/B、optical tweezer、BIC 双实体=接受残留（quote 级可分辨）；GPT-3 尺寸变体入家族=接受（alias 保留表面名）；DLCZ 漏合、引文残渣实体=残留类不动。

**副产品发现**：registry 的 entity_type 对数据集类实体系统性误标 method（Dolma/FineWeb/BEIR 等 39 例实锤）——卡片类型投票的已知噪声，qc suspect_entity_type(66) 抓的就是这类。影响面=typed tools 的类型过滤，**记入首跑观察项**。

## 2. Vocab 判决：**接受**（subject 1381 家族 / setup 3156 / variant 3152 / hyperparam 971）

| 检查 | 结果 |
|---|---|
| (d) 重复规范名（去重降级损伤量化） | **全维度 0 重复**——412 个降级块无测量损伤 |
| 确定性清理 | 28 条 budget-like 出 setup ✓ |
| suspect_family_merge | 453（仲裁队列性质，抽样已入包件 §2f） |
| 合规事件 | 412（主体=跨批去重块的 terse 空答，语义正确） |

## 3. Blocklist 扩域判决：**522 条入库，G3 改全词匹配，金丝雀 PASS**

- 540 提案 → 排 6（IceCube×2 数据/事件、NRHybSur3dq8 方法、RACE-h/m 实验方案、ncbi 机构）→ 排 12 短名（**跨域同形词**：GAP=band gap/BOLD=fMRI/DROP=drop-casting/BBH=双黑洞/Pile=pile-up/ROOTS/BOOKS/NEWS/WIKI/MATH；**实体错标**：Wanda/Ribo-T 是方法）→ **522 条**（`src/kb_compiler/records/data/blocklist_ext_multi431.json`）
- 被 registry 类型噪声冤枉的 39 条真基准（Dolma/FineWeb/BEIR…）经实读甄别后保留
- **G3 门改全词匹配**（裸子串会误杀 "dropout"⊃"drop"、"band gap"⊃"gap"）；单元探针 10/10（含 gap size=0 hits、fever accuracy=1 hit）
- **F35 金丝雀复验 PASS**（C1 role-reversed + C2 normal-rows，G5 硬停线通过）。插曲：首跑 FAIL 系手动调用漏传 --model 吃了 Qwen3.6 旧默认——生产跑一律走 run_stage {MODEL} 占位符，此类雷已封
- bio/photonics 零入库已验证=域特性（2251 subject 表面名无目录类名），非模型偏见

## 4. 总判决

**Registry+Vocab 放行 slot 深抽阶段。** 残留登记：3 处实体混淆、453 suspect_family_merge、entity_type 数据集误标——全部记录在案，归因驱动处置，不阻塞。

下一道门：**slot 冒烟硬门**（8-10 篇分层+RAW 输出形状检查[契约金丝雀，09-21 新纪律]+PS-53 基线指标对照），过门才放行 430 篇马拉松。
