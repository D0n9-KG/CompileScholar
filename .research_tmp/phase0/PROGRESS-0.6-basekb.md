# 阶段 0.6 完结：CS2 热启动基库（综述骨干方案 B，2026-09-28 凌晨）

> **状态：热库建成。** 全部产物在 base_kb/，typed tools 端到端验证
> 通过（实体解析 5/5、lineage 14 边/BERT 含 ERNIE extends BERT 真
> 谱系）。compare/card 空返回是接入调优项（矩阵行结构/卡片路径），
> 不阻塞 CS2 dev20 开跑。

## 热库最终构成

| 层 | 产物 | 规模 |
|---|---|---|
| Tier-1 粗抽 | coarse_records.json | 1,958 篇→7,824 条 |
| **综述层（S1-S4）** | records_survey.json + survey_records_adapted | **136 篇→18,748 条**（lineage 2,294/gap 2,162/snapshot 1,580/claim 12,712） |
| hub 供血层 | records_hub.json | 21 篇→3,798 条一手记录（result 1,020/config 907） |
| registry | registry_v2.json | **15,100 实体**（240 批次 LLM 归并零失败） |
| views | views_cs2.json | 谱系 15,100 节点/2,406 边/1,679 闭包；矩阵 534 表；absence 2,298 |
| 共识层 | views_consensus.json | 11,620 实体的综述共识视图 |
| 全库合流 | records_merged.json + manifest_all.json | **2,100 篇 / 31,312 条记录** |

survey_extract 组件沉淀在 `src/kb_compiler/records/survey_extract.py`
（生产代码位，题内生长摸到综述自动走它——识别路由在粗抽层打标后接）。

## 遗留调优项（CS2 dev20 边测边调清单）

1. compare(Adam) 空：matrix 视图 534 表的行结构与 compare 的
   subject/metric 匹配路径需对齐（survey snapshot 的 comparison_table
   323 表是最可能的数据源）
2. card(Adam) 空：configs/findings 的卡片装配路径
3. registry 15,100 实体未仲裁（growth report 是仲裁队列——
   registry_dedup 应跑一轮防重复，Multi-108 的 G3 教训）
4. vocab 空（轻路线）——分带比较需要 setup/variant 维度词表，
   dev20 后从粗抽 subject 归纳
5. survey 识别路由（粗抽打标→深抽分流）未接线——题内生长用


## 方案 B 定稿（用户裁定链）

1. **作弊质疑成立**：dev 预演检索选深抽池=题目邻域预装，S1' 证据
   被库偏污染；且与题内生长职责重叠 → demand 判据作废
2. **引用量骨干**：用户自行推演恢复设计档原意（"预建骨干面向
   领域，引用量是便宜合理指标"）——A'' 落地时三道闸清掉社科渗漏
   （概念+Physical Sciences 域+全文可得硬筛）
3. **综述骨干（B，终版）**：用户三个连续洞察——①热库关键需求=
   搭领域骨架，综述正是最好信息源 ②综述需要专门抽取器适配系统
   （转述不进事实层在抽取器实现）③综述的参考文献清单=专家策展
   的领域重要论文（跨综述共识频次做 hub 选取主源）

## 三层分工（终版）

- **综述层**：149 篇 arXiv 综述（年代分层 37/45/67，五大类，
  全文可得）→ survey_extract 专门抽取（S1 谱系边/S2 领域快照/
  S3 缺口四档/S4 受控转述）
- **hub 供血层（~80 篇）**：跨综述共识频次 top（纯领域信号）
  +高引补盲——空心锚点变实心（第一方 config/result）
- **题内生长**：中长尾+题目特定（答题时）

## 已完成

| 步骤 | 结果 |
|---|---|
| Tier-1 粗抽 | 1,958 篇→7,824 条（含 bulk 层；7.2 分钟实测） |
| 需求池容器噪声过滤 | 51 条出版物丛书条目清除 |
| 综述池 | 149 篇（survey_arxiv.py：快照路线，OpenAlex 概念路线渗漏 IoT/微电网被否） |
| survey_extract 设计稿 | 冻结（survey_extract_design.md：S1-S4+语义分节+门规则+识别路由） |
| **survey_extract 实现** | `src/kb_compiler/records/survey_extract.py`（复用 chunk_text+call_json；语义路由+S1-S4+确定性门） |
| **真实综述冒烟 3 轮迭代** | 2 篇（icl/dg 综述）终版：**264 条记录全过验收**（lineage 22/snapshot 19/gap 50/claim 173；quote 锚定 100%；claims_about 门零漏；chunk fail 5%） |
| 综述全文获取 | 136/149 篇 arXiv HTML（13 篇老综述无 HTML） |
| **综述深抽马拉松** | **完成：136 篇→18,748 条记录，零错误**（claim 12,712/lineage 2,294/gap 2,162/snapshot 1,580；中位 112 条/综述） |
| views 适配 | survey→views 兼容形态（lineage 2,294+absence 2,162+consensus 14,292；共识层 11,620 实体） |
| hub 共识统计 | 完成：36 篇真骨干（Adam/AlexNet/LSTM/LLaMA/PyTorch/SimCLR/GAN/InstructGPT...；跨子领域阈值修正 ≥2+人工排 3 条渗漏） |
| hub 全文获取 | 进行中（36 篇 arXiv 版路由） |
| entity_queue 合成 | 33,337 表面名（survey from/to/claims_about/subject + coarse subject/mentions） |
| registry growth | 进行中（33k 表面名→registry，LLM 批量归并 417 批次） |

## 冒烟迭代修的问题（4 个，全部在案）

1. **HTML 转换剥掉标题**→chunk 退化固定窗口→语义路由全落 other
   （修：h1-h6→markdown ##；arXiv 页面 chrome 清除）
2. **方法目录式综述盲区**：dg 综述 196/212 落 other——"每方法一节"
   体例（4.6 MASF）没有语义标签可用→新增 method_entry 标签
   （路由 claim+lineage+gap），CS 综述主流体例覆盖
3. **gap 整扫返回裸数组**→只认 dict 的归一化漏了→修（缺口 2→14/
   10→36，修复即翻倍）
4. **引用名当实体名**（"Gu et al. (2023)"）→方法名纪律条（quote
   里方法名必须逐字，作者引用式不输出）
5. References 节大段引文块→输出不可解析（chunk fail 大头）→
   确定性跳过（节名+引文密度启发式）

## 综述池选取的弯路（教训在案）

1. OpenAlex `type:review` 覆盖差（25 篇/目标 150）
2. 标题模式+概念查询 → IoT/微电网/电池综述渗漏（AI 概念太宽）
3. **arXiv 快照标题模式=终解**（领域边界干净+arxiv id 免费）；
   两个排序 bug（升序反了→全 2016 前；纯降序→全 2024）由
   年代分层配额修复

## 待做（按序）

1. **survey_extract 实现**（设计稿→代码，~1 天）：cards 综述变体
   （语义分节）+S1-S4 抽取+postcheck 综述门+views 兼容
2. 冒烟验收（设计稿 §7：1 篇综述四类记录+quote 锚定+无张冠李戴）
3. 跨综述共识频次统计（149 篇 references 交集→hub 池 ~80 篇）
4. 全文获取（arXiv OA 149+80 篇）
5. 双轨深抽马拉松（综述走 survey_extract、hub 走标准管线，过夜）
6. registry 域边界方案（跨子领域 namespace 规则——views 管线
   接入前定）

## 工件

- base_kb/：manifest.json（1,907 篇 Tier-1）+ coarse_records.json
  （7,824 条）+ survey_pool.json（149 篇）+ demand_rehearsal.json
  （存档转题目分布参考）
- base_kb_build/：survey_arxiv.py + survey_extract_design.md +
  全部历史脚本（backbone_pool.py 作废留档）
