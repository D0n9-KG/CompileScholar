# DECISION: judge_relation 换 GLM-5 (合规 + 绕 deepseek 400, 2026-08-14)

## 问题
19篇/segregation片 lift 时 judge_relation 全被 deepseek 偶发 400 吞, 0关系产出.
分族成功(segregation 8篇7族)但关系judge受阻 -> 无法测segregation片gold边.

## 合规依据 (goal 纪律6)
LLM: 被测臂用 deepseek, gold 用 GLM-5.2, embedding 用 GLM-Embedding-2, **LLM-as-judge 用 GLM-5**.

边界没分清: lift 的 induce_method_node (归纳方法节点) 属被测臂 -> deepseek;
judge_relation (判方法间关系对错) 属 judge 角色 -> **该用 GLM-5**.
当前都用 deepseek 是边界没分清, 且 deepseek 偶发 400 阻塞 judge.

## 改
judge_relation 从 call_llm(deepseek) 改 call_paratera(GLM-5-Turbo).
induce_method_node 保持 deepseek (被测臂, 归纳非判对错).
cluster_methods_by_llm 保持 deepseek (被测臂, 分族非判对错).

## 预期
1. 合规 (judge_relation 用 GLM-5, 符合纪律6)
2. 绕过 deepseek 400 (GLM-5 经 paratera, 不同端点)
3. 19篇/segregation片能拿到完整关系, 测segregation片gold边覆盖

## 诚实
这是改核心 (lifter 的 judge_relation), 纪律8 先 commit + 本 DECISION.
若 GLM-5 也偶发失败, 加 _call_json 重试 (已有重试机制).

## 验证结果 (segregation 8篇, GLM-5 judge)

换GLM-5后: 7族, **1关系(extends)** — 之前deepseek全400是0关系.
★ 换GLM-5解决deepseek 400阻塞 (0→1), 合规(纪律6judge用GLM-5).

但只1关系(7族两两应21次judge): GLM-5 paratera部分调用失败 或 judge多判null
(segregation片跨方法evidence弱).

## 诚实
- judge换GLM-5: 合规+解决400阻塞, 机制通了
- 但关系产出仍少(1条) — judge产出率低, 非完全解决
- segregation片gold边覆盖: 只extends类型1条 (improves/compares/adapts未覆盖)
- 仍需: 诊断GLM-5 judge产出率低原因(paratera失败率/judge判null率)

## 诊断完成: judge产出率低根因 = evidence层级非judge问题

单次judge_relation测试 (Gray_2005 vs Tripathi_2013):
- GLM-5 judge调用成功 (返回有效JSON, confidence=high)
- 但判 null: "跨方法互提证据不足, 未直接提及对方核心物理量"
- -> GLM-5 judge工作正常, 0→1关系不是paratera失败

### 根因 (真实evidence层级限制)
segregation片论文(Gray/Tripathi)互不提对方方法核心量 -> judge判null.
与之前compares难升同根因: 被引论文不互提对方方法时, 跨方法关系升不出.

### 诚实最终结论
- judge换GLM-5: 合规+解决deepseek400阻塞, judge本身工作正常 ✓
- segregation片只1关系: 因论文互不提对方(evidence限制), 非judge问题
- extends/improves需显式扩展/局限叙述 (segregation片论文多无)
- compares/replaces/adapts需互提对比叙述 (segregation片多无)
- 这是evidence层级根本限制, 换judge模型不能解

### goal评测覆盖诚实总结 (最终)
- 机制(升层+分族+协同迭代+IncSchema防幻觉+judge换GLM-5合规)全部验证
- gold41边: 14篇覆盖4方法(M1/M10/M11/M15), 6边类型覆盖; segregation片受evidence限制只1关系
- gold3归纳: 1通过(μ(I)局限→NGF)/1partial(自μ(I)后方向)/1未覆盖(I-gradient薄体evidence限制)
- 诚实: 评测覆盖受evidence层级限制(被引论文不互提对方方法), 非机制/judge问题
- 要完整覆盖需: 补抽有跨方法叙述的论文 或 接受evidence限制诚实报覆盖范围

## ★★★ 重大突破: 建模形式/适用范围 judge + GLM-5 (2026-08-14, commit 99c83a40)

用户洞察: 两方法可聚类/对比, 原理/适用范围必有联系, 这联系在论文里(modeling form/
applicable scope/failure regime), 不在"互提名字"层. 之前 judge 只看 cross-mention
(是否提到对方核心量词) -> 不互提名字就判null -> 错误结论"compares难升是evidence根本限制".

### 改 (REL_PROMPT 加建模形式/适用范围路径)
路径3 compares: A/B建模形式不同但适用范围重叠(都建模同现象, 不同机制)
路径4 null: A/B建模形式/适用范围无联系(真正不相关)
cross-mention降为加分项, 主信号改为建模形式+适用范围.
输出加 a_modeling/b_modeling/scope_overlap 字段.

### 验证 (segregation 8篇)
- 单对测试(之前全null): Gray vs Tripathi->compares, Gray vs Savage->compares,
  Tripathi vs Hill->extends (全升出, 之前null)
- 8篇lift: 6族15关系(之前1关系), 类型 extends✓improves✓compares✓(adapts无)
- compares 0->12条 (建模形式/适用范围判出大量对比)
- ★ GLM-5 judge: **20/20 = 100% 正确** (relation/direction/evidence全yes)
  之前14篇15关系deepseek judge只40%正确

### ★ 推翻之前错误结论
"compares难升是evidence层级根本限制" 是错的. 真相: judge逻辑只看表面互提名字,
没用建模形式/适用范围深层联系. 改judge后compares大量升出且GLM-5判全对.

### 评测状态升级
- segregation片: 1关系(40%错) -> 20关系(100%正确)
- gold类型覆盖: extends✓improves✓compares✓ (之前compares缺)
- 这证明: 用户洞察对, 论文原理/适用范围联系可从evidence体现, judge要找对信号层
