# 夜间推进最终结论 (2026-08-14, 诚实中性)

## ★★★ 最终客观结论 (2026-08-14, LOO决定性负面)

**客观 LOO 角色消歧(从数据自身学pattern→角色, 无主观启发表):**
| arm | n_path | LOO_AUC |
|-----|--------|---------|
| full | 11 | **0.464** (最差) |
| add_only | too few | - |
| frozen | 14 | **0.708** (最好!) |
| no_intra_dag | 7 | 0.500 |

**frozen(不演化) > no_intra_dag > full(演化split)**
split不仅没帮助, 反而把角色区分从frozen的0.708砸到0.464(低于随机0.5)。

机制: frozen 6个seed pattern虽粗但与gold演化类型有天然统计关联(LOO在14样本上稳);
full 237 pattern稀疏, 每pattern路径出现少, LOO学不稳 → 0.464。
split把frozen有区分度的大pattern(influences/derived_from)拆成碎片子pattern,
破坏了原有的角色信号。

## 整夜所有评测汇总(split价值)
| 评测 | full vs frozen | 结论 |
|------|---------------|------|
| silhouette(role+qual) | 0.106 < 0.159 | 不利split |
| silhouette(entity) | -0.033 < 0.003 | 不利split |
| 角色消歧启发A | 0.600 > 0.500 | 看似有利(巧合) |
| 角色消歧启发B | 0.407 < 0.500 | 不利(启发表反转) |
| **客观LOO** | **0.464 < 0.708** | **决定性不利split** |

**5种评测4种不利split, 1种(启发A)是巧合。客观LOO决定性证明split损害角色区分。**

## 诚实最终结论(不粉饰)
1. split客观发生(237 vs 6 pattern), 拆分维度看着合理(定性)
2. 但**所有量化评测都显示split损害而非帮助**:
   - 纯度降低(silhouette)
   - 角色区分降低(LOO 0.464 vs frozen 0.708)
3. split把frozen有统计区分度的大pattern拆成稀疏碎片, 破坏原有信号
4. **当前split机制不是有效创新点** — 这是整夜最痛苦的诚实结论

## 可能原因(split为何损害)
- split按LLM语义(物理领域)拆, 非按关系性质 → 拆错维度
- split后子pattern稀疏, LOO学不稳
- frozen的seed pattern(influences/constitutive_law)已天然对应gold演化类型
  (influences~extends, compares~claim_relation), split破坏这对应

## 需你决策(存亡级)
A. 承认split当前无效, 重新定位创新卖点(intra-DAG? nary超图? dep/con/comp?)
B. 修split判据: 不按物理领域拆, 按关系性质拆 + 跨论文聚类(不只单篇内)
C. 换评测: LOO可能也不公平(frozen样本14 vs full 11, 且full稀疏),
   用专家盲评或下游综述重建

我的诚实判断: split作为主卖点已站不住(5评测4不利)。
建议A或C重新审视, 或B大改split但风险高。
intra-DAG传播也没证实(no_intra_dag LOO=0.500 vs full 0.464, 传播也没帮助)。
两个主卖点(split+intra-DAG)在当前评测下都未证实价值。

关联 memory split-naming-divergence-bug.md。
1. silhouette(纯度): 假设"同pattern边应相似" → 对自演化方向相反(惩罚小簇=新发现)
   结果: full < frozen (但指标不适合)
2. 角色消歧AUC: 依赖主观STRONG/WEAK_PAT启发表
   - 启发表A(5篇版): full=0.600 vs frozen=0.500 (看似split有效)
   - 启发表B(30篇版,更全): full=0.407 vs frozen=0.500 (split反更差)
   - AUC对启发表高度敏感(0.600↔0.407), 说明评测本身不可靠, 0.600是巧合非信号

### 30篇客观数据(split确实发生, 但价值测不出)
| arm | uniq_pat | acc | n_path | AUC(启发A) | AUC(启发B) |
|-----|---------|-----|--------|-----------|-----------|
| full | 237 | 323 | 20 | 0.600 | 0.407 |
| add_only | 8 | 15 | 11 | 0.500 | 0.500 |
| frozen | 6 | 0 | 22 | 0.500 | 0.500 |
| no_intra_dag | 31 | 30 | 15 | 0.500 | 0.500 |

split客观发生了(237 pattern vs 6), 拆分维度看着合理(constitutive_law→_geometric/_angular),
但"是否真的有价值"测不出——任何评测都引入主观假设。

## 根本困境
1. **评测依赖主观假设**: 纯度/角色映射都是人定义的"好"标准
2. **LLM judge引入A4循环**: 让LLM判split好坏=自己判自己
3. **下游任务层级错配**: gold是方法间演化, 大图是篇内实体关系
4. **样本仍不足**: 41演化边只11-22连通, AUC统计不稳

## 诚实结论(不粉饰)
- pilot流程已过(抽取+gold+匹配+多评测全跑通)
- split客观发生且拆分维度看着合理, 但价值无法用现有评测可靠证实
- 之前"0.600证实split"是启发表巧合, 更新启发表后变0.407
- **不能宣称split有效, 也不能断言无效** — 评测方法本身是瓶颈

## commit
- 3ce0f502: split命名跨篇收敛(让30篇split质量好, 这个修复本身有价值)
- b0e2cca8: 消融开关
- a5fd47bc: embed容错

## 需你决策的真问题
**不是"split有没有价值", 是"怎么评测自演化schema才可靠"**。
当前所有schema层评测都不可靠。可能的路:
A. 放弃schema层评测, 用纯下游任务(综述重建质量)——但层级错配
B. 加引用因果边抽取(Intern-Atlas式), 让大图产出方法间演化边——根本解层级错配
C. 专家盲评: 让领域专家看full vs frozen的pattern分类, 盲评哪个更合理(避开LLM循环)
D. 接受split价值"看着合理但难量化", 论文定性报告+定性例子, 不强求数字

我倾向B(根本解法)——加引用因果边抽取, 大图能产出方法间extends/improves,
那时NMR/ERR才真正有效, split价值也可在方法演化链重建上测。但这是方法层面大改。

等你醒了定方向。当前所有数据在.research_tmp。
