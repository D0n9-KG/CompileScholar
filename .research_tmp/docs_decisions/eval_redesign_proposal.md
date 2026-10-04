# 评测重新设计 (评独有能力拉开和baseline差距)

Date: 2026-08-14
前提: 用户判断当前评测(边级ERR/PSC)诚实但简单, 不够凸显优势. 先重设计再扩样本.

## 我们独有能力 (代码已实现, 但评测完全没评)
1. **自演化操作集**: add/split/merge/retire/rename + lift_into_schema (schema能从抽取结果生长)
   - 别人(SCION/Agent-K1): 静态schema或一次性抽取, 无schema自演化生长
2. **violation检测**: detect_constraint_violations (查"引用未定义数值/约束违反")
   - 别人: DIAL-KG flat schema无, SCION constraint=domain/range typing非"引用未定义检测"
3. **富拓扑**: composition_edges(组成) + law_constraints(定律约束) + dependencies
   - 别人: 多是单一关系图, 无组成/约束分层

## 之前评测只评了 方法演化边(ERR/PSC), 上面3个独有能力零评测. 这才是"凸显优势"该评的.

## 新评测设计 (4个维度)

### 维度A: 方法演化边 (已有ERR/PSC, 保留作基础)
- 跨论文方法extends/improves/compares关系
- 评: fair_recall + ERR + PSC (已建)

### 维度B: 富拓扑层级 (新增, 评独有富拓扑)
- gold: composition_edges(4条) + law_constraints(41条)
- 评: lift能否抽组成关系(X组成Y) + 定律约束(定律Z约束方法M)
- 指标: 组成边命中率 + 定律约束对齐率(41条里多少定律对上方法)
- baseline对比: SCION/Agent-K1 schema无组成/约束分层 → 应该0或很低

### 维度C: violation检测 (新增, 评独有检测能力)
- 设计: 注入"引用未定义数值"的扰动(在实例超图里故意引用gold没有的方法/数值), 看detect_constraint_violations能否查出
- 指标: violation召回率(注入的扰动查出多少) + precision(没注入的正常实例没误报)
- baseline对比: 别人无此功能 → 0

### 维度D: 自演化能力 (新增, 评独有schema生长)
- 设计: 给N篇论文, 看schema能否从空/种子生长出方法层+演化关系
- 对比: frozen(不演化)=0方法0边 vs full(演化)=N方法 → 证明自演化必需(已有frozen=0对照)
- 指标: 生长出的方法数/边数 vs gold, 跨语料规模递增时schema生长曲线
- baseline对比: 别人静态schema, 无"生长"概念

## 诚实约束
- gold的law_constraints有41条但很多constrains=None(GLM抽漏) → 评维度B/C前需修gold
- 维度C需构造扰动数据(人工注入violation)
- 维度D需"从空schema生长"的实验设置(当前lift_into_schema是加到已有meta)
- 样本量: 扩输入19篇→覆盖更多gold方法后, 维度A/B都受益

## 优先级建议
1. 维度C (violation检测): 独有+可构造扰动+baseline必然0 → 最能拉开差距
2. 维度B (富拓扑): gold有数据(composition+law_constraints), 适配现有eval
3. 维度D (自演化): frozen=0对照已有, 加"生长曲线"实验
4. 维度A扩样本: 拉大coverable

## 待用户定
这个4维度设计方向对吗? 先做哪个维度?
