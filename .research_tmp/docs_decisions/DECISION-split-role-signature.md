# DECISION: split 子 pattern role_slots 约束 = exact signature (非 subset)

## 背景
子代理 review Step 1 底座内核发现 BLOCKER #2：
- 设计补充 (plan_detailed_design_supplement.md L29) 写 "split: 子 pattern role_slots = 父 role_slots 子集(不引入新 role)"
- kernel `_schema_constraint` for SPLIT 实现 `sub_roles.issubset(parent_roles)` (允许丢 role)
- 但底层 `MetaHypergraph.split_pattern` (hypergraph_schema.py L581) 要求 `_role_sig(sp.role_slots) == parent_sig` (exact match)
- 不一致 → kernel 通过的 subset-drop split，底层静默返回 None，pattern 没添加、父没标 abstract

## 决策
**收紧 kernel 到 exact signature，与底层一致。** 不改底层。

## 理由
1. 底层 exact 约束有语义道理：split 是**语义边界拆分**（同一 role 结构不同语义子类），不是**结构拆分**。子 pattern 丢 role = 改 arity = 改结构，超出 split 语义。结构变化该走 add_pattern，不是 split。
2. 不破坏现有代码：底层 split_pattern + 现有 evolution loop + smoke test 都依赖 exact。改底层风险大。
3. 设计补充的"子集(不引入新 role)"理解修正为：**exact 是"不引入新 role"的天然满足**（相同 role 集合 ⊆ trivially）。设计意图核心是"不引入新 role"，exact 满足且更严，不丢失设计意图。
4. 挑战 C 声称"split 不引入新 role"在 exact 下成立（subset-drop 在 exact 下根本不会通过 kernel，所以不会到底层静默失败）。

## 修法
kernel `_schema_constraint` SPLIT 分支：把 `sub_roles.issubset(parent_roles)` 改为与底层一致的 exact signature 检查（role 序列 + type 序列完全相同），保留"不引入新 role"的约束语义。docstring 注明 split 是语义拆分不改结构。

## 影响
- 不影响底层（不改 hypergraph_schema.py）
- kernel split 校验更严，与底层一致，消除静默失败
- 设计补充文档措辞"子集"理解为"不引入新 role"（exact 满足）
