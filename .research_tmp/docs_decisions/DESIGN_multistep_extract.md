# 设计文档：分步抽取（Multi-Step Extraction Agent）

状态：设计阶段，待审。治本——长 prompt 指令遵从退化（21K字符 LLM 不全守）。

## 问题诊断
- 现状：一次 LLM 调用 prompt 21099 字符（完整规则+schema+section text）
- deepseek-chat 对长 prompt 细粒度规则遵守有限（PIV 排除/composed_of 边界/标签判定都漏）
- 不是模型能力差——是 prompt 太长 LLM 不全守（长上下文指令遵从退化）

## 设计：拆成 3+1 步（每步短 prompt + 专一指令 + Pydantic 结构化）

### step 1: 抽节点（surface + evidence，不标 label）
- 输入：section text（chunk）+ 通用节点抽取指令（短）
- 输出：[{nid, surface, evidence_span}]
- prompt 重点：只抽**实体 surface + 逐字 evidence**，不判类型
- 短 prompt（~500 字符指令 + section text）
- LLM 只做一件事：找出论文里的实体，给原文 evidence

### step 2: 标节点 label（每节点判类型，专一 prompt）
- 输入：step1 的节点 list + 标签判定规则（短 + 专一）
- 输出：[{nid, labels: [METHOD/PARAMETER/PHENOMENON/...]}]
- prompt 重点：**只判节点类型**——METHOD 是命名建模方法/律/理论，PARAMETER 是方程里的符号，PHENOMENON 是物理现象...
- 专一 prompt（只判 label，不抽边，~800 字符 + 节点列表）
- LLM 遵守好（短 + 只做一件事）

### step 3: 判超边（节点组合 + pattern_type + role + evidence）
- 输入：step1-2 的节点（带 label）+ 超边判定规则（短 + 专一）
- 输出：[{eid, pattern_type, node_ids, node_roles, qualifiers, evidence_span}]
- prompt 重点：**只判节点间关系**——哪些节点组成律/依赖/组成/演化/定义
- 专一 prompt（只判边，~1000 字符 + 带类型节点列表）

### step 4（可选 critic）: 验证 + 修复
- 输入：step1-3 结果 + 验证规则（专一，短）
- 验证项：
  - "METHOD 节点里有没有工具/几何/泛词误标"（专一验证 prompt）
  - "composition 边里有没有实验设置"
  - "节点有没有孤立（没进任何边）"
- 验出问题 → 修复（重判 label 或删边）
- 短 prompt + 只验一条规则 → LLM 遵守好

## 和现有代码的关系
- 替换 `_run_hg_node` 里的**一次 _call** → 3-4 次 _call（分步）
- 保留：DAG node 分步（已有）+ chunk 分割（已有）+ parse_json_response
- 新增：分步 prompt（短）+ Pydantic 结构化输出（可选，先 JSON parse）
- 不改：schema/meta/evolution/gate（下游不变，输入还是节点+超边）

## 成本
- 现在：1 次 LLM 调用/chunk（长 prompt）
- 改后：3-4 次 LLM 调用/chunk（短 prompt）
- 成本翻 3-4 倍，但每次短 prompt LLM 遵守好——质量提升
- deepseek-chat 快（~5s/次），3-4 次 ~15-20s/chunk（可接受）

## 不确定点
1. step2 标 label：LLM 能只看 surface+evidence 判类型吗？还是要 section 上下文？——倾向给 evidence + 节点列表，不给 full section（短）
2. step3 判边：只给带 label 的节点列表够吗？LLM 能不看 section text 就判关系吗？——倾向给节点 + evidence（不给 section，但要 section 的 evidence）
3. critic 步（step4）：做不做？先做 step1-3，critic 可选
4. Pydantic 结构化：现在 JSON parse 够用吗？还是上 Pydantic？——先 JSON parse（不引入依赖），Pydantic 可后加

## 防过拟合（守原则）
- 分步 prompt 里**不放特例硬规则**（PIV 列表/几何列表/组名）——用**通用的 label 判定原则**（METHOD = 命名建模方法，不靠排除清单）
- step2 label 判定用 LLM 语义判（不靠规则排除清单）——规则列表（_TOOL_RE）只是 critic step 的辅助
- 每步 prompt 例子是**多域 + 标"举例非规则"**

## 实施顺序
1. 写分步 prompt（step1/2/3 各短 prompt + 例子）
2. 改 _run_hg_node 调 3 步 _call
3. smoke 单篇（Jop_2006）看质量改善
4. 加 critic step4（可选）
5. 跑 5 篇验证
