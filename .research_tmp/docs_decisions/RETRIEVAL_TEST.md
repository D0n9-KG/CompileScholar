# 保拓扑检索式测试结论 (2026-08-14)

## 测试: Pouliquen1999 单篇, full(保拓扑检索+宽松gate) vs frozen

| arm | he | influences | influences-family | patterns |
|-----|----|-----------|-------------------|----------|
| frozen | 86 | 43 | 43 | 5(seed) |
| full | 74 | 1 | 34 | 12 |

## 底线达成: 不严重漏抽
- influences-family 34 vs frozen 43 (之前无检索full仅30, 现在34, 缓解)
- 但仍少12条(74 vs 86), 检索式没完全解决漏抽

## ★ 但暴露更严重问题: split命名语义错位
full的influences子类:
- influences_performance_enhancement(23条): 装的是 gravity force→friction force,
  72%→76%→mean velocity, humidity→mean velocity, thickness h→inclination θ
  **这些是参数依赖/几何关系, 根本不是"性能增强"!**
- influences_flow_induced(6条): shear stress→mean shear rate (这个还算对)
- influences_kinematic_bridging(4条): friction coeff→models (勉强)

LLM把23条参数依赖边错误归到"performance_enhancement"——split命名完全错位。
frozen的粗influences至少诚实(都叫影响), full的细分类反而误导。

## 根因: split命名LLM不可靠
- name_split_subpatterns看evidence片段起名, 但没真正理解关系性质
- 跨篇收敛已修(commit 3ce0f502), 但单篇内命名仍错位
- LLM把"参数依赖"叫成"performance_enhancement"——语义理解失败

## 诚实结论
1. 保拓扑检索式: 缓解漏抽(30→34)但没解决(仍<43), 且没让full质量更好
2. split命名语义错位: 比漏抽更严重——错误分类比漏抽更误导
3. 单次LLM命名不可靠, 需更结构化分类

## 指向的备选方案(你提的)
- agent范式: 抽取agent边抽边检索, 按需决定pattern(非预命名)
- 多模型级联: 小模型粗筛pattern类别, 大模型精抽+命名
- 或: split命名不用LLM, 用限定符/角色结构确定性命名(非语义命名)

当前: 保拓扑检索+宽松gate没让full超过frozen, split命名错位是新瓶颈。
