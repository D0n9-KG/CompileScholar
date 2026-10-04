# P1 gold 扩建计划 (3-5篇综述)

Date: 2026-08-14

## 候选综述 (pilot_surveys/ 已有全文)
1. **ARFM2024** (done, 53方法41边) — 致密颗粒介质建模进展, 当前gold
2. **Thornton2026** (87550字, segregation专题, 113 model/182 segregation) — ★推荐做第2篇: 同segregation方法族(M3/M23/M24/M25/M26)跨综述一致性验证, 引用Savage1988/Gray2018/Jenkins1987/Thornton2012/Tunuguntla2014/Hill等
3. **2008 Forterre&Pouliquen** (67641字, local rheology奠基) — 第3篇候选: μ(I)/nonlocal/kinetic/shallow-water, 与ARFM重叠多(一致性), 范围窄(~6方法)
4. **2018 Guazzelli&Pouliquen** (211934字, 悬浮液rheology) — 域偏disjoint(suspension非dry), 泛化验证但方法regime不同
5. **2025 ESR** (滑坡颗粒流流变) — 应用域, geophysical

## 第2篇gold决策: Thornton2026
理由: segregation是ARFM gold里方法最多最细的族(5个方法M3/23/24/25/26), 跨综述一致性测试最有诊断力。若两综述独立标出同一segregation方法族且我们lift在两份输入上都抽到→强证gold可靠性+方法可迁移。

## 执行步骤 (P0完成后)
1. 从Thornton2026抽方法节点(带aliases, 英文)+演化边(src/tgt/type/evidence) — GLM-5.2标注
2. 获取Thornton2026引用的关键源论文PDF(~10-15篇: Savage1988已有, Gray2018/Jenkins1987/Thornton2012/Tunuguntla2014/Hill等需获取)
3. 用extract_hypergraph抽每篇edges → lift_corpus → eval_err_edges评第2份gold的ERR/PSC
4. 跨综述一致性: ARFM gold和Thornton gold共同方法(如Savage-Lun/Gray-Thornton)的lift结果是否一致

## 诚实约束
- 自建gold规模<30篇(Intern-Atlas), 诚实标"规模小但方法可迁移+跨综述一致"
- 标注一致性kappa: Thornton2026 gold 至少两人独立标一部分或LLM辅助+人工核验
- gold来源(哪篇综述怎么抽的)+协议写清
