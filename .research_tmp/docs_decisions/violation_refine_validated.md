# violation refine 验证通过 (Midi_2004, 2026-08-14)

## refine 效果
通用符号排除清单(commit 46a9c9c6)生效:
- 之前 viol=4 含"d"误报(粒径通用符号)
- refine后 viol=2, d被排除, 剩真违规:
  1. "5 to 10 particle diameters"(具体数值被引用未定义)
  2. "1.5"(临界流率系数1.5被引用未定义)
- 这2个是真violation(论文特定数值引用但schema没定义)

## 完整流程数字(Midi单seed)
271节点/73超边/20patterns/ split1/ dep10 cons8 comp4 viol2
- 富拓扑全产出(dep/cons/comp), 自演化split执行, violation检测真违规

## 诚实非确定性问题
Midi两次跑: viol 4→2, dep 2→10(单seed波动大). 和方法演化边一样, violation评测需多seed控方差.
但refine方向对(误报排除), 数字虽波动但语义对(剩的是真违规).

## 下一步
1. violation评测设计: 注入扰动(人工加"引用未定义数值"到实例超图)看检测召回+precision
2. 多seed控violation/富拓扑非确定性
3. 扩19篇走完整流程
4. 富拓扑评测(对gold comp/law_constraints)
5. 自演化评测(frozen对照+生长曲线)
