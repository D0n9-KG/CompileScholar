# POSTRUN-FIXES — 本轮跑完后要修的问题清单（跑中发现的，先记录不interrupt）

> 建档 2026-09-22 21:40。规则：跑中只修阻断性问题，其余全部入此档，
> 跑完统一处理，避免半路重启丢进度。

## A. 效率/结构

1. **postcheck 增量落盘 O(n²) 写放大**：records_checked.json 已 514MB+，
   每篇结算全量重写（430 次 × 500MB 写放大）。改 append-only 或分片。
2. **PaperQA 索引无增量断点**：索引按目录 hash 整体重建（路径常量变 →
   全部重烧）。排查 tantivy index 是否可按文件增量。
3. **LightRAG 入库完成语义靠轮询**：_wait_processed 每 5s 轮询 doc_status，
   430 篇全轮询有开销；探索其 pipeline 回调/事件机制。
4. **build.py 依赖组并行未实装**：PARALLEL_GROUPS 已声明，驱动仍线性
   （当前各站有断点所以影响小；下一轮全量建库要真正并发调度）。
5. **deep_extract WAL 全量重放开销**：WAL 8.2MB/1555 行，每次启动全读。
   规模到 QASA（1375 题×更多篇）时改分片 WAL。
6. **record_count=0 论文**（Laser_speckle 等 2 篇）：源近无正文，0 条是
   诚实反映——但 build.py 应把它们标进产物 manifest 的 known-empty 清单。

## B. 质量/正确性

7. **result 记录 role/scope 填充弱**（实读发现）：bio 题的 result 记录
   role_result/scope_ref 大量空——影响 typed tools 检索。registry_growth
   后重测链接率，若仍低考虑 slot prompt 的字段提醒。
8. **Lipid_Nanoparticles_From_Liposomes 表格综述全进 overflow**（1 记录
   +28 overflow）：表格型综述在类型系统无家。验证 F24 是否接住其
   Table S1；若没有，考虑 overflow→表格通道的路由规则。
9. **fallback 路由 31%**（route stats fallback=2233）：卡图缺题名。补
   cards.json 的 sections 标题覆盖，或 regex 路由增强。
10. **Improving_text_embeddings#c23 永久失败 chunk**：附录指令清单 chunk
    输出超 9k 预算。可试 max_tokens 12k 单点重跑（不入批流）。

## C. 工程/仪表

11. **纯度断言空账本盲区**：check_arm_purity 对零调用账本返回 PURE
    （PaperQA 事故第一轮没被拦住）。加最小调用量断言（如 <10 调用
    = FAIL"insufficient activity"）。
12. **litellm embedding 调用不进账本**：PaperQA 的嵌入调用绕过记账
    （只有 chat 进）。补 embedding callback 通道。
13. **多进程并发写同一 ledger 有竞态窗口**：build-chain 与 baselines 的
    账本是分开的没问题；但同进程多池写同一文件靠行缓冲。低风险，
    观察项。
14. **decompose_points 拆点器与裁判同模型的张力**（用户指出）：概念
    一致性 vs 同模型盲区复制。goldcov 重启用时改为机械规范保证一致性。
15. **monitor 误报三连教训**：监控必须绑定物理产物状态（zip 文件数/
    答案行数），不能 grep 追加式日志——历史行污染假警报。

## D. 观察项（不一定要动）

- LightRAG failed=8：8 篇入库失败待归因（跑完查 doc_status 失败原因）。
- notation 拒率 ~58%：大部分是行内非定义公式，符合预期；跑完抽样核实。
- LightRAG 索引库 graphml 6.6MB/5328 节点：430 篇全入库后重查规模。

## E. 能力边界（用户裁定：能修的跑完要修）

16. **跨篇符号消歧**：notation 记录篇内局部，同符号跨论文不同义
    （C=电容/cooperativity/置信度）。修法方向：registry_growth 给
    notation 建 symbol-namespace 实体（篇内符号→全局消歧键）。
17. **多步公式推理**：平铺定义无推导链，"由式3和式5推出"类答不了。
    修法方向：notation 记录加 references 字段（本公式引用了哪些先行
    符号），形成符号依赖图。
18. **纯视觉图表推理**："图3曲线哪个先达峰"类只有 caption 兜。
    修法方向：figure_channel（识图通道）已有模块未集成，VLM 建库
    预算 ~22M 待批。
19. **postcheck 通道不对称**（用户发现）：五门只检深抽取产物；F24/F35/
    notation 各带通道内轻闸但无统一终检。修法：postcheck 移到所有通道
    汇流后（views 前），per-kind 校验表驱动（notation=LaTeX 折叠锚定、
    表格=单元格锚定——check_record 本就按 kind 分派，架构现成）。
    顺带把 F24 对 checked 的 dedup 基底依赖改为显式合并语义。
