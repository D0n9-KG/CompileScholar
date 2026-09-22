# 09-23 晨报（深夜自治时段 01:16–醒时）

> 自治授权执行完毕的部分。judge 循环与看门狗仍在跑，数字随时间增长；
> 本报数字为起草时快照，醒后 `python _shared/tools/multi_paired_stats.py` 出终值。

## 一、建库终局（全部完成，02:02）

- **registry_growth run6 终局 PASS**：103/103 chunks 全满额 assigned（零 fallback），
  registry v2 = **10,418 实体**（matched 454 / new 7,759）。
  7,759 新实体 = 人工仲裁队列（协议要求 vN+1 转正前人审——晨间待办）。
- **views.json 落盘** check PASS：54,486 records / matrix 5,079 / cards 335 /
  genealogy 971 / absences 1,939+2,014 / notation 214。
- 死因翻案：run5 非换行符问题——salvage 只认 "kind" 键对 assignment 形状永远空手
  （+取证截断 2000 字符掩盖尾部）。修复=salvage 键参数化+chunk WAL 断点+fail_dump
  全量取证（POSTRUN #24）。

## 二、三臂答题（进行中）

| 臂 | 状态 | 快照 |
|---|---|---|
| ours（我方全栈） | 答题中 | 4 题并行，首题 14 步/5107 字符 |
| PaperQA2 | 答题中 | 账本 334 调用入账（异步盲区已修，#27） |
| LightRAG | 入库 252/440 | 入库完自动开查询（Monitor 盯守） |

冒烟阶段揪出并修复三个会毒化整晚的问题（#27-29）：
1. PaperQA 账本零入账（litellm async 回调只读 _async_* 列表）
2. 编译路径引用形态断裂（notes 是 [record_id]，prompt 要 [paper_id]，模型自造
   [Wan#10800]→官方桥译 0 引用=必 0 分）——prompt 改逐字拷贝+桥支持 record_id 解析
3. judge 把超时空答案行判成 0 分（str(TimeoutError())=="" 骗过过滤）+两处冒烟残留

## 三、判分体系（就绪，增量运行中）

- **Citation F1**（主判据）：官方 extract_citations 逐字，每 2 分钟增量判分，
  已在产出。Stage D 配对 bootstrap CI 脚本就绪。
- **rubric 轨一手核验结论**：官方 rubric = allenai/asta-bench 的
  rubrics_v1|v2_recomputed.json，**只覆盖 SQA 轨 100 题，Multi-108 无官方 rubric**
  （Multi 域官方判分 = AutoAIS + Citation F1）。**待你裁**：
  (a) 跳过 rubric 轨（Citation F1 为主判据本就够）；(b) 用官方 rubric.py prompt 逐字
  + 自动生成 ingredients（判分协议决策，我没有单方面定）。
- **AutoAIS**：需本地跑 attrscore-flan-t5-xl，GPU 被三臂占着——白天 GPU 空时跑。
- **gold 探针**（gold 答案交 judge 测天花板）：预注册要求首跑验证，待判分全链
  通了以后补。

## 四、夜间事故（POSTRUN-FIXES.md #24-29 全档）

- LightRAG WinError-5 管线团灭（Defender 句柄撞 rename→199 篇一瞬 failed，
  空 error）——atomic_write 重试补丁，重放成功
- registry_growth salvage 泛化+WAL
- PaperQA 1800s 全超时轮（GPU 争抢）——杀掉重启，垃圾行清除
- 看门狗 build-chain 探针 run_id 脆弱（换 run 必误报）已修

## 五、晨间待办（按优先级）

1. **人工仲裁队列**：registry v2 的 7,759 新实体（REVIEW-PACKAGE 生成器现成）
2. **rubric 轨裁定**（上述 a/b）
3. 配对统计跑一遍看首版结果：`python _shared/tools/multi_paired_stats.py`
4. AutoAIS 接线（GPU 空时）
5. POSTRUN-FIXES 全 29 项过一遍排优先级

## 追记（03:30）：首版分数已在产出

我方臂前 5 题 Citation F1（增量判分实跑数字，非终值）：

| qid | precision | recall | F1 |
|---|---|---|---|
| norman_bio_1 | 1.00 | 0.33 | 0.50 |
| norman_bio_2 | 0.20 | 0.25 | 0.22 |
| norman_bio_3 | 1.00 | 0.75 | 0.86 |
| norman_bio_4 | 0.75 | 0.75 | 0.75 |
| norman_bio_5 | 1.00 | 1.00 | 1.00 |

观察：n_cited_pids 偏少（多数题只引 1-3 篇 gold ctx 内论文）——引用
召回是当前弱轴（答案内容长度 5-8k 字符正常，是"引用哪些"的问题不是
内容问题）。bio_2 精度 0.2=引了题外 ctx。这些是首跑画像，修复方向待
108 题全量后归因。

PaperQA 索引侵蚀案（430→360）根因已定位并治愈中：use_absolute_paper_
directory=True（PS 时代配置债）与 sync 比较集结构性失配→每次重启试图
移除全部索引→边删边崩。修=files.zip 键重写为相对名+开关翻转+PQA_SYNC
治愈跑（重加 70 篇中）。修复后 pqa 重启答题。

## 追记二（05:30）：PaperQA 深夜六层洋葱（终局：真跑通）

PaperQA 答题其实**整夜都是死 agent**——每题都落进"I cannot answer due to
having no papers"的罐头拒绝。六层剥离：

1. ~~超时~~（不是——是秒拒）
2. ~~WinError 5 文件锁~~（症状非根因）
3. **use_absolute_paper_directory 配置债**：索引键=绝对路径 vs sync 比较集
   =相对路径 → 结构性全失配 → 每次重启试图删光索引 → 跨重启侵蚀 430→360
   （修=开关翻转+files.zip 键重写+PQA_SYNC 治愈跑补回 430）
4. **等待探针 os.listdir[0]** 指错索引目录（修=从运行时对象解析）
5. **router shim 签名冲突**：aviary 的 `partial(acompletion, model_name)` 把
   model 绑成位置参数 → 撞 shim 的 messages 参数 → agent 工具选择全程
   TypeError → 每题罐头拒绝（**这是整夜的真根因**）
6. **vLLM 无 --tool-call-parser**：原生工具调用被服务器拒绝
   （修=shim 内提示词模拟工具调用：schema 渲染进 system+JSON/XML 解析还原
   tool_calls——agent 自适应性完整保留，这是 PaperQA2 方法语义的正确等价）

终态（05:25）：15/15 实质答案、引用 12-14 映射零丢弃、~6 分钟/题、
ETA ~4-5h。**晨间建议：给 GPUStack 服务器加 --tool-call-parser hermes**
（然后可撤掉模拟层跑原生工具调用对比）。

 ours 20/108 | pqa 15/108 | lightrag 291/440 入库中。
