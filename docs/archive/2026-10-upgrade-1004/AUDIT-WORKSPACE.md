# 工作区治理审计（2026-10-04，只读，未改动任何文件或进程）

> 范围：`C:\Users\D0n9\Desktop\CompileScholar`。数字全部来自本次实测（du / find / git ls-tree / rev-list / cat-file / count-objects）。
> 审计期间 CS2 test 在跑（run_vnext r2、harness_arm_run、6 个 retrieval_mcp），本审计没有碰它们。
> 体积单位：du 的 MiB 口径，表里写作 MB；GB = 1024 MB。

## 0. 先说结论

1. **main 现在推不上 GitHub。** origin/main 最后一次 push 是 09-20，本地领先 215 个提交。这些提交带了 3,781 个 blob，未压缩 15.40 GB，其中 17 个单文件超过 100 MB（合计 13.79 GB，最大 2.33 GB）。GitHub 对 >100 MB 的文件直接拒收。结果是两周的代码和判决档只存在这一块硬盘上。
2. **.git 占 14 GB，有用的 pack 只有 237 MB。** 10.19 GiB 是 28,553 个散对象（大头是嵌入缓存和 LightRAG 向量库），2.94 GiB 是中断操作留下的临时文件（tmp_pack / tmp_obj），另有一个进程早已不在的 gc.pid。
3. **HEAD 树 5.78 GB，其中 83% 是 16 个向量/嵌入缓存文件**（.bin / .npy / embed_cache/*.json）。10-03 加的 `*.f32` 规则只挡住了 f32。两个 .bin 现在是"已修改"状态，下一次 commit 会再往历史里塞约 0.9 GB。
4. **.gitignore 和实际跟踪状态脱节。** 1,039 个已跟踪文件命中 ignore 规则（是强制 add 进去的）。反过来，RESULTS-LEDGER、paper_drafts、phase0、134 份决策档中的 104 份、test100 答案都没有跟踪。
5. **两个真实密钥写进了已跟踪文件**：MINERU_TOKEN 和 EMBEDDING_API_KEY（GPUStack）。还没推送，目前只在本地历史里。`.env` 本身从未入库。
6. **冻结的 v9b 有一条隐藏依赖。** `cs2/direct_judge.py` 在 FREEZE 哈希清单里，但它用绝对路径 import 了 `scratch/ai2_baseline_2026-09-08/asta/asta-bench-main`，这个目录既没跟踪又被 ignore。
7. **系统代码有一半放在"临时"目录里。** 答题栈 `_shared/tools`（11,089 行）和 `cs2/*.py`（6,626 行）都在 `.research_tmp/` 下；`src/`（27,856 行）只装了建库部分。另有 46 个脚本写死了 `C:\Users\D0n9\...`。
8. **根目录文档整体过期。** README、DIRECTION、ASSET-STATE 停在 09-20～09-25 的方向上。README 指定的"权威台账" RESULTS-LEDGER.md 停在 09-14，而且没入库。当前口径实际分散在 docs_decisions 的 10-03 档、FREEZE 文件和仓库外的 memory 里。

对已知事实的校正：`.research_tmp/` 根下的散文件是 94 个（不是约 150 个；130 是"文件+目录"的总条目数）。docs_decisions 有 134 项（131 个文件 + 3 个子目录，递归共 146 个文件）。cs2 顶层是 394 个文件 + 25 个目录，递归共 2,653 个文件。

## 1. 盘点

### 1.1 总量（磁盘约 65 GB）

| 位置 | 大小 | 文件数 | 说明 |
|---|---|---|---|
| `.git/` | 14 GB | – | 见 §2.2 |
| `.research_tmp/` | ≈49.6 GB（50,800 MB） | ≈172k | 其中 experiments/ 占 48,939 MB |
| `archive/`（根） | 1,110 MB | – | contest_2026-08 311 MB + pre_stageB_root 798 MB，已 ignore |
| `ccfa-workfiles/` | 101 MB | – | 已 ignore |
| `src/` | 3.1 MB | 81 个已跟踪 | 27,856 行 py |
| `tests/` | 0.9 MB | 22 个已跟踪 | |
| `runs/`（根） | 85 KB | 11 个已跟踪 | build.py 生成的 manifest（09-22～09-24），和 `.research_tmp/runs/` 同名但不是一回事 |
| `.claude/` `.playwright-mcp/` `.pytest_cache/` | <1.3 MB | – | 均已 ignore |

C 盘还剩 296 GB，空间本身不紧张。真正的问题是"能不能恢复"和"git 还能不能正常用"。

### 1.2 按类别（.research_tmp，按目录归类，误差约 ±5%）

| 类别 | 约体积 | 主要构成 |
|---|---|---|
| 已退役实验 | ≈28.5 GB | experiments/archive/paperscope_2026-09 19.2 GB（PDF 10.1 GB）、experiments/baselines 4.1 GB（含 bge-m3 权重 2.1 GB）、e2_need_gap 2.3 GB、stageB 1.0 GB、.research_tmp/runs 0.65 GB、benchmark-audit-0926 等 |
| 可重建的缓存（向量 / 索引 / API） | ≈12.5 GB | sqm lightrag/work 6.6 GB、sqm kb/embed_cache 3.4 GB、sqm baselines/ours 1.2 GB、cs2 base_kb/embed_cache 1.7 GB、cs2 arm_ours 向量 0.9 GB、cs2 base_kb_v2 record_vecs.f32 0.45 GB、paperqa 索引 0.37 GB、refgraph_cache + s2_cache 0.36 GB |
| 下载的数据集 / 语料 / 环境 | ≈4.1 GB | litsearch corpus 1.2 GB、deepscholar venv311 ≈1.46 GB（50k 文件）、sqm corpus 0.85 GB、mdaqa 0.19 GB、literature 0.31 GB、corpus_pool 73 MB、qasa 54 MB |
| 实验产出（KB json / 答案 / 判分 / 日志） | ≈3.0 GB | cs2 非缓存部分约 1.1 GB（顶层 json+log 361 MB）、sqm kb/growth + postcheck 等约 1.3 GB、review_1002 0.27 GB、demo_growth_kb* 0.2 GB |
| 网页抓取 / 调研草稿 | ≈0.55 GB | 散文件 37 MB、scratch 168 MB（4,584 文件）、archive_oneoff 131 MB、_trash_pending 61 MB、ppt_edit / talks / *-survey / *-check 共约 150 MB |
| 文档 | <10 MB | docs_decisions 2 MB、paper_drafts 4 MB、phase0 2 MB |
| 代码 | ≈20 MB | 研究区 py；已跟踪的 py 合计 10.3 MB |

### 1.3 `.research_tmp/` 顶层（36 个目录 + 94 个散文件）

| 目录 | MB | 文件数 | 已跟踪 | 状态 |
|---|---|---|---|---|
| experiments/ | 48,939 | 156,550 | 3,181 | 活跃和退役混放 |
| runs/ | 651 | 2,495 | 392 | INDEX 里写的是"死档"，但 kernel_v2 有 392 个文件被跟踪 |
| literature/ | 313 | 1,135 | 4 | 调研档案，117 个 py 未跟踪 |
| review_1002/ | 270 | 2,381 | 553 | 10-02 全面审查；p6 有任务在跑 |
| scratch/ | 168 | 4,584 | 1 | 518 个 py；其中 asta-bench 被冻结的 judge 依赖 |
| archive_oneoff/ | 131 | 3,313 | 0 | 归档 |
| benchmark-audit-0926/ | 64 | 29 | 0 | |
| _trash_pending/ | 61 | 493 | 0 | 09-27 隔离的待删区，一直没有清 |
| 其余 28 个小目录 | ≈190 | ≈700 | 0 | ppt_edit / talks / *-survey / *-check / peerqa / nvm0926 … |
| 根下 94 个散文件 | 37 | 94 | 0 | html 14、json 16、txt 13、py 11、md 27、xml 5、pdf 5、mjs 3。最大的是 emnlp2025.html（11.4 MB）。mdaqa-original.json 和 mdaqa-raw.json 各 5 MB，看起来是同一份数据存了两遍。另有 PUBLISHABILITY-ASSESSMENT-0930.md 这类正式判断和 might_jobhunt.md 这类个人笔记混在一起 |

`.research_tmp/INDEX.md`（09-25）只登记了 36 个顶层目录中的 10 个，违反它自己写的"新增必登记"规则。

### 1.4 experiments/benchmarks/

| 子目录 | MB | 文件数 | 已跟踪 | 要点 |
|---|---|---|---|---|
| scholarqa_multi | 14,613 | 2,586 | 806 | lightrag 6,580 / kb 5,446（embed_cache 3,355、growth 820）/ baselines/ours 1,222 / corpus 853 / paperqa 366 |
| cs2 | 4,258 | 2,653 | 1,783 | base_kb 1,880 / arm_ours 999 / base_kb_v2 509 / arm_memorized 166 / demo_growth_kb* 203（0 跟踪）/ arm_storm_dsb 671 个文件全部跟踪 |
| deepscholar | 1,498 | 50,398 | 0 | **venv311 没有被 ignore**（`.venv/` 规则匹配不到 `venv311/`），在 git status 里产生 33,145 条未跟踪噪声 |
| litsearch | 1,199 | 10 | 2 | corpus 下的 parquet 已正确 ignore |
| _shared | 434 | 8,806 | 526 | refgraph_cache 242 + s2_cache 119（已 ignore）、corpus_pool 73、tools 30 个文件 |
| mdaqa | 189 | 6,560 | 4 | arms/ 5,943 个文件和 texts/ 603 个文件**既没跟踪也没 ignore** |
| qasa | 54 | 1 | 1 | qasa_test.jsonl 53.7 MB 被跟踪 |

experiments/ 下 benchmarks 以外的部分：archive 19,206 MB / 52,610 个文件（paperscope_full 10,236、paperscope_r2 8,332、airqa 638）；baselines 4,105 MB / 23,842 个文件（paper_scion 3,214，其中 bge-m3 的 pytorch_model.bin 占 2,166）；e2_need_gap 2,326；stageB 968；downstream 35；其余 9 个目录合计 <60。

### 1.5 cs2/ 内部

- 顶层 394 个文件：json 185 个（direct_scores_* 91、judge_input_* 85、FREEZE_* 3、其他 6），log 144 个（judge 77 / arm 26 / run 16 / queue 11 …，被 `*.log` 规则 ignore），py 55 个，md 4 个，sh 3 个，txt / xml / jsonl 各 1 个。顶层文件合计 361 MB。
- 55 个 py 里，正式管线（run_vnext / direct_judge / cs2_scoring / harness_arm_run …）和一次性脚本（probe_l1_v2、patch_ch3、verify_cit_bug、cmp15、test_ppr_after …）放在同一层，从文件名看不出哪些在用。
- judge_runs_ours{,2,3,4}、arm_vnext 和 arm_vnext_dsb（后两者 131 个文件只跟踪了 2 个）、_discarded_contended_1003（14 个文件，没有写弃用原因）都平铺在同一层。
- FREEZE_CS2_TEST_1003_v9b.json：git_head 是 26fbd63d，含 14 个文件的 sha256_16。**本次重算 14/14 全部一致。** 其中 record_vecs.f32（453 MB）按设计没有跟踪，所以这块 KB 向量目前只有这块硬盘上的一份。

### 1.6 review_1002/

270 MB，2,381 个文件，跟踪了 553 个（全部是强制 add，父目录被 ignore）。p8 119 MB / 1,011 个文件只跟踪 10 个；p4 95 MB；p6 576 个文件跟踪 403 个，并且有任务在跑；harness_dump 24 个文件没跟踪。按类型：json 1,470、md 594、xml 96、py 89、log 24。

### 1.7 最大的 20 个文件

| # | MB | 路径（省略 `.research_tmp/experiments/` 前缀） | git 状态 |
|---|---|---|---|
| 1 | 3,354 | benchmarks/scholarqa_multi/baselines/lightrag/work/vdb_relationships.json | 未跟踪，但历史里有 2 个版本 |
| 2 | 2,398 | …/lightrag/work/vdb_entities.json | 同上 |
| 3 | 2,166 | baselines/paper_scion/models/bge-m3/pytorch_model.bin | ignore |
| 4 | 1,665 | benchmarks/cs2/base_kb/embed_cache/embed_r2_props.json | **在 HEAD 中跟踪** |
| 5 | 814 | benchmarks/scholarqa_multi/baselines/ours/emb_cache_records.bin | **跟踪** |
| 6 | 787 | benchmarks/scholarqa_multi/kb/growth/registry_growth/embed_cache/embed_r2_canon.json | **跟踪** |
| 7 | 721 | benchmarks/scholarqa_multi/kb/embed_cache/embed_r2_queue.json | ignore |
| 8 | 677 | benchmarks/scholarqa_multi/kb/embed_cache/embed_r2_props.json | ignore |
| 9 | 607 | benchmarks/cs2/arm_ours/emb_cache_records.bin | **跟踪且已修改**（历史里已有 415 MB 和 596 MB 两个版本） |
| 10 | 453 | benchmarks/cs2/base_kb_v2/record_vecs.f32 | ignore；FREEZE 依赖它 |
| 11 | 358 | benchmarks/scholarqa_multi/baselines/ours/text_index/emb.bin | **跟踪** |
| 12 | 332 | benchmarks/scholarqa_multi/kb/embed_cache/embed_surfaces.json | ignore |
| 13 | 306 | benchmarks/cs2/arm_ours/text_index/emb.bin | **跟踪且已修改**（历史里已有 2 个版本） |
| 14 | 296 | benchmarks/scholarqa_multi/kb/embed_cache/embed_dedup_setup.json | ignore |
| 15 | 294 | …/lightrag/work/kv_store_llm_response_cache.json | 未跟踪，历史里有 |
| 16 | 281 | …/kb/embed_cache/embed_dedup_variant.json | ignore |
| 17 | 268 | …/lightrag/work/vdb_chunks.json | 未跟踪，历史里有 |
| 18 | 263 | benchmarks/litsearch/corpus/corpus_clean_0.parquet | ignore（另有 5 个分片，各约 185 MB） |
| 19 | 128 | benchmarks/scholarqa_multi/kb/tmp_qe.npy | **跟踪**，文件名带 tmp |
| 20 | 109 | …/lightrag/work/graph_chunk_entity_relation.graphml | 未跟踪，历史里有 |

`.git` 里另有 3 个 >1.4 GB 的单个文件：tmp_pack_z2dr3e（1,566 MB）、tmp_obj_64u5uS（1,449 MB），以及 lightrag 的散对象（每个 1.1～1.5 GB）。说明：对整棵树的 `find -size +150M` 没有跑出结果（Git Bash 在大目录上超时），上表是在已知的重目录里逐个 `ls -S` 汇总的，可能漏掉零散的大文件。

### 1.8 最大的 10 个目录

experiments/archive/paperscope_2026-09 19.2 GB > .git 14 GB > sqm/baselines/lightrag 6.6 GB > sqm/kb/embed_cache 3.4 GB > baselines/paper_scion 3.2 GB > e2_need_gap 2.3 GB > cs2/base_kb 1.9 GB > deepscholar（venv）1.5 GB > sqm/baselines/ours 1.2 GB > litsearch/corpus 1.2 GB。

## 2. Git 卫生

### 2.1 推送阻塞（最急）

| 项 | 实测 |
|---|---|
| origin/main | cfef2fcc，09-20 push；520 个文件，55 MB |
| 本地 main | 领先 215 个提交，落后 0 |
| 未推送的 blob | 3,781 个，未压缩 15.40 GB；>100 MB 的有 17 个（13.79 GB）；>50 MB 的有 25 个 |
| 已推送历史中最大的 blob | 35.1 MB（dist/submit/*.mp4），>10 MB 的只有 2 个 |

结论：已推送的历史是干净的，问题全部出在 09-20 之后的本地提交上。所以重写可以只限于 `origin/main..main` 这个范围，已推送提交的哈希不会变，事后的 push 仍然是 fast-forward，不需要 force push。

主要来源：
- d5872a0a（09-23）把 lightrag/work 整个提交进去了（vdb_relationships 2.3 GB×2、vdb_entities 1.7 GB×2、kv cache 0.2 GB×2…）。377e1330（09-25）之后不再跟踪，但历史里还在。
- embed_cache/*.json、*.bin、*.npy 被多次提交，每次都是一个完整的新 blob。
- local config 里 `http.postbuffer=524288000`，看起来以前试过硬推大文件。

### 2.2 对象库和垃圾文件

- pack：5 个，共 237.7 MiB。最大的一个 234 MB 来自 LogicKG 时代的导入，属于正常历史。
- 散对象：28,553 个，10.19 GiB。其中 >5 MB 的有 49 个：**37 个可达（7.99 GB）**，也就是 §2.1 的未推送大 blob；**12 个不可达（1.81 GB）**，是 add 之后又被覆盖或 amend 掉的悬空 blob。是否算上 reflog，结果一样。
- 垃圾文件（count-objects 报的 garbage: 3，2.94 GiB）：
  - `.git/objects/pack/tmp_pack_z2dr3e`，1.53 GiB，10-02 20:57。这是 repack/gc 写到一半的 pack。同一时刻还留下了 `.git/gc.pid`（10-02 20:54，PID 50672，进程已经不存在）。推断：auto-gc 在打包 10 GB 散对象时被中断了。
  - `.git/objects/c0/tmp_obj_64u5uS`，1.42 GiB，09-23。这是写散对象时先写的临时文件，正常应该随后改名，被中断就留了下来。时间和 lightrag 大文件入库那次一致。
  - `.git/objects/6c/tmp_obj_GlNXkX`，631 B，10-02。
  - 这三个文件不被任何 ref 或 index 引用。git 只会在 prune 时按过期时间（默认 2 周）清理它们，所以短期内不会自己消失。
- 持续风险：散对象数量是 gc.auto 默认阈值（6,700）的 4 倍多。之后每次 commit 都可能在后台再次触发 auto-gc，去重新打包这 10 GB：跑实验时抢 IO，被打断又会留下新的 tmp_pack。

**安全清理（只是建议，本次没有执行）：**
1. 等 CS2 test 结束；关掉 VS Code（它的 git 扩展会常驻 git.exe）；用 `tasklist | findstr git` 确认没有 git 进程。
2. 先把整个 `.git` 复制到外部盘（robocopy /E），再删除上面 3 个 tmp_* 文件和 gc.pid。可以回收 2.94 GiB，不影响任何历史。
3. **在完成 §7 Phase 3 的历史重写之前，不要跑 `git gc` / `git repack -ad`**。现在 gc 只会把 8 GB 大 blob 压进 pack，白白消耗 IO，而且很可能再次中断。可以考虑临时 `git config gc.auto 0`（改 config，需要你同意）。
4. 重写完成后再执行 `git reflog expire --expire=now --all && git gc --prune=now`，这一步会顺带清掉那 12 个不可达对象。预计 .git 会从 14 GB 降到 1 GB 以下（重写后需要实测确认）。

### 2.3 跟踪了但不该跟踪的（HEAD 树 4,288 个文件，5.78 GB）

| 类别 | 文件数 | MB | 判断 |
|---|---|---|---|
| 向量 / 嵌入缓存（.bin .npy embed_cache） | 16 | 4,777 | 全部取消跟踪，改为 artifacts 加校验和 |
| KB 产物 json（records / views / registry …） | 148 | 428 | 只留冻结快照（可以压缩或进 artifacts），中间版本不进 git |
| 语料 / 数据集文本 | 1,089 | 216 | corpus_pool 的 md 和 qasa_test.jsonl 可以重新下载，改为 MANIFEST |
| 答案 / 判分 / 台账 | 247 | 184 | 只保留冻结 run 的产出，单个文件超过 5 MB 的压缩 |
| 旧 runs（.research_tmp/runs/kernel_v2） | 391 | 49 | 已在 origin 上，保留即可，但应移到 archive |
| 日志 / 索引 / 备份 | 11 | 27 | paperqa run.log.archive1、views.json.bak_v2、*.db、*.zip、*.store、*.pos：全部取消跟踪 |
| 文档笔记 | 1,271 | 37 | 大部分是 corpus md 和 review 输出 |
| 代码 / 配置 | 375 | 10 | 保留 |

具体例子：tmp_qe.npy（128 MB）和 tmp_ce.npy（41 MB），文件名就带 tmp；views.json.bak_v2（18 MB）对应的 ignore 规则写成了 `kb/views.json.bak_v2`，这个模式带斜杠，会锚定在仓库根，所以匹配不到深层路径（而且文件在规则加入前就已经被跟踪）。没有任何 __pycache__ 或 .log 被跟踪，这两条规则是有效的。

**跟踪了又命中 ignore 的文件：1,039 个**，都是 `git add -f` 加进去的：review_1002 553 个（p6 403）、.research_tmp/runs 392 个、docs_decisions 30 个、experiments/downstream 等 41 个、literature 4 个。实际效果是"强制 add 过的才有备份，其余全靠记忆"，规则本身已经失去作用。

当前有 18 个已跟踪文件处于修改状态，其中两个是 cs2/arm_ours 的 emb_cache_records.bin 和 text_index/emb.bin。**在取消跟踪之前不要 `git commit -a`。**

### 2.4 该跟踪但没跟踪的

| 内容 | 现状 | 风险 |
|---|---|---|
| paper_drafts/（RESULTS-LEDGER.md、METHOD-CHAPTER 草稿、作图脚本） | 0 个跟踪，被 ignore | README 指定的"权威台账"没有入库 |
| phase0/（9 份 VERDICT 判决档） | 0 个跟踪 | 阶段 0 的判决全部只有单份 |
| docs_decisions/ | 134 项里只跟踪了 30 项 | 104 份决策档只有单份 |
| scratch/ai2_baseline_2026-09-08/asta/asta-bench-main | 没跟踪，被 ignore | **frozen 的 direct_judge.py 依赖它**（第 28 行绝对路径 sys.path）；verify_cit_bug.py 同样依赖 |
| cs2 test 产出：answers_test100_r1/r2、answers_harness_test100、judge_input_vnext_test100_r1 等 | 没跟踪（任务在跑，等跑完再处理） | 论文主表的原始答案 |
| cs2/arm_vnext*（131 个文件跟踪 2 个）、base_kb_v2 23 个、arm_ours 100 个 | 没跟踪也没 ignore | 现役方法的产出没有入库 |
| mdaqa/hippo_smoke.py、hippo_tiny.py、sqm/baselines/hgr_operate.py | 没跟踪 | 小代码 |
| scratch 518 个、archive_oneoff 449 个、literature 117 个、stageB 100 个、e2_need_gap 32 个、phase0 9 个 py | 没跟踪 | 大多是历史一次性脚本。下结论用过的应在归档时登记 |

同时存在**该 ignore 但没 ignore 的**：deepscholar/venv311（33,145 个文件）、mdaqa/arms 和 texts（6,546 个文件）。`git status --untracked-files=all` 一共 40,225 条，真正该看的内容淹没在里面。

### 2.5 分支、worktree、stash、config

- 分支只有 `main` 和 `origin/main`；`git worktree list` 只有主工作树；stash 为空；611 个提交（03-06 起）；reflog 247 条。
- `.git/config` 有两处异常（建议修，需要你同意后再改）：
  - `branch.main.merge` 有两个值：`refs/heads/main` 和 `refs/heads/feat/global-community-treecomm-migration`（远端没有这个分支）。直接 `git pull` 可能会尝试章鱼合并，或者报错。
  - `branch.main.remote` 和 `branch.main.vscode-merge-base` 都有重复，其中一个值指向已经不存在的 `origin/feature/crossref-batch-resolve`。
  - 修法：`git config --unset-all branch.main.merge && git config branch.main.merge refs/heads/main`，vscode-merge-base 用同样方式处理。
- 没有 pre-commit 钩子，也没有 .gitattributes。git-lfs 3.7.1 已安装但没用。git-filter-repo 没有安装。

## 3. 密钥与安全（只列路径，不列值）

方法：把 `.env` 中每个长度 ≥16 的值逐一在已跟踪文件和全部历史（`git log -S`）里精确匹配；再按 sk- / AKIA / ghp_ / hf_ / AIza / Bearer / `api_key=` 等模式扫描已跟踪的文本文件。

- `.env`：被 `.gitignore:14` 覆盖，`git ls-files` 里没有，历史里也从未出现。OPENROUTER / DEEPSEEK / PARATERA / BOHRIUM / SERPER / POSTGRES / DATABASE_URL 这些值在已跟踪文件和全部历史中命中都是 0。
- **真实泄露 1：MINERU_TOKEN**，硬编码在 `.research_tmp/experiments/benchmarks/scholarqa_multi/promote_to_deep.py:47`。由 dce888a7（09-24）引入，没有推送。
- **真实泄露 2：EMBEDDING_API_KEY**（GPUStack，`gpustack_` 前缀），出现在：
  - `.research_tmp/experiments/benchmarks/_shared/tools/multi_closedbook_recall.py:36`
  - `.research_tmp/experiments/benchmarks/cs2/arm_storm/2bb40aa93ac3a6a673c839bd/Have_specialized_…/run_config.json`。这是 STORM 把带 api_key 的配置原样写进了运行产出，出现 5 次。后续每个 STORM run 都会再写一次。
  - 涉及 4 个提交，全部没有推送。
- 误报：`cs2/judge_run.py:85` 是占位符（`sk-pla…`）；`_shared/corpus_pool/mineru500/Tlsdsb6l9n/*.md` 里的 AKIA 是 MOL-INSTRUCTIONS 论文中的蛋白质序列；其余 sk- 命中是 "task-specific" 这类子串（加上词边界后消失）。
- 处置建议：
  1. 代码改为读 env，STORM 落盘前把 api_key 抹掉；
  2. Phase 3 重写时用 `--replace-text` 从历史中清掉这两个值；
  3. MinerU token 建议轮换（第三方服务）；GPUStack key 如果只在内网，风险较低，也建议换掉。
- 没覆盖到的范围：二进制和日志文件没扫；ignore 目录只抽查了 review_1002 和 scratch（0 命中）；仓库是 public 还是 private 没核实（本机没有 gh）。

## 4. 文档状态

| 文档 | 最后提交 | 状态 | 处置 |
|---|---|---|---|
| README.md | 09-22 | 过期：写的是"冻结 schema v1.4 / 四视图 / 三个入口文档"，指向 ASSET-STATE、DIRECTION、RESULTS-LEDGER 三份旧档 | 按 NARRATIVE-V8 重写 |
| DIRECTION.md | 09-25 | 过期：v3"三考场 / 三家族 PK 矩阵"；09-27 注已经自认被取代，之后又经过 v6→v7→v8 | 归档 |
| ASSET-STATE.md | 09-20 | 过期：停在 Multi-108 预注册之前，STAGEB 路径多数已经归档 | 归档；"现在有什么"改由 README 的布局一节和 EXPERIMENTS 回答 |
| DESIGN-DECISIONS.md | 09-19 | 历史：09-16 的审视靶子清单 | 归档，结论摘要进 DECISIONS |
| CHANGELOG.md | 09-22 | 落后 215 个提交 | 保留，补一条 09-23～10-04 的汇总 |
| .research_tmp/INDEX.md | 09-25 | 只登记了 10/36 个目录 | 迁移后废弃 |
| paper_drafts/RESULTS-LEDGER.md | 09-14（mtime） | 过期且没入库：数字还是旧主场 50 题 / DRL，没有 CS2 / DSB / Multi-108 | 由新的 RESULTS.md 取代 |
| docs_decisions/（134 项） | 08 月 93 份 / 09 月 33 份 / 10 月 8 份 | 19 份自标"取代 / 作废"，没有索引，只入库 30 份 | 建 DECISIONS 索引，旧档整体移入 archive |

当前口径实际分散在 5 处：NARRATIVE-V8-1003、REBUILD-PLAN-1003、PAPER-DRAFT-{INTRO,METHOD,EXPERIMENTS}-1003、cs2/FREEZE_CS2_TEST_1003_v9b.json，以及仓库外的 `~/.claude/.../memory/handoff-1003-rebuild.md`。MEMORY.md 里有 4 条以上同时标着"最新 / 先读这个"，彼此矛盾。memory 不在仓库里，也不随仓库备份。

**最小文档集（建议）：**
- `README.md`：一段话主张（取自 NARRATIVE-V8 §1）、快速上手、目录地图、"当前状态"一行加指向 RESULTS 的链接。
- `docs/ARCHITECTURE.md`：建库（src/kb_compiler）→ 领域状态 → 答题栈（现在的 _shared/tools）→ 基准适配。底稿用 PAPER-DRAFT-METHOD-1003。
- `docs/EXPERIMENTS.md`：每个基准写清楚数据来源和校验和、冻结清单、一条复现命令、已知坑（S2/OpenAlex 限流、Elicit 适配等，从 handoff 搬过来）。
- `docs/RESULTS.md`：只收冻结后的数字。每行带 freeze id、git 哈希、scores 文件路径、n、置信区间。取代 RESULTS-LEDGER。
- `docs/DECISIONS.md`：ADR 式的一行一条（日期 / 决定 / 理由 / 取代了哪条 / 原档路径）。现有 134 份不改写，只建索引。
- `docs/paper/`：论文各章草稿（10-03 那 3 份 PAPER-DRAFT，以及 paper_drafts/ 里的 method 章）。
- 归档：DIRECTION / ASSET-STATE / DESIGN-DECISIONS → `docs/archive/2026-09/`；docs_decisions 中 10-01 之前的部分 → `docs/decisions/archive/`。用 `git mv`；没跟踪的先 add 再 mv。memory 只保留一条指向 docs/ 的"当前指针"，不再复述数字。
- 注意：现在 `.gitignore` 里有 `docs/*` 加白名单的规则，新建 `docs/ARCHITECTURE.md` 等文件会被直接 ignore，必须同时改这段规则。

## 5. 目标布局

```
CompileScholar/
├── README.md  CHANGELOG.md  pyproject.toml  .env.example  .gitignore  .pre-commit-config.yaml
├── conf/                    base.yaml（跟踪）+ local.yaml（ignore）+ paths.yaml（数据 / 缓存根目录，取代硬编码路径）
├── src/compile_scholar/     唯一可 import 的包
│   ├── kb_compiler/ kb_infra/ retrieval/      ← 现有 src/
│   └── answer/              ← _shared/tools 中的 answer_pipeline / refgraph / cutoff / report_* / external_tools
├── bench/                   每个基准一个目录，只放薄壳：runner + scoring + config + freeze
│   ├── common/              judge / 统计 / 配对检验（现在的 multi_judge_* / multi_paired_stats / direct_judge 核心）
│   ├── cs2/  dsb/  multi108/  mdaqa/  litsearch/
│   │   ├── run.py  score.py  configs/*.yaml
│   │   └── freezes/FREEZE_<bench>_<split>_<yyyymmdd>_<ver>.json
│   └── tools/               一次性诊断脚本（probe_* / audit_* / verify_*），每个文件头注明用途和日期
├── third_party/             asta-bench 等：vendored 或用 pip pin，写明版本号和 commit
├── tests/
├── docs/                    §4 的最小文档集 + decisions/ + archive/ + paper/
├── data/        （ignore）  raw/<bench>/、kb/<kb_id>/；只跟踪 data/MANIFEST.tsv
├── artifacts/   （ignore）  冻结用的大文件（向量、KB 快照）；只跟踪 artifacts/MANIFEST.tsv
├── results/     （跟踪）    results/<run_id>/{manifest.json, scores.json, answers.json.gz}，只放冻结的 run，单文件 <5 MB
├── runs/        （ignore）  所有运行中或草稿 run 的输出和日志：runs/<run_id>/...
├── cache/       （ignore）  s2 / openalex / refgraph / embed / llm 响应缓存，可以随时删
├── scratch/     （ignore）  网页抓取和草稿，14 天过期，有用的要主动"晋升"到 docs/ 或 bench/tools/
└── archive/     （ignore）  冷存储入口（真正的数据放外部盘，这里只留 MANIFEST 和 RETIRED.md）
```

## 6. 治理规则

1. **跟踪什么**：代码、配置、测试、docs、`*/MANIFEST.tsv`、`results/` 下冻结 run 的 manifest / scores / 压缩后的答案。**不跟踪**：向量、索引、API 缓存、venv、模型权重、原始数据集、日志、草稿。
2. **大小上限**：单个文件超过 5 MB 不进 git（pre-commit 用 `check-added-large-files --maxkb=5120` 拦截）。确实需要留存的放进 `artifacts/`，在 MANIFEST.tsv 登记一行：`path  sha256  bytes  produced_by(cmd + git sha)  source/license  backup_location`。
3. **ignore 的写法**：数据目录默认忽略、代码白名单放行。禁止 `git add -f`；需要例外就改 `.gitignore` 并在 commit message 里说明原因。补充规则：`**/venv*/`、`**/.venv*/`、`**/embed_cache/`、`*.bin`、`*.npy`、`*.f32`、`*.parquet`、`**/lightrag/work/`、`**/index/`、`*.bak*`、`**/tmp_*`。
4. **run manifest**：每个 runner 启动时写 `runs/<run_id>/manifest.json`，内容包括 git sha、是否 dirty（dirty 时附 diff 的哈希）、argv、解析后的 config、输入文件的完整 sha256（现在的 v9b 用的是 16 位截断）、模型 / judge / endpoint 名称（不写 key）、起止时间、输出文件的哈希。FREEZE 文件就是"被晋升的 manifest"。v9b 的格式已经很接近，补上 dirty 和输出哈希即可。
5. **命名**：`run_id = <bench>-<split>-<arm>-<yyyymmdd>-<tag>`，例如 `cs2-test100-vnext-20261004-v9b-r2`。日期一律写 yyyymmdd，不用 1003 / 0929 这种四位写法，因为跨年会撞。废弃的 run 移到 `runs/_discarded/<run_id>/`，并附一个 REASON.md，说明为什么弃用。
6. **退役实验**：写 `archive/<name>/RETIRED.md`（退役日期、被谁取代、哪些决策档引用过它、关键数字），数据移到外部盘，MANIFEST 里记下哈希和位置。仓库内只留这个说明。
7. **状态的唯一来源**："现在是什么"只写在 README 的状态行和 RESULTS.md 里；"为什么"只写在 DECISIONS.md 里。被取代的档案在文首写一行 `Superseded-by: <路径>`。memory 只放指针，不放数字。
8. **路径**：所有代码都通过 `conf/paths.yaml` 或 `CS_ROOT` 环境变量解析路径。禁止 `C:\Users\D0n9\...`，禁止往 scratch 插 sys.path。
9. **密钥**：只放在 `.env`。pre-commit 加 gitleaks（或 detect-secrets），版本要 pin 住。第三方工具的运行产出在落盘前先抹掉 key。
10. **推送节奏**：每天至少 push 一次。push 失败本身就是警报（这次的问题就是 13 天没推、没人发现），当天要查原因。

## 7. 迁移计划（按阶段走，每一步都可以回退；删除任何东西之前，先做一份经过校验的备份）

**Phase 0（现在，只读）**：本审计。

**Phase 1（等 CS2 test 结束、run_vnext / harness / retrieval_mcp 全部退出，分数已经写进 FREEZE 和台账之后）**
1. 关掉 VS Code，用 tasklist 确认没有 git.exe 和 python.exe 指向本仓库。
2. 整个仓库目录用 `robocopy /E` 复制到外部盘（约 65 GB；也可以只复制 `.git` 加 cs2 / _shared / base_kb_v2 加 docs_decisions 加 paper_drafts 加 phase0）。复制后对 FREEZE v9b 的 14 个文件重新计算哈希，确认一致。
3. 另外做一份 `git bundle create main-20261004.bundle main`（它包含大 blob，约 10 GB），然后在备份盘上跑一次 `git bundle verify`。

**Phase 2（低风险，不改历史）**
1. 删除 3 个 tmp_* 文件和 gc.pid（§2.2），回收 2.94 GiB。
2. 修 `.git/config` 里重复的 merge / remote 项（§2.5）。
3. 两处硬编码的密钥改成读 env；STORM 的 run_config 落盘时抹掉 api_key；轮换 MinerU token。
4. 按 §6.3 补全 `.gitignore`，修好 docs/* 的白名单。
5. 对 §2.3 的向量文件、tmp_*.npy、*.bak*、*.log.archive*、paperqa 索引执行 `git rm --cached`。**文件留在磁盘上**，先把哈希写进 `artifacts/MANIFEST.tsv` 再提交。HEAD 树会从 5.78 GB 降到约 1 GB。
6. 把该跟踪的东西加进来：paper_drafts、phase0、docs_decisions 全部，test100 的答案和分数（大于 5 MB 的先 gzip），mdaqa 和 sqm 那 3 个 py。

**Phase 3（高风险：重写未推送的历史。需要你明确同意后才能做）**

这一步绕不过去：不做的话，这 215 个提交永远推不上去。已推送历史里最大的 blob 只有 35 MB，所以过滤条件不会碰到它们，09-20 之前的提交哈希保持不变，push 仍然是 fast-forward。
1. 安装 git-filter-repo，版本 pin 死（例如 `pip install git-filter-repo==2.47.0`；装之前到 PyPI 上确认这个版本存在）。
2. 在 Phase 1 备份的基础上，先克隆一份副本试跑：
   `git filter-repo --refs origin/main..main --strip-blobs-bigger-than 20M --invert-paths --path-glob '*/lightrag/work/*' --path-glob '*/embed_cache/*' --path-glob '*.bin' --path-glob '*.npy' --path-glob '*/paperqa/index/*' --replace-text secrets.txt`
   secrets.txt 只在本地临时生成，用完删掉。先加 `--dry-run` 看结果。
3. 验收标准：
   - `git rev-list --objects origin/main..main | git cat-file --batch-check` 里没有超过 20 MB 的 blob；
   - `git log -S<密钥>` 命中为 0；
   - 提交数仍然是 215；
   - FREEZE v9b 的 14 个哈希在工作树中仍然一致；
   - `.git` 小于 1 GB（gc 之后）。
4. **哈希会变**：FREEZE 里的 git_head 26fbd63d、memory 里的 48022a46 / 525b5ce0 等都在重写范围内。保存 filter-repo 生成的 `.git/filter-repo/commit-map`，归档成 `docs/archive/commit-map-20261004.tsv`，并在 FREEZE 里补一个字段 `git_head_rewritten`。
5. 先推到一个新分支 `cleanup/20261004`，确认成功后再把 main fast-forward 过去。全程不用 force push。
6. 备选方案（不想重写哈希时）：从 origin/main 新开一个分支，放一个"瘦身后的快照"提交，旧的细粒度历史只保存在 bundle 里。代价是远端看不到 215 个提交的细节。

**Phase 4（目录迁移。等论文的 test 数字定稿之后再做，每步一个 commit）**
1. 打 tag `freeze-cs2-v9b`（指向重写后对应的提交），作为复现锚点。
2. 新建 `conf/paths.yaml` 和一个 `paths.py`，替换掉那 46 处 `C:\Users\D0n9` 绝对路径和各种 sys.path hack。跑 `pytest tests/`，再跑 CS2 dev 的 2 题冒烟（开 cache），逐字段比对迁移前后的输出。
3. `git mv`：`_shared/tools/*` → `src/compile_scholar/answer/`，判分相关 → `bench/common/`，`cs2/*.py` → `bench/cs2/`（诊断脚本放 `bench/cs2/tools/`）。每移一批都重复第 2 步的验证。
4. asta-bench → `third_party/`，或者改成 pip 依赖（pin 版本号）。直接修正 direct_judge 的 import。
5. 没跟踪的大数据（KB、向量）不能用 git mv，做法是复制到 `data/` 或 `artifacts/` 新位置，sha256 核对通过后，旧位置改名为 `*.migrated` 观察 7 天，确认没有任何读取报错再删。
6. 用 §4 的内容新建 5 份文档；旧文档 `git mv` 到 docs/archive/。

**Phase 5（冷存储，释放约 27 GB 以上）**
- 搬到外部盘，并登记 MANIFEST 和 RETIRED.md：
  - experiments/archive/paperscope_2026-09（19.2 GB）
  - experiments/baselines（4.1 GB，含 bge-m3 权重，可以重新下载）
  - e2_need_gap（2.3 GB）
  - stageB（1.0 GB）
  - .research_tmp/runs（0.65 GB）
  - archive_oneoff、benchmark-audit-0926、_trash_pending（隔离期已经过了，复核后可以直接删）
- sqm 的 lightrag/work（6.6 GB）：重建要花 LLM token。如果 Multi-108 还要用 LightRAG 臂，就留在 `cache/`，否则进冷存储。
- deepscholar/venv311：先 `pip freeze > bench/dsb/requirements.lock`，然后移出仓库目录（或者至少 ignore 掉）。
- `.research_tmp` 根下的 94 个散文件按主题移到 `scratch/web/2026-09/`；PUBLISHABILITY-ASSESSMENT-0930 这类正式判断晋升到 docs/decisions/。

**Phase 6（防止复发）**：pre-commit 配置大文件检查加 gitleaks；runner 统一写 manifest；README 状态行和 RESULTS.md 在每次 freeze 时一起更新；每天 push。

**回退方式**：Phase 2 和 4 都是普通提交，可以直接 revert；Phase 3 的回退是用 Phase 1 的 `.git` 备份或 bundle 整体恢复；Phase 5 的回退是按 MANIFEST 从外部盘拷回。

## 8. 这次没有核实的

- 对整棵树的超大文件扫描没跑完（§1.7 是按已知目录汇总的）。没跑 `git fsck`（10 GB 散对象，很慢）。
- 密钥扫描只覆盖了已跟踪的文本文件和 2 个 ignore 目录；二进制和日志没扫；仓库是 public 还是 private 没查。
- 类别体积是按目录归类的，混合目录（cs2/base_kb、sqm/kb）拆分时有约 ±5% 的误差。
- 没有打开 review_1002/p6 和 cs2 正在写的文件，也没有评估在跑任务的进度。







