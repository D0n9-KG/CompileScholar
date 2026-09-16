# LogicKG

面向科研智能体的科学文献知识层：LLM 从论文全文抽取**类型化记录**（冻结 schema v1.4，逐字引文锚 + epistemic 标记），编译为四视图（对比矩阵/方法谱系/覆盖地图/方法卡），agent 经 typed tools 访问。当前形态自 2026-09-03 方向定稿（需求驱动知识模型，不做图）；旧形态（Neo4j 图谱工作台/超图 granular_agent/比赛检索系统）全部归档于 `archive/`（磁盘保留，可逆）。

## 导航（三个入口文档）

| 文档 | 回答什么问题 |
|---|---|
| [ASSET-STATE.md](ASSET-STATE.md) | 现在有什么：活代码/三套 KB/答题栈/五条工作线状态 |
| [DIRECTION.md](DIRECTION.md) | 往哪打：论文大方向三线+监控阈值+执行队列 |
| `.research_tmp/paper_drafts/RESULTS-LEDGER.md` | 权威数字：各考场成绩（唯一可引用口径） |

## 仓库布局

```
src/kb_compiler/     活管线：records（抽取链+表格/语义/识图通道+注册表）
                     + views（四视图编译+typed tools）+ verification
src/kb_infra/        LLM provider 网关（Paratera/CST/LOCAL-vLLM）+ embedding
src/claim_coverage.py  独立工具
tests/               65 个活测试（含 import-gate 机器闸：禁触旧栈+schema 冻结守卫）
archive/             历史形态归档（gitignored，磁盘保留可逆；见 MANIFEST）
.research_tmp/       全部实验工作区/调研/预注册/判决档（gitignored；见其中 INDEX.md）
```

## 运行

```bash
python -m pytest tests/ -q          # 需 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 若本地 logfire 插件损坏
```

LLM 凭据在 `.env`（PARATERA/CST/LOCAL provider）；建库与答题入口见 ASSET-STATE.md §2-§3。
