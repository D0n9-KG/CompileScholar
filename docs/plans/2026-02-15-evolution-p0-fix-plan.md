# Evolution P0修复实施计划

> **创建日期**: 2026-02-15
> **状态**: Phase 1 准备中
> **Git分支**: feature/evolution-p0-fix

## 背景

20论文E2E测试暴露了Evolution功能的关键问题：
- 仅8个SUPPORTS关系（0.56%覆盖率）
- 全部为自环（from==to）
- 0个CONTRADICTS/REFINES关系
- 研究脉络和问题发现功能完全不可用

## 根因分析

| 问题 | 根因 | 优先级 |
|------|------|--------|
| P0-1 | Embedding 502错误静默降级为lexical模式 | A |
| P0-2 | 阈值0.9对lexical模式过高（仅16个候选对） | A |
| P0-3 | 自环bug：相同文本创建SUPPORTS而非合并 | B |
| P0-4 | Claim→Proposition映射不是1:1 | B |
| P0-5 | Extraction噪声高（图表标题、纯定义） | C |
| P0-6 | 缺少硬门禁验证（覆盖率、自环率） | D |

## 实施策略

采用**Evolution基础设施优先 + 增量修复**策略：
- 优先修复推理逻辑和基础设施（Phase 1-2）
- 再修复输入质量（Phase 3）
- 最后添加验证门禁（Phase 4）

---

## Phase 1: 模式稳定性 + 自适应阈值

### 目标
修复embedding降级路径和模式感知阈值

### 文件修改

#### 1.1 backend/app/similarity/service.py
**修改**: 显式标记embedding降级

```python
# 当embedding失败时
if embedding_failed:
    mode = "lexical"
    degradation_reason = f"Embedding failed: {error_code}"
    logger.warning(f"Degraded to lexical mode: {degradation_reason}")
else:
    mode = "embedding"
    degradation_reason = None

# 存储模式元数据
similarity_metadata = {
    "mode": mode,
    "degradation_reason": degradation_reason,
    "threshold_used": get_threshold_by_mode(mode)
}
```

#### 1.2 backend/app/evolution/service.py
**修改**: 实现模式感知阈值

```python
def get_threshold_by_mode(mode: str) -> float:
    """返回模式对应的阈值"""
    thresholds = {
        "embedding": 0.85,  # embedding模式高阈值
        "lexical": 0.70     # lexical模式低阈值
    }
    return thresholds.get(mode, 0.75)  # 默认中值

def filter_candidate_pairs(similar_edges, mode):
    """根据模式动态过滤候选对"""
    threshold = get_threshold_by_mode(mode)
    return [e for e in similar_edges if e.similarity >= threshold]
```

#### 1.3 backend/app/graph/neo4j_client.py
**修改**: 存储相似度模式到SIMILAR_CLAIM边

```python
CREATE (c1)-[:SIMILAR_CLAIM {
    score: $score,
    mode: $mode,  # 新增
    degradation_reason: $degradation_reason  # 新增
}]->(c2)
```

### 测试

#### 1.4 backend/tests/test_embedding_degradation.py
**新增**: 测试embedding降级路径

```python
def test_embedding_502_degradation():
    """测试502错误时降级为lexical模式"""
    # Mock embedding service返回502
    # 验证mode='lexical'
    # 验证degradation_reason包含'502'
    # 验证使用0.70阈值

def test_mode_aware_thresholds():
    """测试模式感知阈值"""
    # 验证embedding模式使用0.85
    # 验证lexical模式使用0.70
```

### 验收标准
- ✅ Embedding 502错误时，系统显式标记mode='lexical'
- ✅ 相似度边包含mode和degradation_reason属性
- ✅ embedding模式使用阈值0.85
- ✅ lexical模式使用阈值0.70
- ✅ 测试覆盖降级路径

### 提交
```bash
git add backend/app/similarity/ backend/app/evolution/ backend/tests/
git commit -m "feat(P0-1,P0-2): add mode-aware thresholds and embedding degradation tracking"
```

---

## Phase 2: 身份正确性

### 目标
修复Claim→Proposition映射和自环bug

### 文件修改

#### 2.1 backend/app/graph/neo4j_client.py
**修改**: 强制1:1映射

```python
def enforce_claim_proposition_mapping():
    """确保每个Claim只映射到1个Proposition"""
    # 查询multi-map claims
    query = """
    MATCH (c:Claim)-[:MAPS_TO]->(p:Proposition)
    WITH c, count(p) AS prop_count
    WHERE prop_count > 1
    RETURN c.claim_id, collect(p.prop_id) AS props
    """

    # 对于每个multi-map claim，合并或选择主Proposition
    # 记录修复操作到audit log
```

#### 2.2 backend/app/evolution/inference.py
**修改**: 自环处理为合并

```python
def infer_relation(prop_a, prop_b, similarity_score):
    """推理关系类型"""
    # 检测文本身份
    if normalize_text(prop_a.text) == normalize_text(prop_b.text):
        if prop_a.prop_id != prop_b.prop_id:
            # 不同ID但文本相同：触发合并
            return ("MERGE", {"reason": "text_identity"})
        else:
            # 同一个proposition：跳过
            return (None, {})

    # 正常推理逻辑
    if similarity_score >= 0.9:
        return infer_support_or_refine(prop_a, prop_b)
    elif contradiction_detected:
        return ("CONTRADICTS", {...})
```

#### 2.3 backend/app/evolution/service.py
**修改**: 添加合并队列

```python
def process_merge_queue(merge_candidates):
    """处理需要合并的Proposition对"""
    for (prop_a, prop_b) in merge_candidates:
        # 选择保留的proposition（更早创建或更多引用）
        primary, secondary = select_primary(prop_a, prop_b)

        # 迁移所有MAPS_TO关系到primary
        migrate_relationships(secondary, primary)

        # 删除secondary节点
        delete_node(secondary)

        logger.info(f"Merged {secondary.prop_id} into {primary.prop_id}")
```

### 测试

#### 2.4 backend/tests/test_claim_mapping_integrity.py
**新增**: 测试1:1映射

```python
def test_no_multi_map_claims():
    """验证无multi-map claims"""
    query = """
    MATCH (c:Claim)-[:MAPS_TO]->(p:Proposition)
    WITH c, count(p) AS prop_count
    WHERE prop_count > 1
    RETURN count(c) AS multi_map_count
    """
    result = run_query(query)
    assert result['multi_map_count'] == 0
```

#### 2.5 backend/tests/test_evolution_self_loop.py
**新增**: 测试自环修复

```python
def test_no_self_loop_supports():
    """验证无自环SUPPORTS关系"""
    query = """
    MATCH (p1:Proposition)-[r:SUPPORTS]->(p2:Proposition)
    WHERE p1.prop_id = p2.prop_id
    RETURN count(r) AS self_loop_count
    """
    result = run_query(query)
    assert result['self_loop_count'] == 0

def test_identical_text_triggers_merge():
    """验证相同文本触发合并而非SUPPORTS"""
    # 创建两个文本相同但ID不同的propositions
    # 运行evolution推理
    # 验证只剩一个proposition
    # 验证没有创建SUPPORTS关系
```

### 验收标准
- ✅ 零multi-map claims（每个Claim恰好映射到1个Proposition）
- ✅ 零自环SUPPORTS关系
- ✅ 文本身份触发合并路径
- ✅ 合并操作记录audit log

### 提交
```bash
git add backend/app/graph/ backend/app/evolution/ backend/tests/
git commit -m "feat(P0-3,P0-4): enforce 1:1 claim mapping and fix self-loop bug"
```

---

## Phase 3: Extraction噪声过滤

### 目标
减少图表标题和纯定义噪声

### 文件修改

#### 3.1 backend/app/llm/logic_claims_v2.py
**修改**: 添加启发式过滤器

```python
def is_figure_caption(text: str) -> bool:
    """检测图表标题"""
    patterns = [
        r"^Figure \d+",
        r"^Fig\. \d+",
        r"^Table \d+",
        r"shows? that",
        r"illustrates? that"
    ]
    return any(re.match(p, text, re.IGNORECASE) for p in patterns)

def is_pure_definition(text: str) -> bool:
    """检测纯定义句（缺乏贡献信号）"""
    # 定义特征：含"is defined as", "refers to"
    # 但不含贡献词：found, demonstrated, showed, proposes
    definition_markers = ["is defined as", "refers to", "is called", "means that"]
    contribution_markers = ["found", "demonstrated", "showed", "proposes", "reveals"]

    has_definition = any(m in text.lower() for m in definition_markers)
    has_contribution = any(m in text.lower() for m in contribution_markers)

    return has_definition and not has_contribution

def filter_noise_claims(claims: List[Claim]) -> List[Claim]:
    """过滤噪声claims"""
    filtered = []
    for claim in claims:
        if is_figure_caption(claim.text):
            logger.debug(f"Filtered figure caption: {claim.text[:50]}...")
            continue
        if is_pure_definition(claim.text):
            logger.debug(f"Filtered pure definition: {claim.text[:50]}...")
            continue
        filtered.append(claim)

    logger.info(f"Filtered {len(claims) - len(filtered)}/{len(claims)} noise claims")
    return filtered
```

#### 3.2 backend/app/extraction/orchestrator.py
**修改**: 在orchestration层应用过滤

```python
async def extract_claims(paper_id: str) -> List[Claim]:
    """提取claims并过滤噪声"""
    # 调用LLM提取
    raw_claims = await llm_extract_claims(paper_id)

    # 应用噪声过滤
    clean_claims = filter_noise_claims(raw_claims)

    # 记录过滤统计
    stats = {
        "raw_count": len(raw_claims),
        "clean_count": len(clean_claims),
        "filter_rate": 1 - len(clean_claims) / len(raw_claims)
    }
    logger.info(f"Claim filtering stats: {stats}")

    return clean_claims
```

### 测试

#### 3.3 backend/tests/test_noise_filtering.py
**新增**: 测试噪声过滤

```python
def test_filter_figure_captions():
    """测试图表标题过滤"""
    captions = [
        "Figure 3 shows the temperature distribution",
        "Table 1 illustrates the results"
    ]
    for caption in captions:
        assert is_figure_caption(caption)

def test_filter_pure_definitions():
    """测试纯定义过滤"""
    definitions = [
        "Tensile strength is defined as the maximum stress",
        "The modulus refers to the slope of the curve"
    ]
    for defn in definitions:
        assert is_pure_definition(defn)

def test_keep_contribution_claims():
    """测试保留贡献性claims"""
    contributions = [
        "We found that temperature affects strength",
        "The study demonstrates improved performance"
    ]
    for contrib in contributions:
        assert not is_pure_definition(contrib)
        assert not is_figure_caption(contrib)
```

### 验收标准
- ✅ 图表标题被过滤（正则匹配"Figure X"等）
- ✅ 纯定义句被过滤（含定义词但无贡献词）
- ✅ 贡献性claims被保留
- ✅ 过滤率可测量（日志记录）

### 提交
```bash
git add backend/app/llm/ backend/app/extraction/ backend/tests/
git commit -m "feat(P0-5): add noise filtering for figure captions and pure definitions"
```

---

## Phase 4: 硬性验证门禁

### 目标
添加质量门禁阻止低质量写入

### 文件修改

#### 4.1 backend/app/evolution/service.py
**修改**: 计算并验证质量指标

```python
def validate_evolution_quality(relations: List[Relation]) -> Tuple[bool, Dict]:
    """验证evolution质量"""
    total_props = count_propositions()

    # 计算覆盖率
    covered_props = count_unique_props_in_relations(relations)
    coverage_ratio = covered_props / total_props if total_props > 0 else 0.0

    # 计算自环率
    self_loops = [r for r in relations if r.from_id == r.to_id]
    self_loop_rate = len(self_loops) / len(relations) if relations else 0.0

    metrics = {
        "coverage_ratio": coverage_ratio,
        "self_loop_rate": self_loop_rate,
        "total_relations": len(relations),
        "covered_propositions": covered_props,
        "total_propositions": total_props
    }

    # 硬门禁验证
    passed = (
        coverage_ratio >= 0.20 and
        self_loop_rate < 0.05
    )

    if not passed:
        logger.error(f"Evolution quality gate FAILED: {metrics}")
    else:
        logger.info(f"Evolution quality gate PASSED: {metrics}")

    return passed, metrics

async def rebuild_evolution():
    """重建evolution关系"""
    # 运行推理
    relations = await infer_all_relations()

    # 验证质量
    passed, metrics = validate_evolution_quality(relations)

    if not passed:
        raise ValueError(
            f"Evolution quality gate failed: "
            f"coverage={metrics['coverage_ratio']:.2%} (need ≥20%), "
            f"self_loop_rate={metrics['self_loop_rate']:.2%} (need <5%)"
        )

    # 写入数据库
    await write_relations(relations)
    return metrics
```

#### 4.2 backend/app/settings.py
**修改**: 添加可配置门禁阈值

```python
# Evolution quality gates
evolution_min_coverage: float = Field(default=0.20, ge=0.0, le=1.0)
evolution_max_self_loop_rate: float = Field(default=0.05, ge=0.0, le=1.0)
```

### 测试

#### 4.3 backend/tests/test_evolution_quality_gates.py
**新增**: 测试质量门禁

```python
def test_quality_gate_blocks_low_coverage():
    """测试低覆盖率被阻止"""
    # 创建覆盖率<20%的relations
    relations = create_low_coverage_relations()
    passed, metrics = validate_evolution_quality(relations)

    assert not passed
    assert metrics['coverage_ratio'] < 0.20

def test_quality_gate_blocks_high_self_loop():
    """测试高自环率被阻止"""
    # 创建自环率≥5%的relations
    relations = create_high_self_loop_relations()
    passed, metrics = validate_evolution_quality(relations)

    assert not passed
    assert metrics['self_loop_rate'] >= 0.05

def test_quality_gate_passes_good_relations():
    """测试高质量relations通过"""
    # 创建覆盖率≥20%且自环率<5%的relations
    relations = create_good_relations()
    passed, metrics = validate_evolution_quality(relations)

    assert passed
    assert metrics['coverage_ratio'] >= 0.20
    assert metrics['self_loop_rate'] < 0.05
```

### 验收标准
- ✅ 覆盖率<20%时阻止写入
- ✅ 自环率≥5%时阻止写入
- ✅ 质量指标记录到日志
- ✅ 阈值可通过settings配置

### 提交
```bash
git add backend/app/evolution/ backend/app/settings.py backend/tests/
git commit -m "feat(P0-6): add hard quality gates for evolution coverage and self-loop rate"
```

---

## 完整E2E验证

### 验证脚本
创建 `backend/verify_evolution_p0_fix.py`：

```python
"""验证P0修复后Evolution功能是否正常"""

def verify_phase1_mode_stability():
    """验证Phase 1：模式稳定性"""
    # 检查相似度边是否包含mode属性
    # 验证阈值是否根据模式调整

def verify_phase2_identity_correctness():
    """验证Phase 2：身份正确性"""
    # 验证无multi-map claims
    # 验证无自环SUPPORTS

def verify_phase3_noise_reduction():
    """验证Phase 3：噪声减少"""
    # 统计过滤率
    # 抽样验证图表标题和纯定义被过滤

def verify_phase4_quality_gates():
    """验证Phase 4：质量门禁"""
    # 验证覆盖率≥20%
    # 验证自环率<5%
    # 验证CONTRADICTS和REFINES存在

def run_full_verification():
    """运行完整验证"""
    verify_phase1_mode_stability()
    verify_phase2_identity_correctness()
    verify_phase3_noise_reduction()
    verify_phase4_quality_gates()

    print("[OK] All P0 fixes verified!")
```

### 20论文重测
```bash
# 清空数据库
python backend/scripts/clear_database.py

# 重新ingest 20论文
# 使用相同的C:\Users\D0n9\Desktop\hzy_paper\selected_20_md_with_images

# 运行evolution rebuild
curl -X POST http://localhost:8000/tasks/rebuild_similarity
curl -X POST http://localhost:8000/evolution/rebuild

# 运行验证
python backend/verify_evolution_p0_fix.py

# 对比指标
python backend/quality_eval_20papers.py
```

### 期望结果
| 指标 | 修复前 | 修复后目标 |
|------|--------|-----------|
| Evolution覆盖率 | 0.56% | ≥20% |
| SUPPORTS数量 | 8 | >200 |
| 自环SUPPORTS | 8 (100%) | 0 (<5%) |
| CONTRADICTS | 0 | >0 |
| REFINES | 0 | >0 |
| Multi-map claims | ? | 0 |
| Claim过滤率 | 0% | 5-10% |

---

## 回滚计划

### Phase级回滚
每个Phase都是独立commit，可单独回滚：
```bash
# 回滚Phase 4
git revert <phase4_commit_sha>

# 回滚Phase 3
git revert <phase3_commit_sha>

# 回滚Phase 2
git revert <phase2_commit_sha>

# 回滚Phase 1
git revert <phase1_commit_sha>
```

### 完全回滚
```bash
git checkout feature/p1-quality-optimization
git branch -D feature/evolution-p0-fix
```

---

## 时间估算

| Phase | 开发 | 测试 | 总计 |
|-------|------|------|------|
| Phase 1 | 2h | 1h | 3h |
| Phase 2 | 3h | 1.5h | 4.5h |
| Phase 3 | 1.5h | 1h | 2.5h |
| Phase 4 | 1h | 0.5h | 1.5h |
| E2E验证 | - | 2h | 2h |
| **总计** | **7.5h** | **6h** | **13.5h** |

---

## 成功标准

**P0修复完成标志**：
1. ✅ 20论文E2E测试通过所有验收标准
2. ✅ Evolution覆盖率≥20%
3. ✅ 自环率<5%
4. ✅ 存在CONTRADICTS和REFINES关系
5. ✅ 研究脉络可视化功能正常
6. ✅ 问题发现功能可用

**部署条件**：
- 所有Phase测试通过
- E2E验证通过
- 质量指标达标
- 回滚计划已测试

---

**计划创建**: 2026-02-15
**预计完成**: Phase 1-2 今天完成，Phase 3-4 明天完成
**负责人**: Claude + Codex 协同实施
