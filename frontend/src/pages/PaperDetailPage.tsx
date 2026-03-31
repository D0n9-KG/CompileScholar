import { useEffect, useMemo, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'

import { apiGet } from '../api'
import MarkdownView from '../components/MarkdownView'
import SignalGraph, { type SignalGraphEdge, type SignalGraphNode } from '../components/SignalGraph'
import { useI18n, type UILocale } from '../i18n'
import type { CitationAct, EvidenceAnchor, L2CompletenessAudit, MoveRelation, PaperLogicTrace, ResearchMove } from '../types/paperLogicTrace'
import { mentionTokens, normalizeText } from '../types/paperLogicTrace'

type TraceTab = 'moves' | 'relations' | 'anchors' | 'citations' | 'original'
type TraceState = { paperId: string; trace: PaperLogicTrace | null; error: string }
type ContentState = { paperId: string; content: string; loaded: boolean }
type GraphMetaSection = {
  label: string
  value?: string
  values?: string[]
  chips?: string[]
}
type GraphMeta = {
  title: string
  typeLabel: string
  detail?: string
  badge?: string
  facts?: Array<{ label: string; value: string }>
  sections?: GraphMetaSection[]
}

function text(locale: UILocale) {
  return locale === 'zh-CN'
    ? {
        loading: '正在加载 PaperLogicTrace...',
        missingPaperId: '缺少论文 ID。',
        graph: '轨迹工作台',
        nodeDetail: '节点详情',
        selectNode: '点击图中的节点查看详情。',
        quality: '质量等级',
        completeness: '完整度',
        community: '社区可用',
        l3: 'L3 可用',
        l4: 'L4 可用',
        missingRoles: '缺失角色',
        sparseSlots: '稀疏槽位',
        qualityFlags: '质量标记',
        auditStatus: '审计状态',
        schema: '轨迹版本',
        paperType: '论文类型',
        builtAt: '构建时间',
        venue: '发表来源',
        year: '年份',
        role: '角色',
        actType: '动作类型',
        confidence: '置信度',
        focusGraph: '在图中定位',
        summaryLabel: '摘要',
        evidenceAnchorsLabel: '证据锚点',
        signalTagsLabel: '关键信号',
        quoteLabel: '证据摘录',
        sourceSpansLabel: '来源位置',
        emptyContent: '暂时没有原文内容。',
        noRelations: '没有动作关系。',
        noCitations: '没有引用行为。',
        noiseMoves: '噪声动作',
        tabs: {
          moves: '研究动作',
          relations: '动作关系',
          anchors: '证据锚点',
          citations: '引用行为',
          original: '原文内容',
        },
      }
    : {
        loading: 'Loading PaperLogicTrace...',
        missingPaperId: 'Missing paper id.',
        graph: 'Trace Workspace',
        nodeDetail: 'Node Detail',
        selectNode: 'Select a node to inspect it.',
        quality: 'Quality Tier',
        completeness: 'Completeness',
        community: 'Community Ready',
        l3: 'Ready for L3',
        l4: 'Ready for L4',
        missingRoles: 'Missing Roles',
        sparseSlots: 'Sparse Slots',
        qualityFlags: 'Quality Flags',
        auditStatus: 'Audit Status',
        schema: 'Trace Version',
        paperType: 'Paper Type',
        builtAt: 'Built At',
        venue: 'Venue',
        year: 'Year',
        role: 'Role',
        actType: 'Act Type',
        confidence: 'Confidence',
        focusGraph: 'Focus in Graph',
        summaryLabel: 'Summary',
        evidenceAnchorsLabel: 'Evidence Anchors',
        signalTagsLabel: 'Signals',
        quoteLabel: 'Quoted Evidence',
        sourceSpansLabel: 'Source Spans',
        emptyContent: 'Original content unavailable.',
        noRelations: 'No move relations.',
        noCitations: 'No citation acts.',
        noiseMoves: 'Noise Moves',
        tabs: {
          moves: 'Research Moves',
          relations: 'Move Relations',
          anchors: 'Evidence Anchors',
          citations: 'Citation Acts',
          original: 'Source Content',
        },
      }
}

function labelMap(locale: UILocale, kind: 'role' | 'act' | 'paperType', value: string | null | undefined) {
  const key = normalizeText(value)
  const zh = {
    role: {
      problem: '问题',
      background: '背景',
      hypothesis: '假设',
      method: '方法',
      experiment: '实验',
      result: '结果',
      interpretation: '解释',
      limitation: '局限',
      future_work: '未来工作',
    },
    act: {
      identify_gap: '识别缺口',
      define_task: '定义任务',
      formulate_hypothesis: '提出假设',
      propose_method: '提出方法',
      adapt_method: '改造方法',
      build_resource: '构建资源',
      set_condition: '设定条件',
      run_experiment: '执行实验',
      measure_outcome: '测量结果',
      compare_baseline: '对比基线',
      report_effect: '报告效果',
      explain_mechanism: '解释机制',
      diagnose_failure: '诊断失败',
      state_limitation: '说明局限',
      suggest_extension: '提出扩展',
    },
    paperType: {
      empirical: '经验论文',
      theoretical: '理论论文',
      review: '综述论文',
      software: '软件论文',
      benchmark: '基准论文',
      case_study: '案例研究',
      unknown: '未知类型',
    },
  }
  const en = {
    role: {
      problem: 'Problem',
      background: 'Background',
      hypothesis: 'Hypothesis',
      method: 'Method',
      experiment: 'Experiment',
      result: 'Result',
      interpretation: 'Interpretation',
      limitation: 'Limitation',
      future_work: 'Future Work',
    },
    act: {
      identify_gap: 'Identify Gap',
      define_task: 'Define Task',
      formulate_hypothesis: 'Formulate Hypothesis',
      propose_method: 'Propose Method',
      adapt_method: 'Adapt Method',
      build_resource: 'Build Resource',
      set_condition: 'Set Condition',
      run_experiment: 'Run Experiment',
      measure_outcome: 'Measure Outcome',
      compare_baseline: 'Compare Baseline',
      report_effect: 'Report Effect',
      explain_mechanism: 'Explain Mechanism',
      diagnose_failure: 'Diagnose Failure',
      state_limitation: 'State Limitation',
      suggest_extension: 'Suggest Extension',
    },
    paperType: {
      empirical: 'Empirical',
      theoretical: 'Theoretical',
      review: 'Review',
      software: 'Software',
      benchmark: 'Benchmark',
      case_study: 'Case Study',
      unknown: 'Unknown',
    },
  }
  const dict = locale === 'zh-CN' ? zh : en
  return (dict[kind] as Record<string, string>)[key] ?? (key || (kind === 'paperType' ? dict.paperType.unknown : 'Unknown'))
}

function qualityTierLabel(locale: UILocale, value: string | null | undefined) {
  const key = normalizeText(value)
  const zh: Record<string, string> = {
    green: '高可信',
    yellow: '可用待补强',
    red: '受限',
  }
  const en: Record<string, string> = {
    green: 'High Confidence',
    yellow: 'Usable with Gaps',
    red: 'Limited',
  }
  const dict = locale === 'zh-CN' ? zh : en
  return dict[key] ?? (key || '-')
}

function auditStatusLabel(locale: UILocale, value: string | null | undefined) {
  const key = normalizeText(value)
  const zh: Record<string, string> = {
    not_needed: '无需补审',
    pending: '待补审',
    completed: '已补审',
  }
  const en: Record<string, string> = {
    not_needed: 'No Further Audit Needed',
    pending: 'Pending Audit',
    completed: 'Audited',
  }
  const dict = locale === 'zh-CN' ? zh : en
  return dict[key] ?? (key || '-')
}

function relationTypeLabel(locale: UILocale, value: string | null | undefined) {
  const key = normalizeText(value)
  const zh: Record<string, string> = {
    motivates: '提供动机',
    addresses: '回应问题',
    implements: '展开实现',
    evaluates: '进行评估',
    yields: '产出结果',
    explains: '解释机制',
    limits: '指出限制',
    extends: '提出扩展',
  }
  const en: Record<string, string> = {
    motivates: 'Motivates',
    addresses: 'Addresses',
    implements: 'Implements',
    evaluates: 'Evaluates',
    yields: 'Yields',
    explains: 'Explains',
    limits: 'Limits',
    extends: 'Extends',
  }
  const dict = locale === 'zh-CN' ? zh : en
  return dict[key] ?? (key || '-')
}

const fmtPct = (value: number | null | undefined) => (Number.isFinite(Number(value)) ? `${Math.round(Number(value) * 100)}%` : '-')
const fmtReady = (ready: boolean | null | undefined, locale: UILocale) => (ready ? (locale === 'zh-CN' ? '已就绪' : 'Ready') : locale === 'zh-CN' ? '暂缓' : 'Hold')
const fmtList = (values: string[] | null | undefined) => {
  const list = (values ?? []).map((item) => normalizeText(item)).filter(Boolean)
  return list.length ? list.join(', ') : '-'
}

function moveTagSummary(move: ResearchMove) {
  return Array.from(
    new Set([
      ...mentionTokens(move.methods),
      ...mentionTokens(move.research_objects),
      ...mentionTokens(move.metrics),
      ...mentionTokens(move.comparators),
      ...mentionTokens(move.conditions),
    ]),
  ).slice(0, 8)
}

function moveLabel(move: ResearchMove, locale: UILocale) {
  return `${move.sequence_no}. ${labelMap(locale, 'role', move.role)}`
}

function locator(anchor: EvidenceAnchor, locale: UILocale) {
  const start = Number(anchor.locator?.start_line)
  const end = Number(anchor.locator?.end_line)
  if (Number.isFinite(start) && Number.isFinite(end) && end >= start) return locale === 'zh-CN' ? `第 ${start}-${end} 行` : `Lines ${start}-${end}`
  if (Number.isFinite(start)) return locale === 'zh-CN' ? `第 ${start} 行` : `Line ${start}`
  return normalizeText(anchor.source_ref) || '-'
}

function formatBuiltAt(value: string | null | undefined, locale: UILocale) {
  const raw = normalizeText(value)
  if (!raw) return '-'
  const timestamp = new Date(raw)
  if (Number.isNaN(timestamp.getTime())) return raw
  return timestamp.toLocaleString(locale === 'zh-CN' ? 'zh-CN' : 'en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function graphData(trace: PaperLogicTrace, locale: UILocale) {
  const nodes = new Map<string, SignalGraphNode>()
  const edges = new Map<string, SignalGraphEdge>()
  const meta = new Map<string, GraphMeta>()
  const summaryLabel = locale === 'zh-CN' ? '摘要' : 'Summary'
  const evidenceAnchorsLabel = locale === 'zh-CN' ? '证据锚点' : 'Evidence Anchors'
  const signalTagsLabel = locale === 'zh-CN' ? '关键信号' : 'Signals'
  const quoteLabel = locale === 'zh-CN' ? '证据摘录' : 'Quoted Evidence'
  const sourceSpansLabel = locale === 'zh-CN' ? '来源位置' : 'Source Spans'
  const rootId = 'paper:root'
  const anchors = new Map(trace.canonical_core.evidence_anchors.map((anchor) => [anchor.anchor_id, anchor]))

  const put = (node: SignalGraphNode, value: GraphMeta) => {
    nodes.set(node.id, node)
    meta.set(node.id, value)
  }

  put(
    { id: rootId, label: normalizeText(trace.paper_metadata.title) || trace.paper_metadata.paper_id, kind: 'root', weight: 1 },
      {
        title: normalizeText(trace.paper_metadata.title) || trace.paper_metadata.paper_id,
        typeLabel: locale === 'zh-CN' ? '论文轨迹' : 'Paper Trace',
        detail: [normalizeText(trace.paper_metadata.paper_id), normalizeText(trace.paper_metadata.canonical_doi), normalizeText(trace.paper_metadata.venue)].filter(Boolean).join(' | '),
        badge: qualityTierLabel(locale, trace.quality?.quality_tier),
      },
    )

  for (const move of trace.canonical_core.moves) {
    const moveId = `move:${move.move_id}`
    const clusterId = `cluster:${move.move_id}`
    const moveTags = moveTagSummary(move)
    const anchorPreview = (move.anchor_ids ?? [])
      .map((anchorId) => anchors.get(anchorId))
      .filter((anchor): anchor is EvidenceAnchor => Boolean(anchor))
      .slice(0, 4)
      .map((anchor) => {
        const sectionPath = (anchor.section_path ?? []).map((part) => normalizeText(part)).filter(Boolean).join(' > ')
        const loc = locator(anchor, locale)
        if (sectionPath && sectionPath !== '-') return `${loc} · ${sectionPath}`
        return loc
      })
    put(
      { id: clusterId, label: '', kind: 'cluster', weight: 0.4 },
      { title: moveLabel(move, locale), typeLabel: locale === 'zh-CN' ? '动作簇' : 'Move Cluster' },
    )
    put(
      { id: moveId, label: moveLabel(move, locale), kind: 'move', weight: 0.76 },
      {
        title: moveLabel(move, locale),
        typeLabel: locale === 'zh-CN' ? '研究动作' : 'Research Move',
        detail: normalizeText(move.summary),
        badge: labelMap(locale, 'act', move.act_type),
        facts: [
          { label: locale === 'zh-CN' ? '角色' : 'Role', value: labelMap(locale, 'role', move.role) },
          { label: locale === 'zh-CN' ? '动作类型' : 'Act Type', value: labelMap(locale, 'act', move.act_type) },
          { label: locale === 'zh-CN' ? '置信度' : 'Confidence', value: fmtPct(move.confidence) },
          { label: evidenceAnchorsLabel, value: String((move.anchor_ids ?? []).length) },
        ],
        sections: [
          { label: summaryLabel, value: normalizeText(move.summary) },
          ...(anchorPreview.length ? [{ label: sourceSpansLabel, values: anchorPreview }] : []),
          ...(moveTags.length ? [{ label: signalTagsLabel, chips: moveTags }] : []),
        ],
      },
    )

    edges.set(`${rootId}->${moveId}`, { id: `${rootId}->${moveId}`, source: rootId, target: moveId, kind: 'contains', weight: 0.7 })

    for (const anchorId of move.anchor_ids ?? []) {
      const anchor = anchors.get(anchorId)
      if (!anchor) continue
      const anchorNodeId = `anchor:${anchor.anchor_id}`
      if (!nodes.has(anchorNodeId)) {
        put(
          { id: anchorNodeId, label: anchor.anchor_id, kind: 'anchor', weight: anchor.weak ? 0.38 : 0.5 },
          {
            title: anchor.anchor_id,
            typeLabel: locale === 'zh-CN' ? '证据锚点' : 'Evidence Anchor',
            detail: normalizeText(anchor.quote),
            badge: locator(anchor, locale),
            facts: [
              { label: locale === 'zh-CN' ? '支持类型' : 'Support Type', value: normalizeText(anchor.support_type) || '-' },
              {
                label: locale === 'zh-CN' ? '章节' : 'Section',
                value: (anchor.section_path ?? []).map((part) => normalizeText(part)).filter(Boolean).join(' > ') || '-',
              },
            ],
            sections: [{ label: quoteLabel, value: normalizeText(anchor.quote) }],
          },
        )
      }
      edges.set(`${moveId}->${anchorNodeId}`, {
        id: `${moveId}->${anchorNodeId}`,
        source: moveId,
        target: anchorNodeId,
        kind: 'evidenced_by',
        weight: anchor.weak ? 0.4 : 0.58,
      })
    }
  }

  for (const relation of trace.canonical_core.move_relations) {
    const source = `move:${relation.source_move_id}`
    const target = `move:${relation.target_move_id}`
    if (!nodes.has(source) || !nodes.has(target)) continue
    edges.set(relation.relation_id, {
      id: relation.relation_id,
      source,
      target,
      kind: relation.relation_type,
      weight: relation.confidence ?? 0.55,
    })
  }

  for (const citation of trace.canonical_core.citation_acts) {
    const moveId = citation.source_move_id ? `move:${citation.source_move_id}` : ''
    if (!moveId || !nodes.has(moveId)) continue
    const citationId = `citation:${citation.citation_act_id}`
    put(
      { id: citationId, label: citation.citation_act_id, kind: 'citation', weight: 0.42 },
      {
        title: citation.citation_act_id,
        typeLabel: locale === 'zh-CN' ? '引用行为' : 'Citation Act',
        detail: [normalizeText(citation.target_paper_id), normalizeText(citation.purpose), normalizeText(citation.semantic_signal)].filter(Boolean).join(' | '),
        badge: normalizeText(citation.target_scope),
        facts: [{ label: locale === 'zh-CN' ? '置信度' : 'Confidence', value: fmtPct(citation.confidence) }],
      },
    )
    edges.set(`${moveId}->${citationId}`, {
      id: `${moveId}->${citationId}`,
      source: moveId,
      target: citationId,
      kind: 'cites',
      weight: citation.confidence ?? 0.45,
    })
  }

  return { nodes: Array.from(nodes.values()), edges: Array.from(edges.values()), meta }
}

function auditCards(audit: L2CompletenessAudit | undefined, labels: ReturnType<typeof text>) {
  if (!audit) return []
  return [
    { label: labels.missingRoles, value: fmtList(audit.missing_expected_roles), testId: 'trace-missing-roles' },
    { label: labels.sparseSlots, value: fmtList(audit.sparse_expected_slot_fields), testId: 'trace-sparse-slots' },
    { label: labels.noiseMoves, value: String((audit.noise_move_ids ?? []).length) },
  ]
}

export default function PaperDetailPage() {
  const { locale } = useI18n()
  const copy = text(locale)
  const { paperId = '' } = useParams<{ paperId: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const [traceState, setTraceState] = useState<TraceState>({ paperId: '', trace: null, error: '' })
  const [contentState, setContentState] = useState<ContentState>({ paperId: '', content: '', loaded: false })
  const [selectedNodeId, setSelectedNodeId] = useState('')

  const tab = useMemo<TraceTab>(() => {
    const raw = normalizeText(searchParams.get('tab'))
    return (['moves', 'relations', 'anchors', 'citations', 'original'] as TraceTab[]).find((item) => item === raw) ?? 'moves'
  }, [searchParams])

  useEffect(() => {
    if (!paperId) return
    let cancelled = false
    apiGet<PaperLogicTrace>(`/papers/${encodeURIComponent(paperId)}/logic-trace`)
      .then((trace) => {
        if (cancelled) return
        setTraceState({ paperId, trace, error: '' })
        setSelectedNodeId('')
      })
      .catch((cause: unknown) => {
        if (cancelled) return
        setTraceState({
          paperId,
          trace: null,
          error: String((cause as { message?: unknown } | null)?.message ?? cause),
        })
      })
    return () => {
      cancelled = true
    }
  }, [paperId])

  useEffect(() => {
    if (!paperId || tab !== 'original' || (contentState.paperId === paperId && contentState.loaded)) return
    let cancelled = false
    apiGet<string>(`/papers/${encodeURIComponent(paperId)}/content`)
      .then((content) => {
        if (!cancelled) setContentState({ paperId, content: String(content ?? ''), loaded: true })
      })
      .catch(() => {
        if (!cancelled) setContentState({ paperId, content: '', loaded: true })
      })
    return () => {
      cancelled = true
    }
  }, [contentState.loaded, contentState.paperId, paperId, tab])

  const trace = traceState.paperId === paperId ? traceState.trace : null
  const content = contentState.paperId === paperId ? contentState.content : ''
  const quality = trace?.quality
  const audit = quality?.l2_completeness_audit
  const pageTitle = trace?.paper_metadata.title ?? (locale === 'zh-CN' ? '论文轨迹' : 'Paper Trace')
  const graph = useMemo(
    () => (trace ? graphData(trace, locale) : { nodes: [] as SignalGraphNode[], edges: [] as SignalGraphEdge[], meta: new Map<string, GraphMeta>() }),
    [locale, trace],
  )
  const selectedMeta = graph.meta.get(selectedNodeId) ?? graph.meta.get('paper:root') ?? null
  const graphHeight = useMemo(() => {
    if (!trace) return 420
    const moveCount = trace.canonical_core.moves.length
    const anchorCount = trace.canonical_core.evidence_anchors.length
    return Math.min(720, Math.max(540, 330 + moveCount * 12 + Math.min(anchorCount, 28) * 2))
  }, [trace])

  const setTab = (nextTab: TraceTab) => {
    const next = new URLSearchParams(searchParams)
    next.set('tab', nextTab)
    setSearchParams(next, { replace: true })
  }

  if (!paperId) return <div className="page">{copy.missingPaperId}</div>

  return (
    <div className="page paperTracePage">
      <div className="pageHeader paperTraceHeader">
        <div>
          <h2 className="pageTitle">{pageTitle}</h2>
          <div className="pageSubtitle">{trace?.paper_metadata.paper_id ?? paperId}</div>
        </div>
        {trace ? (
          <div className="pageActions paperTraceHeaderMeta">
            <span className="pill">
              <span className="kicker">{copy.schema}</span> {trace.schema_version}
            </span>
            <span className="pill">
              <span className="kicker">{copy.paperType}</span> {labelMap(locale, 'paperType', trace.paper_metadata.paper_type)}
            </span>
            <span className="pill">
              <span className="kicker">{copy.venue}</span> {normalizeText(trace.paper_metadata.venue) || '-'}
            </span>
          </div>
        ) : null}
      </div>

      {traceState.error ? <div className="errorBox">{traceState.error}</div> : null}
      {!trace ? (
        <div className="panel">
          <div className="panelBody">{copy.loading}</div>
        </div>
      ) : (
        <>
          <section className="panel">
            <div className="panelHeader">
              <div className="panelTitle">{locale === 'zh-CN' ? 'L2 完整性体检' : 'L2 Completeness Audit'}</div>
            </div>
            <div className="panelBody">
              <div className="kgStatGrid">
                <div className="kgStatCard">
                  <div className="kgStatLabel">{copy.quality}</div>
                  <div className="kgStatValue" data-testid="trace-quality-tier">{qualityTierLabel(locale, quality?.quality_tier)}</div>
                </div>
                <div className="kgStatCard">
                  <div className="kgStatLabel">{copy.completeness}</div>
                  <div className="kgStatValue" data-testid="trace-completeness-score">{fmtPct(audit?.completeness_score)}</div>
                </div>
                <div className="kgStatCard">
                  <div className="kgStatLabel">{copy.community}</div>
                  <div className="kgStatValue">{fmtReady(audit?.ready_for_community, locale)}</div>
                </div>
                <div className="kgStatCard">
                  <div className="kgStatLabel">{copy.l3}</div>
                  <div className="kgStatValue" data-testid="trace-ready-l3">{fmtReady(audit?.ready_for_l3, locale)}</div>
                </div>
                <div className="kgStatCard">
                  <div className="kgStatLabel">{copy.l4}</div>
                  <div className="kgStatValue" data-testid="trace-ready-l4">{fmtReady(audit?.ready_for_l4, locale)}</div>
                </div>
              </div>
              <div className="list" style={{ marginTop: 12 }}>
                {auditCards(audit, copy).map((card) => (
                  <div key={card.label} className="itemCard">
                    <div className="itemTitle">{card.label}</div>
                    <div className="itemBody" data-testid={card.testId}>{card.value}</div>
                  </div>
                ))}
                <div className="itemCard">
                  <div className="itemTitle">{copy.qualityFlags}</div>
                  <div className="itemBody">{fmtList(quality?.quality_flags)}</div>
                </div>
                <div className="itemCard">
                  <div className="itemTitle">{copy.auditStatus}</div>
                  <div className="itemBody">{auditStatusLabel(locale, quality?.audit_status)}</div>
                </div>
                <div className="itemCard">
                  <div className="itemTitle">{copy.builtAt}</div>
                  <div className="itemBody">{formatBuiltAt(trace.built_at, locale)}</div>
                </div>
                <div className="itemCard">
                  <div className="itemTitle">{copy.year}</div>
                  <div className="itemBody">{trace.paper_metadata.year ?? '-'}</div>
                </div>
              </div>
            </div>
          </section>

          <div className="row paperTraceTabRow">
            {(Object.keys(copy.tabs) as TraceTab[]).map((traceTab) => (
              <button key={traceTab} className={`chip ${tab === traceTab ? 'chipActive' : ''}`} type="button" onClick={() => setTab(traceTab)}>
                {copy.tabs[traceTab]}
              </button>
            ))}
          </div>

          <section className="panel paperTraceWorkbench">
            <div className="panelHeader">
              <div className="split">
                <div className="panelTitle">{copy.graph}</div>
                <div className="metaLine">
                  {copy.paperType}: {labelMap(locale, 'paperType', trace.paper_metadata.paper_type)} | {copy.builtAt}:{' '}
                  {formatBuiltAt(trace.built_at, locale)}
                </div>
              </div>
            </div>
            <div className="panelBody">
              <div className="paperTraceWorkbenchGrid">
                <div className="paperTraceGraphPane">
                  <div className="paperTraceGraphMeta">
                    <div className="paperTraceGraphStats">
                      <span className="paperTraceGraphStat">
                        {trace.canonical_core.moves.length} {locale === 'zh-CN' ? '动作' : 'Moves'}
                      </span>
                      <span className="paperTraceGraphStat">
                        {trace.canonical_core.evidence_anchors.length} {locale === 'zh-CN' ? '锚点' : 'Anchors'}
                      </span>
                      <span className="paperTraceGraphStat">
                        {trace.canonical_core.move_relations.length} {locale === 'zh-CN' ? '关系' : 'Relations'}
                      </span>
                    </div>
                    <div className="paperTraceGraphHint">
                      {locale === 'zh-CN'
                        ? '编号节点表示研究动作，外围小节点表示对应证据锚点。'
                        : 'Numbered nodes are research moves, and the outer dots are supporting evidence anchors.'}
                    </div>
                  </div>
                  <SignalGraph
                    nodes={graph.nodes}
                    edges={graph.edges}
                    selectedId={selectedNodeId || undefined}
                    onSelect={(id) => setSelectedNodeId(id)}
                    height={graphHeight}
                  />
                </div>
                <aside className="paperTraceDetailCard">
                  <div className="itemTitle">{copy.nodeDetail}</div>
                  {selectedMeta ? (
                    <div className="list">
                      <div className="itemCard">
                        <div className="metaLine">{selectedMeta.typeLabel}</div>
                        <div className="itemBody" style={{ fontWeight: 700 }}>
                          {selectedMeta.title}
                        </div>
                        {selectedMeta.badge ? <div className="badge" style={{ marginTop: 10 }}>{selectedMeta.badge}</div> : null}
                      </div>
                      {selectedMeta.facts?.length ? (
                        <div className="itemCard">
                          {selectedMeta.facts.map((fact) => (
                            <div key={`${selectedMeta.title}:${fact.label}`} className="metaLine">
                              <strong>{fact.label}</strong>: {fact.value}
                            </div>
                          ))}
                        </div>
                      ) : null}
                      {selectedMeta.sections?.map((section) => (
                        <div key={`${selectedMeta.title}:${section.label}`} className="itemCard">
                          <div className="itemTitle">{section.label}</div>
                          {section.value ? <div className="itemBody">{section.value}</div> : null}
                          {section.values?.length ? (
                            <div className="stack" style={{ gap: 6 }}>
                              {section.values.map((value, index) => (
                                <div key={`${selectedMeta.title}:${section.label}:${index}`} className="metaLine">
                                  {value}
                                </div>
                              ))}
                            </div>
                          ) : null}
                          {section.chips?.length ? (
                            <div className="row" style={{ gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
                              {section.chips.map((chip) => (
                                <span key={`${selectedMeta.title}:${section.label}:${chip}`} className="chip">
                                  {chip}
                                </span>
                              ))}
                            </div>
                          ) : null}
                        </div>
                      ))}
                      {selectedMeta.detail && !selectedMeta.sections?.some((section) => section.value === selectedMeta.detail) ? (
                        <div className="itemCard">
                          <div className="itemTitle">{copy.summaryLabel}</div>
                          <div className="itemBody">{selectedMeta.detail}</div>
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div className="metaLine">{copy.selectNode}</div>
                  )}
                </aside>
              </div>
            </div>
          </section>

          {tab === 'moves' ? (
            <section className="panel">
              <div className="panelHeader">
                <div className="panelTitle">{copy.tabs.moves}</div>
              </div>
              <div className="panelBody">
                <div className="list">
                  {trace.canonical_core.moves
                    .slice()
                    .sort((a, b) => a.sequence_no - b.sequence_no)
                    .map((move) => (
                      <article key={move.move_id} className="itemCard">
                        <div className="split">
                          <div>
                            <div className="itemTitle">{moveLabel(move, locale)}</div>
                            <div className="metaLine">
                              {copy.role}: {labelMap(locale, 'role', move.role)} | {copy.actType}:{' '}
                              {labelMap(locale, 'act', move.act_type)}
                            </div>
                          </div>
                          <div className="row" style={{ gap: 8 }}>
                            <span className="badge">{fmtPct(move.confidence)}</span>
                            <button className="btn btnSmall" type="button" onClick={() => setSelectedNodeId(`move:${move.move_id}`)}>
                              {copy.focusGraph}
                            </button>
                          </div>
                        </div>
                        <div className="itemBody">{normalizeText(move.summary)}</div>
                        {moveTagSummary(move).length ? (
                          <div className="row" style={{ gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
                            {moveTagSummary(move).map((token) => (
                              <span key={`${move.move_id}:${token}`} className="chip">
                                {token}
                              </span>
                            ))}
                          </div>
                        ) : null}
                      </article>
                    ))}
                </div>
              </div>
            </section>
          ) : null}

          {tab === 'relations' ? (
            <section className="panel">
              <div className="panelHeader">
                <div className="panelTitle">{copy.tabs.relations}</div>
              </div>
              <div className="panelBody">
                <div className="list">
                  {trace.canonical_core.move_relations.length ? (
                    trace.canonical_core.move_relations.map((relation: MoveRelation) => {
                      const sourceMove = trace.canonical_core.moves.find((move) => move.move_id === relation.source_move_id)
                      const targetMove = trace.canonical_core.moves.find((move) => move.move_id === relation.target_move_id)
                      const relationSummary =
                        sourceMove && targetMove
                          ? `${moveLabel(sourceMove, locale)} -> ${moveLabel(targetMove, locale)}`
                          : `${normalizeText(relation.source_move_id)} -> ${normalizeText(relation.target_move_id)}`
                      return (
                      <article key={`${relation.relation_id}:${relation.source_move_id}:${relation.target_move_id}`} className="itemCard">
                        <div className="split">
                          <div className="itemTitle">{relationTypeLabel(locale, relation.relation_type)}</div>
                          <span className="badge">{fmtPct(relation.confidence)}</span>
                        </div>
                        <div className="metaLine">{relationSummary}</div>
                      </article>
                    )})
                  ) : (
                    <div className="metaLine">{copy.noRelations}</div>
                  )}
                </div>
              </div>
            </section>
          ) : null}

          {tab === 'anchors' ? (
            <section className="panel">
              <div className="panelHeader">
                <div className="panelTitle">{copy.tabs.anchors}</div>
              </div>
              <div className="panelBody">
                <div className="list">
                  {trace.canonical_core.evidence_anchors.map((anchor: EvidenceAnchor) => (
                    <article key={anchor.anchor_id} className="itemCard">
                      <div className="split">
                        <div>
                          <div className="itemTitle">{anchor.anchor_id}</div>
                          <div className="metaLine">{locator(anchor, locale)}</div>
                        </div>
                        <button className="btn btnSmall" type="button" onClick={() => setSelectedNodeId(`anchor:${anchor.anchor_id}`)}>
                          {copy.focusGraph}
                        </button>
                      </div>
                      <div className="itemBody">{normalizeText(anchor.quote)}</div>
                    </article>
                  ))}
                </div>
              </div>
            </section>
          ) : null}

          {tab === 'citations' ? (
            <section className="panel">
              <div className="panelHeader">
                <div className="panelTitle">{copy.tabs.citations}</div>
              </div>
              <div className="panelBody">
                <div className="list">
                  {trace.canonical_core.citation_acts.length ? (
                    trace.canonical_core.citation_acts.map((citation: CitationAct) => (
                      <article key={`${citation.citation_act_id}:${citation.source_move_id || 'none'}`} className="itemCard">
                        <div className="split">
                          <div className="itemTitle">{citation.citation_act_id}</div>
                          <button className="btn btnSmall" type="button" onClick={() => setSelectedNodeId(`citation:${citation.citation_act_id}`)}>
                            {copy.focusGraph}
                          </button>
                        </div>
                        <div className="metaLine">
                          {normalizeText(citation.source_move_id) || '-'} | {normalizeText(citation.target_scope) || '-'}
                        </div>
                        <div className="itemBody">
                          {[normalizeText(citation.target_paper_id), normalizeText(citation.purpose), normalizeText(citation.polarity), normalizeText(citation.semantic_signal)]
                            .filter(Boolean)
                            .join(' | ')}
                        </div>
                      </article>
                    ))
                  ) : (
                    <div className="metaLine">{copy.noCitations}</div>
                  )}
                </div>
              </div>
            </section>
          ) : null}

          {tab === 'original' ? (
            <section className="panel">
              <div className="panelHeader">
                <div className="panelTitle">{copy.tabs.original}</div>
              </div>
              <div className="panelBody">{content ? <MarkdownView markdown={content} /> : <div className="metaLine">{copy.emptyContent}</div>}</div>
            </section>
          ) : null}
        </>
      )}
    </div>
  )
}
