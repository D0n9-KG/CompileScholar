import { type CSSProperties, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { apiGet } from '../api'
import ContentGraph, { type ContentEdge, type ContentNode } from '../components/ContentGraph'
import { applyScopeToUrl, loadScope, saveScope, scopeFromUrl, scopeLabel, type Scope } from '../scope'
import MarkdownView from '../components/MarkdownView'

type PaperDetail = {
  paper: {
    paper_id: string
    title?: string
    doi?: string
    year?: number
    paper_source?: string
  }
  logic_steps?: Array<{ step_type: string; summary: string; order?: number | null }>
  claims?: Array<{
    claim_id?: string | null
    claim_key: string
    text: string
    confidence?: number | null
    step_type?: string | null
    source?: string | null
    kinds?: string[] | null
  }>
}

function short(s: string, n: number) {
  const t = String(s ?? '').trim().replace(/\s+/g, ' ')
  if (t.length <= n) return t
  return `${t.slice(0, n)}...`
}

function safeStepType(st: string | null | undefined) {
  const s = String(st ?? '').trim()
  return s || 'Unassigned'
}

function contentKindLabel(kind: ContentNode['kind']) {
  if (kind === 'paper') return '论文'
  if (kind === 'logic') return '逻辑步骤'
  if (kind === 'claim_group') return 'Claim 组'
  return 'Claim'
}

type LaneTheme = {
  accent: string
  paperFill: string
  paperBorder: string
  logicFill: string
  logicBorder: string
  groupFill: string
  groupBorder: string
  claimFill: string
  claimBorder: string
  edgeColor: string
  edgeSoftColor: string
  chipBg: string
  chipBorder: string
}

const LANE_ACCENTS = ['#62f1c9', '#7f92ff', '#ffcb7a', '#ff93c7', '#77d4ff', '#9def9d']

function hexToRgba(hex: string, alpha: number) {
  const normalized = hex.replace('#', '').trim()
  if (normalized.length !== 6) return `rgba(124,255,203,${alpha})`
  const r = Number.parseInt(normalized.slice(0, 2), 16)
  const g = Number.parseInt(normalized.slice(2, 4), 16)
  const b = Number.parseInt(normalized.slice(4, 6), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

function getLaneTheme(lane: number): LaneTheme {
  const accent = LANE_ACCENTS[((lane % LANE_ACCENTS.length) + LANE_ACCENTS.length) % LANE_ACCENTS.length]
  return {
    accent,
    paperFill: hexToRgba(accent, 0.34),
    paperBorder: hexToRgba(accent, 0.88),
    logicFill: hexToRgba(accent, 0.23),
    logicBorder: hexToRgba(accent, 0.54),
    groupFill: hexToRgba(accent, 0.14),
    groupBorder: hexToRgba(accent, 0.48),
    claimFill: hexToRgba(accent, 0.28),
    claimBorder: hexToRgba(accent, 0.62),
    edgeColor: hexToRgba(accent, 0.5),
    edgeSoftColor: hexToRgba(accent, 0.32),
    chipBg: hexToRgba(accent, 0.17),
    chipBorder: hexToRgba(accent, 0.46),
  }
}

export default function ContentPage() {
  const nav = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  const [scope, setScope] = useState<Scope>(() => scopeFromUrl(searchParams) ?? loadScope())
  const [error, setError] = useState<string>('')
  const [busy, setBusy] = useState<boolean>(false)

  const paperIds = useMemo(() => {
    if (scope.mode !== 'papers') return []
    return (scope.paperIds ?? []).map(String).filter(Boolean)
  }, [scope.mode, scope.paperIds])

  const [details, setDetails] = useState<PaperDetail[]>([])
  const [simLogic, setSimLogic] = useState<Array<{ source: string; target: string; score?: number }>>([])
  const [simClaims, setSimClaims] = useState<Array<{ source: string; target: string; score?: number }>>([])

  const [defaultClaimsN, setDefaultClaimsN] = useState<number>(5)
  const [groupExpanded, setGroupExpanded] = useState<Record<string, boolean>>({})
  const [groupNOverride, setGroupNOverride] = useState<Record<string, number>>({})

  const [selectedId, setSelectedId] = useState<string>('')

  useEffect(() => {
    const fromUrl = scopeFromUrl(searchParams)
    if (fromUrl) {
      setScope(fromUrl)
      saveScope(fromUrl)
      setSearchParams(applyScopeToUrl(searchParams, fromUrl), { replace: true })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!paperIds.length) {
      setDetails([])
      setSimLogic([])
      setSimClaims([])
      return
    }
    let alive = true
    setBusy(true)
    setError('')
    Promise.all(paperIds.map((id) => apiGet<PaperDetail>(`/graph/paper/${encodeURIComponent(id)}`)))
      .then((rows) => {
        if (!alive) return
        setDetails(rows)
      })
      .catch((e: unknown) => {
        if (!alive) return
        setError(String((e as { message?: unknown } | null)?.message ?? e))
      })
      .finally(() => {
        if (!alive) return
        setBusy(false)
      })

    const pidParam = paperIds.join(',')
    apiGet<{ edges: Array<{ source: string; target: string; score?: number }> }>(
      `/graph/similarity/logic?paper_ids=${encodeURIComponent(pidParam)}&min_score=0.75&limit_per_source=2`,
    )
      .then((r) => {
        if (!alive) return
        setSimLogic(r.edges ?? [])
      })
      .catch(() => {})

    apiGet<{ edges: Array<{ source: string; target: string; score?: number }> }>(
      `/graph/similarity/claims?paper_ids=${encodeURIComponent(pidParam)}&min_score=0.78&limit_per_source=2`,
    )
      .then((r) => {
        if (!alive) return
        setSimClaims(r.edges ?? [])
      })
      .catch(() => {})

    return () => {
      alive = false
    }
  }, [paperIds])

  const graph = useMemo(() => {
    const nodes: Array<ContentNode & { x: number; y: number }> = []
    const edges: ContentEdge[] = []

    const claimNodeIds = new Set<string>()
    const logicNodeIds = new Set<string>()
    if (!details.length) return { nodes, edges }

    const laneColumns = details.length >= 9 ? 3 : details.length >= 4 ? 2 : 1
    const laneWidth = 520
    const laneGapX = 126
    const gridStartX = 150
    const gridStartY = 44
    const paperOffsetY = 0
    const logicOffsetY = 124
    const stepGap = 110
    const groupOffsetY = 168
    const claimColumns = laneColumns >= 3 ? 2 : 3
    const claimRowGap = 44
    const claimColGap = laneColumns >= 3 ? 102 : 88
    const rowGap = 150

    const rowHeights: number[] = []
    const rowOffsets: number[] = []

    type ClaimRow = NonNullable<PaperDetail['claims']>[number]
    type LanePlan = {
      lane: number
      paperId: string
      paper: PaperDetail['paper']
      theme: LaneTheme
      steps: Array<{ step_type: string; summary: string; order?: number | null }>
      claimsByStep: Map<string, ClaimRow[]>
      orderedStepGroups: Array<[string, ClaimRow[]]>
      logicYByStep: Map<string, number>
      groupYByStep: Map<string, number>
      visibleCountByStep: Map<string, number>
      laneHeight: number
    }

    const visibleClaimsForGroup = (groupId: string, total: number) => {
      if (!groupExpanded[groupId]) return 0
      const n = Math.max(1, Math.min(30, Number(groupNOverride[groupId] ?? defaultClaimsN) || defaultClaimsN))
      return Math.min(total, n)
    }

    const lanePlans: LanePlan[] = details.map((d, lane) => {
      const theme = getLaneTheme(lane)
      const steps = [...(d.logic_steps ?? [])].sort((a, b) => (Number(a.order ?? 999) || 999) - (Number(b.order ?? 999) || 999))
      const paperId = d.paper.paper_id
      const claims = [...(d.claims ?? [])]
      const claimsByStep = new Map<string, ClaimRow[]>()
      for (const claim of claims) {
        const st = safeStepType(claim.step_type)
        const arr = claimsByStep.get(st) ?? []
        arr.push(claim)
        claimsByStep.set(st, arr)
      }

      const stepOrder = new Map<string, number>(steps.map((s, idx) => [safeStepType(s.step_type), idx]))
      const orderedStepGroups = Array.from(claimsByStep.entries()).sort(
        (a, b) => (stepOrder.get(a[0]) ?? 999) - (stepOrder.get(b[0]) ?? 999) || a[0].localeCompare(b[0]),
      )

      const logicYByStep = new Map<string, number>()
      const groupYByStep = new Map<string, number>()
      const visibleCountByStep = new Map<string, number>()

      let cursorY = logicOffsetY
      for (const step of steps) {
        const st = safeStepType(step.step_type)
        logicYByStep.set(st, cursorY)

        const claimsInStep = claimsByStep.get(st) ?? []
        if (!claimsInStep.length) {
          cursorY += stepGap
          continue
        }

        const groupId = `claimgrp:${paperId}:${st}`
        const visibleCount = visibleClaimsForGroup(groupId, claimsInStep.length)
        visibleCountByStep.set(st, visibleCount)
        groupYByStep.set(st, cursorY + groupOffsetY)

        const claimRows = visibleCount > 0 ? Math.ceil(visibleCount / claimColumns) : 0
        const claimTail = claimRows > 0 ? 84 + (claimRows - 1) * claimRowGap + 50 : 70
        cursorY += Math.max(stepGap, groupOffsetY + claimTail)
      }

      for (const [st, claimsInStep] of orderedStepGroups) {
        if (logicYByStep.has(st)) continue
        const groupId = `claimgrp:${paperId}:${st}`
        const visibleCount = visibleClaimsForGroup(groupId, claimsInStep.length)
        visibleCountByStep.set(st, visibleCount)
        groupYByStep.set(st, cursorY + 72)

        const claimRows = visibleCount > 0 ? Math.ceil(visibleCount / claimColumns) : 0
        const claimTail = claimRows > 0 ? 84 + (claimRows - 1) * claimRowGap + 50 : 70
        cursorY += Math.max(160, 72 + claimTail)
      }

      return {
        lane,
        paperId,
        paper: d.paper,
        theme,
        steps,
        claimsByStep,
        orderedStepGroups,
        logicYByStep,
        groupYByStep,
        visibleCountByStep,
        laneHeight: cursorY + 92,
      }
    })

    for (const lanePlan of lanePlans) {
      const row = Math.floor(lanePlan.lane / laneColumns)
      rowHeights[row] = Math.max(rowHeights[row] ?? 0, lanePlan.laneHeight)
    }

    let offsetY = gridStartY
    for (let row = 0; row < rowHeights.length; row++) {
      rowOffsets[row] = offsetY
      offsetY += (rowHeights[row] ?? 760) + rowGap
    }

    for (const lanePlan of lanePlans) {
      const { lane, paperId, paper, theme, steps, claimsByStep, orderedStepGroups, logicYByStep, groupYByStep, visibleCountByStep } = lanePlan
      const row = Math.floor(lane / laneColumns)
      const col = lane % laneColumns
      const x = gridStartX + col * (laneWidth + laneGapX)
      const laneTopY = rowOffsets[row] ?? gridStartY

      const paperNodeId = `paper:${paperId}`
      nodes.push({
        id: paperNodeId,
        kind: 'paper',
        label: short(paper.title || paper.paper_source || paper.doi || paperId, details.length > 4 ? 36 : 52),
        paperId,
        lane,
        fillColor: theme.paperFill,
        borderColor: theme.paperBorder,
        textColor: 'rgba(245,251,255,0.98)',
        glowColor: theme.edgeColor,
        x,
        y: laneTopY + paperOffsetY,
      })

      const stepNodeIds: string[] = []
      for (const step of steps) {
        const stepType = safeStepType(step.step_type)
        const logicY = logicYByStep.get(stepType)
        if (typeof logicY !== 'number') continue
        const logicId = `${paperId}:${stepType}`
        logicNodeIds.add(logicId)
        stepNodeIds.push(logicId)
        nodes.push({
          id: logicId,
          kind: 'logic',
          label: stepType,
          paperId,
          lane,
          stepType,
          fillColor: theme.logicFill,
          borderColor: theme.logicBorder,
          textColor: 'rgba(239,248,255,0.96)',
          glowColor: theme.edgeSoftColor,
          text: step.summary ?? '',
          x,
          y: laneTopY + logicY,
        })
      }
      for (let i = 0; i < stepNodeIds.length - 1; i++) {
        edges.push({
          id: `next:${stepNodeIds[i]}->${stepNodeIds[i + 1]}`,
          kind: 'next',
          source: stepNodeIds[i],
          target: stepNodeIds[i + 1],
          color: theme.edgeColor,
        })
      }

      for (const [st, arr] of orderedStepGroups) {
        const groupY = groupYByStep.get(st)
        if (typeof groupY !== 'number') continue
        const total = arr.length
        const groupId = `claimgrp:${paperId}:${st}`

        nodes.push({
          id: groupId,
          kind: 'claim_group',
          label: `${short(st, 18)} · ${total}`,
          paperId,
          lane,
          stepType: st,
          count: total,
          fillColor: theme.groupFill,
          borderColor: theme.groupBorder,
          textColor: 'rgba(229,240,255,0.92)',
          glowColor: theme.edgeSoftColor,
          x,
          y: laneTopY + groupY,
        })

        if (logicYByStep.has(st)) {
          edges.push({
            id: `group:${paperId}:${st}`,
            kind: 'group',
            source: `${paperId}:${st}`,
            target: groupId,
            color: theme.edgeSoftColor,
          })
        }

        const expandedCount = visibleCountByStep.get(st) ?? 0
        if (!expandedCount) continue

        const picked = [...(claimsByStep.get(st) ?? arr)]
          .sort((a, b) => (Number(b.confidence ?? 0) || 0) - (Number(a.confidence ?? 0) || 0))
          .slice(0, expandedCount)

        for (let i = 0; i < picked.length; i++) {
          const claim = picked[i]
          const cid = String(claim.claim_id ?? '') || `claim:${paperId}:${claim.claim_key}`
          claimNodeIds.add(cid)
          const rowInGroup = Math.floor(i / claimColumns)
          const colInGroup = i % claimColumns
          const columnOffset = (colInGroup - (claimColumns - 1) / 2) * claimColGap
          const cx = x + columnOffset
          const cy = laneTopY + groupY + 84 + rowInGroup * claimRowGap

          nodes.push({
            id: cid,
            kind: 'claim',
            label: `C${i + 1}`,
            paperId,
            lane,
            stepType: st,
            fillColor: theme.claimFill,
            borderColor: theme.claimBorder,
            textColor: 'rgba(246,252,255,0.94)',
            glowColor: theme.edgeSoftColor,
            text: claim.text ?? '',
            confidence: Number(claim.confidence ?? 0) || 0,
            claimKey: claim.claim_key,
            source: claim.source ?? undefined,
            kinds: (claim.kinds ?? undefined) as string[] | undefined,
            x: cx,
            y: cy,
          })
          edges.push({
            id: `claim:${groupId}->${cid}`,
            kind: 'claim',
            source: groupId,
            target: cid,
            color: theme.edgeSoftColor,
          })
        }
      }
    }

    for (const e of simLogic) {
      if (!logicNodeIds.has(e.source) || !logicNodeIds.has(e.target)) continue
      edges.push({
        id: `sim_logic:${e.source}->${e.target}`,
        kind: 'sim_logic',
        source: e.source,
        target: e.target,
        score: e.score ?? 0,
        color: 'rgba(156,136,255,0.38)',
        textColor: 'rgba(230,221,255,0.96)',
      })
    }
    for (const e of simClaims) {
      if (!claimNodeIds.has(e.source) || !claimNodeIds.has(e.target)) continue
      edges.push({
        id: `sim_claim:${e.source}->${e.target}`,
        kind: 'sim_claim',
        source: e.source,
        target: e.target,
        score: e.score ?? 0,
        color: 'rgba(132,160,255,0.36)',
        textColor: 'rgba(214,229,255,0.96)',
      })
    }

    return { nodes, edges }
  }, [defaultClaimsN, details, groupExpanded, groupNOverride, simClaims, simLogic])

  const selected = useMemo(() => graph.nodes.find((n) => n.id === selectedId), [graph.nodes, selectedId])
  const claimGroupIds = useMemo(() => graph.nodes.filter((node) => node.kind === 'claim_group').map((node) => node.id), [graph.nodes])
  const contentGraphHeight = useMemo(() => {
    if (!graph.nodes.length) return 780
    let minY = Number.POSITIVE_INFINITY
    let maxY = Number.NEGATIVE_INFINITY
    for (const node of graph.nodes) {
      minY = Math.min(minY, node.y)
      maxY = Math.max(maxY, node.y)
    }
    const span = Math.max(0, maxY - minY)
    return Math.min(3400, Math.max(780, Math.ceil(span + 260)))
  }, [graph.nodes])
  const laneSummaries = useMemo(
    () =>
      details.map((d, lane) => {
        const theme = getLaneTheme(lane)
        return {
          lane,
          paperId: d.paper.paper_id,
          label: short(d.paper.title || d.paper.paper_source || d.paper.doi || d.paper.paper_id, 44),
          style: {
            '--lane-chip-bg': theme.chipBg,
            '--lane-chip-border': theme.chipBorder,
            '--lane-chip-accent': theme.accent,
          } as CSSProperties,
        }
      }),
    [details],
  )

  function toggleGroup(groupId: string) {
    setGroupExpanded((prev) => ({ ...prev, [groupId]: !prev[groupId] }))
  }

  function setGroupN(groupId: string, n: number) {
    const v = Math.max(1, Math.min(30, Number(n) || 5))
    setGroupNOverride((prev) => ({ ...prev, [groupId]: v }))
  }

  function expandAllGroups() {
    if (!claimGroupIds.length) return
    setGroupExpanded((prev) => {
      const next = { ...prev }
      for (const id of claimGroupIds) next[id] = true
      return next
    })
  }

  function collapseAllGroups() {
    setGroupExpanded((prev) => {
      if (!Object.keys(prev).length) return prev
      const next = { ...prev }
      for (const id of claimGroupIds) delete next[id]
      return next
    })
  }

  if (scope.mode !== 'papers') {
    return (
      <div className="page">
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">内容模式</h2>
            <div className="pageSubtitle">请先在图谱页选择一篇或多篇论文进入内容模式。</div>
            <div className="metaLine">当前范围：{scopeLabel(scope)}</div>
          </div>
          <div className="pageActions">
            <button className="btn btnPrimary" onClick={() => nav('/graph')}>
              去图谱页
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">内容模式</h2>
          <div className="pageSubtitle">按论文泳道展示“论文 → 逻辑步骤 → Claim 组 → Claim”，支持多论文并排对比。</div>
          <div className="metaLine">已加载 {paperIds.length} 篇论文，颜色与泳道一一对应。</div>
          <div className="contentLaneList">
            {laneSummaries.map((lane) => (
              <span key={lane.paperId} className="contentLaneChip" style={lane.style} title={lane.paperId}>
                <span className="contentLaneChipIndex">Lane {lane.lane + 1}</span>
                <span className="contentLaneChipTitle">{lane.label}</span>
              </span>
            ))}
          </div>
        </div>
        <div className="pageActions">
          <span className="pill">
            <span className="kicker">默认 Top</span>
            <input
              className="input"
              name="content_default_claim_topn"
              style={{ width: 80 }}
              type="number"
              min={1}
              max={30}
              value={defaultClaimsN}
              onChange={(e) => setDefaultClaimsN(Math.max(1, Math.min(30, Number(e.target.value) || 5)))}
            />
          </span>
          <button className="btn btnSmall" onClick={expandAllGroups} disabled={claimGroupIds.length === 0}>
            展开全部 Claim 组
          </button>
          <button className="btn btnSmall btnGhost" onClick={collapseAllGroups} disabled={claimGroupIds.length === 0}>
            收起全部 Claim 组
          </button>
          <button className="btn" onClick={() => nav('/graph')}>
            返回图谱
          </button>
        </div>
      </div>

      {error && <div className="errorBox">{error}</div>}

      <div className="panel">
        <div className="panelHeader">
          <div className="contentGraphHeader">
            <div className="panelTitle">内容图谱</div>
            <div className="contentNodeLegend">
              <span className="contentLegendItem">
                <span className="contentLegendDot contentLegendDot--paper" />
                论文
              </span>
              <span className="contentLegendItem">
                <span className="contentLegendDot contentLegendDot--logic" />
                逻辑步骤
              </span>
              <span className="contentLegendItem">
                <span className="contentLegendDot contentLegendDot--group" />
                Claim 组
              </span>
              <span className="contentLegendItem">
                <span className="contentLegendDot contentLegendDot--claim" />
                Claim
              </span>
              <span className="contentLegendItem">
                <span className="contentLegendDot contentLegendDot--sim" />
                跨论文相似
              </span>
            </div>
          </div>
        </div>
        <div className="panelBody">
          <div className="contentModeLayout">
            <div>
              <ContentGraph
                nodes={graph.nodes}
                edges={graph.edges}
                height={contentGraphHeight}
                onSelect={(id) => {
                  setSelectedId(id)
                }}
              />
              {busy && <div className="metaLine">加载中…</div>}
            </div>

            <div className="panel contentModeSidePanel">
              <div className="panelHeader">
                <div className="panelTitle">节点详情</div>
              </div>
              <div className="panelBody">
                {!selected ? (
                  <div className="metaLine">单击节点查看详情；Claim 组支持按组或一键展开/收起，便于快速扫读多论文内容。</div>
                ) : (
                  <div className="stack">
                    <div className="itemCard">
                      <div className="itemTitle">
                        {contentKindLabel(selected.kind)} · {selected.paperId}
                      </div>
                      {selected.stepType && <div className="itemMeta">step: {selected.stepType}</div>}
                      {selected.kind === 'claim' && (
                        <div className="metaLine" style={{ marginTop: 6 }}>
                          {(() => {
                            const parts: string[] = []
                            if (selected.claimKey) parts.push(`key: ${selected.claimKey}`)
                            if ((selected.kinds?.length ?? 0) > 0) parts.push(`kinds: ${(selected.kinds ?? []).join(', ')}`)
                            if (typeof selected.confidence === 'number') parts.push(`confidence: ${(selected.confidence ?? 0).toFixed(2)}`)
                            if (selected.source) parts.push(`source: ${selected.source}`)
                            return parts.join(' · ')
                          })()}
                        </div>
                      )}
                      {selected.kind === 'claim_group' && (
                        <div className="row" style={{ marginTop: 10, gap: 8 }}>
                          <button className="btn btnSmall" onClick={() => toggleGroup(selected.id)}>
                            {groupExpanded[selected.id] ? '收起' : '展开'}
                          </button>
                          <span className="pill">
                            <span className="kicker">Top</span>
                            <input
                              className="input"
                              name={`content_group_claim_topn_${selected.id}`}
                              style={{ width: 80 }}
                              type="number"
                              min={1}
                              max={30}
                              value={groupNOverride[selected.id] ?? defaultClaimsN}
                              onChange={(e) => setGroupN(selected.id, Number(e.target.value))}
                            />
                          </span>
                        </div>
                      )}
                      {(selected.text ?? '').trim() && (
                        <div style={{ marginTop: 10 }}>
                          <MarkdownView markdown={selected.text ?? ''} paperId={selected.paperId} />
                        </div>
                      )}
                      {selected.kind === 'paper' && (
                        <div className="row" style={{ marginTop: 10 }}>
                          <button className="btn btnPrimary" onClick={() => nav(`/paper/${encodeURIComponent(selected.paperId)}`)}>
                            打开论文详情
                          </button>
                        </div>
                      )}
                    </div>

                    {selected.kind === 'paper' && (
                      <GraphNodeDrawerShim paperId={selected.paperId} onOpenDetail={(id) => nav(`/paper/${encodeURIComponent(id)}`)} />
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function GraphNodeDrawerShim({ paperId, onOpenDetail }: { paperId: string; onOpenDetail: (id: string) => void }) {
  const [node, setNode] = useState<{ id: string; doi?: string; title?: string; paper_source?: string; year?: number; ingested?: boolean } | null>(null)

  useEffect(() => {
    let alive = true
    apiGet<PaperDetail>(`/graph/paper/${encodeURIComponent(paperId)}`)
      .then((d) => {
        if (!alive) return
        setNode({
          id: d.paper.paper_id,
          doi: d.paper.doi,
          title: d.paper.title,
          paper_source: d.paper.paper_source,
          year: d.paper.year,
          ingested: true,
        })
      })
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [paperId])

  if (!node) return null
  return (
    <div className="itemCard">
      <div className="itemTitle">快捷操作</div>
      <div className="row" style={{ marginTop: 8 }}>
        <button className="btn btnPrimary" onClick={() => onOpenDetail(node.id)}>
          打开论文详情
        </button>
      </div>
    </div>
  )
}
