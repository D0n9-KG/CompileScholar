import { useEffect, useMemo, useState } from 'react'
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

export default function ContentPage() {
  const nav = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  const [scope, setScope] = useState<Scope>(() => scopeFromUrl(searchParams) ?? loadScope())
  const [error, setError] = useState<string>('')
  const [busy, setBusy] = useState<boolean>(false)

  const paperIds = useMemo(() => {
    if (scope.mode !== 'papers') return []
    return (scope.paperIds ?? []).map(String).filter(Boolean).slice(0, 3)
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

    const laneW = 420
    const laneX0 = 120
    const topY = 40
    const logicY0 = 120
    const stepGap = 105
    const groupOffsetY = 160
    const claimGapY = 48
    const claimGapX = 58

    const claimNodeIds = new Set<string>()
    const logicNodeIds = new Set<string>()

    for (let lane = 0; lane < details.length; lane++) {
      const d = details[lane]
      const p = d.paper
      const paperId = p.paper_id
      const x = laneX0 + laneW * lane

      const paperNodeId = `paper:${paperId}`
      nodes.push({
        id: paperNodeId,
        kind: 'paper',
        label: short(p.title || p.paper_source || p.doi || paperId, 54),
        paperId,
        lane,
        x,
        y: topY,
      })

      const steps = [...(d.logic_steps ?? [])].sort((a, b) => (Number(a.order ?? 999) || 999) - (Number(b.order ?? 999) || 999))
      const stepNodeIds: string[] = []
      for (let idx = 0; idx < steps.length; idx++) {
        const s = steps[idx]
        const logicId = `${paperId}:${s.step_type}`
        logicNodeIds.add(logicId)
        stepNodeIds.push(logicId)
        const y = logicY0 + idx * stepGap
        nodes.push({
          id: logicId,
          kind: 'logic',
          label: s.step_type,
          paperId,
          lane,
          stepType: s.step_type,
          text: s.summary ?? '',
          x,
          y,
        })
      }
      for (let i = 0; i < stepNodeIds.length - 1; i++) {
        edges.push({ id: `next:${stepNodeIds[i]}->${stepNodeIds[i + 1]}`, kind: 'next', source: stepNodeIds[i], target: stepNodeIds[i + 1] })
      }

      const claims = [...(d.claims ?? [])]
      const byStep = new Map<string, typeof claims>()
      for (const c of claims) {
        const st = safeStepType(c.step_type)
        const arr = byStep.get(st) ?? []
        arr.push(c)
        byStep.set(st, arr)
      }

      for (const [st, arr] of byStep.entries()) {
        const groupId = `claimgrp:${paperId}:${st}`
        const anchorIdx = steps.findIndex((x2) => x2.step_type === st)
        const baseY = anchorIdx >= 0 ? logicY0 + anchorIdx * stepGap : logicY0 + steps.length * stepGap
        const gy = baseY + groupOffsetY

        const total = arr.length
        nodes.push({
          id: groupId,
          kind: 'claim_group',
          label: `${st} · ${total}`,
          paperId,
          lane,
          stepType: st,
          count: total,
          x,
          y: gy,
        })

        if (anchorIdx >= 0) {
          edges.push({ id: `group:${paperId}:${st}`, kind: 'group', source: `${paperId}:${st}`, target: groupId })
        }

        const expanded = Boolean(groupExpanded[groupId])
        if (!expanded) continue

        const n = Math.max(1, Math.min(30, Number(groupNOverride[groupId] ?? defaultClaimsN) || defaultClaimsN))
        const picked = [...arr].sort((a, b) => (Number(b.confidence ?? 0) || 0) - (Number(a.confidence ?? 0) || 0)).slice(0, n)

        for (let i = 0; i < picked.length; i++) {
          const c = picked[i]
          const cid = String(c.claim_id ?? '') || `claim:${paperId}:${c.claim_key}`
          claimNodeIds.add(cid)
          const drift = (i % 2 === 0 ? -1 : 1) * Math.min(3, Math.floor(i / 6)) * claimGapX
          const cx = x + drift
          nodes.push({
            id: cid,
            kind: 'claim',
            label: `C${i + 1}`,
            paperId,
            lane,
            stepType: st,
            text: c.text ?? '',
            confidence: Number(c.confidence ?? 0) || 0,
            claimKey: c.claim_key,
            source: c.source ?? undefined,
            kinds: (c.kinds ?? undefined) as string[] | undefined,
            x: cx,
            y: gy + 86 + i * claimGapY,
          })
          edges.push({ id: `claim:${groupId}->${cid}`, kind: 'claim', source: groupId, target: cid })
        }
      }
    }

    for (const e of simLogic) {
      if (!logicNodeIds.has(e.source) || !logicNodeIds.has(e.target)) continue
      edges.push({ id: `sim_logic:${e.source}->${e.target}`, kind: 'sim_logic', source: e.source, target: e.target, score: e.score ?? 0 })
    }
    for (const e of simClaims) {
      if (!claimNodeIds.has(e.source) || !claimNodeIds.has(e.target)) continue
      edges.push({ id: `sim_claim:${e.source}->${e.target}`, kind: 'sim_claim', source: e.source, target: e.target, score: e.score ?? 0 })
    }

    return { nodes, edges }
  }, [defaultClaimsN, details, groupExpanded, groupNOverride, simClaims, simLogic])

  const selected = useMemo(() => graph.nodes.find((n) => n.id === selectedId), [graph.nodes, selectedId])

  function toggleGroup(groupId: string) {
    setGroupExpanded((prev) => ({ ...prev, [groupId]: !prev[groupId] }))
  }

  function setGroupN(groupId: string, n: number) {
    const v = Math.max(1, Math.min(30, Number(n) || 5))
    setGroupNOverride((prev) => ({ ...prev, [groupId]: v }))
  }

  if (scope.mode !== 'papers') {
    return (
      <div className="page">
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">内容模式</h2>
            <div className="pageSubtitle">请先在图谱页选择 1–3 篇论文进入内容模式。</div>
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
          <div className="pageSubtitle">上层：逻辑链条；下层：按步骤分组的 Claims（点击节点后在右侧查看/展开）</div>
          <div className="metaLine">论文：{paperIds.join(' · ')}</div>
        </div>
        <div className="pageActions">
          <span className="pill">
            <span className="kicker">默认 Top</span>
            <input
              className="input"
              style={{ width: 80 }}
              type="number"
              min={1}
              max={30}
              value={defaultClaimsN}
              onChange={(e) => setDefaultClaimsN(Math.max(1, Math.min(30, Number(e.target.value) || 5)))}
            />
          </span>
          <button className="btn" onClick={() => nav('/graph')}>
            返回图谱
          </button>
        </div>
      </div>

      {error && <div className="errorBox">{error}</div>}

      <div className="panel">
        <div className="panelHeader">
          <div className="panelTitle">内容图谱</div>
        </div>
        <div className="panelBody">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: 12, alignItems: 'start' }}>
            <div>
              <ContentGraph
                nodes={graph.nodes}
                edges={graph.edges}
                onSelect={(id) => {
                  setSelectedId(id)
                }}
              />
              {busy && <div className="metaLine">加载中…</div>}
            </div>

            <div className="panel" style={{ position: 'sticky', top: 14, alignSelf: 'start' }}>
              <div className="panelHeader">
                <div className="panelTitle">详情</div>
              </div>
              <div className="panelBody">
                {!selected ? (
                  <div className="metaLine">单击节点查看详情；Claim 展开/收起在右侧操作。</div>
                ) : (
                  <div className="stack">
                    <div className="itemCard">
                      <div className="itemTitle">
                        {selected.kind} · {selected.paperId}
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
