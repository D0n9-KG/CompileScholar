import { useEffect, useMemo, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'

import { apiGet } from '../api'
import MarkdownView from '../components/MarkdownView'
import SignalGraph, { type SignalGraphEdge, type SignalGraphNode } from '../components/SignalGraph'
import type {
  CitationAct,
  EvidenceAnchor,
  MoveRelation,
  PaperLogicTrace,
  ResearchMove,
} from '../types/paperLogicTrace'
import {
  formatMoveLabel,
  mentionTokens,
  moveActLabel,
  moveRoleLabel,
  normalizeText,
} from '../types/paperLogicTrace'

type TraceTab = 'moves' | 'relations' | 'anchors' | 'citations' | 'original'

type GraphMeta = {
  title: string
  typeLabel: string
  detail?: string
  badge?: string
  facts?: Array<{ label: string; value: string }>
  moveId?: string
  anchorId?: string
  citationActId?: string
}

type TraceLoadState = {
  paperId: string
  trace: PaperLogicTrace | null
  error: string
}

type OriginalContentState = {
  paperId: string
  content: string
  loaded: boolean
}

type SelectionState = {
  paperId: string
  nodeId: string
}

function formatLocator(anchor: EvidenceAnchor): string {
  const locator = anchor.locator ?? {}
  const startLine = Number(locator.start_line)
  const endLine = Number(locator.end_line)
  const page = Number(locator.page)
  if (Number.isFinite(startLine) && Number.isFinite(endLine) && endLine >= startLine) {
    return `Lines ${startLine}-${endLine}`
  }
  if (Number.isFinite(startLine)) return `Line ${startLine}`
  if (Number.isFinite(page)) return `Page ${page}`
  return anchor.source_ref
}

function formatConfidence(value: number | null | undefined): string {
  if (!Number.isFinite(Number(value))) return '-'
  return `${Math.round(Number(value) * 100)}%`
}

function moveTagSummary(move: ResearchMove): string[] {
  const tags = [
    ...mentionTokens(move.methods),
    ...mentionTokens(move.research_objects),
    ...mentionTokens(move.metrics),
    ...mentionTokens(move.comparators),
    ...mentionTokens(move.conditions),
  ]
  return Array.from(new Set(tags)).slice(0, 6)
}

function evidenceSummary(anchor: EvidenceAnchor): string {
  const section = (anchor.section_path ?? []).filter(Boolean).join(' / ')
  const location = formatLocator(anchor)
  return [section, location].filter(Boolean).join(' · ')
}

function buildTraceGraph(trace: PaperLogicTrace): {
  nodes: SignalGraphNode[]
  edges: SignalGraphEdge[]
  metaMap: Map<string, GraphMeta>
} {
  const nodes = new Map<string, SignalGraphNode>()
  const edges = new Map<string, SignalGraphEdge>()
  const metaMap = new Map<string, GraphMeta>()
  const rootId = 'paper:root'
  const anchorById = new Map(trace.canonical_core.evidence_anchors.map((anchor) => [anchor.anchor_id, anchor]))

  const putNode = (node: SignalGraphNode, meta: GraphMeta) => {
    nodes.set(node.id, node)
    metaMap.set(node.id, meta)
  }

  putNode(
    {
      id: rootId,
      label: normalizeText(trace.paper_metadata.title) || trace.paper_metadata.paper_id,
      kind: 'root',
      weight: 1,
    },
    {
      title: normalizeText(trace.paper_metadata.title) || trace.paper_metadata.paper_id,
      typeLabel: 'Paper Logic Trace',
      detail: [trace.paper_metadata.paper_id, trace.paper_metadata.canonical_doi, trace.paper_metadata.venue]
        .filter(Boolean)
        .join(' · '),
      badge: String(trace.quality?.quality_tier ?? ''),
    },
  )

  for (const move of trace.canonical_core.moves) {
    const clusterId = `cluster:${move.move_id}`
    const moveId = `move:${move.move_id}`
    putNode(
      {
        id: clusterId,
        label: '',
        kind: 'cluster',
        weight: 0.45,
      },
      {
        title: formatMoveLabel(move),
        typeLabel: 'Move Cluster',
        detail: `${move.anchor_ids?.length ?? 0} anchors`,
      },
    )
    putNode(
      {
        id: moveId,
        label: formatMoveLabel(move),
        kind: 'move',
        weight: 0.76,
      },
      {
        title: formatMoveLabel(move),
        typeLabel: 'Research Move',
        detail: move.summary,
        badge: moveActLabel(move.act_type),
        facts: [
          { label: 'Role', value: moveRoleLabel(move.role) },
          { label: 'Act Type', value: moveActLabel(move.act_type) },
        ],
        moveId: move.move_id,
      },
    )
    edges.set(`${rootId}->${moveId}`, {
      id: `${rootId}->${moveId}`,
      source: rootId,
      target: moveId,
      kind: 'contains',
      weight: 0.7,
    })

    for (const anchorId of move.anchor_ids ?? []) {
      const anchor = anchorById.get(anchorId)
      if (!anchor) continue
      const anchorNodeId = `anchor:${anchor.anchor_id}`
      if (!nodes.has(anchorNodeId)) {
        putNode(
          {
            id: anchorNodeId,
            label: anchor.anchor_id,
            kind: 'anchor',
            weight: anchor.weak ? 0.38 : 0.5,
          },
          {
            title: anchor.anchor_id,
            typeLabel: 'Evidence Anchor',
            detail: anchor.quote,
            badge: evidenceSummary(anchor),
            anchorId: anchor.anchor_id,
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
    const sourceId = `move:${relation.source_move_id}`
    const targetId = `move:${relation.target_move_id}`
    if (!nodes.has(sourceId) || !nodes.has(targetId)) continue
    edges.set(relation.relation_id, {
      id: relation.relation_id,
      source: sourceId,
      target: targetId,
      kind: relation.relation_type,
      weight: relation.confidence ?? 0.55,
    })
  }

  for (const citation of trace.canonical_core.citation_acts) {
    const moveId = citation.source_move_id ? `move:${citation.source_move_id}` : ''
    if (!moveId || !nodes.has(moveId)) continue
    const citationNodeId = `citation:${citation.citation_act_id}`
    putNode(
      {
        id: citationNodeId,
        label: citation.citation_act_id,
        kind: 'citation',
        weight: 0.42,
      },
      {
        title: citation.citation_act_id,
        typeLabel: 'Citation Act',
        detail: [citation.target_paper_id, citation.purpose, citation.semantic_signal].filter(Boolean).join(' · '),
        badge: citation.target_scope ?? '',
        citationActId: citation.citation_act_id,
      },
    )
    edges.set(`${moveId}->${citationNodeId}`, {
      id: `${moveId}->${citationNodeId}`,
      source: moveId,
      target: citationNodeId,
      kind: 'cites',
      weight: citation.confidence ?? 0.45,
    })
  }

  return {
    nodes: Array.from(nodes.values()),
    edges: Array.from(edges.values()),
    metaMap,
  }
}

export default function PaperDetailPage() {
  const { paperId = '' } = useParams<{ paperId: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const [traceState, setTraceState] = useState<TraceLoadState>({ paperId: '', trace: null, error: '' })
  const [contentState, setContentState] = useState<OriginalContentState>({ paperId: '', content: '', loaded: false })
  const [selectionState, setSelectionState] = useState<SelectionState>({ paperId: '', nodeId: 'paper:root' })

  const tab = useMemo<TraceTab>(() => {
    const raw = normalizeText(searchParams.get('tab'))
    const allowed: TraceTab[] = ['moves', 'relations', 'anchors', 'citations', 'original']
    return (allowed.find((item) => item === raw) ?? 'moves') as TraceTab
  }, [searchParams])

  useEffect(() => {
    if (!paperId) return
    let cancelled = false

    apiGet<PaperLogicTrace>(`/papers/${encodeURIComponent(paperId)}/logic-trace`)
      .then((payload) => {
        if (!cancelled) setTraceState({ paperId, trace: payload, error: '' })
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setTraceState({
            paperId,
            trace: null,
            error: String((cause as { message?: unknown } | null)?.message ?? cause),
          })
        }
      })

    return () => {
      cancelled = true
    }
  }, [paperId])

  useEffect(() => {
    if (!paperId || tab !== 'original' || (contentState.paperId === paperId && contentState.loaded)) return
    let cancelled = false

    apiGet<string>(`/papers/${encodeURIComponent(paperId)}/content`)
      .then((payload) => {
        if (!cancelled) {
          setContentState({ paperId, content: String(payload ?? ''), loaded: true })
        }
      })
      .catch(() => {
        if (!cancelled) setContentState({ paperId, content: '', loaded: true })
      })

    return () => {
      cancelled = true
    }
  }, [contentState.loaded, contentState.paperId, paperId, tab])

  const trace = traceState.paperId === paperId ? traceState.trace : null
  const error = traceState.paperId === paperId ? traceState.error : ''
  const content = contentState.paperId === paperId ? contentState.content : ''
  const selectedGraphNodeId = selectionState.paperId === paperId ? selectionState.nodeId : 'paper:root'

  const graph = useMemo(() => (trace ? buildTraceGraph(trace) : { nodes: [], edges: [], metaMap: new Map<string, GraphMeta>() }), [trace])
  const selectedMeta = useMemo(() => graph.metaMap.get(selectedGraphNodeId) ?? null, [graph.metaMap, selectedGraphNodeId])

  function selectTab(nextTab: TraceTab) {
    const next = new URLSearchParams(searchParams)
    next.set('tab', nextTab)
    setSearchParams(next, { replace: true })
  }

  function selectGraphNode(nodeId: string) {
    setSelectionState({ paperId, nodeId })
  }

  if (!paperId) return <div className="page">Missing paper id.</div>

  return (
    <div className="page paperTracePage">
      <div className="pageHeader paperTraceHeader">
        <div>
          <h2 className="pageTitle">{trace?.paper_metadata.title ?? 'Paper Logic Trace'}</h2>
          <div className="pageSubtitle">{trace?.paper_metadata.paper_id ?? paperId}</div>
        </div>
        {trace && (
          <div className="pageActions paperTraceHeaderMeta">
            <span className="pill">
              <span className="kicker">Schema</span> {trace.schema_version}
            </span>
            <span className="pill">
              <span className="kicker">Moves</span> {trace.canonical_core.moves.length}
            </span>
            <span className="pill">
              <span className="kicker">Anchors</span> {trace.canonical_core.evidence_anchors.length}
            </span>
            <span className="pill">
              <span className="kicker">Relations</span> {trace.canonical_core.move_relations.length}
            </span>
            <span className="pill">
              <span className="kicker">Quality</span> {String(trace.quality?.quality_tier ?? '-')}
            </span>
          </div>
        )}
      </div>

      {error && <div className="errorBox">{error}</div>}
      {!trace ? (
        <div className="panel">
          <div className="panelBody">Loading paper logic trace…</div>
        </div>
      ) : (
        <>
          <div className="row paperTraceTabRow">
            <button className={`chip ${tab === 'moves' ? 'chipActive' : ''}`} onClick={() => selectTab('moves')}>
              Research Moves
            </button>
            <button className={`chip ${tab === 'relations' ? 'chipActive' : ''}`} onClick={() => selectTab('relations')}>
              Move Relations
            </button>
            <button className={`chip ${tab === 'anchors' ? 'chipActive' : ''}`} onClick={() => selectTab('anchors')}>
              Evidence Anchors
            </button>
            <button className={`chip ${tab === 'citations' ? 'chipActive' : ''}`} onClick={() => selectTab('citations')}>
              Citation Acts
            </button>
            <button className={`chip ${tab === 'original' ? 'chipActive' : ''}`} onClick={() => selectTab('original')}>
              Source Content
            </button>
          </div>

          <section className="panel paperTraceWorkbench">
            <div className="panelHeader">
              <div className="split">
                <div className="panelTitle">Trace Graph</div>
                <div className="metaLine">
                  {trace.paper_metadata.paper_type} · {trace.paper_metadata.year ?? 'unknown year'}
                </div>
              </div>
            </div>
            <div className="panelBody">
              <div className="paperTraceWorkbenchGrid">
                <div className="paperTraceGraphPane">
                  <SignalGraph
                    nodes={graph.nodes}
                    edges={graph.edges}
                    selectedId={selectedGraphNodeId}
                    onSelect={selectGraphNode}
                    height={420}
                  />
                </div>
                <aside className="paperTraceDetailCard">
                  <div className="itemTitle">Node Detail</div>
                  {selectedMeta ? (
                    <div className="stack">
                      <div>
                        <div className="metaLine">{selectedMeta.typeLabel}</div>
                        <div className="itemBody" style={{ fontWeight: 700 }}>{selectedMeta.title}</div>
                      </div>
                      {selectedMeta.badge && <div className="badge">{selectedMeta.badge}</div>}
                      {selectedMeta.facts?.map((fact) => (
                        <div key={`${selectedMeta.title}:${fact.label}`} className="metaLine">
                          <strong>{fact.label}</strong> · {fact.value}
                        </div>
                      ))}
                      {selectedMeta.detail && <div className="itemBody">{selectedMeta.detail}</div>}
                    </div>
                  ) : (
                    <div className="metaLine">Select a move, anchor, or citation node.</div>
                  )}
                </aside>
              </div>
            </div>
          </section>

          {tab === 'moves' && (
            <section className="panel">
              <div className="panelHeader">
                <div className="panelTitle">Research Moves</div>
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
                            <div className="itemTitle">{formatMoveLabel(move)}</div>
                            <div className="metaLine">
                              Role · {moveRoleLabel(move.role)} | Act Type · {moveActLabel(move.act_type)}
                            </div>
                          </div>
                          <div className="row" style={{ gap: 8 }}>
                            <span className="badge">{formatConfidence(move.confidence)}</span>
                            <button className="btn btnSmall" onClick={() => selectGraphNode(`move:${move.move_id}`)}>
                              Focus in Graph
                            </button>
                          </div>
                        </div>
                        <div className="itemBody">{move.summary}</div>
                        {moveTagSummary(move).length > 0 && (
                          <div className="row" style={{ gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
                            {moveTagSummary(move).map((token) => (
                              <span key={`${move.move_id}:${token}`} className="chip">
                                {token}
                              </span>
                            ))}
                          </div>
                        )}
                      </article>
                    ))}
                </div>
              </div>
            </section>
          )}

          {tab === 'relations' && (
            <section className="panel">
              <div className="panelHeader">
                <div className="panelTitle">Move Relations</div>
              </div>
              <div className="panelBody">
                <div className="list">
                  {trace.canonical_core.move_relations.map((relation: MoveRelation) => (
                    <article key={relation.relation_id} className="itemCard">
                      <div className="split">
                        <div className="itemTitle">{relation.relation_type}</div>
                        <span className="badge">{formatConfidence(relation.confidence)}</span>
                      </div>
                      <div className="metaLine">{relation.source_move_id}</div>
                      <div className="metaLine">{relation.target_move_id}</div>
                    </article>
                  ))}
                  {trace.canonical_core.move_relations.length === 0 && <div className="metaLine">No move relations.</div>}
                </div>
              </div>
            </section>
          )}

          {tab === 'anchors' && (
            <section className="panel">
              <div className="panelHeader">
                <div className="panelTitle">Evidence Anchors</div>
              </div>
              <div className="panelBody">
                <div className="list">
                  {trace.canonical_core.evidence_anchors.map((anchor: EvidenceAnchor) => (
                    <article key={anchor.anchor_id} className="itemCard">
                      <div className="split">
                        <div>
                          <div className="itemTitle">{anchor.anchor_id}</div>
                          <div className="metaLine">{evidenceSummary(anchor)}</div>
                        </div>
                        <button className="btn btnSmall" onClick={() => selectGraphNode(`anchor:${anchor.anchor_id}`)}>
                          Focus in Graph
                        </button>
                      </div>
                      <div className="itemBody">{anchor.quote}</div>
                    </article>
                  ))}
                </div>
              </div>
            </section>
          )}

          {tab === 'citations' && (
            <section className="panel">
              <div className="panelHeader">
                <div className="panelTitle">Citation Acts</div>
              </div>
              <div className="panelBody">
                <div className="list">
                  {trace.canonical_core.citation_acts.map((citation: CitationAct) => (
                    <article key={citation.citation_act_id} className="itemCard">
                      <div className="split">
                        <div className="itemTitle">{citation.citation_act_id}</div>
                        <button className="btn btnSmall" onClick={() => selectGraphNode(`citation:${citation.citation_act_id}`)}>
                          Focus in Graph
                        </button>
                      </div>
                      <div className="metaLine">
                        {citation.source_move_id ?? '-'} · {citation.target_scope ?? '-'}
                      </div>
                      <div className="itemBody">
                        {[citation.target_paper_id, citation.purpose, citation.polarity, citation.semantic_signal]
                          .filter(Boolean)
                          .join(' · ')}
                      </div>
                    </article>
                  ))}
                  {trace.canonical_core.citation_acts.length === 0 && <div className="metaLine">No citation acts.</div>}
                </div>
              </div>
            </section>
          )}

          {tab === 'original' && (
            <section className="panel">
              <div className="panelHeader">
                <div className="panelTitle">Source Content</div>
              </div>
              <div className="panelBody">
                {content ? <MarkdownView markdown={content} /> : <div className="metaLine">Original content unavailable.</div>}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  )
}
