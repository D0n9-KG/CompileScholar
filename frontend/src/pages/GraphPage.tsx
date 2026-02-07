import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { apiGet } from '../api'
import GraphNodeDrawer from '../components/GraphNodeDrawer'
import NetworkGraph, { type GraphNodeHoverPayload, type NetworkEdge, type NetworkNode } from '../components/NetworkGraph'
import { applyScopeToUrl, loadScope, saveScope, scopeFromUrl, scopeLabel, type Scope } from '../scope'
import { TERMS } from '../ui/terms'

type CollectionRow = { collection_id: string; name: string }
type SearchRow = { paper_id: string; title?: string; doi?: string; paper_source?: string; year?: number; ingested?: boolean }
type GraphLayoutMode = 'balanced' | 'compact' | 'spread'
type EdgeLabelMode = 'off' | 'mentions'
type NodeLabelMode = 'focus' | 'minimal' | 'full'
type GraphViewState = { zoom: number; nodes: number; edges: number }
type ExpansionRecord = { nodeIds: Set<string>; edgeKeys: Set<string> }

const DEFAULT_MIN_MENTIONS = 0
const DEFAULT_LAYOUT_MODE: GraphLayoutMode = 'spread'
const DEFAULT_EDGE_LABEL_MODE: EdgeLabelMode = 'off'
const DEFAULT_NODE_LABEL_MODE: NodeLabelMode = 'focus'
const DEFAULT_CHIP_LABEL_MAX = 20

function toEdgeKey(edge: Pick<NetworkEdge, 'source' | 'target'>) {
  return `${edge.source}->${edge.target}`
}

type CompactNodeLabel = Pick<NetworkNode, 'id' | 'paper_source' | 'title' | 'doi'>

function getNodeDisplayLabel(node: CompactNodeLabel) {
  return node.paper_source ?? node.title ?? node.doi ?? node.id
}

function shortenText(value: string | null | undefined, maxLen: number) {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim()
  if (!text) return ''
  if (text.length <= maxLen) return text
  return `${text.slice(0, Math.max(1, maxLen - 3))}...`
}

function formatNodeChipLabel(node: CompactNodeLabel, maxLen: number = DEFAULT_CHIP_LABEL_MAX) {
  return shortenText(getNodeDisplayLabel(node), maxLen)
}

const scopeModeLabel: Record<Scope['mode'], string> = {
  all: '全部论文',
  collection: '论文集',
  papers: '已选论文',
}

export default function GraphPage() {
  const nav = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  const [scope, setScope] = useState<Scope>(() => scopeFromUrl(searchParams) ?? loadScope())
  const [collections, setCollections] = useState<CollectionRow[]>([])
  const [showOutside, setShowOutside] = useState<boolean>(true)

  const [nodes, setNodes] = useState<NetworkNode[]>([])
  const [edges, setEdges] = useState<NetworkEdge[]>([])
  const [error, setError] = useState<string>('')
  const [busy, setBusy] = useState<boolean>(false)
  const [focusedId, setFocusedId] = useState<string>('')

  const [selectedIds, setSelectedIds] = useState<string[]>([])

  const [searchQ, setSearchQ] = useState<string>('')
  const [searchBusy, setSearchBusy] = useState<boolean>(false)
  const [searchResults, setSearchResults] = useState<SearchRow[]>([])
  const [searchRunCount, setSearchRunCount] = useState<number>(0)

  const [layoutMode, setLayoutMode] = useState<GraphLayoutMode>(DEFAULT_LAYOUT_MODE)
  const [edgeLabelMode, setEdgeLabelMode] = useState<EdgeLabelMode>(DEFAULT_EDGE_LABEL_MODE)
  const [nodeLabelMode, setNodeLabelMode] = useState<NodeLabelMode>(DEFAULT_NODE_LABEL_MODE)
  const [minMentions, setMinMentions] = useState<number>(DEFAULT_MIN_MENTIONS)
  const [fitTrigger, setFitTrigger] = useState<number>(0)
  const [relayoutTrigger, setRelayoutTrigger] = useState<number>(0)
  const [clearHighlightTrigger, setClearHighlightTrigger] = useState<number>(0)
  const [graphView, setGraphView] = useState<GraphViewState>({ zoom: 1, nodes: 0, edges: 0 })
  const [hoverNode, setHoverNode] = useState<GraphNodeHoverPayload | null>(null)
  const baseGraphRef = useRef<{ nodes: Map<string, NetworkNode>; edges: Map<string, NetworkEdge> }>({
    nodes: new Map(),
    edges: new Map(),
  })
  const expansionByNodeRef = useRef<Map<string, ExpansionRecord>>(new Map())

  const filteredEdges = useMemo(() => {
    if (minMentions <= 0) return edges
    return edges.filter((edge) => (edge.total_mentions ?? 0) >= minMentions)
  }, [edges, minMentions])

  const filteredNodes = useMemo(() => {
    if (minMentions <= 0) return nodes
    const visibleIds = new Set<string>()
    for (const edge of filteredEdges) {
      visibleIds.add(edge.source)
      visibleIds.add(edge.target)
    }
    if (focusedId) visibleIds.add(focusedId)
    for (const id of selectedIds) visibleIds.add(id)
    return nodes.filter((node) => visibleIds.has(node.id))
  }, [nodes, filteredEdges, minMentions, focusedId, selectedIds])

  const stats = useMemo(
    () => ({
      rawNodes: nodes.length,
      rawEdges: edges.length,
      nodes: filteredNodes.length,
      edges: filteredEdges.length,
      selected: selectedIds.length,
    }),
    [nodes.length, edges.length, filteredNodes.length, filteredEdges.length, selectedIds.length],
  )

  const focused = useMemo(() => {
    if (!focusedId) return null
    return nodes.find((n) => n.id === focusedId) ?? { id: focusedId }
  }, [nodes, focusedId])

  const focusedDegree = useMemo(() => {
    if (!focusedId) return 0
    return filteredEdges.reduce((count, edge) => {
      if (edge.source === focusedId || edge.target === focusedId) return count + 1
      return count
    }, 0)
  }, [filteredEdges, focusedId])

  const focusNeighbors = useMemo(() => {
    if (!focusedId) return []
    const weightById = new Map<string, number>()
    for (const edge of filteredEdges) {
      if (edge.source !== focusedId && edge.target !== focusedId) continue
      const otherId = edge.source === focusedId ? edge.target : edge.source
      if (!otherId) continue
      weightById.set(otherId, (weightById.get(otherId) ?? 0) + Math.max(1, Number(edge.total_mentions ?? 0)))
    }
    return Array.from(weightById.entries())
      .map(([id, weight]) => ({ id, weight, node: nodes.find((n) => n.id === id) ?? ({ id } as NetworkNode) }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 4)
  }, [filteredEdges, focusedId, nodes])

  const hubNodes = useMemo(() => {
    const degreeById = new Map<string, number>()
    const mentionsById = new Map<string, number>()
    for (const edge of filteredEdges) {
      const mentions = Math.max(1, Number(edge.total_mentions ?? 0))
      degreeById.set(edge.source, (degreeById.get(edge.source) ?? 0) + 1)
      degreeById.set(edge.target, (degreeById.get(edge.target) ?? 0) + 1)
      mentionsById.set(edge.source, (mentionsById.get(edge.source) ?? 0) + mentions)
      mentionsById.set(edge.target, (mentionsById.get(edge.target) ?? 0) + mentions)
    }
    return filteredNodes
      .map((node) => ({
        node,
        degree: degreeById.get(node.id) ?? 0,
        mentions: mentionsById.get(node.id) ?? 0,
      }))
      .sort((a, b) => b.degree - a.degree || b.mentions - a.mentions)
      .slice(0, 4)
  }, [filteredEdges, filteredNodes])

  const selectedBasket = useMemo(
    () => selectedIds.map((id) => nodes.find((n) => n.id === id) ?? { id }),
    [nodes, selectedIds],
  )
  const hoverNodeData = useMemo(
    () => (hoverNode ? nodes.find((node) => node.id === hoverNode.id) ?? { id: hoverNode.id } : null),
    [hoverNode, nodes],
  )
  const hoverNodeExpanded = Boolean(hoverNode && expansionByNodeRef.current.has(hoverNode.id))
  const hoverNodeSelected = Boolean(hoverNode && selectedIds.includes(hoverNode.id))
  const hoverMenuPlacement = useMemo(() => {
    if (!hoverNode) return null
    const margin = 14
    const x = Math.max(margin, Math.min(hoverNode.x, Math.max(margin, hoverNode.canvasWidth - margin)))
    const y = Math.max(margin, Math.min(hoverNode.y, Math.max(margin, hoverNode.canvasHeight - margin)))
    const alignLeft = x > hoverNode.canvasWidth * 0.62
    return { x, y, alignLeft }
  }, [hoverNode])

  const filteredByThreshold = minMentions > 0 && (filteredEdges.length !== edges.length || filteredNodes.length !== nodes.length)

  function setScopeAndRefresh(next: Scope) {
    setScope(next)
    saveScope(next)
    setSearchParams(applyScopeToUrl(searchParams, next), { replace: true })
    void refresh(next, showOutside)
  }

  async function refresh(nextScope: Scope = scope, includeOutside: boolean = showOutside) {
    setBusy(true)
    setError('')
    try {
      const qs = new URLSearchParams()
      qs.set('limit_papers', '200')
      qs.set('limit_edges', '800')
      if (nextScope.mode === 'collection' && nextScope.collectionId) qs.set('collection_id', nextScope.collectionId)
      if (nextScope.mode === 'papers' && (nextScope.paperIds ?? []).length) qs.set('paper_ids', (nextScope.paperIds ?? []).join(','))

      const r = await apiGet<{ nodes: NetworkNode[]; edges: NetworkEdge[] }>(`/graph/network?${qs.toString()}`)
      const allNodes = r.nodes ?? []
      const inScope = new Set<string>(
        allNodes.filter((n) => (n as unknown as { in_scope?: boolean }).in_scope !== false).map((n) => n.id),
      )

      const ns = includeOutside ? allNodes : allNodes.filter((n) => inScope.has(n.id))
      const es = includeOutside ? (r.edges ?? []) : (r.edges ?? []).filter((e) => inScope.has(e.source) && inScope.has(e.target))

      setNodes(ns)
      setEdges(es)
      baseGraphRef.current = {
        nodes: new Map<string, NetworkNode>(ns.map((node) => [node.id, node])),
        edges: new Map<string, NetworkEdge>(es.map((edge) => [toEdgeKey(edge), edge])),
      }
      expansionByNodeRef.current.clear()
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    let alive = true
    const initial = scopeFromUrl(searchParams)

    if (initial) {
      setScope(initial)
      saveScope(initial)
      setSearchParams(applyScopeToUrl(searchParams, initial), { replace: true })
    }

    apiGet<{ collections: CollectionRow[] }>('/collections?limit=200')
      .then((r) => {
        if (!alive) return
        setCollections(r.collections ?? [])
      })
      .catch(() => {})

    void refresh(initial ?? scope, showOutside)

    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const toggleSelected = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const has = prev.includes(id)
      return has ? prev.filter((x) => x !== id) : [...prev, id]
    })
  }, [])

  const handleSelectNode = useCallback((id: string) => {
    setFocusedId(id)
  }, [])

  const handleGraphViewChange = useCallback((view: GraphViewState) => {
    setGraphView(view)
  }, [])

  const handleNodeHover = useCallback(
    (payload: GraphNodeHoverPayload) => {
      setHoverNode(payload)
    },
    [],
  )

  const handleNodeHoverOut = useCallback(() => {
    setHoverNode(null)
  }, [])

  useEffect(() => {
    if (!hoverNode) return
    if (nodes.some((node) => node.id === hoverNode.id)) return
    setHoverNode(null)
  }, [hoverNode, nodes])

  function enterContentMode() {
    if (!selectedIds.length) return
    const next: Scope = { mode: 'papers', paperIds: selectedIds }
    saveScope(next)
    nav(`/content?${applyScopeToUrl(new URLSearchParams(), next).toString()}`)
  }

  function openContentModeForNode(id: string) {
    const next: Scope = { mode: 'papers', paperIds: [id] }
    saveScope(next)
    nav(`/content?${applyScopeToUrl(new URLSearchParams(), next).toString()}`)
  }

  function clearGraphHighlight() {
    setFocusedId('')
    setHoverNode(null)
    setClearHighlightTrigger((n) => n + 1)
  }

  function resetGraphControls() {
    setSearchQ('')
    setSearchResults([])
    setSearchRunCount(0)
    setMinMentions(DEFAULT_MIN_MENTIONS)
    setLayoutMode(DEFAULT_LAYOUT_MODE)
    setEdgeLabelMode(DEFAULT_EDGE_LABEL_MODE)
    setNodeLabelMode(DEFAULT_NODE_LABEL_MODE)
    setFocusedId('')
    setSelectedIds([])
    setHoverNode(null)
    setClearHighlightTrigger((n) => n + 1)
    setRelayoutTrigger((n) => n + 1)
    setFitTrigger((n) => n + 1)
  }

  async function runSearch() {
    const q = searchQ.trim()
    if (!q) return
    setSearchRunCount((n) => n + 1)
    setSearchBusy(true)
    setError('')
    try {
      const qs = new URLSearchParams()
      qs.set('q', q)
      qs.set('limit', '30')
      if (scope.mode === 'collection' && scope.collectionId) qs.set('collection_id', scope.collectionId)
      const r = await apiGet<{ papers: SearchRow[] }>(`/graph/search?${qs.toString()}`)
      setSearchResults(r.papers ?? [])
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setSearchBusy(false)
    }
  }

  async function expandNeighbors(id: string) {
    const previousNodeIds = new Set<string>(nodes.map((node) => node.id))
    const previousEdgeKeys = new Set<string>(edges.map((edge) => toEdgeKey(edge)))
    const qs = new URLSearchParams()
    qs.set('paper_id', id)
    qs.set('depth', '1')
    qs.set('limit_nodes', '260')
    qs.set('limit_edges', '700')
    if (scope.mode === 'collection' && scope.collectionId) qs.set('collection_id', scope.collectionId)

    const r = await apiGet<{ nodes: NetworkNode[]; edges: NetworkEdge[] }>(`/graph/neighborhood?${qs.toString()}`)

    const byId = new Map<string, NetworkNode>()
    for (const n of nodes) byId.set(n.id, n)
    for (const n of r.nodes ?? []) byId.set(n.id, n)

    const edgeMap = new Map<string, NetworkEdge>()
    for (const e of edges) edgeMap.set(toEdgeKey(e), e)
    for (const e of r.edges ?? []) edgeMap.set(toEdgeKey(e), e)

    const mergedNodes = Array.from(byId.values())
    const inScope = new Set<string>(
      mergedNodes.filter((n) => (n as unknown as { in_scope?: boolean }).in_scope !== false).map((n) => n.id),
    )
    const finalNodes = showOutside ? mergedNodes : mergedNodes.filter((n) => inScope.has(n.id))
    const finalEdges = showOutside
      ? Array.from(edgeMap.values())
      : Array.from(edgeMap.values()).filter((e) => inScope.has(e.source) && inScope.has(e.target))

    setNodes(finalNodes)
    setEdges(finalEdges)
    const expansion = expansionByNodeRef.current.get(id) ?? { nodeIds: new Set<string>(), edgeKeys: new Set<string>() }
    for (const node of finalNodes) {
      if (!previousNodeIds.has(node.id)) expansion.nodeIds.add(node.id)
    }
    for (const edge of finalEdges) {
      const edgeKey = toEdgeKey(edge)
      if (!previousEdgeKeys.has(edgeKey)) expansion.edgeKeys.add(edgeKey)
    }
    if (expansion.nodeIds.size || expansion.edgeKeys.size) {
      expansionByNodeRef.current.set(id, expansion)
    }
    setFocusedId(id)
  }

  function collapseNeighbors(id: string) {
    if (!expansionByNodeRef.current.has(id)) return
    expansionByNodeRef.current.delete(id)

    const currentNodeMap = new Map<string, NetworkNode>(nodes.map((node) => [node.id, node]))
    const currentEdgeMap = new Map<string, NetworkEdge>(edges.map((edge) => [toEdgeKey(edge), edge]))
    const edgeOrder = new Map<string, number>(edges.map((edge, index) => [toEdgeKey(edge), index]))
    const nodeOrder = new Map<string, number>(nodes.map((node, index) => [node.id, index]))

    const keepNodeIds = new Set<string>(baseGraphRef.current.nodes.keys())
    const keepEdgeKeys = new Set<string>(baseGraphRef.current.edges.keys())
    for (const expansion of expansionByNodeRef.current.values()) {
      for (const nodeId of expansion.nodeIds) keepNodeIds.add(nodeId)
      for (const edgeKey of expansion.edgeKeys) keepEdgeKeys.add(edgeKey)
    }
    for (const selectedId of selectedIds) keepNodeIds.add(selectedId)
    if (focusedId) keepNodeIds.add(focusedId)
    keepNodeIds.add(id)

    const nextEdges: NetworkEdge[] = []
    for (const edgeKey of keepEdgeKeys) {
      const edge = currentEdgeMap.get(edgeKey) ?? baseGraphRef.current.edges.get(edgeKey)
      if (!edge) continue
      nextEdges.push(edge)
      keepNodeIds.add(edge.source)
      keepNodeIds.add(edge.target)
    }
    nextEdges.sort((a, b) => (edgeOrder.get(toEdgeKey(a)) ?? Number.MAX_SAFE_INTEGER) - (edgeOrder.get(toEdgeKey(b)) ?? Number.MAX_SAFE_INTEGER))

    const nextNodes = Array.from(keepNodeIds)
      .map((nodeId) => currentNodeMap.get(nodeId) ?? baseGraphRef.current.nodes.get(nodeId))
      .filter((node): node is NetworkNode => Boolean(node))
      .sort((a, b) => (nodeOrder.get(a.id) ?? Number.MAX_SAFE_INTEGER) - (nodeOrder.get(b.id) ?? Number.MAX_SAFE_INTEGER))

    setNodes(nextNodes)
    setEdges(nextEdges)
    setFocusedId(id)
  }

  return (
    <div className="page graphPage">
      <div className="graphHeader">
        <div>
          <h2 className="pageTitle">图谱工作台</h2>
          <div className="pageSubtitle">探索论文引用网络，按论文集筛选范围，并把关键节点送入内容模式深挖。</div>
          <div className="graphHeaderMeta">
            <span className="pill">
              <span className="kicker">范围</span> {scopeLabel(scope)}
            </span>
            <span className="pill">
              <span className="kicker">模式</span> {scopeModeLabel[scope.mode]}
            </span>
            {filteredByThreshold && (
              <span className="pill graphToolbarBadge">
                <span className="kicker">过滤</span> 已按引用强度阈值裁剪
              </span>
            )}
          </div>
        </div>
        <div className="pageActions graphHeaderActions">
          <div className="graphStatCard">
            <span className="kicker">展示节点</span>
            <span className="graphStatValue">{stats.nodes}</span>
          </div>
          <div className="graphStatCard">
            <span className="kicker">展示边</span>
            <span className="graphStatValue">{stats.edges}</span>
          </div>
          <div className="graphStatCard">
            <span className="kicker">缩放</span>
            <span className="graphStatValue">{Math.round(graphView.zoom * 100)}%</span>
          </div>
          <button className="btn" disabled={busy} onClick={() => refresh().catch(() => {})}>
            {busy ? '加载中…' : '刷新网络'}
          </button>
        </div>
      </div>

      {error && <div className="errorBox">{error}</div>}

      <div className="panel graphControlPanel">
        <div className="panelBody">
          <div className="graphControlRow">
            <span className="kicker">范围过滤</span>
            <select
              className="select graphScopeSelect"
              name="graph_scope_mode"
              value={scope.mode}
              onChange={(e) => {
                const m = e.target.value as Scope['mode']
                if (m === 'all') setScopeAndRefresh({ mode: 'all' })
                else if (m === 'collection') {
                  setScopeAndRefresh({ mode: 'collection', collectionId: scope.collectionId ?? collections[0]?.collection_id ?? '' })
                } else {
                  setScopeAndRefresh({ mode: 'papers', paperIds: scope.paperIds ?? [] })
                }
              }}
            >
              <option value="all">全部论文</option>
              <option value="collection">论文集</option>
              <option value="papers">已选论文</option>
            </select>

            {scope.mode === 'collection' && (
              <select className="select graphCollectionSelect" name="graph_scope_collection" value={scope.collectionId ?? ''} onChange={(e) => setScopeAndRefresh({ mode: 'collection', collectionId: e.target.value })}>
                {collections.map((c) => (
                  <option key={c.collection_id} value={c.collection_id}>
                    {c.name}
                  </option>
                ))}
              </select>
            )}

            <label className="pill graphToggle" title="显示/隐藏集合外（stub）论文节点">
              <input
                type="checkbox"
                name="graph_show_outside"
                checked={showOutside}
                onChange={(e) => {
                  const checked = e.target.checked
                  setShowOutside(checked)
                  void refresh(scope, checked)
                }}
              />
              <span>显示集合外节点（{TERMS.stub}）</span>
            </label>
          </div>

          <div className="graphControlRow">
            <input
              className="input graphSearchInput"
              name="graph_search_query"
              value={searchQ}
              onChange={(e) => setSearchQ(e.target.value)}
              placeholder="搜索论文：标题 / DOI"
              onKeyDown={(e) => {
                if (e.key === 'Enter') void runSearch()
              }}
            />
            <button className="btn" disabled={searchBusy || !searchQ.trim()} onClick={() => runSearch().catch(() => {})}>
              {searchBusy ? '搜索中…' : '搜索'}
            </button>
            <button
              className="btn btnGhost"
              disabled={searchResults.length === 0 && !searchQ.trim()}
              onClick={() => {
                setSearchResults([])
                setSearchRunCount(0)
                setSearchQ('')
              }}
            >
              清空结果
            </button>

            <button className="btn" disabled={selectedIds.length === 0} onClick={() => setScopeAndRefresh({ mode: 'papers', paperIds: selectedIds })}>
              仅看已选
            </button>
            <button className="btn btnPrimary" disabled={selectedIds.length === 0} onClick={enterContentMode}>
              进入内容模式
            </button>
          </div>

          <div className="graphControlRow graphControlRow--tools">
            <span className="kicker">展示参数</span>
            <label className="graphInlineStat">
              <span>最小引用次数（0 = 不过滤）</span>
              <input
                className="input"
                name="graph_min_mentions"
                type="number"
                min={0}
                step={1}
                value={String(minMentions)}
                onChange={(e) => {
                  const next = Number(e.target.value)
                  if (!Number.isFinite(next)) return
                  setMinMentions(Math.max(0, Math.floor(next)))
                }}
              />
            </label>
            <label className="graphInlineStat">
              <span>布局密度</span>
              <select
                className="select"
                name="graph_layout_mode"
                value={layoutMode}
                onChange={(e) => {
                  setLayoutMode(e.target.value as GraphLayoutMode)
                  setRelayoutTrigger((n) => n + 1)
                }}
              >
                <option value="balanced">平衡</option>
                <option value="compact">紧凑</option>
                <option value="spread">展开</option>
              </select>
            </label>
            <label className="graphInlineStat">
              <span>节点标签</span>
              <select className="select" name="graph_node_label_mode" value={nodeLabelMode} onChange={(e) => setNodeLabelMode(e.target.value as NodeLabelMode)}>
                <option value="focus">只看当前节点（推荐）</option>
                <option value="minimal">少量标签</option>
                <option value="full">显示更多（易拥挤）</option>
              </select>
            </label>
            <label className="graphInlineStat">
              <span>边标签</span>
              <select className="select" name="graph_edge_label_mode" value={edgeLabelMode} onChange={(e) => setEdgeLabelMode(e.target.value as EdgeLabelMode)}>
                <option value="mentions">显示高权重边</option>
                <option value="off">关闭</option>
              </select>
            </label>

            <div className="graphToolsRow">
              <button className="btn btnSmall" onClick={() => setFitTrigger((n) => n + 1)}>
                适配视图
              </button>
              <button className="btn btnSmall" onClick={() => setRelayoutTrigger((n) => n + 1)}>
                重新布局
              </button>
              <button className="btn btnSmall btnGhost" onClick={clearGraphHighlight}>
                清空选中
              </button>
            </div>
          </div>
          <div className="graphControlRow graphControlRow--meta">
            <div className="graphQuickActions">
              <button className="btn btnSmall" onClick={resetGraphControls}>
                重置视图
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="graphLayout">
        <section className="panel graphMainPanel">
          <div className="panelHeader">
            <div className="graphMainHeader">
              <div className="panelTitle">引用网络</div>
              <div className="graphMainHint">
                单击节点会在图上打开功能菜单（展开/收回邻居、选中、内容模式）并在右侧显示详情。当前展示 {stats.nodes}/{stats.rawNodes} 节点，{stats.edges}/{stats.rawEdges} 边。
              </div>
            </div>
          </div>
          <div className="panelBody">
            <div className="graphCanvasWrap">
              <NetworkGraph
                nodes={filteredNodes}
                edges={filteredEdges}
                selectedIds={selectedIds}
                focusNodeId={focusedId}
                layoutMode={layoutMode}
                edgeLabelMode={edgeLabelMode}
                nodeLabelMode={nodeLabelMode}
                fitTrigger={fitTrigger}
                relayoutTrigger={relayoutTrigger}
                clearHighlightTrigger={clearHighlightTrigger}
                onSelectNode={handleSelectNode}
                onToggleSelect={toggleSelected}
                onNodeHover={handleNodeHover}
                onNodeHoverOut={handleNodeHoverOut}
                onViewChange={handleGraphViewChange}
                height={760}
              />
              {hoverNodeData && hoverMenuPlacement && (
                <div
                  className={`graphNodeHoverMenu ${hoverMenuPlacement.alignLeft ? 'graphNodeHoverMenu--left' : ''}`}
                  style={{ left: hoverMenuPlacement.x, top: hoverMenuPlacement.y }}
                >
                  <div className="graphNodeHoverMenuTitle">{formatNodeChipLabel(hoverNodeData, 34)}</div>
                  <div className="graphNodeHoverMenuActions">
                    <button
                      className="btn btnSmall"
                      onClick={() => {
                        setFocusedId(hoverNodeData.id)
                      }}
                    >
                      查看详情
                    </button>
                    <button
                      className="btn btnSmall"
                      onClick={() => {
                        toggleSelected(hoverNodeData.id)
                      }}
                    >
                      {hoverNodeSelected ? '取消选中' : '选中节点'}
                    </button>
                    <button
                      className="btn btnSmall"
                      onClick={() => {
                        expandNeighbors(hoverNodeData.id).catch(() => {})
                      }}
                    >
                      展开邻居
                    </button>
                    <button
                      className="btn btnSmall"
                      disabled={!hoverNodeExpanded}
                      onClick={() => {
                        collapseNeighbors(hoverNodeData.id)
                      }}
                    >
                      收回邻居
                    </button>
                    <button
                      className="btn btnSmall btnPrimary"
                      onClick={() => {
                        openContentModeForNode(hoverNodeData.id)
                      }}
                    >
                      内容模式
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div className="graphLegend">
              <span className="graphLegendItem">
                <span className="graphLegendDot graphLegendDot--ingested" />
                已导入论文
              </span>
              <span className="graphLegendItem">
                <span className="graphLegendDot graphLegendDot--stub" />
                {TERMS.stub}
              </span>
              <span className="graphLegendItem">
                <span className="graphLegendDot graphLegendDot--selected" />
                当前选中
              </span>
              <span className="graphLegendItem">
                <span className="graphLegendDot graphLegendDot--strong-edge" />
                高频引用边
              </span>
            </div>
            <div className="hint">提示：需要 Neo4j 运行并完成导入/重建后才能看到网络。</div>

            {searchResults.length > 0 && (
              <div className="graphSearchResults">
                <div className="panelTitle">搜索结果（{searchResults.length}）</div>
                <div className="graphSearchList graphSearchListSpaced">
                  {searchResults.slice(0, 30).map((p) => (
                    <div key={p.paper_id} className="graphSearchCard">
                      <div className="itemTitle">{p.title || p.paper_source || p.paper_id}</div>
                      <div className="itemMeta">
                        {p.doi ?? ''} {p.year ? `· ${p.year}` : ''}
                      </div>
                      <div className="graphSearchActions">
                        <button className="btn btnSmall" onClick={() => setFocusedId(p.paper_id)}>
                          定位
                        </button>
                        <button className="btn btnSmall" onClick={() => toggleSelected(p.paper_id)}>
                          {selectedIds.includes(p.paper_id) ? '移出已选' : '加入已选'}
                        </button>
                        <button className="btn btnSmall" onClick={() => expandNeighbors(p.paper_id).catch(() => {})}>
                          展开邻居
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {searchBusy && (
              <div className="graphSearchStatus">
                <div className="graphEmptyState">Searching papers in current scope...</div>
              </div>
            )}

            {!searchBusy && searchRunCount > 0 && searchResults.length === 0 && (
              <div className="graphSearchStatus">
                <div className="graphEmptyState">No papers matched this query in current scope.</div>
              </div>
            )}
          </div>
        </section>

        <aside className="panel graphSidePanel">
          <div className="panelHeader">
            <div className="split">
              <div className="panelTitle">节点详情与补全</div>
              {focused && (
                <span className="badge" title={focused.ingested ? '已导入' : `${TERMS.stub}`}>
                  {focused.ingested ? '已导入' : TERMS.stub}
                </span>
              )}
            </div>
          </div>
          <div className="panelBody">
            <div className="itemCard">
              <div className="split graphSelectionHeader">
                <div>
                  <div className="itemTitle">已选论文</div>
                  <div className="itemMeta">{selectedIds.length ? `已选择 ${selectedIds.length} 篇` : '尚未选择论文'}</div>
                </div>
                <span className="badge">{stats.selected}</span>
              </div>
              {selectedBasket.length > 0 ? (
                <div className="graphSelectedList">
                  {selectedBasket.map((n) => (
                    <button key={n.id} className="chip chipActive graphNodeChipButton" onClick={() => toggleSelected(n.id)} title={getNodeDisplayLabel(n)}>
                      <span className="graphNodeChip">{formatNodeChipLabel(n, 26)}</span>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="graphEmptyState graphEmptyStateSpaced">
                  从左侧图谱选择论文，可快速切换到“内容模式”做更深层分析。
                </div>
              )}
            </div>

            {hubNodes.length > 0 && (
              <div className="itemCard graphHubCard">
                <div className="split graphSelectionHeader">
                  <div className="itemTitle">热点节点</div>
                  <span className="badge">Top {hubNodes.length}</span>
                </div>
                <div className="graphHubList">
                  {hubNodes.map((item) => (
                    <button
                      key={item.node.id}
                      className={`chip graphNodeChipButton ${focusedId === item.node.id ? 'chipActive' : ''}`}
                      onClick={() => setFocusedId(item.node.id)}
                      title={`${getNodeDisplayLabel(item.node)} | degree=${item.degree}, mentions=${item.mentions}`}
                    >
                      <span className="graphNodeChip">{formatNodeChipLabel(item.node)}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {!focused ? (
              <div className="graphEmptyState">点击左侧图谱中的节点，即可查看论文元数据、导入状态与补全操作。</div>
            ) : (
              <div className="stack">
                <div className="itemCard graphFocusCard">
                  <div className="graphFocusTitle">{focused.title ?? focused.paper_source ?? focused.doi ?? focused.id}</div>
                  <div className="graphFocusMeta">
                    <span>节点 ID：{focused.id}</span>
                    <span>年份：{focused.year ?? '未知'}</span>
                    <span>关联边：{focusedDegree}</span>
                  </div>
                </div>
                <div className="graphHoverHint">提示：节点的展开/收回邻居、选中与内容模式操作已移到图上单击菜单。</div>
                {focusNeighbors.length > 0 && (
                  <div className="itemCard graphNeighborCard">
                    <div className="split graphSelectionHeader">
                      <div className="itemTitle">邻居节点</div>
                      <span className="badge">{focusNeighbors.length}</span>
                    </div>
                    <div className="graphNeighborList">
                      {focusNeighbors.map((item) => (
                        <button
                          key={item.id}
                          className="chip graphNodeChipButton"
                          onClick={() => setFocusedId(item.id)}
                          title={`${getNodeDisplayLabel(item.node)} | total mentions=${item.weight}`}
                        >
                          <span className="graphNodeChip">{formatNodeChipLabel(item.node)}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                <GraphNodeDrawer node={focused} onOpenDetail={(id) => nav(`/paper/${encodeURIComponent(id)}`)} onRefreshNetwork={refresh} />
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}
