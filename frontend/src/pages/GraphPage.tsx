import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { apiGet } from '../api'
import NetworkGraph, { type NetworkEdge, type NetworkNode } from '../components/NetworkGraph'
import GraphNodeDrawer from '../components/GraphNodeDrawer'
import { TERMS } from '../ui/terms'
import { applyScopeToUrl, loadScope, saveScope, scopeFromUrl, scopeLabel, type Scope } from '../scope'

type CollectionRow = { collection_id: string; name: string }
type SearchRow = { paper_id: string; title?: string; doi?: string; paper_source?: string; year?: number; ingested?: boolean }

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

  const stats = useMemo(() => ({ nodes: nodes.length, edges: edges.length }), [edges.length, nodes.length])
  const focused = useMemo(() => {
    if (!focusedId) return null
    return nodes.find((n) => n.id === focusedId) ?? { id: focusedId }
  }, [nodes, focusedId])

  const selectedBasket = useMemo(
    () => selectedIds.map((id) => nodes.find((n) => n.id === id) ?? { id }),
    [nodes, selectedIds],
  )

  function setScopeAndRefresh(next: Scope) {
    setScope(next)
    saveScope(next)
    setSearchParams(applyScopeToUrl(searchParams, next), { replace: true })
    void refresh(next)
  }

  async function refresh(nextScope: Scope = scope) {
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
      const ns = showOutside ? allNodes : allNodes.filter((n) => inScope.has(n.id))
      const es = showOutside ? (r.edges ?? []) : (r.edges ?? []).filter((e) => inScope.has(e.source) && inScope.has(e.target))
      setNodes(ns)
      setEdges(es)
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

    void refresh(initial ?? scope)

    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function toggleSelected(id: string) {
    setSelectedIds((prev) => {
      const has = prev.includes(id)
      const next = has ? prev.filter((x) => x !== id) : [...prev, id]
      return next.slice(0, 3)
    })
  }

  function enterContentMode() {
    if (!selectedIds.length) return
    const next: Scope = { mode: 'papers', paperIds: selectedIds.slice(0, 3) }
    saveScope(next)
    nav(`/content?${applyScopeToUrl(new URLSearchParams(), next).toString()}`)
  }

  async function runSearch() {
    const q = searchQ.trim()
    if (!q) return
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

    const edgeKey = (e: NetworkEdge) => `${e.source}->${e.target}`
    const edgeMap = new Map<string, NetworkEdge>()
    for (const e of edges) edgeMap.set(edgeKey(e), e)
    for (const e of r.edges ?? []) edgeMap.set(edgeKey(e), e)

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
  }

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">图谱</h2>
          <div className="pageSubtitle">论文引用网络（点击节点查看信息；支持多选进入内容模式）</div>
          <div className="metaLine">范围：{scopeLabel(scope)}</div>
        </div>
        <div className="pageActions">
          <span className="pill">
            <span className="kicker">节点</span> {stats.nodes}
          </span>
          <span className="pill">
            <span className="kicker">边</span> {stats.edges}
          </span>
          <button className="btn" disabled={busy} onClick={() => refresh().catch(() => {})}>
            {busy ? '加载中…' : '刷新'}
          </button>
        </div>
      </div>

      {error && <div className="errorBox">{error}</div>}

      <div className="panel">
        <div className="panelHeader">
          <div className="split">
            <div className="panelTitle">网络</div>
            <div className="row" style={{ gap: 8 }}>
              <select
                className="select"
                style={{ width: 180 }}
                value={scope.mode}
                onChange={(e) => {
                  const m = e.target.value as Scope['mode']
                  if (m === 'all') setScopeAndRefresh({ mode: 'all' })
                  else if (m === 'collection')
                    setScopeAndRefresh({ mode: 'collection', collectionId: scope.collectionId ?? collections[0]?.collection_id ?? '' })
                  else setScopeAndRefresh({ mode: 'papers', paperIds: scope.paperIds ?? [] })
                }}
              >
                <option value="all">全部论文</option>
                <option value="collection">论文集</option>
                <option value="papers">已选论文</option>
              </select>

              {scope.mode === 'collection' && (
                <select
                  className="select"
                  style={{ width: 220 }}
                  value={scope.collectionId ?? ''}
                  onChange={(e) => setScopeAndRefresh({ mode: 'collection', collectionId: e.target.value })}
                >
                  {collections.map((c) => (
                    <option key={c.collection_id} value={c.collection_id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              )}

              <label className="pill" style={{ cursor: 'pointer' }} title="显示/隐藏集合外（stub）论文节点">
                <span className="kicker">集合外</span>
                <input
                  type="checkbox"
                  style={{ marginLeft: 8 }}
                  checked={showOutside}
                  onChange={(e) => {
                    setShowOutside(e.target.checked)
                    void refresh()
                  }}
                />
              </label>

              <div className="row" style={{ gap: 6 }}>
                <input className="input" style={{ width: 220 }} value={searchQ} onChange={(e) => setSearchQ(e.target.value)} placeholder="搜索：标题 / DOI" />
                <button className="btn" disabled={searchBusy || !searchQ.trim()} onClick={() => runSearch().catch(() => {})}>
                  {searchBusy ? '搜索中…' : '搜索'}
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="panelBody">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: 12, alignItems: 'start' }}>
            <div>
              <NetworkGraph nodes={nodes} edges={edges} selectedIds={selectedIds} onSelectNode={(id) => setFocusedId(id)} onToggleSelect={(id) => toggleSelected(id)} />
              <div className="hint">提示：需要 Neo4j 运行并完成导入/重建后才能看到网络。</div>

              {searchResults.length > 0 && (
                <div className="panel" style={{ marginTop: 12, borderRadius: 14 }}>
                  <div className="panelHeader">
                    <div className="split">
                      <div className="panelTitle">搜索结果</div>
                      <button className="btn btnSmall" onClick={() => setSearchResults([])}>
                        清空
                      </button>
                    </div>
                  </div>
                  <div className="panelBody">
                    <div className="list">
                      {searchResults.slice(0, 30).map((p) => (
                        <div key={p.paper_id} className="itemCard">
                          <div className="itemTitle">{p.title || p.paper_source || p.paper_id}</div>
                          <div className="itemMeta">
                            {p.doi ?? ''} {p.year ? `· ${p.year}` : ''}
                          </div>
                          <div className="row" style={{ marginTop: 8, gap: 8 }}>
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
                </div>
              )}
            </div>

            <div className="panel" style={{ position: 'sticky', top: 14, alignSelf: 'start' }}>
              <div className="panelHeader">
                <div className="split">
                  <div className="panelTitle">节点信息</div>
                  {focused && (
                    <span className="badge" title={focused.ingested ? '已导入' : `${TERMS.stub}`}>
                      {focused.ingested ? '已导入' : TERMS.stub}
                    </span>
                  )}
                </div>
              </div>
              <div className="panelBody">
                <div className="itemCard" style={{ marginBottom: 10 }}>
                  <div className="split">
                    <div>
                      <div className="itemTitle">已选论文（最多 3 篇）</div>
                      <div className="itemMeta">{selectedIds.length ? selectedIds.join(' · ') : '（空）'}</div>
                    </div>
                    <div className="row" style={{ gap: 8 }}>
                      <button className="btn" disabled={selectedIds.length === 0} onClick={() => setScopeAndRefresh({ mode: 'papers', paperIds: selectedIds })}>
                        仅看已选
                      </button>
                      <button className="btn btnPrimary" disabled={selectedIds.length === 0} onClick={enterContentMode}>
                        内容模式
                      </button>
                    </div>
                  </div>
                  {selectedBasket.length > 0 && (
                    <div className="row" style={{ marginTop: 8, gap: 8, flexWrap: 'wrap' }}>
                      {selectedBasket.map((n) => (
                        <button key={n.id} className="chip chipActive" onClick={() => toggleSelected(n.id)}>
                          {n.paper_source ?? n.doi ?? n.id.slice(0, 12)}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {!focused ? (
                  <div className="metaLine">点击图谱中的节点以查看详情。</div>
                ) : (
                  <div className="stack">
                    <div className="row" style={{ gap: 8 }}>
                      <button className="btn btnSmall" onClick={() => toggleSelected(focused.id)}>
                        {selectedIds.includes(focused.id) ? '移出已选' : '加入已选'}
                      </button>
                      <button className="btn btnSmall" onClick={() => expandNeighbors(focused.id).catch(() => {})}>
                        展开邻居
                      </button>
                    </div>
                    <GraphNodeDrawer node={focused} onOpenDetail={(id) => nav(`/paper/${encodeURIComponent(id)}`)} onRefreshNetwork={refresh} />
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
