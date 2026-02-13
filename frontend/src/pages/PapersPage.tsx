import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiDelete, apiGet, apiPatch, apiPost } from '../api'

type CollectionRow = { collection_id: string; name: string }
type PaperRow = { paper_id: string; paper_source: string; title?: string; year?: number; doi?: string; collections?: CollectionRow[] }

export default function PapersPage() {
  const [papers, setPapers] = useState<PaperRow[]>([])
  const [collections, setCollections] = useState<CollectionRow[]>([])
  const [query, setQuery] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [info, setInfo] = useState<string>('')
  const [collectionFilter, setCollectionFilter] = useState<string>('all')
  const [collectionEditOpen, setCollectionEditOpen] = useState<boolean>(false)
  const [collectionEditMode, setCollectionEditMode] = useState<'create' | 'rename'>('create')
  const [collectionEditName, setCollectionEditName] = useState<string>('')
  const [collectionEditBusy, setCollectionEditBusy] = useState<boolean>(false)
  const [collectionDeleteOpen, setCollectionDeleteOpen] = useState<boolean>(false)
  const [collectionDeleteBusy, setCollectionDeleteBusy] = useState<boolean>(false)
  const [assignOpen, setAssignOpen] = useState<boolean>(false)
  const [assignPaper, setAssignPaper] = useState<PaperRow | null>(null)
  const [assignSelected, setAssignSelected] = useState<Record<string, boolean>>({})
  const [assignBusy, setAssignBusy] = useState<boolean>(false)
  const [selectedPaperIds, setSelectedPaperIds] = useState<Record<string, boolean>>({})
  const [batchAssignOpen, setBatchAssignOpen] = useState<boolean>(false)
  const [batchAssignSelected, setBatchAssignSelected] = useState<Record<string, boolean>>({})
  const [batchAssignBusy, setBatchAssignBusy] = useState<boolean>(false)
  const [batchDeleteOpen, setBatchDeleteOpen] = useState<boolean>(false)
  const [batchDeleteBusy, setBatchDeleteBusy] = useState<boolean>(false)
  const [deleteBusy, setDeleteBusy] = useState<string>('')
  const [paperDeleteOpen, setPaperDeleteOpen] = useState<boolean>(false)
  const [paperDeleteId, setPaperDeleteId] = useState<string>('')

  const reloadCollections = useCallback(async () => {
    const r = await apiGet<{ collections: CollectionRow[] }>('/collections?limit=200')
    setCollections(r.collections ?? [])
  }, [])

  const reloadPapers = useCallback(async () => {
    setError('')
    const cid = collectionFilter === 'all' ? '' : collectionFilter
    const qs = cid ? `&collection_id=${encodeURIComponent(cid)}` : ''
    const r = await apiGet<{ papers: PaperRow[] }>(`/graph/papers?limit=600${qs}`)
    setPapers(r.papers ?? [])
  }, [collectionFilter])

  useEffect(() => {
    reloadCollections().catch((e: unknown) => setError(String((e as { message?: unknown } | null)?.message ?? e)))
  }, [reloadCollections])

  useEffect(() => {
    reloadPapers().catch((e: unknown) => setError(String((e as { message?: unknown } | null)?.message ?? e)))
  }, [reloadPapers])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const rows = q
      ? papers.filter((p) => {
          const hay = `${p.title ?? ''} ${p.doi ?? ''} ${p.paper_source ?? ''}`.toLowerCase()
          return hay.includes(q)
        })
      : papers
    return [...rows].sort((a, b) => (b.year ?? 0) - (a.year ?? 0))
  }, [papers, query])

  const selectedPaperIdList = useMemo(() => {
    const valid = new Set(papers.map((p) => String(p.paper_id)))
    return Object.entries(selectedPaperIds)
      .filter(([paperId, checked]) => !!checked && valid.has(String(paperId)))
      .map(([paperId]) => String(paperId))
  }, [papers, selectedPaperIds])

  const selectedPaperCount = selectedPaperIdList.length
  const selectedCollectionCount = useMemo(
    () => Object.values(batchAssignSelected).filter((v) => !!v).length,
    [batchAssignSelected],
  )
  const selectedInFilteredCount = useMemo(
    () => filtered.filter((p) => !!selectedPaperIds[p.paper_id]).length,
    [filtered, selectedPaperIds],
  )
  const allFilteredSelected = filtered.length > 0 && selectedInFilteredCount === filtered.length

  useEffect(() => {
    const valid = new Set(papers.map((p) => String(p.paper_id)))
    setSelectedPaperIds((prev) => {
      let changed = false
      const next: Record<string, boolean> = {}
      for (const [k, v] of Object.entries(prev)) {
        if (!v || !valid.has(String(k))) {
          if (v) changed = true
          continue
        }
        next[k] = true
      }
      if (!changed && Object.keys(next).length === Object.keys(prev).length) return prev
      return next
    })
  }, [papers])

  function openCreateCollection() {
    setCollectionEditMode('create')
    setCollectionEditName('')
    setCollectionEditOpen(true)
  }

  function openRenameCollection() {
    if (collectionFilter === 'all' || collectionFilter === '__uncategorized__') return
    const current = collections.find((c) => c.collection_id === collectionFilter)
    setCollectionEditMode('rename')
    setCollectionEditName(current?.name ?? '')
    setCollectionEditOpen(true)
  }

  function openDeleteCollection() {
    if (collectionFilter === 'all' || collectionFilter === '__uncategorized__') return
    setCollectionDeleteOpen(true)
  }

  async function submitCollectionEdit() {
    const name = collectionEditName.trim()
    if (!name) return
    setCollectionEditBusy(true)
    setError('')
    setInfo('')
    try {
      if (collectionEditMode === 'create') {
        const r = await apiPost<{ collection_id: string }>('/collections', { name })
        await reloadCollections()
        setCollectionFilter(String(r.collection_id ?? 'all'))
        setInfo(`已创建论文集：${name}`)
      } else {
        await apiPatch<Record<string, unknown>>(`/collections/${encodeURIComponent(collectionFilter)}`, { name })
        await reloadCollections()
        await reloadPapers()
        setInfo('已重命名论文集。')
      }
      setCollectionEditOpen(false)
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setCollectionEditBusy(false)
    }
  }

  async function confirmDeleteCollection() {
    if (collectionFilter === 'all' || collectionFilter === '__uncategorized__') return
    setCollectionDeleteBusy(true)
    setError('')
    setInfo('')
    try {
      await apiDelete<Record<string, unknown>>(`/collections/${encodeURIComponent(collectionFilter)}`)
      setCollectionFilter('all')
      await reloadCollections()
      await reloadPapers()
      setInfo('已删除论文集。')
      setCollectionDeleteOpen(false)
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setCollectionDeleteBusy(false)
    }
  }

  function openAssign(p: PaperRow) {
    const selected: Record<string, boolean> = {}
    for (const c of p.collections ?? []) selected[String(c.collection_id)] = true
    setAssignSelected(selected)
    setAssignPaper(p)
    setAssignOpen(true)
  }

  async function saveAssign() {
    if (!assignPaper) return
    setAssignBusy(true)
    setError('')
    setInfo('')
    try {
      const before = new Set((assignPaper.collections ?? []).map((c) => String(c.collection_id)))
      const after = new Set(Object.entries(assignSelected).filter(([, v]) => !!v).map(([k]) => String(k)))

      const toAdd = [...after].filter((x) => !before.has(x))
      const toRemove = [...before].filter((x) => !after.has(x))

      for (const cid of toAdd) {
        await apiPost<Record<string, unknown>>(`/collections/${encodeURIComponent(cid)}/papers/${encodeURIComponent(assignPaper.paper_id)}`, {})
      }
      for (const cid of toRemove) {
        await apiDelete<Record<string, unknown>>(`/collections/${encodeURIComponent(cid)}/papers/${encodeURIComponent(assignPaper.paper_id)}`)
      }

      setInfo('已更新论文分类。')
      setAssignOpen(false)
      setAssignPaper(null)
      await reloadPapers()
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setAssignBusy(false)
    }
  }

  function togglePaperSelected(paperId: string, checked: boolean) {
    setSelectedPaperIds((prev) => {
      const next = { ...prev }
      if (checked) next[paperId] = true
      else delete next[paperId]
      return next
    })
  }

  function toggleFilteredSelected(checked: boolean) {
    setSelectedPaperIds((prev) => {
      const next = { ...prev }
      for (const p of filtered) {
        const paperId = String(p.paper_id)
        if (checked) next[paperId] = true
        else delete next[paperId]
      }
      return next
    })
  }

  function openBatchAssign() {
    if (!selectedPaperCount) return
    setBatchAssignSelected({})
    setBatchAssignOpen(true)
  }

  async function saveBatchAssign() {
    if (!selectedPaperCount) return
    const collectionIds = Object.entries(batchAssignSelected)
      .filter(([, v]) => !!v)
      .map(([k]) => String(k))
    if (!collectionIds.length) {
      setError('\u8bf7\u81f3\u5c11\u9009\u62e9\u4e00\u4e2a\u8bba\u6587\u96c6\u3002')
      return
    }
    setBatchAssignBusy(true)
    setError('')
    setInfo('')
    let ok = 0
    const failed: string[] = []
    try {
      for (const paperId of selectedPaperIdList) {
        try {
          for (const cid of collectionIds) {
            await apiPost<Record<string, unknown>>(`/collections/${encodeURIComponent(cid)}/papers/${encodeURIComponent(paperId)}`, {})
          }
          ok += 1
        } catch {
          failed.push(paperId)
        }
      }
      await reloadPapers()
      const msg = `\u6279\u91cf\u5206\u7c7b\u5b8c\u6210\uff1a${ok}/${selectedPaperIdList.length}\u3002`
      if (failed.length > 0) {
        setError(`${msg}\n\u5931\u8d25 ${failed.length} \u7bc7\uff1a${failed.slice(0, 5).join(', ')}${failed.length > 5 ? ' ...' : ''}`)
      } else {
        setInfo(msg)
      }
      if (failed.length === 0) setBatchAssignOpen(false)
    } finally {
      setBatchAssignBusy(false)
    }
  }

  function openBatchDelete() {
    if (!selectedPaperCount) return
    setBatchDeleteOpen(true)
  }

  async function confirmBatchDelete() {
    if (!selectedPaperCount) return
    setBatchDeleteBusy(true)
    setError('')
    setInfo('')
    let ok = 0
    const failed: string[] = []
    try {
      for (const paperId of selectedPaperIdList) {
        try {
          await apiDelete<Record<string, unknown>>(`/papers/${encodeURIComponent(paperId)}`)
          ok += 1
        } catch {
          failed.push(paperId)
        }
      }
      await reloadPapers()
      setSelectedPaperIds({})
      const msg = `\u6279\u91cf\u5220\u9664\u5b8c\u6210\uff1a${ok}/${selectedPaperIdList.length}\u3002`
      if (failed.length > 0) {
        setError(`${msg}\n\u5931\u8d25 ${failed.length} \u7bc7\uff1a${failed.slice(0, 5).join(', ')}${failed.length > 5 ? ' ...' : ''}`)
      } else {
        setInfo(msg)
      }
      setBatchDeleteOpen(false)
    } finally {
      setBatchDeleteBusy(false)
    }
  }

  function openDeletePaper(paperId: string) {
    setPaperDeleteId(paperId)
    setPaperDeleteOpen(true)
  }

  async function confirmDeletePaper() {
    const paperId = paperDeleteId
    if (!paperId) return
    setDeleteBusy(paperId)
    setError('')
    setInfo('')
    try {
      await apiDelete<Record<string, unknown>>(`/papers/${encodeURIComponent(paperId)}`)
      setInfo('已删除论文（含 Neo4j 图数据）。建议随后重建全局 FAISS 以清理检索索引。')
      setPaperDeleteOpen(false)
      setPaperDeleteId('')
      await reloadPapers()
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setDeleteBusy('')
    }
  }

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">论文</h2>
          <div className="pageSubtitle">按 DOI 去重后的论文列表（点击进入论文详情）</div>
        </div>
        <div className="pageActions">
          <span className="pill">
            <span className="kicker">数量</span> {filtered.length}
          </span>
          <select className="select" style={{ width: 200, maxWidth: '60vw' }} value={collectionFilter} onChange={(e) => setCollectionFilter(e.target.value)}>
            <option value="all">全部论文</option>
            <option value="__uncategorized__">未分类</option>
            {collections.map((c) => (
              <option key={c.collection_id} value={c.collection_id}>
                {c.name}
              </option>
            ))}
          </select>
          <button className="btn" onClick={() => openCreateCollection()}>
            新建论文集
          </button>
          <button className="btn" disabled={collectionFilter === 'all' || collectionFilter === '__uncategorized__'} onClick={() => openRenameCollection()}>
            重命名
          </button>
          <button className="btn btnDanger" disabled={collectionFilter === 'all' || collectionFilter === '__uncategorized__'} onClick={() => openDeleteCollection()}>
            删除论文集
          </button>
          <button className="btn" disabled={!selectedPaperCount} onClick={() => openBatchAssign()}>
            {'\u6279\u91cf\u5206\u7c7b'}
          </button>
          <button className="btn btnDanger" disabled={!selectedPaperCount} onClick={() => openBatchDelete()}>
            {'\u6279\u91cf\u5220\u9664'}
          </button>
          <input className="input" style={{ width: 340, maxWidth: '70vw' }} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="搜索标题 / DOI…" />
        </div>
      </div>

      {error && <div className="errorBox">{error}</div>}
      {info && (
        <div className="infoBox" style={{ marginTop: 12 }}>
          <div className="split">
            <div style={{ whiteSpace: 'pre-wrap' }}>{info}</div>
            <button className="btn btnSmall" onClick={() => setInfo('')}>
              清除
            </button>
          </div>
        </div>
      )}

      <div className="panel">
        <div className="panelHeader">
          <div className="split">
            <div className="panelTitle">列表</div>
            <div className="row" style={{ gap: 10 }}>
              <label className="row" style={{ gap: 8 }}>
                <input type="checkbox" checked={allFilteredSelected} disabled={filtered.length === 0} onChange={(e) => toggleFilteredSelected(e.target.checked)} />
                <span className="kicker">{'\u5168\u9009\u5f53\u524d\u7ed3\u679c'}</span>
              </label>
              <span className="pill">
                <span className="kicker">{'\u5df2\u9009'}</span> {selectedPaperCount}
              </span>
            </div>
          </div>
        </div>
        <div className="panelBody">
          <div className="list">
            {filtered.map((p) => (
              <div key={p.paper_id} className="itemCard">
                <div className="split">
                  <div className="row" style={{ gap: 10, minWidth: 0 }}>
                    <input type="checkbox" checked={!!selectedPaperIds[p.paper_id]} onChange={(e) => togglePaperSelected(String(p.paper_id), e.target.checked)} />
                    <div className="itemTitle">
                      <Link to={`/paper/${encodeURIComponent(p.paper_id)}`}>{p.title ?? p.paper_source}</Link>
                    </div>
                  </div>
                  <div className="row" style={{ gap: 8 }}>
                    <span className="badge">{p.year ?? ''}</span>
                    <button className="btn btnSmall" onClick={() => openAssign(p)}>
                      分类…
                    </button>
                    <button className="btn btnSmall btnDanger" disabled={deleteBusy === p.paper_id} onClick={() => openDeletePaper(p.paper_id)}>
                      删除
                    </button>
                  </div>
                </div>
                <div className="itemMeta">
                  <code>{p.paper_id}</code> {p.doi ? <span> · {p.doi}</span> : null}
                </div>
                {(p.collections ?? []).length > 0 && (
                  <div className="row" style={{ gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
                    {(p.collections ?? []).map((c) => (
                      <span key={c.collection_id} className="badge" title={c.collection_id}>
                        {c.name}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
          {filtered.length === 0 && <div className="metaLine">暂无论文。请先去导入页导入数据。</div>}
        </div>
      </div>

      {assignOpen && assignPaper && (
        <div className="modalOverlay" onClick={() => !assignBusy && setAssignOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div className="modalTitle">论文分类</div>
              <button className="btn btnSmall" disabled={assignBusy} onClick={() => setAssignOpen(false)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              <div className="hint" style={{ marginBottom: 12 }}>
                论文：<code>{assignPaper.paper_id}</code>
              </div>
              <div className="row" style={{ marginBottom: 10 }}>
                <button className="btn btnSmall" disabled={assignBusy} onClick={() => openCreateCollection()}>
                  新建论文集
                </button>
              </div>
              <div className="list">
                {collections.map((c) => {
                  const checked = !!assignSelected[c.collection_id]
                  return (
                    <div key={c.collection_id} className="itemCard">
                      <div className="split">
                        <div className="itemTitle">{c.name}</div>
                        <label className="row" style={{ gap: 10 }}>
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={(e) => setAssignSelected((m) => ({ ...m, [c.collection_id]: e.target.checked }))}
                          />
                          <span className="kicker">{checked ? '已加入' : '未加入'}</span>
                        </label>
                      </div>
                      <div className="itemMeta">
                        <code>{c.collection_id}</code>
                      </div>
                    </div>
                  )
                })}
                {collections.length === 0 && <div className="metaLine">暂无论文集。你可以先点击“新建论文集”。</div>}
              </div>
              <div className="row" style={{ marginTop: 12 }}>
                <button className="btn btnPrimary" disabled={assignBusy} onClick={() => saveAssign().catch(() => {})}>
                  {assignBusy ? '保存中…' : '保存'}
                </button>
                <button className="btn" disabled={assignBusy} onClick={() => setAssignOpen(false)}>
                  取消
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {batchAssignOpen && (
        <div className="modalOverlay" onClick={() => !batchAssignBusy && setBatchAssignOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div className="modalTitle">{'\u6279\u91cf\u5206\u7c7b'}</div>
              <button className="btn btnSmall" disabled={batchAssignBusy} onClick={() => setBatchAssignOpen(false)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              <div className="hint" style={{ marginBottom: 12 }}>
                {'\u5df2\u9009\u8bba\u6587\uff1a'}
                <b>{selectedPaperCount}</b>
                {'\u3002\u9009\u4e2d\u540e\u4f1a\u5c06\u8bba\u6587\u52a0\u5165\u5bf9\u5e94\u8bba\u6587\u96c6\u3002'}
              </div>
              <div className="row" style={{ marginBottom: 10 }}>
                <button className="btn btnSmall" disabled={batchAssignBusy} onClick={() => openCreateCollection()}>
                  新建论文集
                </button>
              </div>
              <div className="list">
                {collections.map((c) => {
                  const checked = !!batchAssignSelected[c.collection_id]
                  return (
                    <div key={c.collection_id} className="itemCard">
                      <div className="split">
                        <div className="itemTitle">{c.name}</div>
                        <label className="row" style={{ gap: 10 }}>
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={(e) => setBatchAssignSelected((m) => ({ ...m, [c.collection_id]: e.target.checked }))}
                          />
                          <span className="kicker">{checked ? '\u5df2\u9009\u4e2d' : '\u672a\u9009\u4e2d'}</span>
                        </label>
                      </div>
                      <div className="itemMeta">
                        <code>{c.collection_id}</code>
                      </div>
                    </div>
                  )
                })}
                {collections.length === 0 && <div className="metaLine">暂无论文集。你可以先点击“新建论文集”。</div>}
              </div>
              <div className="row" style={{ marginTop: 12 }}>
                <button className="btn btnPrimary" disabled={batchAssignBusy || selectedCollectionCount === 0} onClick={() => saveBatchAssign().catch(() => {})}>
                  {batchAssignBusy ? '保存中…' : '保存'}
                </button>
                <button className="btn" disabled={batchAssignBusy} onClick={() => setBatchAssignOpen(false)}>
                  取消
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {batchDeleteOpen && (
        <div className="modalOverlay" onClick={() => !batchDeleteBusy && setBatchDeleteOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div className="modalTitle">{'\u6279\u91cf\u5220\u9664\u8bba\u6587'}</div>
              <button className="btn btnSmall" disabled={batchDeleteBusy} onClick={() => setBatchDeleteOpen(false)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              <div className="hint" style={{ whiteSpace: 'pre-wrap' }}>
                {`\u786e\u5b9a\u5220\u9664\u5df2\u9009\u4e2d\u7684 ${selectedPaperCount} \u7bc7\u8bba\u6587\u5417\uff1f\n\u8fd9\u4f1a\u76f4\u63a5\u5220\u9664\u8fd9\u4e9b\u8bba\u6587\u7684 Neo4j \u56fe\u6570\u636e\uff0c\u5e76\u6e05\u7406\u5bf9\u5e94\u6d3e\u751f\u6587\u4ef6\u3002`}
              </div>
              <div className="row" style={{ marginTop: 12 }}>
                <button className="btn btnDanger" disabled={batchDeleteBusy} onClick={() => confirmBatchDelete().catch(() => {})}>
                  {batchDeleteBusy ? '删除中…' : '确定删除'}
                </button>
                <button className="btn" disabled={batchDeleteBusy} onClick={() => setBatchDeleteOpen(false)}>
                  取消
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {collectionEditOpen && (
        <div className="modalOverlay" onClick={() => !collectionEditBusy && setCollectionEditOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div className="modalTitle">{collectionEditMode === 'create' ? '新建论文集' : '重命名论文集'}</div>
              <button className="btn btnSmall" disabled={collectionEditBusy} onClick={() => setCollectionEditOpen(false)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              <div className="hint" style={{ marginBottom: 10 }}>
                {collectionEditMode === 'create' ? '请输入论文集名称（可随时重命名）。' : '请输入新的论文集名称。'}
              </div>
              <input className="input" value={collectionEditName} onChange={(e) => setCollectionEditName(e.target.value)} placeholder="例如：摩擦/颗粒/悬浮液" />
              <div className="row" style={{ marginTop: 12 }}>
                <button className="btn btnPrimary" disabled={collectionEditBusy || !collectionEditName.trim()} onClick={() => submitCollectionEdit().catch(() => {})}>
                  {collectionEditBusy ? '提交中…' : '确定'}
                </button>
                <button className="btn" disabled={collectionEditBusy} onClick={() => setCollectionEditOpen(false)}>
                  取消
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {collectionDeleteOpen && (
        <div className="modalOverlay" onClick={() => !collectionDeleteBusy && setCollectionDeleteOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div className="modalTitle">删除论文集</div>
              <button className="btn btnSmall" disabled={collectionDeleteBusy} onClick={() => setCollectionDeleteOpen(false)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              <div className="hint">
                确定要删除当前论文集吗？不会删除论文本身，只会删除该“分类容器”。
              </div>
              <div className="row" style={{ marginTop: 12 }}>
                <button className="btn btnDanger" disabled={collectionDeleteBusy} onClick={() => confirmDeleteCollection().catch(() => {})}>
                  {collectionDeleteBusy ? '删除中…' : '确定删除'}
                </button>
                <button className="btn" disabled={collectionDeleteBusy} onClick={() => setCollectionDeleteOpen(false)}>
                  取消
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {paperDeleteOpen && (
        <div className="modalOverlay" onClick={() => !(deleteBusy === paperDeleteId) && setPaperDeleteOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div className="modalTitle">删除论文</div>
              <button className="btn btnSmall" disabled={deleteBusy === paperDeleteId} onClick={() => setPaperDeleteOpen(false)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              <div className="hint" style={{ whiteSpace: 'pre-wrap' }}>
                确定要删除这篇“已导入论文”吗？
                {'\n'}该操作会直接删除该论文在 Neo4j 中的图数据（含关系）以及本地派生文件。
              </div>
              <div className="row" style={{ marginTop: 12 }}>
                <button className="btn btnDanger" disabled={deleteBusy === paperDeleteId} onClick={() => confirmDeletePaper().catch(() => {})}>
                  {deleteBusy === paperDeleteId ? '删除中…' : '确定删除'}
                </button>
                <button className="btn" disabled={deleteBusy === paperDeleteId} onClick={() => setPaperDeleteOpen(false)}>
                  取消
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
