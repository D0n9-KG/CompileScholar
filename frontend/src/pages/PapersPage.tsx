import { useEffect, useMemo, useState } from 'react'
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
  const [deleteBusy, setDeleteBusy] = useState<string>('')
  const [paperDeleteOpen, setPaperDeleteOpen] = useState<boolean>(false)
  const [paperDeleteId, setPaperDeleteId] = useState<string>('')

  useEffect(() => {
    reloadCollections()
      .then(() => reloadPapers())
      .catch((e: unknown) => setError(String((e as { message?: unknown } | null)?.message ?? e)))
  }, [])

  useEffect(() => {
    reloadPapers().catch((e: unknown) => setError(String((e as { message?: unknown } | null)?.message ?? e)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collectionFilter])

  async function reloadCollections() {
    const r = await apiGet<{ collections: CollectionRow[] }>('/collections?limit=200')
    setCollections(r.collections ?? [])
  }

  async function reloadPapers() {
    setError('')
    const cid = collectionFilter === 'all' ? '' : collectionFilter
    const qs = cid ? `&collection_id=${encodeURIComponent(cid)}` : ''
    const r = await apiGet<{ papers: PaperRow[] }>(`/graph/papers?limit=600${qs}`)
    setPapers(r.papers ?? [])
  }

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
      setInfo('已删除论文（回退为 stub）。建议随后重建全局 FAISS 以清理检索索引。')
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
          <div className="panelTitle">列表</div>
        </div>
        <div className="panelBody">
          <div className="list">
            {filtered.map((p) => (
              <div key={p.paper_id} className="itemCard">
                <div className="split">
                  <div className="itemTitle">
                    <Link to={`/paper/${encodeURIComponent(p.paper_id)}`}>{p.title ?? p.paper_source}</Link>
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
                {'\n'}该操作会删除该论文的抽取结果/派生文件，并把论文回退为 stub（用于保留其它论文对它的引用）。
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
