import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiDelete, apiGet, apiPost } from '../api'

type TextbookRow = {
  textbook_id: string
  title: string
  authors: string[]
  year: number | null
  doc_type: string
  chapter_count: number
  entity_count: number
  ingested: string
}

export default function TextbooksPage() {
  const [textbooks, setTextbooks] = useState<TextbookRow[]>([])
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')

  // Ingest form
  const [showIngest, setShowIngest] = useState(false)
  const [ingestPath, setIngestPath] = useState('')
  const [ingestTitle, setIngestTitle] = useState('')
  const [ingestAuthors, setIngestAuthors] = useState('')
  const [ingestYear, setIngestYear] = useState('')
  const [ingestDocType, setIngestDocType] = useState('textbook')
  const [ingestBusy, setIngestBusy] = useState(false)

  // Delete
  const [deleteId, setDeleteId] = useState('')
  const [deleteBusy, setDeleteBusy] = useState(false)

  const reload = useCallback(async () => {
    setError('')
    const r = await apiGet<{ textbooks: TextbookRow[] }>('/textbooks?limit=200')
    setTextbooks(r.textbooks ?? [])
  }, [])

  useEffect(() => {
    reload().catch((e: unknown) => setError(String((e as { message?: unknown })?.message ?? e)))
  }, [reload])

  const handleIngest = async () => {
    if (!ingestPath.trim() || !ingestTitle.trim()) return
    setIngestBusy(true)
    setError('')
    try {
      const authors = ingestAuthors.split(',').map(a => a.trim()).filter(Boolean)
      const r = await apiPost<{ task_id: string }>('/textbooks/ingest', {
        path: ingestPath.trim(),
        title: ingestTitle.trim(),
        authors,
        year: ingestYear ? Number(ingestYear) : null,
        doc_type: ingestDocType,
      })
      setInfo(`Task submitted: ${r.task_id}`)
      setShowIngest(false)
      setIngestPath('')
      setIngestTitle('')
      setIngestAuthors('')
      setIngestYear('')
    } catch (e: unknown) {
      setError(String((e as { message?: unknown })?.message ?? e))
    } finally {
      setIngestBusy(false)
    }
  }

  const handleDelete = async (id: string) => {
    setDeleteBusy(true)
    try {
      await apiDelete(`/textbooks/${encodeURIComponent(id)}`)
      setDeleteId('')
      await reload()
    } catch (e: unknown) {
      setError(String((e as { message?: unknown })?.message ?? e))
    } finally {
      setDeleteBusy(false)
    }
  }

  const handleFusionLink = async (id: string) => {
    try {
      const r = await apiPost<{ ok: boolean; entities: number; propositions: number }>('/textbooks/fusion/link', { textbook_id: id })
      setInfo(`Fusion: ${r.propositions} propositions created from ${r.entities} entities`)
    } catch (e: unknown) {
      setError(String((e as { message?: unknown })?.message ?? e))
    }
  }

  return (
    <div className="page">
      <div className="pageHeader">
        <h2 className="pageTitle">教科书</h2>
        <div className="pageActions">
          <button className="btn" onClick={() => setShowIngest(!showIngest)}>
            {showIngest ? '取消' : '+ 导入教科书'}
          </button>
        </div>
      </div>

      {error && <div className="errorBox">{error}</div>}
      {info && <div className="infoBox">{info}</div>}

      {showIngest && (
        <div className="panel" style={{ maxWidth: 500, marginBottom: 14 }}>
          <div className="panelHeader"><div className="panelTitle">导入教科书</div></div>
          <div className="panelBody stack">
            <div>
              <label className="kicker">Markdown 路径</label>
              <input className="input" value={ingestPath} onChange={e => setIngestPath(e.target.value)} placeholder="/path/to/textbook.md" />
            </div>
            <div>
              <label className="kicker">书名</label>
              <input className="input" value={ingestTitle} onChange={e => setIngestTitle(e.target.value)} placeholder="DEM Fundamentals" />
            </div>
            <div>
              <label className="kicker">作者 (逗号分隔)</label>
              <input className="input" value={ingestAuthors} onChange={e => setIngestAuthors(e.target.value)} placeholder="Author A, Author B" />
            </div>
            <div className="row">
              <div style={{ flex: 1 }}>
                <label className="kicker">年份</label>
                <input className="input" value={ingestYear} onChange={e => setIngestYear(e.target.value)} placeholder="2020" />
              </div>
              <div style={{ flex: 1 }}>
                <label className="kicker">类型</label>
                <select className="select" value={ingestDocType} onChange={e => setIngestDocType(e.target.value)}>
                  <option value="textbook">教科书</option>
                  <option value="standard">标准</option>
                  <option value="specification">规范</option>
                </select>
              </div>
            </div>
            <button className="btn btnPrimary" onClick={handleIngest} disabled={ingestBusy || !ingestPath.trim() || !ingestTitle.trim()}>
              {ingestBusy ? '提交中...' : '提交摄入任务'}
            </button>
          </div>
        </div>
      )}

      {textbooks.length === 0 ? (
        <div className="metaLine">暂无教科书。点击「导入教科书」开始。</div>
      ) : (
        <div className="textbookGrid">
          {textbooks.map(tb => (
            <div key={tb.textbook_id} className="itemCard">
              <div className="itemTitle">
                <Link to={`/textbooks/${encodeURIComponent(tb.textbook_id)}`}>{tb.title}</Link>
              </div>
              <div className="metaLine" style={{ marginTop: 4 }}>
                {tb.authors?.join(', ') || '—'} {tb.year ? `(${tb.year})` : ''}
              </div>
              <div className="row" style={{ marginTop: 8, gap: 12 }}>
                <span className="badge">{tb.chapter_count} 章</span>
                <span className="badge">{tb.entity_count} 实体</span>
                <span className="badge" style={{ textTransform: 'capitalize' }}>{tb.doc_type}</span>
              </div>
              <div className="row" style={{ marginTop: 8 }}>
                <button className="btn btnSmall" onClick={() => handleFusionLink(tb.textbook_id)}>融合到演化层</button>
                <button className="btn btnSmall btnDanger" onClick={() => setDeleteId(tb.textbook_id)}>删除</button>
              </div>
              {deleteId === tb.textbook_id && (
                <div className="row" style={{ marginTop: 8 }}>
                  <span className="metaLine">确认删除？</span>
                  <button className="btn btnSmall btnDanger" onClick={() => handleDelete(tb.textbook_id)} disabled={deleteBusy}>
                    {deleteBusy ? '删除中...' : '确认'}
                  </button>
                  <button className="btn btnSmall" onClick={() => setDeleteId('')}>取消</button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
