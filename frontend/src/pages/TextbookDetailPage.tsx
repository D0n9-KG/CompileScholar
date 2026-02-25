import { useCallback, useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { apiGet } from '../api'

type ChapterRow = {
  chapter_id: string
  chapter_num: number
  title: string
  entity_count: number
  relation_count: number
}

type TextbookDetail = {
  textbook_id: string
  title: string
  authors: string[]
  year: number | null
  edition: string | null
  doc_type: string
  total_chapters: number
  chapters: ChapterRow[]
}

type EntityRow = {
  entity_id: string
  name: string
  entity_type: string
  description: string
}

type RelationRow = {
  source_id: string
  target_id: string
  rel_type: string
}

type ChapterData = {
  entities: EntityRow[]
  relations: RelationRow[]
}

export default function TextbookDetailPage() {
  const { textbookId } = useParams<{ textbookId: string }>()
  const [detail, setDetail] = useState<TextbookDetail | null>(null)
  const [error, setError] = useState('')
  const [selectedChapter, setSelectedChapter] = useState<string>('')
  const [chapterData, setChapterData] = useState<ChapterData | null>(null)
  const [loadingChapter, setLoadingChapter] = useState(false)

  const loadDetail = useCallback(async () => {
    if (!textbookId) return
    setError('')
    try {
      const r = await apiGet<TextbookDetail>(`/textbooks/${encodeURIComponent(textbookId)}`)
      setDetail(r)
    } catch (e: unknown) {
      setError(String((e as { message?: unknown })?.message ?? e))
    }
  }, [textbookId])

  useEffect(() => { loadDetail() }, [loadDetail])

  const loadChapter = useCallback(async (chapterId: string) => {
    if (!textbookId || !chapterId) return
    setLoadingChapter(true)
    setChapterData(null)
    try {
      const r = await apiGet<ChapterData>(`/textbooks/${encodeURIComponent(textbookId)}/chapters/${encodeURIComponent(chapterId)}/entities`)
      setChapterData(r)
    } catch (e: unknown) {
      setError(String((e as { message?: unknown })?.message ?? e))
    } finally {
      setLoadingChapter(false)
    }
  }, [textbookId])

  const handleSelectChapter = (chapterId: string) => {
    setSelectedChapter(chapterId)
    loadChapter(chapterId)
  }

  if (!detail) {
    return <div className="page">{error ? <div className="errorBox">{error}</div> : <div className="metaLine">Loading...</div>}</div>
  }

  const typeColors: Record<string, string> = {
    concept: '#6ec6ff', theory: '#ff9e80', equation: '#b39ddb',
    method: '#81c784', model: '#ffcc80', material: '#90caf9',
    parameter: '#ce93d8', phenomenon: '#ef9a9a', tool: '#80cbc4',
    condition: '#fff59d',
  }

  return (
    <div className="page textbookDetailLayout">
      {/* Left: Chapter tree */}
      <div className="textbookChapterTree">
        <Link to="/textbooks" className="metaLine">← 教科书列表</Link>
        <h2 className="pageTitle" style={{ marginTop: 8, fontSize: 20 }}>{detail.title}</h2>
        <div className="metaLine" style={{ marginTop: 4 }}>
          {detail.authors?.join(', ')} {detail.year ? `(${detail.year})` : ''}
          {detail.edition ? ` · ${detail.edition}` : ''}
        </div>
        <div style={{ marginTop: 14 }}>
          {detail.chapters.map(ch => (
            <div
              key={ch.chapter_id}
              className={`textbookChapterItem${selectedChapter === ch.chapter_id ? ' textbookChapterItem--active' : ''}`}
              onClick={() => handleSelectChapter(ch.chapter_id)}
            >
              <div className="textbookChapterTitle">Ch.{ch.chapter_num}: {ch.title}</div>
              <div className="textbookChapterMeta">{ch.entity_count} 实体 · {ch.relation_count} 关系</div>
            </div>
          ))}
        </div>
      </div>

      {/* Right: Entity list + relations */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {!selectedChapter && <div className="metaLine">选择左侧章节查看实体</div>}
        {loadingChapter && <div className="metaLine">Loading...</div>}
        {chapterData && (
          <>
            <div className="panelTitle" style={{ marginBottom: 12 }}>
              实体 ({chapterData.entities.length}) · 关系 ({chapterData.relations.length})
            </div>
            <div className="entityGrid">
              {chapterData.entities.map(e => (
                <div key={e.entity_id} className="entityCard" style={{ border: `1px solid ${typeColors[e.entity_type] ?? 'rgba(146,168,217,0.25)'}` }}>
                  <div className="entityName" style={{ color: typeColors[e.entity_type] ?? 'rgba(242,248,255,0.94)' }}>{e.name}</div>
                  <div className="kicker">{e.entity_type}</div>
                  {e.description && (
                    <div className="entityDesc">
                      {e.description.length > 120 ? e.description.slice(0, 120) + '...' : e.description}
                    </div>
                  )}
                </div>
              ))}
            </div>
            {chapterData.relations.length > 0 && (
              <div style={{ marginTop: 14 }}>
                <div className="panelTitle" style={{ marginBottom: 8 }}>关系</div>
                <table className="relTable">
                  <thead>
                    <tr>
                      <th>源实体</th>
                      <th>关系</th>
                      <th>目标实体</th>
                    </tr>
                  </thead>
                  <tbody>
                    {chapterData.relations.map((r, i) => {
                      const src = chapterData.entities.find(e => e.entity_id === r.source_id)
                      const tgt = chapterData.entities.find(e => e.entity_id === r.target_id)
                      return (
                        <tr key={i}>
                          <td>{src?.name ?? r.source_id}</td>
                          <td className="metaLine">{r.rel_type}</td>
                          <td>{tgt?.name ?? r.target_id}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
