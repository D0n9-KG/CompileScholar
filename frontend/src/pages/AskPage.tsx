import { useEffect, useMemo, useState } from 'react'

import { apiPost } from '../api'
import MarkdownView from '../components/MarkdownView'
import { loadScope, scopeLabel, type Scope } from '../scope'

type EvidenceRow = {
  paper_id?: string
  paper_source?: string
  md_path?: string
  start_line?: number
  end_line?: number
  score?: number
  snippet?: string
}

const EXAMPLES = [
  '这篇论文的主要方法是什么？',
  '这篇论文引入的 Θ（Theta）是什么？它为什么重要？',
  '这篇论文的核心结论是什么？',
  '用一句话概括这篇论文的贡献。',
]

type AskItem = {
  id: string
  question: string
  k: number
  createdAt: number
  status: 'running' | 'done' | 'error'
  answer?: string
  evidence?: EvidenceRow[]
  notice?: string
  insufficientScopeEvidence?: boolean
  error?: string
}

type AskStore = {
  draft: { question: string; k: number }
  currentId?: string
  items: AskItem[]
}

const LS_KEY = 'logickg.ask.v1'
const inflight = new Map<string, Promise<void>>()

function emitChanged() {
  window.dispatchEvent(new Event('logickg:ask_state_changed'))
}

function loadStore(): AskStore {
  try {
    const raw = localStorage.getItem(LS_KEY)
    if (!raw) throw new Error('empty')
    const data = JSON.parse(raw) as Partial<AskStore>
    const items = Array.isArray(data.items) ? (data.items as AskItem[]) : []
    const currentId =
      typeof data.currentId === 'string' ? data.currentId : typeof items[0]?.id === 'string' ? items[0].id : undefined
    return {
      draft: {
        question: String(data.draft?.question ?? EXAMPLES[0]),
        k: Math.max(1, Math.min(20, Number(data.draft?.k ?? 8) || 8)),
      },
      currentId,
      items,
    }
  } catch {
    return { draft: { question: EXAMPLES[0], k: 8 }, items: [], currentId: undefined }
  }
}

function saveStore(s: AskStore) {
  localStorage.setItem(LS_KEY, JSON.stringify(s))
}

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

async function runAsk(item: AskItem) {
  if (inflight.has(item.id)) return

  const p = (async () => {
    try {
      const scope = loadScope()
      const res = await apiPost<{
        answer?: string
        evidence?: EvidenceRow[]
        message?: string
        insufficient_scope_evidence?: boolean
      }>('/rag/ask', {
        question: item.question,
        k: item.k,
        scope:
          scope.mode === 'collection'
            ? { mode: 'collection', collection_id: scope.collectionId ?? '' }
            : scope.mode === 'papers'
              ? { mode: 'papers', paper_ids: scope.paperIds ?? [] }
              : { mode: 'all' },
      })

      const s = loadStore()
      s.items = s.items.map((x) =>
        x.id === item.id
          ? { ...x, status: 'done', answer: res.answer ?? '', evidence: res.evidence ?? [], error: '' }
          : x,
      )
      s.items = s.items.map((x) =>
        x.id === item.id
          ? { ...x, notice: res.message ?? '', insufficientScopeEvidence: Boolean(res.insufficient_scope_evidence) }
          : x,
      )
      saveStore(s)
      emitChanged()
    } catch (e: unknown) {
      const msg = String((e as { message?: unknown } | null)?.message ?? e)
      const s = loadStore()
      s.items = s.items.map((x) => (x.id === item.id ? { ...x, status: 'error', error: msg } : x))
      saveStore(s)
      emitChanged()
    } finally {
      inflight.delete(item.id)
    }
  })()

  inflight.set(item.id, p)
}

export default function AskPage() {
  const [store, setStore] = useState<AskStore>(() => loadStore())
  const [scope, setScope] = useState<Scope>(() => loadScope())

  const current = useMemo(
    () => (store.currentId ? store.items.find((x) => x.id === store.currentId) : undefined),
    [store.currentId, store.items],
  )

  const busy = current?.status === 'running'
  const canAsk = useMemo(() => store.draft.question.trim().length > 0 && !busy, [busy, store.draft.question])

  useEffect(() => {
    function refresh() {
      setStore(loadStore())
      setScope(loadScope())
    }
    window.addEventListener('storage', refresh)
    window.addEventListener('logickg:ask_state_changed', refresh)
    window.addEventListener('logickg:scope_changed', refresh)
    return () => {
      window.removeEventListener('storage', refresh)
      window.removeEventListener('logickg:ask_state_changed', refresh)
      window.removeEventListener('logickg:scope_changed', refresh)
    }
  }, [])

  useEffect(() => {
    if (current?.status === 'running' && !inflight.has(current.id)) {
      // page refresh / service restart: allow user to retry
      const s = loadStore()
      s.items = s.items.map((x) =>
        x.id === current.id ? { ...x, status: 'error', error: '请求已中断（可能是页面刷新或服务重启）。请重新提问。' } : x,
      )
      saveStore(s)
      emitChanged()
    }
  }, [current?.id, current?.status])

  function setDraft(next: Partial<AskStore['draft']>) {
    setStore((prev) => {
      const s: AskStore = {
        ...prev,
        draft: {
          question: next.question ?? prev.draft.question,
          k: next.k ?? prev.draft.k,
        },
      }
      saveStore(s)
      return s
    })
  }

  function ask() {
    const q = store.draft.question.trim()
    if (!q || busy) return

    const id = makeId()
    const item: AskItem = {
      id,
      question: q,
      k: store.draft.k,
      createdAt: Date.now(),
      status: 'running',
      answer: '',
      evidence: [],
      error: '',
    }

    const next: AskStore = {
      ...store,
      currentId: id,
      items: [item, ...store.items].slice(0, 30),
    }

    setStore(next)
    saveStore(next)
    emitChanged()
    void runAsk(item)
  }

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">问答</h2>
          <div className="pageSubtitle">图谱增强问答（GraphRAG：全局 FAISS + DeepSeek）— 返回答案与可追溯证据</div>
          <div className="metaLine">范围：{scopeLabel(scope)}</div>
        </div>
        <div className="pageActions">
          <button className="btn btnPrimary" disabled={!canAsk} onClick={ask}>
            {busy ? '思考中…' : '提问'}
          </button>
        </div>
      </div>

      {current?.status === 'done' && current?.insufficientScopeEvidence && current?.notice && (
        <div className="errorBox">{current.notice}</div>
      )}
      {current?.status === 'error' && current?.error && <div className="errorBox">{current.error}</div>}

      <div className="grid2">
        <div className="panel">
          <div className="panelHeader">
            <div className="panelTitle">问题</div>
          </div>
          <div className="panelBody">
            <div className="stack">
              <textarea
                className="textarea"
                value={store.draft.question}
                onChange={(e) => setDraft({ question: e.target.value })}
                placeholder="输入你的问题…"
              />

              <div className="row">
                <span className="pill">
                  <span className="kicker">k</span>
                  <input
                    className="input"
                    style={{ width: 92 }}
                    type="number"
                    min={1}
                    max={20}
                    value={store.draft.k}
                    onChange={(e) => setDraft({ k: Math.max(1, Math.min(20, Number(e.target.value) || 8)) })}
                  />
                </span>
                <span className="pill">
                  <span className="kicker">提示</span> 导入 / 重建后证据更稳定
                </span>
              </div>

              <div className="row" style={{ gap: 8 }}>
                {EXAMPLES.map((x) => (
                  <button key={x} className="chip" onClick={() => setDraft({ question: x })}>
                    {x}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panelHeader">
            <div className="panelTitle">答案</div>
          </div>
          <div className="panelBody">
            {current?.status === 'running' && <div className="metaLine">思考中…（可切换页面，回答完成后会保留）</div>}
            {current?.status !== 'running' && !(current?.answer ?? '').trim() ? (
              <div className="metaLine">还没有回答。点击“提问”开始。</div>
            ) : (
              <MarkdownView markdown={current?.answer ?? ''} />
            )}
          </div>
        </div>
      </div>

      {(current?.evidence ?? []).length > 0 && (
        <div className="panel" style={{ marginTop: 14 }}>
          <div className="panelHeader">
            <div className="panelTitle">证据</div>
          </div>
          <div className="panelBody">
            <div className="list">
              {(current?.evidence ?? []).map((e, idx) => (
                <div key={idx} className="itemCard">
                  <div className="itemMeta">
                    {e.paper_source ?? ''} · {e.md_path ?? ''}:{e.start_line ?? ''}-{e.end_line ?? ''} · 相似度{' '}
                    {Number(e.score ?? 0).toFixed(3)}
                  </div>
                  <div className="itemBody">
                    <MarkdownView markdown={e.snippet ?? ''} paperId={e.paper_id} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
