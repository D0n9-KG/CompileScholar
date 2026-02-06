import { useEffect, useMemo, useState } from 'react'
import { apiGet, apiPost } from '../api'
import { TERMS } from '../ui/terms'

type TaskRow = {
  task_id: string
  type: string
  status: string
  stage?: string
  progress?: number
  message?: string | null
  error?: string | null
  created_at?: string
  started_at?: string | null
  finished_at?: string | null
}

function badgeClass(status: string) {
  if (status === 'succeeded') return 'badge badgeOk'
  if (status === 'failed') return 'badge badgeDanger'
  if (status === 'running') return 'badge badgeWarn'
  if (status === 'queued') return 'badge badgeWarn'
  return 'badge'
}

function statusLabel(status: string) {
  if (status === 'queued') return '排队中'
  if (status === 'running') return '进行中'
  if (status === 'succeeded') return '成功'
  if (status === 'failed') return '失败'
  if (status === 'canceled') return '已取消'
  return status
}

function typeLabel(type: string) {
  if (type === 'ingest_path') return '导入（旧：本地路径）'
  if (type === 'ingest_upload_ready') return '导入（上传可导入项）'
  if (type === 'upload_replace') return '替换论文（按 DOI）'
  if (type === 'rebuild_paper') return '重建论文'
  if (type === 'rebuild_faiss') return '重建全局 FAISS'
  if (type === 'rebuild_all') return '全链路重建（所有论文）'
  if (type === 'rebuild_similarity') return '重建相似度关系'
  if (type === 'rebuild_evolution') return '重算演化关系/状态'
  if (type === 'update_similarity_paper') return '更新单论文相似度'
  return type
}

function stageLabel(stage: string | null | undefined) {
  const s = String(stage ?? '')
  if (!s) return ''
  if (s.includes('crossref')) return `${TERMS.crossref} 解析`
  if (s.includes('neo4j_clear')) return '清理 Neo4j'
  if (s.includes('neo4j_write')) return '写入 Neo4j'
  if (s.includes('llm')) return `${TERMS.llm} 抽取`
  if (s.includes('faiss')) return `${TERMS.faiss} 重建`
  if (s.includes('evolution')) return '演化关系/状态重算'
  if (s === 'done') return '完成'
  if (s === 'canceled') return '已取消'
  if (s === 'failed') return '失败'
  return s
}

export default function TasksPage() {
  const [tasks, setTasks] = useState<TaskRow[]>([])
  const [error, setError] = useState<string>('')
  const [info, setInfo] = useState<string>('')
  const [busy, setBusy] = useState<string>('')
  const [actionBusy, setActionBusy] = useState<string>('')
  const [query, setQuery] = useState<string>('')
  const [statusFilter, setStatusFilter] = useState<string>('all')

  async function refresh() {
    const r = await apiGet<{ tasks: TaskRow[] }>('/tasks?limit=120&keep_finished=10&prune_finished=true')
    setTasks(r.tasks ?? [])
  }

  useEffect(() => {
    refresh().catch((e: unknown) => setError(String((e as { message?: unknown } | null)?.message ?? e)))
  }, [])

  const hasActive = useMemo(() => tasks.some((t) => ['queued', 'running'].includes(String(t.status))), [tasks])

  useEffect(() => {
    let canceled = false
    let inFlight = false

    async function tick() {
      if (canceled || inFlight) return
      inFlight = true
      try {
        const r = await apiGet<{ tasks: TaskRow[] }>('/tasks?limit=120&keep_finished=10&prune_finished=true')
        if (!canceled) setTasks(r.tasks ?? [])
      } catch {
        // keep polling silent
      } finally {
        inFlight = false
      }
    }

    const iv = setInterval(() => tick().catch(() => {}), hasActive ? 1200 : 5000)
    return () => {
      canceled = true
      clearInterval(iv)
    }
  }, [hasActive])

  async function cancel(taskId: string) {
    setBusy(taskId)
    setError('')
    try {
      await apiPost<Record<string, unknown>>(`/tasks/${encodeURIComponent(taskId)}/cancel`, {})
      await refresh()
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setBusy('')
    }
  }

  async function submitRebuildFaiss() {
    setActionBusy('rebuild_faiss')
    setError('')
    setInfo('')
    try {
      const res = await apiPost<{ task_id: string }>('/tasks/rebuild/faiss', {})
      setInfo(`已提交任务：重建全局 FAISS（${res.task_id ?? ''}）`)
      await refresh()
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setActionBusy('')
    }
  }

  async function submitRebuildAll() {
    if (!window.confirm('确定要“全链路重建（所有论文）”吗？\n这会重新解析/抽取并写图谱，可能耗时较长。')) return
    setActionBusy('rebuild_all')
    setError('')
    setInfo('')
    try {
      const res = await apiPost<{ task_id: string }>('/tasks/rebuild/all', {})
      setInfo(`已提交任务：全链路重建（${res.task_id ?? ''}）`)
      await refresh()
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setActionBusy('')
    }
  }

  async function submitRebuildEvolution() {
    setActionBusy('rebuild_evolution')
    setError('')
    setInfo('')
    try {
      const res = await apiPost<{ task_id: string }>('/tasks/rebuild/evolution', {})
      setInfo(`已提交任务：重算演化关系/状态（${res.task_id ?? ''}）`)
      await refresh()
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setActionBusy('')
    }
  }

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return tasks.filter((t) => {
      if (statusFilter !== 'all' && t.status !== statusFilter) return false
      if (!q) return true
      const hay = `${t.task_id} ${t.type} ${t.status} ${t.stage ?? ''} ${t.message ?? ''} ${t.error ?? ''}`.toLowerCase()
      return hay.includes(q)
    })
  }, [query, statusFilter, tasks])

  const runningCount = useMemo(() => tasks.filter((t) => ['queued', 'running'].includes(t.status)).length, [tasks])

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">任务</h2>
          <div className="pageSubtitle">后台队列任务（导入 / 替换 / 重建 / 演化重算）</div>
        </div>
        <div className="pageActions">
          <span className="pill">
            <span className="kicker">进行中</span> {runningCount}
          </span>
          <button className="btn" disabled={!!actionBusy} onClick={submitRebuildFaiss}>
            {actionBusy === 'rebuild_faiss' ? '提交中…' : '重建全局 FAISS'}
          </button>
          <button className="btn btnDanger" disabled={!!actionBusy} onClick={submitRebuildAll}>
            {actionBusy === 'rebuild_all' ? '提交中…' : '全链路重建'}
          </button>
          <button className="btn" disabled={!!actionBusy} onClick={submitRebuildEvolution}>
            {actionBusy === 'rebuild_evolution' ? '提交中…' : '重算演化关系/状态'}
          </button>
          <button className="btn" onClick={() => refresh().catch((e: unknown) => setError(String((e as { message?: unknown } | null)?.message ?? e)))}>
            刷新
          </button>
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
            <div className="panelTitle">队列</div>
            <div className="row">
              <select className="select" style={{ width: 160 }} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                <option value="all">全部</option>
                <option value="queued">排队中</option>
                <option value="running">进行中</option>
                <option value="succeeded">成功</option>
                <option value="failed">失败</option>
                <option value="canceled">已取消</option>
              </select>
              <input className="input" style={{ width: 260, maxWidth: '70vw' }} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="搜索任务…" />
            </div>
          </div>
        </div>
        <div className="panelBody">
          <div className="list">
            {filtered.map((t) => {
              const pct = Math.round(Math.max(0, Math.min(1, Number(t.progress ?? 0))) * 100)
              const cancelable = ['queued', 'running'].includes(t.status)
              return (
                <div key={t.task_id} className="itemCard">
                  <div className="split">
                    <div className="itemTitle" style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                      <code>{t.task_id}</code>
                      <span className={badgeClass(t.status)} title={t.status}>
                        {statusLabel(t.status)}
                      </span>
                      <span className="badge" title={t.type}>
                        {typeLabel(t.type)}
                      </span>
                    </div>
                    <button className="btn btnSmall btnDanger" disabled={!cancelable || busy === t.task_id} onClick={() => cancel(t.task_id)}>
                      取消
                    </button>
                  </div>

                  <div className="itemMeta">
                    阶段: <code title={t.stage ?? ''}>{stageLabel(t.stage)}</code> · 进度: {pct}%
                  </div>

                  <div className="progress" style={{ marginTop: 10 }}>
                    <div className="progressBar" style={{ width: `${pct}%` }} />
                  </div>

                  {(t.message || t.error) && <div className="itemBody">{t.error ? `ERROR: ${t.error}` : t.message}</div>}
                </div>
              )
            })}
          </div>
          {filtered.length === 0 && <div className="metaLine">暂无任务。</div>}
        </div>
      </div>
    </div>
  )
}
