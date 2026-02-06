import { useEffect, useMemo, useState } from 'react'

import { apiGet, apiPost } from '../api'

type PropositionRow = {
  prop_id: string
  canonical_text?: string
  current_state?: string
  current_score?: number
  mention_count?: number
  supports?: number
  challenges?: number
  supersedes?: number
}

type HotspotRow = {
  prop_id: string
  canonical_text?: string
  current_state?: string
  current_score?: number
  challenge_events?: number
  supersede_events?: number
  conflict_events?: number
  source_paper_count?: number
}

type PropositionDetail = {
  proposition?: {
    prop_id?: string
    canonical_text?: string
    current_state?: string
    current_score?: number
    score_updated_at?: string
  }
  events?: Array<{
    event_id?: string
    event_type?: string
    status?: string
    confidence?: number
    strength?: number
    event_time?: string
    origin?: string
    source_prop_id?: string
    target_prop_id?: string
    claim_text?: string
    paper_id?: string
    paper_title?: string
    paper_year?: number
  }>
  neighbors?: Array<{
    relation_type?: string
    target_prop_id?: string
    target_text?: string
    score?: number
    evidence_count?: number
  }>
}

function stateLabel(state: string | undefined) {
  const s = String(state ?? '').toLowerCase()
  if (s === 'stable') return '稳定'
  if (s === 'challenged') return '受挑战'
  if (s === 'superseded') return '被替代'
  return s || '-'
}

function eventLabel(kind: string | undefined) {
  const k = String(kind ?? '').toUpperCase()
  if (k === 'SUPPORTS') return '支持'
  if (k === 'CHALLENGES') return '挑战'
  if (k === 'SUPERSEDES') return '替代'
  return k || '-'
}

export default function EvolutionPage() {
  const [propositions, setPropositions] = useState<PropositionRow[]>([])
  const [hotspots, setHotspots] = useState<HotspotRow[]>([])
  const [selectedId, setSelectedId] = useState<string>('')
  const [detail, setDetail] = useState<PropositionDetail | null>(null)
  const [stateFilter, setStateFilter] = useState<string>('all')
  const [searchQ, setSearchQ] = useState<string>('')
  const [busy, setBusy] = useState<boolean>(false)
  const [detailBusy, setDetailBusy] = useState<boolean>(false)
  const [actionBusy, setActionBusy] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [info, setInfo] = useState<string>('')

  async function loadPropositions() {
    const qs = new URLSearchParams()
    qs.set('limit', '300')
    if (stateFilter !== 'all') qs.set('state', stateFilter)
    if (searchQ.trim()) qs.set('q', searchQ.trim())
    const res = await apiGet<{ propositions: PropositionRow[] }>(`/evolution/propositions?${qs.toString()}`)
    const items = res.propositions ?? []
    setPropositions(items)
    if (!selectedId && items.length) {
      setSelectedId(items[0].prop_id)
    } else if (selectedId && !items.some((x) => x.prop_id === selectedId)) {
      setSelectedId(items[0]?.prop_id ?? '')
    }
  }

  async function loadHotspots() {
    const res = await apiGet<{ hotspots: HotspotRow[] }>('/evolution/hotspots?limit=120&min_events=1')
    setHotspots(res.hotspots ?? [])
  }

  async function loadDetail(propId: string) {
    const pid = String(propId || '').trim()
    if (!pid) {
      setDetail(null)
      return
    }
    setDetailBusy(true)
    try {
      const d = await apiGet<PropositionDetail>(`/evolution/proposition/${encodeURIComponent(pid)}?limit_events=300`)
      setDetail(d)
    } finally {
      setDetailBusy(false)
    }
  }

  async function refreshAll() {
    setBusy(true)
    setError('')
    try {
      await Promise.all([loadPropositions(), loadHotspots()])
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setBusy(false)
    }
  }

  async function submitRebuildEvolution() {
    setActionBusy(true)
    setError('')
    setInfo('')
    try {
      const res = await apiPost<{ task_id: string }>('/tasks/rebuild/evolution', {})
      setInfo(`已提交演化重算任务：${res.task_id ?? ''}`)
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setActionBusy(false)
    }
  }

  useEffect(() => {
    refreshAll().catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    loadPropositions().catch((e: unknown) => setError(String((e as { message?: unknown } | null)?.message ?? e)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stateFilter])

  useEffect(() => {
    loadDetail(selectedId).catch((e: unknown) => setError(String((e as { message?: unknown } | null)?.message ?? e)))
  }, [selectedId])

  const counts = useMemo(() => {
    const out = { total: propositions.length, stable: 0, challenged: 0, superseded: 0 }
    for (const p of propositions) {
      const s = String(p.current_state ?? '').toLowerCase()
      if (s === 'stable') out.stable += 1
      else if (s === 'challenged') out.challenged += 1
      else if (s === 'superseded') out.superseded += 1
    }
    return out
  }, [propositions])

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">演化追踪</h2>
          <div className="pageSubtitle">跨论文命题状态、冲突热点与事件时间线</div>
        </div>
        <div className="pageActions">
          <span className="pill">
            <span className="kicker">命题</span> {counts.total}
          </span>
          <span className="pill">
            <span className="kicker">稳定</span> {counts.stable}
          </span>
          <span className="pill">
            <span className="kicker">受挑战</span> {counts.challenged}
          </span>
          <span className="pill">
            <span className="kicker">被替代</span> {counts.superseded}
          </span>
          <button className="btn" disabled={busy || actionBusy} onClick={() => refreshAll().catch(() => {})}>
            {busy ? '加载中…' : '刷新'}
          </button>
          <button className="btn btnPrimary" disabled={actionBusy} onClick={submitRebuildEvolution}>
            {actionBusy ? '提交中…' : '重算演化关系/状态'}
          </button>
        </div>
      </div>

      {error && <div className="errorBox">{error}</div>}
      {info && <div className="infoBox">{info}</div>}

      <div className="panel">
        <div className="panelHeader">
          <div className="split">
            <div className="panelTitle">过滤</div>
            <div className="row" style={{ gap: 8 }}>
              <select className="select" style={{ width: 160 }} value={stateFilter} onChange={(e) => setStateFilter(e.target.value)}>
                <option value="all">全部状态</option>
                <option value="stable">稳定</option>
                <option value="challenged">受挑战</option>
                <option value="superseded">被替代</option>
              </select>
              <input className="input" style={{ width: 320, maxWidth: '70vw' }} value={searchQ} onChange={(e) => setSearchQ(e.target.value)} placeholder="按命题文本搜索…" />
              <button className="btn" disabled={busy} onClick={() => loadPropositions().catch(() => {})}>
                搜索
              </button>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr 380px', gap: 12, alignItems: 'start' }}>
        <div className="panel">
          <div className="panelHeader">
            <div className="panelTitle">命题列表</div>
          </div>
          <div className="panelBody">
            <div className="list">
              {propositions.map((p) => (
                <button
                  key={p.prop_id}
                  className={`itemCard ${selectedId === p.prop_id ? 'itemCardActive' : ''}`}
                  style={{ textAlign: 'left', width: '100%', cursor: 'pointer' }}
                  onClick={() => setSelectedId(p.prop_id)}
                >
                  <div className="itemTitle">{p.canonical_text || p.prop_id}</div>
                  <div className="itemMeta">
                    状态: {stateLabel(p.current_state)} · 分值: {Number(p.current_score ?? 0).toFixed(3)} · 提及: {p.mention_count ?? 0}
                  </div>
                </button>
              ))}
            </div>
            {!propositions.length && <div className="metaLine">暂无命题数据。</div>}
          </div>
        </div>

        <div className="panel">
          <div className="panelHeader">
            <div className="panelTitle">事件时间线</div>
          </div>
          <div className="panelBody">
            {!selectedId && <div className="metaLine">请先从左侧选择一个命题。</div>}
            {selectedId && detailBusy && <div className="metaLine">加载命题详情中…</div>}
            {selectedId && !detailBusy && detail && (
              <div className="stack">
                <div className="itemCard">
                  <div className="itemTitle">{detail.proposition?.canonical_text || detail.proposition?.prop_id}</div>
                  <div className="itemMeta">
                    状态: {stateLabel(detail.proposition?.current_state)} · 分值: {Number(detail.proposition?.current_score ?? 0).toFixed(3)}
                  </div>
                </div>
                <div className="list">
                  {(detail.events ?? []).map((ev) => (
                    <div key={ev.event_id} className="itemCard">
                      <div className="split">
                        <div className="itemTitle">{eventLabel(ev.event_type)}</div>
                        <span className={`badge ${ev.status === 'accepted' ? 'badgeOk' : 'badgeWarn'}`}>{ev.status || '-'}</span>
                      </div>
                      <div className="itemMeta">
                        时间: {ev.event_time ?? '-'} · 置信: {Number(ev.confidence ?? 0).toFixed(3)} · 强度: {Number(ev.strength ?? 0).toFixed(3)}
                      </div>
                      {(ev.paper_title || ev.paper_id) && (
                        <div className="itemMeta">
                          来源论文: {ev.paper_title || ev.paper_id}
                          {ev.paper_year ? ` (${ev.paper_year})` : ''}
                        </div>
                      )}
                      {ev.claim_text && <div className="itemBody">{ev.claim_text}</div>}
                    </div>
                  ))}
                  {!detail.events?.length && <div className="metaLine">没有事件记录。</div>}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panelHeader">
            <div className="panelTitle">冲突雷达</div>
          </div>
          <div className="panelBody">
            <div className="list">
              {hotspots.map((h) => (
                <div key={h.prop_id} className="itemCard">
                  <div className="split">
                    <div className="itemTitle">{h.canonical_text || h.prop_id}</div>
                    <button className="btn btnSmall" onClick={() => setSelectedId(h.prop_id)}>
                      查看
                    </button>
                  </div>
                  <div className="itemMeta">
                    冲突事件: {h.conflict_events ?? 0} · 挑战: {h.challenge_events ?? 0} · 替代: {h.supersede_events ?? 0}
                  </div>
                  <div className="itemMeta">
                    状态: {stateLabel(h.current_state)} · 分值: {Number(h.current_score ?? 0).toFixed(3)} · 涉及论文: {h.source_paper_count ?? 0}
                  </div>
                </div>
              ))}
            </div>
            {!hotspots.length && <div className="metaLine">暂无冲突热点。</div>}
          </div>
        </div>
      </div>
    </div>
  )
}

