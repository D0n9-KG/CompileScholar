import { useEffect, useMemo, useState } from 'react'

import { apiGet, apiPost } from '../api'

type PropositionGroupRow = {
  group_id: string
  label_text?: string
  proposition_count?: number
  actual_prop_count?: number
  paper_count?: number
  embedding_model?: string
  model_version?: string
  similarity_threshold?: number
  clustering_method?: string
  build_status?: string
  updated_at?: string
  created_at?: string
}

type GroupMemberRow = {
  prop_id: string
  canonical_text?: string
  similarity_score?: number
  paper_count?: number
  current_state?: string
  current_score?: number
}

type GroupDetail = PropositionGroupRow & { propositions?: GroupMemberRow[] }

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
  const [groups, setGroups] = useState<PropositionGroupRow[]>([])
  const [selectedGroupId, setSelectedGroupId] = useState<string>('')
  const [groupDetail, setGroupDetail] = useState<GroupDetail | null>(null)
  const [groupBusy, setGroupBusy] = useState<boolean>(false)
  const [selectedPropId, setSelectedPropId] = useState<string>('')

  const [hotspots, setHotspots] = useState<HotspotRow[]>([])
  const [detail, setDetail] = useState<PropositionDetail | null>(null)
  const [searchQ, setSearchQ] = useState<string>('')
  const [busy, setBusy] = useState<boolean>(false)
  const [detailBusy, setDetailBusy] = useState<boolean>(false)
  const [actionBusy, setActionBusy] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [info, setInfo] = useState<string>('')

  async function loadGroups() {
    const qs = new URLSearchParams()
    qs.set('limit', '300')
    if (searchQ.trim()) qs.set('q', searchQ.trim())
    const res = await apiGet<{ groups: PropositionGroupRow[] }>(`/evolution/groups?${qs.toString()}`)
    const items = res.groups ?? []
    setGroups(items)
    if (!selectedGroupId && items.length) {
      setSelectedGroupId(items[0].group_id)
    } else if (selectedGroupId && !items.some((x) => x.group_id === selectedGroupId)) {
      setSelectedGroupId(items[0]?.group_id ?? '')
    }
  }

  async function loadGroupDetail(groupId: string) {
    const gid = String(groupId || '').trim()
    if (!gid) {
      setGroupDetail(null)
      setSelectedPropId('')
      return
    }
    setGroupBusy(true)
    try {
      const data = await apiGet<GroupDetail>(`/evolution/group/${encodeURIComponent(gid)}?limit_propositions=200`)
      setGroupDetail(data)
      const members = data.propositions ?? []
      if (!selectedPropId && members.length) {
        setSelectedPropId(members[0].prop_id)
      } else if (selectedPropId && !members.some((m) => m.prop_id === selectedPropId)) {
        setSelectedPropId(members[0]?.prop_id ?? '')
      }
    } finally {
      setGroupBusy(false)
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
      await Promise.all([loadGroups(), loadHotspots()])
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setBusy(false)
    }
  }

  async function submitRebuildGroups() {
    setActionBusy(true)
    setError('')
    setInfo('')
    try {
      const res = await apiPost<{ status?: string; groups_created?: number; propositions_clustered?: number; error?: string }>(
        '/evolution/rebuild-groups',
        {},
      )
      setInfo(
        `Groups rebuilt: status=${res.status ?? '-'}, groups=${res.groups_created ?? 0}, propositions=${res.propositions_clustered ?? 0}` +
          (res.error ? `, error=${res.error}` : ''),
      )
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
    loadGroupDetail(selectedGroupId).catch((e: unknown) => setError(String((e as { message?: unknown } | null)?.message ?? e)))
  }, [selectedGroupId])

  useEffect(() => {
    loadDetail(selectedPropId).catch((e: unknown) => setError(String((e as { message?: unknown } | null)?.message ?? e)))
  }, [selectedPropId])

  const counts = useMemo(() => {
    let propositionTotal = 0
    for (const g of groups) {
      propositionTotal += Number(g.actual_prop_count ?? g.proposition_count ?? 0)
    }
    return { groups: groups.length, propositions: propositionTotal }
  }, [groups])

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">演化追踪</h2>
          <div className="pageSubtitle">Group-level clustering + proposition event timeline</div>
        </div>
        <div className="pageActions">
          <span className="pill">
            <span className="kicker">Groups</span> {counts.groups}
          </span>
          <span className="pill">
            <span className="kicker">Props</span> {counts.propositions}
          </span>
          <button className="btn" disabled={busy || actionBusy} onClick={() => refreshAll().catch(() => {})}>
            {busy ? '加载中…' : '刷新'}
          </button>
          <button className="btn btnPrimary" disabled={actionBusy} onClick={submitRebuildGroups}>
            {actionBusy ? 'Submitting...' : 'Rebuild Groups'}
          </button>
        </div>
      </div>

      {error && <div className="errorBox">{error}</div>}
      {info && <div className="infoBox">{info}</div>}

      <div className="panel">
        <div className="panelHeader">
          <div className="split">
            <div className="panelTitle">Search Groups</div>
            <div className="row" style={{ gap: 8 }}>
              <input className="input" style={{ width: 420, maxWidth: '70vw' }} value={searchQ} onChange={(e) => setSearchQ(e.target.value)} placeholder="Search by group label text..." />
              <button className="btn" disabled={busy} onClick={() => loadGroups().catch(() => {})}>
                Search
              </button>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr 380px', gap: 12, alignItems: 'start' }}>
        <div className="panel">
          <div className="panelHeader">
            <div className="panelTitle">命题组列表</div>
          </div>
          <div className="panelBody">
            <div className="list">
              {groups.map((g) => (
                <button
                  key={g.group_id}
                  className={`itemCard ${selectedGroupId === g.group_id ? 'itemCardActive' : ''}`}
                  style={{ textAlign: 'left', width: '100%', cursor: 'pointer' }}
                  onClick={() => setSelectedGroupId(g.group_id)}
                >
                  <div className="itemTitle">{g.label_text || g.group_id}</div>
                  <div className="itemMeta">
                    Group: {g.group_id} · Propositions: {Number(g.actual_prop_count ?? g.proposition_count ?? 0)} · Papers:{' '}
                    {g.paper_count ?? 0}
                  </div>
                </button>
              ))}
            </div>
            {!groups.length && <div className="metaLine">No groups found.</div>}
          </div>
        </div>

        <div className="panel">
          <div className="panelHeader">
            <div className="panelTitle">组内命题 / 事件时间线</div>
          </div>
          <div className="panelBody">
            {!selectedGroupId && <div className="metaLine">Select a group from the left panel.</div>}
            {selectedGroupId && groupBusy && <div className="metaLine">Loading group details...</div>}
            {selectedGroupId && !groupBusy && groupDetail && (
              <div className="stack">
                <div className="itemCard">
                  <div className="itemTitle">{groupDetail.label_text || groupDetail.group_id}</div>
                  <div className="itemMeta">
                    Group ID: {groupDetail.group_id} · Propositions:{' '}
                    {Number(groupDetail.actual_prop_count ?? groupDetail.proposition_count ?? 0)} · Paper count:{' '}
                    {groupDetail.paper_count ?? 0}
                  </div>
                </div>
                <div className="list">
                  {(groupDetail.propositions ?? []).map((p) => (
                    <button
                      key={p.prop_id}
                      className={`itemCard ${selectedPropId === p.prop_id ? 'itemCardActive' : ''}`}
                      style={{ textAlign: 'left', width: '100%', cursor: 'pointer' }}
                      onClick={() => setSelectedPropId(p.prop_id)}
                    >
                      <div className="itemTitle">{p.canonical_text || p.prop_id}</div>
                      <div className="itemMeta">
                        Similarity: {Number(p.similarity_score ?? 0).toFixed(3)} · State: {stateLabel(p.current_state)} · Score:{' '}
                        {Number(p.current_score ?? 0).toFixed(3)}
                      </div>
                    </button>
                  ))}
                  {!groupDetail.propositions?.length && <div className="metaLine">No propositions in this group.</div>}
                </div>
              </div>
            )}

            {!selectedPropId && <div className="metaLine">Select a proposition to view timeline.</div>}
            {selectedPropId && detailBusy && <div className="metaLine">Loading proposition detail...</div>}
            {selectedPropId && !detailBusy && detail && (
              <div className="stack">
                <div className="itemCard">
                  <div className="itemTitle">{detail.proposition?.canonical_text || detail.proposition?.prop_id}</div>
                  <div className="itemMeta">
                    State: {stateLabel(detail.proposition?.current_state)} · Score: {Number(detail.proposition?.current_score ?? 0).toFixed(3)}
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
                          Source paper: {ev.paper_title || ev.paper_id}
                          {ev.paper_year ? ` (${ev.paper_year})` : ''}
                        </div>
                      )}
                      {ev.claim_text && <div className="itemBody">{ev.claim_text}</div>}
                    </div>
                  ))}
                  {!detail.events?.length && <div className="metaLine">No events.</div>}
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
                    <button className="btn btnSmall" onClick={() => setSelectedPropId(h.prop_id)}>
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
