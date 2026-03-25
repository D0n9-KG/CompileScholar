import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { apiPost } from '../api'
import { useI18n } from '../i18n'
import { invalidateOverviewStatsCache, invalidatePaperDataCache, loadOverviewStatsSnapshot } from '../loaders/panelData'
import {
  OVERVIEW_GRAPH_DEFAULT_LIMIT_EDGES,
  OVERVIEW_GRAPH_DEFAULT_LIMIT_PAPERS,
  invalidateOverviewGraphCache,
  loadOverviewGraph,
} from '../loaders/overview'
import { useGlobalState } from '../state/store'

type Stats = {
  paperCount: number
  researchMoveCount: number
  globalCommunityCount: number
  readyForL3Count: number
  readyForL4Count: number
}

export default function OverviewPanel() {
  const nav = useNavigate()
  const { t } = useI18n()
  const { dispatch } = useGlobalState()
  const [stats, setStats] = useState<Stats>({
    paperCount: 0,
    researchMoveCount: 0,
    globalCommunityCount: 0,
    readyForL3Count: 0,
    readyForL4Count: 0,
  })
  const [ingestPath, setIngestPath] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [refreshingStats, setRefreshingStats] = useState(false)
  const refreshingGraphRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    async function loadGraph() {
      dispatch({ type: 'SET_TRANSITIONING', value: true })
      try {
        const elements = await loadOverviewGraph()
        if (!cancelled) dispatch({ type: 'SET_GRAPH', elements, layout: 'cose' })
      } finally {
        if (!cancelled) dispatch({ type: 'SET_TRANSITIONING', value: false })
      }
    }
    void loadGraph()
    return () => {
      cancelled = true
    }
  }, [dispatch])

  const refreshStats = async (force = false) => {
    setRefreshingStats(true)
    try {
      if (force) invalidateOverviewStatsCache()
      const snapshot = await loadOverviewStatsSnapshot()
      setStats({
        paperCount: snapshot.paperCount,
        researchMoveCount: snapshot.researchMoveCount,
        globalCommunityCount: snapshot.globalCommunityCount,
        readyForL3Count: snapshot.readyForL3Count,
        readyForL4Count: snapshot.readyForL4Count,
      })
    } finally {
      setRefreshingStats(false)
    }
  }

  useEffect(() => {
    void refreshStats().catch(() => {})
  }, [])

  const submitIngest = async () => {
    const path = ingestPath.trim()
    if (!path || busy) return
    setBusy(true)
    setMessage('')
    setError('')
    try {
      const result = await apiPost<{ task_id: string }>('/tasks/ingest/path', { path })
      setMessage(t(`已提交导入任务：${result.task_id}`, `Ingest task submitted: ${result.task_id}`))
      invalidateOverviewGraphCache()
      invalidatePaperDataCache()
      void refreshStats(true).catch(() => {})
    } catch (cause: unknown) {
      setError(String((cause as { message?: unknown } | null)?.message ?? cause))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="kgPanelBody kgStack">
      <div className="kgStatGrid">
        <div className="kgStatCard"><div className="kgStatLabel">{t('论文数', 'Papers')}</div><div className="kgStatValue">{stats.paperCount}</div></div>
        <div className="kgStatCard"><div className="kgStatLabel">{t('ResearchMove', 'Research Moves')}</div><div className="kgStatValue">{stats.researchMoveCount}</div></div>
        <div className="kgStatCard"><div className="kgStatLabel">{t('跨论文社区', 'Cross-paper Communities')}</div><div className="kgStatValue">{stats.globalCommunityCount}</div></div>
        <div className="kgStatCard"><div className="kgStatLabel">{t('L3 就绪', 'L3 Ready')}</div><div className="kgStatValue">{stats.readyForL3Count}</div></div>
        <div className="kgStatCard"><div className="kgStatLabel">{t('L4 就绪', 'L4 Ready')}</div><div className="kgStatValue">{stats.readyForL4Count}</div></div>
      </div>

      <div className="kgCard">
        <div className="kgCardTitle">{t('导入中心', 'Import Center')}</div>
        <div className="kgStack" style={{ marginTop: 8 }}>
          <label className="sr-only" htmlFor="overview-ingest-path">{t('论文目录路径', 'Paper folder path')}</label>
          <input
            id="overview-ingest-path"
            name="overview_ingest_path"
            className="kgInput"
            aria-label={t('论文目录路径', 'Paper folder path')}
            placeholder={t('输入要导入的论文目录...', 'Enter the paper folder path...')}
            value={ingestPath}
            onChange={(event) => setIngestPath(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') void submitIngest()
            }}
          />
          <button className="kgBtn kgBtn--primary kgBtn--sm" type="button" disabled={busy || !ingestPath.trim()} onClick={() => void submitIngest()}>
            {busy ? t('提交中...', 'Submitting...') : t('快速导入', 'Quick Ingest')}
          </button>
          <button className="kgBtn kgBtn--sm" type="button" onClick={() => nav('/ingest')}>
            {t('打开完整导入工作台', 'Open Full Ingest Workbench')}
          </button>
          {message ? <div style={{ fontSize: 10.5, color: 'var(--success)' }}>{message}</div> : null}
          {error ? <div style={{ fontSize: 10.5, color: 'var(--danger)' }}>{error}</div> : null}
        </div>
      </div>

      <div className="kgRow" style={{ flexWrap: 'wrap' }}>
        <button
          className="kgBtn kgBtn--sm"
          type="button"
          onClick={() => {
            if (refreshingGraphRef.current) return
            refreshingGraphRef.current = true
            invalidateOverviewGraphCache()
            void loadOverviewGraph(OVERVIEW_GRAPH_DEFAULT_LIMIT_PAPERS, OVERVIEW_GRAPH_DEFAULT_LIMIT_EDGES, { force: true })
              .then((elements) => dispatch({ type: 'SET_GRAPH', elements, layout: 'cose' }))
              .finally(() => {
                refreshingGraphRef.current = false
              })
          }}
        >
          {t('刷新图谱', 'Refresh Graph')}
        </button>
        <button className="kgBtn kgBtn--sm" type="button" disabled={refreshingStats} onClick={() => void refreshStats(true).catch(() => {})}>
          {refreshingStats ? t('刷新中...', 'Refreshing...') : t('刷新统计', 'Refresh Stats')}
        </button>
      </div>
    </div>
  )
}
