import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import { BrowserRouter, Navigate, Route, Routes, useNavigate } from 'react-router-dom'

import GraphCanvas from './components/GraphCanvas'
import LeftPanel from './components/LeftPanel'
import PageWorkbench from './components/PageWorkbench'
import RightPanel from './components/RightPanel'
import StatusBar from './components/StatusBar'
import TopBar from './components/TopBar'
import { resolveOverview3DPanelState, type OverviewMode } from './components/overview3dLayout'
import './components/layout.css'
import { I18nProvider, useI18n } from './i18n'
import ConfigCenterPage from './pages/ConfigCenterPage'
import IngestPage from './pages/IngestPage'
import OpsWorkbench from './pages/OpsWorkbench'
import PaperDetailPage from './pages/PaperDetailPage'
import TasksPage from './pages/TasksPage'
import TextbookDetailPage from './pages/TextbookDetailPage'
import UnresolvedPage from './pages/UnresolvedPage'
import { GlobalStateProvider, useGlobalState } from './state/store'
import type { ModuleId, SelectedNode } from './state/types'

type WorkspacePreset = 'focus' | 'balanced' | 'analysis'
type VisibleGraphStats = { nodeCount: number; edgeCount: number }

function clamp(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min
  if (value < min) return min
  if (value > max) return max
  return value
}

function shortText(value: string, max = 30) {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim()
  if (!text || text.length <= max) return text
  return `${text.slice(0, Math.max(1, max - 3))}...`
}

function Shell() {
  const { state, dispatch } = useGlobalState()
  const { t } = useI18n()
  const { activeModule, graphElements, graphLayout, layoutTrigger, transitioning, selectedNode } = state
  const nav = useNavigate()

  const [leftCollapsed, setLeftCollapsed] = useState(false)
  const [rightCollapsed, setRightCollapsed] = useState(false)
  const [workspacePreset, setWorkspacePreset] = useState<WorkspacePreset>('balanced')
  const [overviewGraphMode, setOverviewGraphMode] = useState<OverviewMode>('3d')
  const [leftDrawerOpen, setLeftDrawerOpen] = useState(false)
  const [rightDrawerOpen, setRightDrawerOpen] = useState(false)
  const [leftWidth, setLeftWidth] = useState(300)
  const [rightWidth, setRightWidth] = useState(340)
  const [resizing, setResizing] = useState<null | 'left' | 'right'>(null)
  const [visibleGraphStats, setVisibleGraphStats] = useState<VisibleGraphStats>({ nodeCount: 0, edgeCount: 0 })

  const panelState = resolveOverview3DPanelState({
    activeModule,
    overviewMode: overviewGraphMode,
    hasGraphData: graphElements.length > 0,
    leftCollapsed,
    rightCollapsed,
    leftDrawerOpen,
    rightDrawerOpen,
  })

  const floatingPanelMode = panelState.immersive
  const layoutLeftCollapsed = panelState.layoutLeftCollapsed
  const layoutRightCollapsed = panelState.layoutRightCollapsed

  const handleSelectNode = useCallback(
    (node: SelectedNode | null) => {
      dispatch({ type: 'SET_SELECTED', node })
      if (!node) return
      if (floatingPanelMode) {
        setRightDrawerOpen(true)
        return
      }
      if (rightCollapsed) setRightCollapsed(false)
    },
    [dispatch, floatingPanelMode, rightCollapsed],
  )

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (workspacePreset === 'focus') {
        setLeftCollapsed(true)
        setRightCollapsed(true)
        return
      }
      if (workspacePreset === 'analysis') {
        setLeftCollapsed(false)
        setRightCollapsed(false)
        setLeftWidth(320)
        setRightWidth(430)
        return
      }
      setLeftCollapsed(false)
      setRightCollapsed(false)
      setLeftWidth(300)
      setRightWidth(340)
    }, 0)
    return () => window.clearTimeout(timer)
  }, [workspacePreset])

  useEffect(() => {
    if (!resizing) return
    const handleMove = (evt: MouseEvent) => {
      if (resizing === 'left') setLeftWidth(clamp(evt.clientX - 8, 240, 620))
      else setRightWidth(clamp(window.innerWidth - evt.clientX - 8, 280, 620))
    }
    const handleUp = () => setResizing(null)
    document.body.style.userSelect = 'none'
    window.addEventListener('mousemove', handleMove)
    window.addEventListener('mouseup', handleUp)
    return () => {
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('mouseup', handleUp)
    }
  }, [resizing])

  const frameStyle = {
    '--left-panel-w': `${leftWidth}px`,
    '--right-panel-w': `${rightWidth}px`,
  } as CSSProperties

  const frameClass = [
    'kgFrame',
    activeModule === 'ask' ? 'is-ask-mode' : '',
    layoutLeftCollapsed ? 'left-collapsed' : '',
    layoutRightCollapsed ? 'right-collapsed' : '',
    floatingPanelMode ? 'is-floating-panel-mode' : '',
    resizing ? 'is-resizing' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div className="kgShell">
      <TopBar graphStats={visibleGraphStats} />
      <div className="kgWorkBar">
        <div className="kgWorkBarPrimary">
          <button className="kgBtn kgBtn--sm kgBtn--primary" type="button" onClick={() => nav('/ingest')}>
            {t('导入中心', 'Import Center')}
          </button>
          <button className="kgBtn kgBtn--sm" type="button" onClick={() => dispatch({ type: 'RELAYOUT' })}>
            {t('重新布局', 'Re-layout')}
          </button>
        </div>
        <div className="kgWorkBarSecondary">
          <div className="kgPresetWrap">
            <button
              className={`kgBtn kgBtn--sm${workspacePreset === 'focus' ? ' kgBtn--primary' : ''}`}
              type="button"
              onClick={() => setWorkspacePreset('focus')}
            >
              {t('专注', 'Focus')}
            </button>
            <button
              className={`kgBtn kgBtn--sm${workspacePreset === 'balanced' ? ' kgBtn--primary' : ''}`}
              type="button"
              onClick={() => setWorkspacePreset('balanced')}
            >
              {t('均衡', 'Balanced')}
            </button>
            <button
              className={`kgBtn kgBtn--sm${workspacePreset === 'analysis' ? ' kgBtn--primary' : ''}`}
              type="button"
              onClick={() => setWorkspacePreset('analysis')}
            >
              {t('分析', 'Analysis')}
            </button>
          </div>
          <div className="kgWorkHint">
            {selectedNode
              ? t(`当前节点: ${shortText(selectedNode.label)}`, `Current Node: ${shortText(selectedNode.label)}`)
              : t('未选中节点', 'No Node Selected')}
          </div>
        </div>
      </div>

      <div className={frameClass} style={frameStyle}>
        <LeftPanel
          collapsed={panelState.leftPanelCollapsed}
          floating={floatingPanelMode}
          onToggle={() => (floatingPanelMode ? setLeftDrawerOpen((value) => !value) : setLeftCollapsed((value) => !value))}
        />
        <div
          className={`kgResize kgResize--left${layoutLeftCollapsed ? ' is-hidden' : ''}`}
          onMouseDown={() => {
            if (!layoutLeftCollapsed) setResizing('left')
          }}
          title={t('拖动调整左侧面板宽度', 'Drag to resize left panel')}
        />
        <GraphCanvas
          elements={graphElements}
          layout={graphLayout}
          layoutTrigger={layoutTrigger}
          overviewMode={overviewGraphMode}
          onOverviewModeChange={setOverviewGraphMode}
          transitioning={transitioning}
          onSelectNode={handleSelectNode}
          onVisibleGraphStatsChange={setVisibleGraphStats}
        />
        <div
          className={`kgResize kgResize--right${layoutRightCollapsed ? ' is-hidden' : ''}`}
          onMouseDown={() => {
            if (!layoutRightCollapsed) setResizing('right')
          }}
          title={t('拖动调整右侧面板宽度', 'Drag to resize right panel')}
        />
        <RightPanel
          collapsed={panelState.rightPanelCollapsed}
          floating={floatingPanelMode}
          onToggle={() => (floatingPanelMode ? setRightDrawerOpen((value) => !value) : setRightCollapsed((value) => !value))}
        />
      </div>
      <StatusBar graphStats={visibleGraphStats} />
    </div>
  )
}

function ShellRoute({ module }: { module?: ModuleId }) {
  const { state, switchModule } = useGlobalState()
  const initializedRef = useRef(false)

  useEffect(() => {
    if (initializedRef.current) return
    initializedRef.current = true
    if (module && state.activeModule !== module) switchModule(module)
  }, [module, state.activeModule, switchModule])

  return <Shell />
}

function PaperDetailWrapper() {
  const nav = useNavigate()
  const { t } = useI18n()
  return (
    <div style={{ height: '100vh', overflow: 'auto', background: 'var(--bg)', color: 'var(--text)', padding: 20 }}>
      <button className="kgBtn kgBtn--sm" type="button" onClick={() => nav(-1)} style={{ marginBottom: 16 }}>
        {t('返回图谱', 'Back to Graph')}
      </button>
      <PaperDetailPage />
    </div>
  )
}

function TextbookDetailWrapper() {
  const nav = useNavigate()
  const { t } = useI18n()
  return (
    <div style={{ height: '100vh', overflow: 'auto', background: 'var(--bg)', color: 'var(--text)', padding: 20 }}>
      <button className="kgBtn kgBtn--sm" type="button" onClick={() => nav(-1)} style={{ marginBottom: 16 }}>
        {t('返回图谱', 'Back to Graph')}
      </button>
      <TextbookDetailPage />
    </div>
  )
}

function AppRoutes() {
  const { t } = useI18n()
  return (
    <Routes>
      <Route path="/ask" element={<ShellRoute module="ask" />} />
      <Route path="/ask/workbench" element={<Navigate to="/ask" replace />} />
      <Route path="/fusion" element={<Navigate to="/ask" replace />} />
      <Route
        path="/ops"
        element={
          <PageWorkbench title={t('运维工作台', 'Operations Workbench')}>
            <OpsWorkbench />
          </PageWorkbench>
        }
      />
      <Route
        path="/tasks"
        element={
          <PageWorkbench title={t('任务队列', 'Task Queue')}>
            <TasksPage />
          </PageWorkbench>
        }
      />
      <Route
        path="/config-center"
        element={
          <PageWorkbench title={t('配置中心', 'Config Center')}>
            <ConfigCenterPage />
          </PageWorkbench>
        }
      />
      <Route
        path="/unresolved"
        element={
          <PageWorkbench title={t('未解析引文', 'Unresolved Cites')}>
            <UnresolvedPage />
          </PageWorkbench>
        }
      />
      <Route
        path="/ingest"
        element={
          <PageWorkbench title={t('导入中心', 'Import Center')}>
            <IngestPage />
          </PageWorkbench>
        }
      />
      <Route path="/imported-sources" element={<Navigate to="/ingest" replace />} />
      <Route path="/discovery" element={<Navigate to="/ops" replace />} />
      <Route path="/paper/:paperId" element={<PaperDetailWrapper />} />
      <Route path="/textbooks/:textbookId" element={<TextbookDetailWrapper />} />
      <Route path="*" element={<ShellRoute />} />
    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <I18nProvider>
        <GlobalStateProvider>
          <AppRoutes />
        </GlobalStateProvider>
      </I18nProvider>
    </BrowserRouter>
  )
}
