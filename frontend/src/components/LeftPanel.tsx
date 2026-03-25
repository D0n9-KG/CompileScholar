import { Suspense, lazy, useEffect, useRef, useState } from 'react'
import { useI18n } from '../i18n'
import { useGlobalState } from '../state/store'

const OverviewPanel = lazy(() => import('../panels/OverviewPanel'))
const PapersPanel = lazy(() => import('../panels/PapersPanel'))
const AskPanel = lazy(() => import('../panels/AskPanel'))
const TextbooksPanel = lazy(() => import('../panels/TextbooksPanel'))
const OpsPanel = lazy(() => import('../panels/OpsPanel'))

const PANEL_META: Record<string, { badge: string; zh: string; en: string }> = {
  overview: { badge: '总', zh: '总览', en: 'Overview' },
  papers: { badge: '文', zh: '论文', en: 'Papers' },
  ask: { badge: '问', zh: '问答', en: 'Ask' },
  textbooks: { badge: '教', zh: '教材', en: 'Textbooks' },
  ops: { badge: '运', zh: '运维', en: 'Ops' },
}

type Props = {
  collapsed: boolean
  floating?: boolean
  onToggle: () => void
}

export default function LeftPanel({ collapsed, floating = false, onToggle }: Props) {
  const { state } = useGlobalState()
  const { t } = useI18n()
  const { activeModule } = state
  const [transitioning, setTransitioning] = useState(false)
  const prevModuleRef = useRef(activeModule)
  const activeMeta = PANEL_META[activeModule] ?? { badge: '模', zh: '模块', en: 'Module' }

  useEffect(() => {
    if (prevModuleRef.current === activeModule) return
    prevModuleRef.current = activeModule
    let stopTimer = 0
    const startTimer = window.setTimeout(() => {
      setTransitioning(true)
      stopTimer = window.setTimeout(() => setTransitioning(false), 150)
    }, 0)
    return () => {
      window.clearTimeout(startTimer)
      if (stopTimer) window.clearTimeout(stopTimer)
    }
  }, [activeModule])

  function renderContent() {
    if (activeModule === 'overview') return <OverviewPanel />
    if (activeModule === 'papers') return <PapersPanel />
    if (activeModule === 'ask') return <AskPanel />
    if (activeModule === 'textbooks') return <TextbooksPanel />
    if (activeModule === 'ops') return <OpsPanel />
    return null
  }

  if (collapsed) {
    return (
      <aside className="kgPanel kgPanel--left">
        <div className="kgPanelIcon" title={t('展开左侧面板', 'Expand Left Panel')}>
          <button
            className="kgPanelIconBtn"
            type="button"
            onClick={onToggle}
            aria-label={t('展开左侧面板', 'Expand Left Panel')}
            title={t('展开左侧面板', 'Expand Left Panel')}
          >
            {'>'}
          </button>
          <button
            className="kgPanelIconBtn"
            type="button"
            onClick={onToggle}
            aria-label={t(`${activeMeta.zh} 面板`, `${activeMeta.en} panel`)}
            title={t(`${activeMeta.zh} 面板`, `${activeMeta.en} panel`)}
          >
            {activeMeta.badge}
          </button>
        </div>
        <div aria-hidden="true" style={{ display: 'none' }}>
          <Suspense fallback={null}>{renderContent()}</Suspense>
        </div>
      </aside>
    )
  }

  const panelClass = ['kgPanel', 'kgPanel--left', floating ? 'kgPanel--floating kgPanel--floating-left' : '']
    .filter(Boolean)
    .join(' ')

  return (
    <aside className={panelClass}>
      <div className="kgPanelHeader">
        <span className="kgPanelTitle">
          {activeMeta.badge} {t(activeMeta.zh, activeMeta.en)}
        </span>
        <button
          className="kgPanelCollapseBtn"
          type="button"
          onClick={onToggle}
          title={t('收起', 'Collapse')}
          aria-label={t('收起左侧面板', 'Collapse left panel')}
        >
          {'<'}
        </button>
      </div>
      <div className={`kgPanelContent${transitioning ? ' is-transitioning' : ''}`}>
        <Suspense
          fallback={
            <PanelFallback
              label={t(activeMeta.zh, activeMeta.en)}
              message={t('正在加载模块...', 'Loading module...')}
            />
          }
        >
          {renderContent()}
        </Suspense>
      </div>
    </aside>
  )
}

function PanelFallback({ label, message }: { label: string; message: string }) {
  return (
    <div className="kgPanelBody kgStack" aria-busy="true">
      <div className="kgCard">
        <div className="kgCardTitle">{label}</div>
        <div className="text-faint" style={{ fontSize: 11 }}>
          {message}
        </div>
      </div>
    </div>
  )
}
