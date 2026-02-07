import { Link, NavLink, Outlet } from 'react-router-dom'
import { apiBaseUrl } from '../api'
import './layout.css'

type NavIconName = 'ingest' | 'papers' | 'unresolved' | 'graph' | 'evolution' | 'ask' | 'tasks' | 'schema'

type NavSection = {
  title: string
  items: Array<{
    to: string
    label: string
    icon: NavIconName
  }>
}

const navSections: NavSection[] = [
  {
    title: '数据',
    items: [
      { to: '/ingest', label: '导入', icon: 'ingest' },
      { to: '/papers', label: '论文', icon: 'papers' },
      { to: '/unresolved', label: '待解析引用', icon: 'unresolved' },
    ],
  },
  {
    title: '探索',
    items: [
      { to: '/graph', label: '图谱', icon: 'graph' },
      { to: '/evolution', label: '演化', icon: 'evolution' },
      { to: '/ask', label: '问答', icon: 'ask' },
    ],
  },
  {
    title: '运维',
    items: [
      { to: '/tasks', label: '任务', icon: 'tasks' },
      { to: '/schema', label: '配置', icon: 'schema' },
    ],
  },
]

function iconPath(icon: NavIconName) {
  switch (icon) {
    case 'ingest':
      return <path d="M12 3v10m0 0 4-4m-4 4-4-4M5 15.5V18a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2.5" />
    case 'papers':
      return (
        <>
          <path d="M8 3h9l4 4v12a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" />
          <path d="M17 3v5h5M10 12h7M10 16h7" />
        </>
      )
    case 'unresolved':
      return (
        <>
          <path d="M10 3h10a1 1 0 0 1 1 1v15a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" />
          <path d="M4 7h4v12H6a2 2 0 0 1-2-2V7Zm9 4h4m-4 4h2" />
        </>
      )
    case 'graph':
      return (
        <>
          <circle cx="6.5" cy="12" r="2.5" />
          <circle cx="17.5" cy="6.5" r="2.5" />
          <circle cx="17.5" cy="17.5" r="2.5" />
          <path d="M8.8 10.9 15.2 7.7M8.8 13.1l6.4 3.2" />
        </>
      )
    case 'evolution':
      return (
        <>
          <path d="M4 18h16M6 15l3-3 3 2 4-5 2 2" />
          <path d="m16 9 2-2 2 2" />
        </>
      )
    case 'ask':
      return (
        <>
          <path d="M5 9a7 7 0 1 1 13 3.6A7 7 0 0 1 12 20l-4 1 1-4a7 7 0 0 1-4-8Z" />
          <path d="M12 7.8a2.2 2.2 0 0 1 2.2 2.2c0 1.8-2.2 1.8-2.2 3.5m0 2.5h.01" />
        </>
      )
    case 'tasks':
      return (
        <>
          <rect x="4" y="5" width="16" height="14" rx="2.5" />
          <path d="m8 9 2 2 4-4m-6 8h8" />
        </>
      )
    case 'schema':
      return (
        <>
          <rect x="4" y="4" width="7" height="7" rx="1.6" />
          <rect x="13" y="4" width="7" height="4.5" rx="1.6" />
          <rect x="13" y="11.5" width="7" height="8.5" rx="1.6" />
          <path d="M11 7.5h2M9 11v2h4" />
        </>
      )
    default:
      return null
  }
}

function NavIcon({ icon }: { icon: NavIconName }) {
  return (
    <svg viewBox="0 0 24 24" className="navIcon" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
      {iconPath(icon)}
    </svg>
  )
}

function BrandGlyph() {
  return (
    <svg viewBox="0 0 28 28" className="brandGlyph" aria-hidden="true" fill="none">
      <defs>
        <linearGradient id="brandPulse" x1="6" x2="23" y1="5" y2="22" gradientUnits="userSpaceOnUse">
          <stop stopColor="#7CFFCB" />
          <stop offset="1" stopColor="#6E73FF" />
        </linearGradient>
      </defs>
      <rect x="2.5" y="2.5" width="23" height="23" rx="7.5" stroke="url(#brandPulse)" strokeWidth="1.6" />
      <path d="M8 14h5l3-4 4 8" stroke="url(#brandPulse)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="8" cy="14" r="1.2" fill="#7CFFCB" />
      <circle cx="13" cy="14" r="1.2" fill="#A4B4FF" />
      <circle cx="16" cy="10" r="1.2" fill="#7CFFCB" />
      <circle cx="20" cy="18" r="1.2" fill="#A4B4FF" />
    </svg>
  )
}

export default function Layout() {
  const apiUrl = apiBaseUrl()

  return (
    <div className="appShell">
      <header className="topBar">
        <div className="topLeft">
          <Link to="/" className="brand">
            <BrandGlyph />
            <span>LogicKG</span>
          </Link>
          <div className="tagline">逻辑链 · 引文网络 · 演化追踪 · GraphRAG</div>
        </div>
        <div className="topRight">
          <div className="apiPill" title={apiUrl}>
            <span className="apiDot" aria-hidden="true" />
            <span className="apiLabel">API</span>
            <code className="apiValue">{apiUrl}</code>
          </div>
        </div>
      </header>
      <div className="body">
        <nav className="sideNav" aria-label="主导航">
          {navSections.map((section) => (
            <section key={section.title} className="navGroup">
              <div className="navGroupTitle">{section.title}</div>
              {section.items.map((item) => (
                <NavLink key={item.to} to={item.to} className={({ isActive }) => `navLink${isActive ? ' active' : ''}`}>
                  <NavIcon icon={item.icon} />
                  <span>{item.label}</span>
                </NavLink>
              ))}
            </section>
          ))}
        </nav>
        <main className="main">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
