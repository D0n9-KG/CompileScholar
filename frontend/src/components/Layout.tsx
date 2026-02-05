import { Link, NavLink, Outlet } from 'react-router-dom'
import { apiBaseUrl } from '../api'
import './layout.css'

export default function Layout() {
  return (
    <div className="appShell">
      <header className="topBar">
        <div className="topLeft">
          <Link to="/" className="brand">
            LogicKG
          </Link>
          <div className="tagline">逻辑链 · 引文网络 · 图谱问答（GraphRAG）</div>
        </div>
        <div className="topRight">
          <div className="apiPill">
            <span className="apiDot" aria-hidden="true" />
            <span className="apiLabel">接口</span>
            <code className="apiValue">{apiBaseUrl()}</code>
          </div>
        </div>
      </header>
      <div className="body">
        <nav className="sideNav">
          <div className="navGroup">
            <div className="navGroupTitle">数据</div>
            <NavLink to="/ingest">导入</NavLink>
            <NavLink to="/papers">论文</NavLink>
            <NavLink to="/unresolved">待解析</NavLink>
          </div>
          <div className="navGroup">
            <div className="navGroupTitle">探索</div>
            <NavLink to="/graph">图谱</NavLink>
            <NavLink to="/ask">问答</NavLink>
          </div>
          <div className="navGroup">
            <div className="navGroupTitle">运维</div>
            <NavLink to="/tasks">任务</NavLink>
            <NavLink to="/schema">配置</NavLink>
          </div>
        </nav>
        <main className="main">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
