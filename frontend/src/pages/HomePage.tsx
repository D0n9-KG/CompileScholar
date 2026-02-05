import { Link } from 'react-router-dom'

export default function HomePage() {
  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">LogicKG</h2>
          <div className="pageSubtitle">MinerU Markdown → 逻辑链 / 要点 → Neo4j 引文网络 → 图谱问答（GraphRAG）</div>
        </div>
        <div className="pageActions">
          <Link className="btn btnPrimary" to="/ingest">
            开始导入
          </Link>
          <Link className="btn" to="/graph">
            打开图谱
          </Link>
        </div>
      </div>

      <div className="grid2">
        <div className="panel">
          <div className="panelHeader">
            <div className="panelTitle">推荐流程</div>
          </div>
          <div className="panelBody">
            <div className="stack">
              <div className="itemCard">
                <div className="itemTitle">1) 导入论文文件夹 / ZIP</div>
                <div className="itemBody">进入导入页，上传包含多个论文子文件夹的数据集（支持分片/断点续传）。</div>
              </div>
              <div className="itemCard">
                <div className="itemTitle">2) 解决 DOI 冲突 / 缺失 DOI</div>
                <div className="itemBody">同一 DOI 只能保留一套信息；重复导入时由你选择保留哪一套。</div>
              </div>
              <div className="itemCard">
                <div className="itemTitle">3) 浏览论文详情与图片</div>
                <div className="itemBody">论文详情页展示逻辑链条、要点、引文目的标签，以及图片缩略图与图注。</div>
              </div>
              <div className="itemCard">
                <div className="itemTitle">4) 图谱问答（GraphRAG）</div>
                <div className="itemBody">问答页用全局 FAISS + DeepSeek 生成回答，并返回可追溯的证据。</div>
              </div>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panelHeader">
            <div className="panelTitle">快捷入口</div>
          </div>
          <div className="panelBody">
            <div className="stack">
              <Link className="btn btnPrimary" to="/ingest">
                导入
              </Link>
              <Link className="btn" to="/papers">
                论文
              </Link>
              <Link className="btn" to="/unresolved">
                待解析
              </Link>
              <Link className="btn" to="/ask">
                问答
              </Link>
              <Link className="btn" to="/tasks">
                任务
              </Link>
              <div className="hint">提示：如果图谱为空，通常是 Neo4j 未启动或还没完成导入 / 重建。</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
