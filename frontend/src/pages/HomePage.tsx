import { Link } from 'react-router-dom'

const workflow = [
  {
    title: '导入论文数据',
    body: '上传包含多个论文子目录的 ZIP / 文件夹，支持大文件分片与断点续传。',
  },
  {
    title: '处理 DOI 冲突',
    body: '自动扫描缺失 DOI 与重复 DOI，按条目决策保留、替换或补全后继续导入。',
  },
  {
    title: '构建知识图谱',
    body: '写入 Neo4j 后自动生成引文网络、逻辑链、要点与图文索引。',
  },
  {
    title: '检索与问答',
    body: '在图谱与语义索引上进行问答，返回可追溯证据与来源节点。',
  },
]

const quickLinks = [
  { to: '/ingest', title: '导入中心', hint: '批量导入、冲突处理、补全上传' },
  { to: '/graph', title: '图谱视图', hint: '探索引文关系、筛选范围、选题分析' },
  { to: '/papers', title: '论文列表', hint: '按 DOI / 标题检索，进入结构化详情页' },
  { to: '/ask', title: '图谱问答', hint: 'GraphRAG 检索增强生成与证据追踪' },
]

export default function HomePage() {
  return (
    <div className="page homePage">
      <section className="panel homeHero">
        <div className="panelBody homeHeroBody">
          <div>
            <span className="homeOverline">Scientific Knowledge Graph Workspace</span>
            <h2 className="homeHeroTitle">LogicKG：从论文原文到可推理的知识图谱工作流</h2>
            <p className="homeHeroSubtitle">
              MinerU Markdown → 逻辑链 / 要点抽取 → Neo4j 引文网络 → GraphRAG 问答，全流程都在同一界面完成。
            </p>
            <div className="homeHeroActions">
              <Link className="btn btnPrimary" to="/ingest">
                开始导入
              </Link>
              <Link className="btn" to="/graph">
                打开图谱工作台
              </Link>
              <Link className="btn btnGhost" to="/ask">
                进入图谱问答
              </Link>
            </div>
          </div>

          <div className="homeMetrics">
            <div className="homeMetricCard">
              <div className="homeMetricValue">4</div>
              <div className="homeMetricLabel">核心流程阶段：导入 / 解析 / 图谱 / 问答</div>
            </div>
            <div className="homeMetricCard">
              <div className="homeMetricValue">Neo4j</div>
              <div className="homeMetricLabel">结构化图数据库驱动节点关系检索</div>
            </div>
            <div className="homeMetricCard">
              <div className="homeMetricValue">GraphRAG</div>
              <div className="homeMetricLabel">回答附证据，可追溯回原始文本片段</div>
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panelHeader">
          <div className="panelTitle">推荐流程</div>
        </div>
        <div className="panelBody">
          <div className="homeWorkflowGrid">
            {workflow.map((item, idx) => (
              <div key={item.title} className="homeStepCard">
                <span className="homeStepIndex">{idx + 1}</span>
                <div className="itemTitle" style={{ marginTop: 8 }}>
                  {item.title}
                </div>
                <div className="itemBody">{item.body}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panelHeader">
          <div className="panelTitle">快捷入口</div>
        </div>
        <div className="panelBody">
          <div className="homeQuickGrid">
            {quickLinks.map((item) => (
              <Link key={item.to} className="homeQuickCard" to={item.to}>
                <div className="homeQuickCardTitle">{item.title}</div>
                <div className="homeQuickCardHint">{item.hint}</div>
              </Link>
            ))}
          </div>
          <div className="hint">提示：如果图谱为空，通常是 Neo4j 未启动或尚未完成导入 / 重建。</div>
        </div>
      </section>
    </div>
  )
}
