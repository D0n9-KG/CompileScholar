export function term(zh: string, en?: string) {
  return en ? `${zh}(${en})` : zh
}

export const TERMS = {
  stub: term('仅元数据', 'stub'),
  llm: term('大模型', 'LLM'),
  faiss: term('向量索引', 'FAISS'),
  crossref: term('文献元数据', 'Crossref'),
  evidenceAnchor: term('证据锚点', 'Evidence Anchor'),
  researchMove: term('研究动作', 'Research Move'),
  evidence: term('证据', 'Evidence'),
  graphRag: term('图谱检索增强生成', 'GraphRAG'),
} as const
