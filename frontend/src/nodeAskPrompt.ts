import type { UILocale } from './i18n'
import { normalizeGraphKind } from './graphKinds'

function normalize(value: unknown): string {
  return String(value ?? '').trim()
}

export function buildNodeAskQuestion(nodeKind: string, nodeLabel: string, locale: UILocale): string {
  const kind = normalizeGraphKind(nodeKind)
  const label = normalize(nodeLabel)

  if (kind === 'paper') {
    return locale === 'zh-CN'
      ? '请围绕这篇论文提炼核心方法、关键发现，并标出可核验的证据。'
      : "Summarize this paper's core methods, key findings, and verifiable evidence."
  }
  if (kind === 'move') {
    return locale === 'zh-CN'
      ? `请解释这个研究动作在论文中的作用、证据来源与局限：${label}`
      : `Explain this research move in the paper, including role, evidence sources, and limitations: ${label}`
  }
  if (kind === 'anchor') {
    return locale === 'zh-CN'
      ? `请评估这个证据锚点的支撑充分性与可核验性：${label}`
      : `Assess this evidence anchor for evidence sufficiency and verifiability: ${label}`
  }
  return locale === 'zh-CN'
    ? `请基于这个节点给出解释，并梳理证据链：${label}`
    : `Explain this node and outline its evidence chain: ${label}`
}
