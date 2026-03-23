import type { GraphNodeData } from './state/types'

type AskScopeNode = Pick<GraphNodeData, 'id' | 'kind' | 'paperId'>

function normalize(value: unknown): string {
  return String(value ?? '').trim()
}

function normalizePrefixedPaperRef(value: unknown): string {
  let text = normalize(value)
  if (!text) return ''
  if (text.startsWith('paper:')) {
    text = normalize(text.slice('paper:'.length))
  } else if (text.startsWith('paper_source:')) {
    text = normalize(text.slice('paper_source:'.length))
  }
  return text
}

function looksLikePaperId(value: string): boolean {
  return /^doi:/i.test(value) || /^[0-9a-f]{40,128}$/i.test(value)
}

function extractPaperRefFromNodeId(rawId: string): string {
  const parts = rawId.split(':')
  if (parts.length < 3) return ''
  const prefix = parts[0]?.trim().toLowerCase()
  if (prefix !== 'move' && prefix !== 'anchor') return ''
  return normalize(parts[1])
}

export function paperRefForAskScope(node: AskScopeNode): string {
  const direct = normalizePrefixedPaperRef(node.paperId)
  if (direct) return direct

  const rawId = normalize(node.id)
  const derivedFromNode = extractPaperRefFromNodeId(rawId)
  if (derivedFromNode) return derivedFromNode
  const normalizedId = normalizePrefixedPaperRef(rawId)
  if (!normalizedId) return ''

  if (rawId.startsWith('paper:') || rawId.startsWith('paper_source:')) return normalizedId
  if (normalize(node.kind) === 'paper' && looksLikePaperId(normalizedId)) return normalizedId
  return ''
}
