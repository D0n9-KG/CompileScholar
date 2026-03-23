import type { UILocale } from './i18n'

type LocalizedText = {
  zh: string
  en: string
}

const KIND_LABELS: Record<string, LocalizedText> = {
  textbook: { zh: '教材', en: 'Textbook' },
  chapter: { zh: '章节', en: 'Chapter' },
  community: { zh: '社区', en: 'Community' },
  paper: { zh: '论文', en: 'Paper' },
  move: { zh: '研究动作', en: 'Research Move' },
  anchor: { zh: '证据锚点', en: 'Evidence Anchor' },
  group: { zh: '分组', en: 'Group' },
  entity: { zh: '实体', en: 'Entity' },
  citation: { zh: '引用', en: 'Citation' },
}

function normalize(value: unknown): string {
  return String(value ?? '')
    .trim()
    .toLowerCase()
}

export function normalizeGraphKind(value: unknown): string {
  const kind = normalize(value)
  if (kind === 'research_move' || kind === 'researchmove' || kind === 'move') return 'move'
  if (kind === 'evidence_anchor' || kind === 'evidenceanchor' || kind === 'anchor') return 'anchor'
  if (kind === 'knowledge_entity' || kind === 'knowledgeentity' || kind === 'entity') return 'entity'
  return kind
}

export function isMoveGraphKind(value: unknown): boolean {
  return normalizeGraphKind(value) === 'move'
}

export function isAnchorGraphKind(value: unknown): boolean {
  return normalizeGraphKind(value) === 'anchor'
}

export function graphKindLabel(value: unknown, locale: UILocale): string {
  const kind = normalizeGraphKind(value)
  const label = KIND_LABELS[kind]
  if (!label) return kind || 'unknown'
  return locale === 'zh-CN' ? label.zh : label.en
}
