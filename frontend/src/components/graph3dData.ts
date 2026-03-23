import { buildOverview3DVisibleLinks } from './graph3dLinkBudget'
import { seedClusteredPositions } from './graph3dModel'
import { applyCommunityThemeLinks, applyCommunityThemeNodes, buildCommunityRoleAssignments, type CommunityRoleAssignment } from './graph3dTheme'
import type { GraphElement } from '../state/types'
import { normalizeGraphKind } from '../graphKinds'

export type Graph3DNode = {
  id: string
  label: string
  kind: string
  description?: string
  communityId?: string
  clusterKey?: string
  qualityTier?: string
  ingested?: boolean
  paperId?: string
  paperSource?: string
  paperTitle?: string
  role?: string
  textbookId?: string
  chapterId?: string
  keywords?: string[]
  val: number
  color: string
  auraColor?: string
  ringColor?: string
  x?: number
  y?: number
  z?: number
}

export type Graph3DLink = {
  id: string
  source: string
  target: string
  kind: string
  weight: number
  emphasis?: 'default' | 'background' | 'primary' | 'focus'
  visible?: boolean
  displayColor?: string
}

export type Graph3DBaseData = {
  nodes: Graph3DNode[]
  links: Graph3DLink[]
  nodeSignature: string
  communityAssignments: Map<string, CommunityRoleAssignment>
}

function clamp(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min
  if (value < min) return min
  if (value > max) return max
  return value
}

function nodeColor(kind: string, tier?: string, ingested?: boolean): string {
  const normalizedKind = normalizeGraphKind(kind)
  if (normalizedKind === 'paper') {
    if (ingested === false) return '#4f6d89'
    if (tier === 'A1') return '#7dd3fc'
    if (tier === 'A2') return '#38bdf8'
    if (tier === 'B1') return '#0ea5e9'
    if (tier === 'B2') return '#0284c7'
    if (tier === 'C') return '#0369a1'
    return '#0ea5e9'
  }
  if (normalizedKind === 'textbook') return '#f59e0b'
  if (normalizedKind === 'chapter') return '#22c55e'
  if (normalizedKind === 'community') return '#fb7185'
  if (normalizedKind === 'move') return '#34d399'
  if (normalizedKind === 'anchor') return '#fb923c'
  if (normalizedKind === 'group') return '#2dd4bf'
  if (normalizedKind === 'entity') return '#14b8a6'
  return '#94a3b8'
}

function nodeSize(kind: string, degree?: number, ingested?: boolean): number {
  const normalizedKind = normalizeGraphKind(kind)
  const d = clamp(Number(degree ?? 0), 0, 20)
  if (normalizedKind === 'textbook') return 10.5 + d * 0.32
  if (normalizedKind === 'chapter') return 7.2 + d * 0.24
  if (normalizedKind === 'community') return 6.4 + d * 0.24
  if (normalizedKind === 'paper') return ingested === false ? 3.8 + d * 0.2 : 5.8 + d * 0.34
  if (normalizedKind === 'group') return 6.2 + d * 0.28
  if (normalizedKind === 'move' || normalizedKind === 'anchor') return 4.8 + d * 0.24
  if (normalizedKind === 'citation') return 3.2 + d * 0.14
  return 4 + d * 0.2
}

export function buildGraph3DBaseData(elements: GraphElement[]): Graph3DBaseData {
  const degreeMap = new Map<string, number>()
  for (const el of elements) {
    if (el.group !== 'edges') continue
    const source = String(el.data.source ?? '')
    const target = String(el.data.target ?? '')
    degreeMap.set(source, (degreeMap.get(source) ?? 0) + 1)
    degreeMap.set(target, (degreeMap.get(target) ?? 0) + 1)
  }

  const nodes: Graph3DNode[] = elements
    .filter((element) => element.group === 'nodes')
    .map((element) => ({
      id: element.data.id,
      label: element.data.label,
      kind: element.data.kind,
      description: element.data.description,
      communityId: element.data.communityId,
      clusterKey: element.data.clusterKey,
      qualityTier: element.data.qualityTier,
      ingested: element.data.ingested,
      paperId: element.data.paperId,
      paperSource: element.data.paperSource,
      paperTitle: element.data.paperTitle,
      role: element.data.role,
      textbookId: element.data.textbookId,
      chapterId: element.data.chapterId,
      keywords: element.data.keywords,
      val: nodeSize(element.data.kind, degreeMap.get(element.data.id), element.data.ingested),
      color: nodeColor(element.data.kind, element.data.qualityTier, element.data.ingested),
    }))

  const rawLinks: Graph3DLink[] = elements
    .filter((element) => element.group === 'edges')
    .map((element) => ({
      id: element.data.id,
      source: element.data.source,
      target: element.data.target,
      kind: element.data.kind,
      weight: clamp(Number(element.data.weight ?? 0.5), 0.1, 1),
    }))
  seedClusteredPositions(nodes)
  const assignments = buildCommunityRoleAssignments(nodes, rawLinks)
  const themedNodes = applyCommunityThemeNodes(nodes, assignments)
  const links = applyCommunityThemeLinks(buildOverview3DVisibleLinks(rawLinks), assignments)

  return {
    nodes: themedNodes,
    links,
    nodeSignature: themedNodes.map((node) => node.id).sort((a, b) => a.localeCompare(b)).join('|'),
    communityAssignments: assignments,
  }
}

export function buildGraph3DVisibleData(baseData: Graph3DBaseData, selectedNodeId?: string | null) {
  return {
    nodes: baseData.nodes,
    links: applyCommunityThemeLinks(buildOverview3DVisibleLinks(baseData.links, selectedNodeId), baseData.communityAssignments),
    nodeSignature: baseData.nodeSignature,
  }
}
