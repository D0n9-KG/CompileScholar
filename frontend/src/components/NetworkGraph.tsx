import cytoscape from 'cytoscape'
import { useEffect, useMemo, useRef } from 'react'

export type NetworkNode = {
  id: string
  paper_source?: string
  title?: string
  year?: number
  doi?: string
  ingested?: boolean
  in_scope?: boolean
}

export type NetworkEdge = {
  source: string
  target: string
  total_mentions?: number
  purpose_labels?: string[]
}

export type GraphNodeHoverPayload = {
  id: string
  x: number
  y: number
  canvasWidth: number
  canvasHeight: number
}

type LayoutMode = 'balanced' | 'compact' | 'spread'
type EdgeLabelMode = 'off' | 'mentions'
type NodeLabelMode = 'focus' | 'minimal' | 'full'
type DensityBand = 'normal' | 'dense' | 'veryDense'

const LABEL_LENGTH_LIMIT = 12
const MINIMAL_LABEL_MENTION_FLOOR = 22
const LABEL_MIN_PIXEL_GAP = 142

type GraphViewMemory = {
  positions: Map<string, { x: number; y: number }>
  zoom: number
  pan: { x: number; y: number }
}

function getDensityBand(nodeCount: number, edgeCount: number): DensityBand {
  if (nodeCount <= 0) return 'normal'
  const edgePerNode = edgeCount / nodeCount
  if (nodeCount >= 18 || edgePerNode >= 1.45) return 'veryDense'
  if (nodeCount >= 10 || edgePerNode >= 0.95) return 'dense'
  return 'normal'
}

function getDensityPenalty(densityBand: DensityBand) {
  return densityBand === 'veryDense' ? 2 : densityBand === 'dense' ? 1 : 0
}

function shortenLabel(value: string | null | undefined, maxLen: number) {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim()
  if (!text) return ''
  if (text.length <= maxLen) return text
  return `${text.slice(0, Math.max(1, maxLen - 3))}...`
}

function selectSpacedNodes(
  nodes: cytoscape.NodeSingular[],
  maxCount: number,
  minPixelGap: number,
): cytoscape.NodeSingular[] {
  if (!nodes.length || maxCount <= 0) return []

  const picked: cytoscape.NodeSingular[] = []
  const minGapSq = minPixelGap * minPixelGap

  for (const candidate of nodes) {
    const { x, y } = candidate.renderedPosition()
    const isFarEnough = picked.every((selected) => {
      const pos = selected.renderedPosition()
      const dx = pos.x - x
      const dy = pos.y - y
      return dx * dx + dy * dy >= minGapSq
    })

    if (!isFarEnough) continue
    picked.push(candidate)
    if (picked.length >= maxCount) break
  }

  if (picked.length >= maxCount) return picked

  for (const candidate of nodes) {
    if (picked.includes(candidate)) continue
    picked.push(candidate)
    if (picked.length >= maxCount) break
  }

  return picked
}

function getMinimalLabelAnchors(cy: cytoscape.Core) {
  const zoom = cy.zoom()
  const nodeCount = cy.nodes().length
  const edgeCount = cy.edges().length
  const densityBand = getDensityBand(nodeCount, edgeCount)
  const densityPenalty = getDensityPenalty(densityBand)
  const zoomGate = densityBand === 'veryDense' ? 1.34 : densityBand === 'dense' ? 1.2 : 1.04
  if (zoom < zoomGate) return []

  const baseMaxAnchors = zoom >= 1.72 ? 4 : zoom >= 1.44 ? 2 : 1
  const maxAnchors = Math.max(1, baseMaxAnchors - densityPenalty)
  const degreeFloor = (zoom >= 1.72 ? 4 : zoom >= 1.44 ? 5 : 6) + densityPenalty
  const mentionFloor =
    (zoom >= 1.58 ? MINIMAL_LABEL_MENTION_FLOOR : MINIMAL_LABEL_MENTION_FLOOR + 4) + densityPenalty * 8

  const candidates = cy
    .nodes()
    .toArray()
    .filter((node) => Number(node.data('degree') ?? 0) >= degreeFloor || Number(node.data('mentions') ?? 0) >= mentionFloor)
    .sort(
      (a, b) =>
        Number(b.data('mentions') ?? 0) - Number(a.data('mentions') ?? 0) ||
        Number(b.data('degree') ?? 0) - Number(a.data('degree') ?? 0) ||
        Number(b.degree(false)) - Number(a.degree(false)),
    )

  return selectSpacedNodes(candidates, maxAnchors, LABEL_MIN_PIXEL_GAP + densityPenalty * 36)
}

function applyNodeLabels(cy: cytoscape.Core, nodeLabelMode: NodeLabelMode) {
  cy.nodes().removeClass('showLabel anchorLabel')

  if (nodeLabelMode === 'full') {
    const nodeCount = cy.nodes().length
    const densityBand = getDensityBand(nodeCount, cy.edges().length)

    if (densityBand === 'normal' && nodeCount <= 10) {
      cy.nodes().addClass('showLabel')
      return
    }

    const zoom = cy.zoom()
    const baseCap = densityBand === 'veryDense' ? 2 : densityBand === 'dense' ? 3 : 5
    const zoomBoost = zoom >= 1.42 ? 2 : zoom >= 1.18 ? 1 : 0
    const labelCap = Math.min(nodeCount, baseCap + zoomBoost)
    const labelGap = LABEL_MIN_PIXEL_GAP + (densityBand === 'veryDense' ? 42 : densityBand === 'dense' ? 28 : 14)

    const candidates = cy
      .nodes()
      .toArray()
      .sort(
        (a, b) =>
          Number(b.data('mentions') ?? 0) - Number(a.data('mentions') ?? 0) ||
          Number(b.data('degree') ?? 0) - Number(a.data('degree') ?? 0) ||
          Number(b.degree(false)) - Number(a.degree(false)),
      )

    const anchors = selectSpacedNodes(candidates, labelCap, labelGap)
    for (const node of anchors) node.addClass('showLabel anchorLabel')
    return
  }

  if (nodeLabelMode === 'minimal') {
    const anchors = getMinimalLabelAnchors(cy)
    for (const node of anchors) node.addClass('showLabel anchorLabel')
  }
}

function resolveNodeOverlaps(cy: cytoscape.Core) {
  const nodes = cy.nodes().toArray()
  if (nodes.length < 2) return

  const densityBand = getDensityBand(nodes.length, cy.edges().length)
  const baseGapPx = densityBand === 'veryDense' ? 30 : densityBand === 'dense' ? 24 : 18
  const iterations = densityBand === 'veryDense' ? 34 : densityBand === 'dense' ? 26 : 18
  const zoom = Math.max(0.45, cy.zoom())
  const gapInModel = baseGapPx / zoom

  for (let iter = 0; iter < iterations; iter += 1) {
    let moved = false
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const a = nodes[i]
        const b = nodes[j]
        const pa = a.position()
        const pb = b.position()

        let dx = pb.x - pa.x
        let dy = pb.y - pa.y
        let dist = Math.hypot(dx, dy)

        if (dist >= gapInModel) continue

        if (dist < 1e-3) {
          const seed = (i + 1) * 73856093 + (j + 1) * 19349663
          const angle = (seed % 360) * (Math.PI / 180)
          dx = Math.cos(angle)
          dy = Math.sin(angle)
          dist = 1
        }

        const push = (gapInModel - dist) * 0.5
        const ux = dx / dist
        const uy = dy / dist

        a.position({ x: pa.x - ux * push, y: pa.y - uy * push })
        b.position({ x: pb.x + ux * push, y: pb.y + uy * push })
        moved = true
      }
    }
    if (!moved) break
  }
}

function getLayout(mode: LayoutMode, nodeCount: number, edgeCount: number): cytoscape.LayoutOptions {
  const densityBand = getDensityBand(nodeCount, edgeCount)
  const densityBoost = densityBand === 'veryDense' ? 1.8 : densityBand === 'dense' ? 1.4 : 1
  const overlapFloor = densityBand === 'veryDense' ? 50 : densityBand === 'dense' ? 36 : 24
  const layoutPadding = densityBand === 'veryDense' ? 68 : densityBand === 'dense' ? 58 : 48

  if (mode === 'compact') {
    return {
      name: 'cose',
      animate: false,
      randomize: true,
      nodeRepulsion: Math.round(130000 * densityBoost * 1.82),
      nodeOverlap: overlapFloor + 10,
      idealEdgeLength: Math.round(194 * (densityBand === 'veryDense' ? 1.22 : densityBand === 'dense' ? 1.14 : 1)),
      edgeElasticity: 0.12,
      gravity: densityBand === 'veryDense' ? 0.028 : densityBand === 'dense' ? 0.045 : 0.07,
      numIter: densityBand === 'veryDense' ? 4200 : densityBand === 'dense' ? 3400 : 2600,
      componentSpacing: densityBand === 'veryDense' ? 320 : densityBand === 'dense' ? 240 : 180,
      nodeDimensionsIncludeLabels: false,
      fit: true,
      padding: Math.max(44, layoutPadding),
    } as unknown as cytoscape.LayoutOptions
  }

  if (mode === 'spread') {
    return {
      name: 'cose',
      animate: false,
      randomize: true,
      nodeRepulsion: Math.round(440000 * densityBoost * 1.45),
      nodeOverlap: overlapFloor,
      idealEdgeLength: Math.round(448 * (densityBand === 'veryDense' ? 1.28 : densityBand === 'dense' ? 1.18 : 1.04)),
      edgeElasticity: 0.06,
      gravity: densityBand === 'veryDense' ? 0.009 : densityBand === 'dense' ? 0.015 : 0.024,
      numIter: densityBand === 'veryDense' ? 7600 : densityBand === 'dense' ? 6200 : 4600,
      componentSpacing: densityBand === 'veryDense' ? 620 : densityBand === 'dense' ? 470 : 360,
      nodeDimensionsIncludeLabels: false,
      fit: true,
      padding: layoutPadding + 20,
    } as unknown as cytoscape.LayoutOptions
  }

  return {
    name: 'cose',
    animate: false,
    randomize: true,
    nodeRepulsion: Math.round(250000 * densityBoost * 1.48),
    nodeOverlap: overlapFloor + 6,
    idealEdgeLength: Math.round(318 * (densityBand === 'veryDense' ? 1.28 : densityBand === 'dense' ? 1.2 : 1.04)),
    edgeElasticity: 0.08,
    gravity: densityBand === 'veryDense' ? 0.014 : densityBand === 'dense' ? 0.026 : 0.044,
    numIter: densityBand === 'veryDense' ? 5600 : densityBand === 'dense' ? 4400 : 3300,
    componentSpacing: densityBand === 'veryDense' ? 450 : densityBand === 'dense' ? 340 : 260,
    nodeDimensionsIncludeLabels: false,
    fit: true,
    padding: layoutPadding + 6,
  } as unknown as cytoscape.LayoutOptions
}

export default function NetworkGraph({
  nodes,
  edges,
  onSelectNode,
  onToggleSelect,
  onNodeHover,
  onNodeHoverOut,
  selectedIds,
  height = 620,
  focusNodeId,
  layoutMode = 'balanced',
  edgeLabelMode = 'mentions',
  nodeLabelMode = 'minimal',
  fitTrigger = 0,
  relayoutTrigger = 0,
  clearHighlightTrigger = 0,
  onViewChange,
}: {
  nodes: NetworkNode[]
  edges: NetworkEdge[]
  onSelectNode?: (paperId: string) => void
  onToggleSelect?: (paperId: string) => void
  onNodeHover?: (payload: GraphNodeHoverPayload) => void
  onNodeHoverOut?: () => void
  selectedIds?: string[]
  height?: number
  focusNodeId?: string
  layoutMode?: LayoutMode
  edgeLabelMode?: EdgeLabelMode
  nodeLabelMode?: NodeLabelMode
  fitTrigger?: number
  relayoutTrigger?: number
  clearHighlightTrigger?: number
  onViewChange?: (view: { zoom: number; nodes: number; edges: number }) => void
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const cyRef = useRef<cytoscape.Core | null>(null)
  const viewMemoryRef = useRef<GraphViewMemory | null>(null)

  const elements = useMemo(() => {
    const nodeMap = new Map<string, NetworkNode>()
    const degreeMap = new Map<string, number>()
    const mentionMap = new Map<string, number>()

    for (const n of nodes) nodeMap.set(n.id, n)
    for (const e of edges) {
      if (!nodeMap.has(e.source)) nodeMap.set(e.source, { id: e.source })
      if (!nodeMap.has(e.target)) nodeMap.set(e.target, { id: e.target })

      degreeMap.set(e.source, (degreeMap.get(e.source) ?? 0) + 1)
      degreeMap.set(e.target, (degreeMap.get(e.target) ?? 0) + 1)

      mentionMap.set(e.source, (mentionMap.get(e.source) ?? 0) + (e.total_mentions ?? 0))
      mentionMap.set(e.target, (mentionMap.get(e.target) ?? 0) + (e.total_mentions ?? 0))
    }

    const nodeEls = Array.from(nodeMap.values()).map((n) => {
      const degree = degreeMap.get(n.id) ?? 0
      const mentions = mentionMap.get(n.id) ?? 0
      const label = n.paper_source ?? n.title ?? n.doi ?? (n.id.startsWith('doi:') ? n.id.slice(4, 18) : n.id.slice(0, 12))
      const shortLabel = shortenLabel(label, LABEL_LENGTH_LIMIT)
      return {
        data: {
          id: n.id,
          label,
          short_label: shortLabel,
          title: n.title,
          year: n.year,
          ingested: n.ingested ? 'true' : 'false',
          in_scope: n.in_scope === false ? 'false' : 'true',
          degree,
          mentions,
        },
      }
    })

    const edgeEls = edges.map((e, idx) => ({
      data: {
        id: `e:${idx}:${e.source}->${e.target}`,
        source: e.source,
        target: e.target,
        total_mentions: e.total_mentions ?? 0,
        purpose_labels: (e.purpose_labels ?? []).join(','),
      },
    }))

    return [...nodeEls, ...edgeEls]
  }, [nodes, edges])

  useEffect(() => {
    if (!ref.current) return

    const previousView = viewMemoryRef.current
    const previousPositions = previousView?.positions ?? new Map<string, { x: number; y: number }>()
    const hasPreviousView = previousPositions.size > 0

    const nodeElements = elements.filter((ele) => !('source' in (ele.data as Record<string, unknown>)))
    const edgeElements = elements.filter((ele) => 'source' in (ele.data as Record<string, unknown>))

    const neighborMap = new Map<string, string[]>()
    for (const edge of edgeElements) {
      const sourceId = String((edge.data as { source?: string }).source ?? '')
      const targetId = String((edge.data as { target?: string }).target ?? '')
      if (!sourceId || !targetId) continue
      const sourceNeighbors = neighborMap.get(sourceId) ?? []
      sourceNeighbors.push(targetId)
      neighborMap.set(sourceId, sourceNeighbors)
      const targetNeighbors = neighborMap.get(targetId) ?? []
      targetNeighbors.push(sourceId)
      neighborMap.set(targetId, targetNeighbors)
    }

    let centerX = 0
    let centerY = 0
    if (previousPositions.size) {
      for (const pos of previousPositions.values()) {
        centerX += pos.x
        centerY += pos.y
      }
      centerX /= previousPositions.size
      centerY /= previousPositions.size
    }

    const seededPositions = new Map(previousPositions)
    const positionedElements = hasPreviousView
      ? [
          ...nodeElements.map((nodeEl, index) => {
            const id = String((nodeEl.data as { id?: string }).id ?? '')
            if (!id) return nodeEl
            const existing = seededPositions.get(id)
            if (existing) return { ...nodeEl, position: existing }

            const neighbors = neighborMap.get(id) ?? []
            const positionedNeighbors = neighbors
              .map((neighborId) => seededPositions.get(neighborId))
              .filter((pos): pos is { x: number; y: number } => Boolean(pos))

            const anchor = positionedNeighbors.length
              ? {
                  x: positionedNeighbors.reduce((sum, pos) => sum + pos.x, 0) / positionedNeighbors.length,
                  y: positionedNeighbors.reduce((sum, pos) => sum + pos.y, 0) / positionedNeighbors.length,
                }
              : { x: centerX, y: centerY }

            const angle = (((index + 1) * 137.5) % 360) * (Math.PI / 180)
            const radius = 64 + (index % 7) * 12
            const position = {
              x: anchor.x + Math.cos(angle) * radius,
              y: anchor.y + Math.sin(angle) * radius,
            }
            seededPositions.set(id, position)
            return { ...nodeEl, position }
          }),
          ...edgeElements,
        ]
      : elements

    const cy = cytoscape({
      container: ref.current,
      elements: positionedElements,
      style: [
        {
          selector: 'node',
          style: {
            width: 'mapData(degree, 0, 12, 10, 18)',
            height: 'mapData(degree, 0, 12, 10, 18)',
            'background-color': 'rgba(128,238,197,0.9)',
            'background-opacity': 0.94,
            'border-width': 1,
            'border-color': 'rgba(226,248,255,0.82)',
          },
        },
        {
          selector: 'node.showLabel',
          style: {
            label: 'data(short_label)',
            color: 'rgba(240,247,255,0.95)',
            'font-size': '5px',
            'font-weight': 500,
            'text-wrap': 'none',
            'min-zoomed-font-size': 5,
            'text-background-color': 'rgba(5,10,22,0.82)',
            'text-background-opacity': 1,
            'text-background-padding': '1px',
            'text-background-shape': 'roundrectangle',
            'text-border-color': 'rgba(146,168,217,0.22)',
            'text-border-width': 1,
            'text-border-opacity': 0.78,
          },
        },
        {
          selector: 'node.anchorLabel',
          style: {
            'text-background-opacity': 0.92,
            color: 'rgba(225,236,255,0.9)',
          },
        },
        {
          selector: 'node[ingested = "false"]',
          style: {
            'background-color': 'rgba(187,201,228,0.5)',
            'background-opacity': 1,
            'border-color': 'rgba(196,212,242,0.72)',
            'border-width': 1.5,
            'border-style': 'dashed',
          },
        },
        {
          selector: 'node[in_scope = "false"]',
          style: {
            'background-color': 'rgba(171,188,221,0.34)',
            'border-color': 'rgba(188,207,242,0.62)',
            'border-style': 'dashed',
            opacity: 0.74,
          },
        },
        {
          selector: 'edge',
          style: {
            width: 'mapData(total_mentions, 0, 9, 0.54, 2)',
            'line-color': 'rgba(162,183,227,0.2)',
            'line-opacity': 0.48,
            'target-arrow-color': 'rgba(162,183,227,0.2)',
            'target-arrow-shape': 'none',
            'curve-style': 'bezier',
          },
        },
        {
          selector: 'edge[total_mentions > 24]',
          style: {
            'line-color': 'rgba(126,139,255,0.72)',
            'target-arrow-color': 'rgba(126,139,255,0.72)',
            label: edgeLabelMode === 'mentions' ? 'data(total_mentions)' : '',
            color: 'rgba(222,231,252,0.88)',
            'font-size': '7px',
            'text-background-color': 'rgba(5,10,22,0.72)',
            'text-background-opacity': 1,
            'text-background-padding': '2px',
            'text-background-shape': 'roundrectangle',
            'text-rotation': 'autorotate',
          },
        },
        {
          selector: '.faded',
          style: {
            opacity: 0.04,
            'text-opacity': 0,
            'line-opacity': 0.04,
          },
        },
        {
          selector: '.highlighted',
          style: {
            opacity: 1,
            'text-opacity': 1,
            'line-opacity': 1,
          },
        },
        {
          selector: '.picked',
          style: {
            'border-color': 'rgba(126,139,255,0.96)',
            'border-width': 2.8,
            'background-color': 'rgba(126,139,255,0.86)',
            color: 'rgba(252,255,255,0.98)',
          },
        },
        {
          selector: '.focusCenter',
          style: {
            'border-color': 'rgba(124,255,203,0.98)',
            'border-width': 3,
            'background-color': 'rgba(112,220,178,0.95)',
            'z-index-compare': 'manual',
            'z-index': 999,
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': 'rgba(124,255,203,0.98)',
            'border-width': 3,
            'overlay-color': 'rgba(124,255,203,0.16)',
            'overlay-padding': '6px',
            'overlay-opacity': 1,
          },
        },
      ],
      layout: hasPreviousView
        ? ({ name: 'preset', fit: false, padding: 40 } as unknown as cytoscape.LayoutOptions)
        : getLayout(layoutMode, nodes.length, edges.length),
      minZoom: 0.38,
      maxZoom: 2.4,
    })

    if (hasPreviousView && previousView) {
      cy.zoom(Math.max(0.38, Math.min(2.4, previousView.zoom)))
      cy.pan(previousView.pan)
    }

    cyRef.current = cy
    if (typeof window !== 'undefined' && import.meta.env.DEV) {
      ;(window as Window & { __logicKgCy?: cytoscape.Core }).__logicKgCy = cy
    }

    const emitViewState = () => {
      onViewChange?.({
        zoom: cy.zoom(),
        nodes: cy.nodes().length,
        edges: cy.edges().length,
      })
    }

    const applyBaseLabels = () => applyNodeLabels(cy, nodeLabelMode)

    const selectForInspect = (node: cytoscape.NodeSingular) => {
      cy.elements().removeClass('faded highlighted')
      cy.nodes().removeClass('focusCenter').unselect()
      applyBaseLabels()
      node.select()
      node.addClass('focusCenter showLabel')
    }

    const emitNodeHover = (node: cytoscape.NodeSingular) => {
      const id = String(node.data('id') ?? '')
      if (!id) return
      const canvas = ref.current
      if (!canvas) return
      const pos = node.renderedPosition()
      onNodeHover?.({
        id,
        x: pos.x,
        y: pos.y,
        canvasWidth: canvas.clientWidth,
        canvasHeight: canvas.clientHeight,
      })
    }

    cy.on('tap', 'node', (evt) => {
      const node = evt.target
      const id = String(node.data('id') ?? '')
      if (!id) return

      const oe = (evt as unknown as { originalEvent?: MouseEvent | PointerEvent }).originalEvent
      const multi = Boolean(
        oe &&
          ((typeof (oe as MouseEvent).shiftKey === 'boolean' && (oe as MouseEvent).shiftKey) ||
            (typeof (oe as MouseEvent).ctrlKey === 'boolean' && (oe as MouseEvent).ctrlKey) ||
            (typeof (oe as MouseEvent).metaKey === 'boolean' && (oe as MouseEvent).metaKey)),
      )

      selectForInspect(node)

      if (multi) {
        onToggleSelect?.(id)
      }
      onSelectNode?.(id)
      emitNodeHover(node)
    })

    cy.on('tap', (evt) => {
      if (evt.target !== cy) return
      cy.elements().removeClass('faded highlighted')
      cy.nodes().removeClass('focusCenter').unselect()
      applyBaseLabels()
      onSelectNode?.('')
      onNodeHoverOut?.()
    })

    cy.on('zoom', () => {
      emitViewState()
      applyBaseLabels()
      const center = cy.nodes('.focusCenter').first()
      if (center && center.length) center.addClass('showLabel')
    })
    const settleGraphReadability = () => {
      if (!hasPreviousView) resolveNodeOverlaps(cy)
      applyBaseLabels()
      const center = cy.nodes('.focusCenter').first()
      if (center && center.length) center.addClass('showLabel')
      emitViewState()
    }

    const settleTimerA = window.setTimeout(settleGraphReadability, 0)
    const settleTimerB = window.setTimeout(settleGraphReadability, 140)

    return () => {
      window.clearTimeout(settleTimerA)
      window.clearTimeout(settleTimerB)
      const positions = new Map<string, { x: number; y: number }>()
      for (const node of cy.nodes().toArray()) {
        const pos = node.position()
        positions.set(node.id(), { x: pos.x, y: pos.y })
      }
      viewMemoryRef.current = {
        positions,
        zoom: cy.zoom(),
        pan: cy.pan(),
      }
      if (typeof window !== 'undefined' && import.meta.env.DEV) {
        const devWindow = window as Window & { __logicKgCy?: cytoscape.Core }
        if (devWindow.__logicKgCy === cy) delete devWindow.__logicKgCy
      }
      cyRef.current = null
      cy.destroy()
    }
  }, [
    elements,
    onSelectNode,
    onToggleSelect,
    onNodeHover,
    onNodeHoverOut,
    edgeLabelMode,
    nodeLabelMode,
    onViewChange,
    layoutMode,
    nodes.length,
    edges.length,
  ])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return

    const set = new Set<string>((selectedIds ?? []).map(String))
    cy.nodes().removeClass('picked')
    for (const id of set) {
      const node = cy.$id(id)
      if (node && node.length) node.addClass('picked')
    }
  }, [selectedIds])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy || !focusNodeId) return
    const node = cy.$id(focusNodeId)
    if (!node || !node.length) return

    cy.elements().removeClass('faded highlighted')
    cy.nodes().removeClass('focusCenter').unselect()
    applyNodeLabels(cy, nodeLabelMode)
    node.removeClass('faded').addClass('highlighted')
    node.addClass('focusCenter showLabel')
    node.select()
    cy.animate(
      {
        center: { eles: node },
      },
      { duration: 180 },
    )
  }, [focusNodeId, nodeLabelMode])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.fit(cy.elements(), 40)
  }, [fitTrigger])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.layout(getLayout(layoutMode, nodes.length, edges.length)).run()
    const settleTimer = window.setTimeout(() => {
      resolveNodeOverlaps(cy)
      applyNodeLabels(cy, nodeLabelMode)
    }, 140)
    return () => window.clearTimeout(settleTimer)
    // We intentionally avoid depending on node/edge counts here so expand/collapse keeps current positions.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [relayoutTrigger, layoutMode, nodeLabelMode])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.elements().removeClass('faded highlighted')
    cy.nodes().removeClass('focusCenter').unselect()
    applyNodeLabels(cy, nodeLabelMode)
    onSelectNode?.('')
    onNodeHoverOut?.()
  }, [clearHighlightTrigger, nodeLabelMode, onSelectNode, onNodeHoverOut])

  return <div ref={ref} className="networkGraphCanvas" style={{ height }} />
}
