import cytoscape from 'cytoscape'
import { useEffect, useMemo, useRef } from 'react'

export type ContentNodeKind = 'paper' | 'logic' | 'claim_group' | 'claim'

export type ContentNode = {
  id: string
  kind: ContentNodeKind
  label: string
  paperId: string
  lane: number
  fillColor?: string
  borderColor?: string
  textColor?: string
  glowColor?: string
  stepType?: string
  text?: string
  count?: number
  confidence?: number
  claimKey?: string
  source?: string
  kinds?: string[]
}

export type ContentEdgeKind = 'next' | 'group' | 'claim' | 'sim_logic' | 'sim_claim'

export type ContentEdge = {
  id: string
  source: string
  target: string
  kind: ContentEdgeKind
  score?: number
  color?: string
  textColor?: string
}

export default function ContentGraph({
  nodes,
  edges,
  height = 720,
  onSelect,
}: {
  nodes: Array<ContentNode & { x: number; y: number }>
  edges: ContentEdge[]
  height?: number
  onSelect?: (id: string) => void
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const cyRef = useRef<cytoscape.Core | null>(null)
  const viewMemoryRef = useRef<{ zoom: number; pan: { x: number; y: number } } | null>(null)

  const elements = useMemo(() => {
    const nodeEls = nodes.map((n) => ({
      data: {
        id: n.id,
        kind: n.kind,
        label: n.label,
        paperId: n.paperId,
        lane: String(n.lane),
        fillColor: n.fillColor ?? 'rgba(124,255,203,0.16)',
        borderColor: n.borderColor ?? 'rgba(198,216,248,0.34)',
        textColor: n.textColor ?? 'rgba(245,250,255,0.94)',
        glowColor: n.glowColor ?? 'rgba(124,255,203,0.26)',
        stepType: n.stepType ?? '',
        count: n.count ?? 0,
      },
      position: { x: n.x, y: n.y },
    }))
    const edgeEls = edges.map((e) => ({
      data: {
        id: e.id,
        source: e.source,
        target: e.target,
        kind: e.kind,
        score: e.score ?? 0,
        scoreLabel: typeof e.score === 'number' && Number.isFinite(e.score) ? e.score.toFixed(2) : '',
        color: e.color ?? 'rgba(192,210,244,0.24)',
        textColor: e.textColor ?? 'rgba(210,224,248,0.84)',
      },
      classes: e.kind.startsWith('sim_') ? 'sim' : '',
    }))
    return [...nodeEls, ...edgeEls]
  }, [edges, nodes])

  useEffect(() => {
    if (!ref.current) return
    const previousView = viewMemoryRef.current
    const cy = cytoscape({
      container: ref.current,
      elements,
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(label)',
            color: 'data(textColor)',
            'font-size': '10.5px',
            'font-weight': 500,
            'text-wrap': 'wrap',
            'text-max-width': '186px',
            'text-valign': 'center',
            'text-halign': 'center',
            'border-width': 1.2,
            'border-color': 'data(borderColor)',
            'background-color': 'data(fillColor)',
            'background-opacity': 1,
            'text-background-color': 'rgba(6,12,24,0.72)',
            'text-background-opacity': 0.92,
            'text-background-padding': '2px',
            'text-background-shape': 'roundrectangle',
            'text-border-color': 'rgba(188,208,244,0.3)',
            'text-border-width': 1,
            'text-border-opacity': 0.85,
          },
        },
        {
          selector: 'node[kind = "paper"]',
          style: {
            shape: 'roundrectangle',
            width: '300px',
            height: '48px',
            'font-size': '11.2px',
            'font-weight': 600,
            'text-wrap': 'wrap',
            'text-max-width': '282px',
            'text-background-opacity': 0.88,
            'border-width': 1.6,
          },
        },
        {
          selector: 'node[kind = "logic"]',
          style: {
            shape: 'roundrectangle',
            width: '220px',
            height: '50px',
            'font-size': '10.5px',
            'font-weight': 600,
            'text-max-width': '206px',
            'border-width': 1.4,
          },
        },
        {
          selector: 'node[kind = "claim_group"]',
          style: {
            shape: 'roundrectangle',
            width: '182px',
            height: '38px',
            'border-style': 'dashed',
            'font-size': '10px',
            'text-wrap': 'wrap',
            'text-max-width': '168px',
            'border-width': 1.4,
          },
        },
        {
          selector: 'node[kind = "claim"]',
          style: {
            shape: 'ellipse',
            width: '36px',
            height: '36px',
            'font-size': '9.5px',
            'font-weight': 700,
            'text-background-opacity': 0,
            'border-width': 1.3,
          },
        },
        {
          selector: 'edge',
          style: {
            width: 1.4,
            'line-color': 'data(color)',
            'target-arrow-color': 'data(color)',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'control-point-step-size': 26,
            'arrow-scale': 0.72,
            opacity: 0.76,
          },
        },
        {
          selector: 'edge[kind = "next"]',
          style: {
            width: 2.2,
            'control-point-step-size': 20,
            opacity: 0.84,
          },
        },
        {
          selector: 'edge[kind = "group"], edge[kind = "claim"]',
          style: {
            'target-arrow-shape': 'none',
            width: 1.12,
            'line-style': 'dotted',
            opacity: 0.66,
          },
        },
        {
          selector: 'edge.sim',
          style: {
            width: 1.2,
            'line-style': 'dashed',
            'line-color': 'data(color)',
            'target-arrow-color': 'data(color)',
            'target-arrow-shape': 'none',
            opacity: 0.24,
          },
        },
        {
          selector: 'edge.simActive',
          style: {
            width: 2.3,
            label: 'data(scoreLabel)',
            color: 'data(textColor)',
            'font-size': '8.5px',
            'font-weight': 600,
            'text-background-color': 'rgba(8,12,24,0.82)',
            'text-background-opacity': 1,
            'text-background-padding': '1px',
            'text-background-shape': 'roundrectangle',
            opacity: 0.95,
          },
        },
        {
          selector: '.faded',
          style: {
            opacity: 0.14,
            'text-opacity': 0.16,
            'line-opacity': 0.1,
          },
        },
        {
          selector: '.active',
          style: {
            opacity: 1,
            'text-opacity': 1,
            'line-opacity': 1,
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 3.2,
            'border-color': 'rgba(124,255,203,0.95)',
            'z-index-compare': 'manual',
            'z-index': 999,
          },
        },
      ],
      layout: { name: 'preset', fit: !previousView, padding: 40 } as unknown as cytoscape.LayoutOptions,
      minZoom: 0.24,
      maxZoom: 2.5,
      boxSelectionEnabled: true,
    })

    cyRef.current = cy
    if (typeof window !== 'undefined' && import.meta.env.DEV) {
      ;(window as Window & { __logicKgContentCy?: cytoscape.Core }).__logicKgContentCy = cy
    }
    if (previousView) {
      cy.zoom(previousView.zoom)
      cy.pan(previousView.pan)
    }

    function highlightSimilar(nodeId: string) {
      cy.edges('.sim').removeClass('simActive')
      cy.edges('.sim').filter((e) => e.data('source') === nodeId || e.data('target') === nodeId).addClass('simActive')
    }

    function focusNode(node: cytoscape.NodeSingular) {
      cy.edges('.sim').removeClass('simActive')
      cy.elements().removeClass('faded active')
      cy.elements().addClass('faded')
      const neighborhood = node.closedNeighborhood()
      neighborhood.removeClass('faded').addClass('active')
      highlightSimilar(node.id())
      cy.edges('.simActive').removeClass('faded').addClass('active')
      cy.nodes().unselect()
      node.select()
    }

    cy.on('tap', 'node', (evt) => {
      const id = String(evt.target.data('id') ?? '')
      if (!id) return
      focusNode(evt.target)
      onSelect?.(id)
    })

    cy.on('tap', (evt) => {
      if (evt.target !== cy) return
      cy.elements().removeClass('faded active')
      cy.edges('.sim').removeClass('simActive')
      cy.nodes().unselect()
      onSelect?.('')
    })

    return () => {
      viewMemoryRef.current = { zoom: cy.zoom(), pan: cy.pan() }
      if (typeof window !== 'undefined' && import.meta.env.DEV) {
        const devWindow = window as Window & { __logicKgContentCy?: cytoscape.Core }
        if (devWindow.__logicKgContentCy === cy) delete devWindow.__logicKgContentCy
      }
      cyRef.current = null
      cy.destroy()
    }
  }, [elements, onSelect])

  return (
    <div
      ref={ref}
      className="contentGraphCanvas"
      style={{
        height,
      }}
    />
  )
}
