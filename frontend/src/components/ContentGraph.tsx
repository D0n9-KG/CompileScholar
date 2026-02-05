import cytoscape from 'cytoscape'
import { useEffect, useMemo, useRef } from 'react'

export type ContentNodeKind = 'paper' | 'logic' | 'claim_group' | 'claim'

export type ContentNode = {
  id: string
  kind: ContentNodeKind
  label: string
  paperId: string
  lane: number
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

  const elements = useMemo(() => {
    const nodeEls = nodes.map((n) => ({
      data: {
        id: n.id,
        kind: n.kind,
        label: n.label,
        paperId: n.paperId,
        lane: String(n.lane),
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
      },
      classes: e.kind.startsWith('sim_') ? 'sim' : '',
    }))
    return [...nodeEls, ...edgeEls]
  }, [edges, nodes])

  useEffect(() => {
    if (!ref.current) return
    const cy = cytoscape({
      container: ref.current,
      elements,
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(label)',
            color: 'rgba(255,255,255,0.92)',
            'font-size': '11px',
            'text-wrap': 'none',
            'text-valign': 'center',
            'text-halign': 'center',
            'border-width': 1,
            'border-color': 'rgba(255,255,255,0.18)',
            'background-color': 'rgba(124,255,203,0.10)',
            'background-opacity': 1,
          },
        },
        {
          selector: 'node[kind = "paper"]',
          style: {
            shape: 'roundrectangle',
            width: '220px',
            height: '36px',
            'background-color': 'rgba(110,115,255,0.22)',
            'border-color': 'rgba(110,115,255,0.55)',
            'font-size': '12px',
            'text-wrap': 'wrap',
            'text-max-width': '210px',
          },
        },
        {
          selector: 'node[kind = "logic"]',
          style: {
            shape: 'roundrectangle',
            width: '200px',
            height: '44px',
            'background-color': 'rgba(124,255,203,0.16)',
            'border-color': 'rgba(124,255,203,0.35)',
            'font-weight': 600,
          },
        },
        {
          selector: 'node[kind = "claim_group"]',
          style: {
            shape: 'roundrectangle',
            width: '140px',
            height: '34px',
            'background-color': 'rgba(255,255,255,0.06)',
            'border-color': 'rgba(255,255,255,0.22)',
            'border-style': 'dashed',
            'font-size': '10.5px',
            'text-wrap': 'wrap',
            'text-max-width': '130px',
          },
        },
        {
          selector: 'node[kind = "claim"]',
          style: {
            shape: 'ellipse',
            width: '38px',
            height: '38px',
            'background-color': 'rgba(110,115,255,0.16)',
            'border-color': 'rgba(110,115,255,0.35)',
            'font-size': '10px',
            'font-weight': 700,
          },
        },
        {
          selector: 'edge',
          style: {
            width: 1.5,
            'line-color': 'rgba(255,255,255,0.18)',
            'target-arrow-color': 'rgba(255,255,255,0.18)',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
          },
        },
        {
          selector: 'edge[kind = "next"]',
          style: {
            width: 2,
            'line-color': 'rgba(124,255,203,0.45)',
            'target-arrow-color': 'rgba(124,255,203,0.45)',
          },
        },
        {
          selector: 'edge[kind = "group"], edge[kind = "claim"]',
          style: {
            'target-arrow-shape': 'none',
            width: 1.2,
            'line-color': 'rgba(255,255,255,0.14)',
          },
        },
        {
          selector: 'edge.sim',
          style: {
            width: 1,
            'line-style': 'dashed',
            'line-color': 'rgba(110,115,255,0.10)',
            'target-arrow-color': 'rgba(110,115,255,0.10)',
            opacity: 0.12,
          },
        },
        {
          selector: 'edge.simActive',
          style: {
            width: 2.5,
            'line-color': 'rgba(110,115,255,0.72)',
            'target-arrow-color': 'rgba(110,115,255,0.72)',
            opacity: 0.95,
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 3,
            'border-color': 'rgba(124,255,203,0.95)',
            'background-color': 'rgba(110,115,255,0.22)',
          },
        },
      ],
      layout: { name: 'preset', fit: true, padding: 20 } as unknown as cytoscape.LayoutOptions,
      wheelSensitivity: 0.16,
      boxSelectionEnabled: true,
    })

    cyRef.current = cy

    function highlightSimilar(nodeId: string) {
      cy.edges('.sim').removeClass('simActive')
      cy.edges('.sim').filter((e) => e.data('source') === nodeId || e.data('target') === nodeId).addClass('simActive')
    }

    cy.on('tap', 'node', (evt) => {
      const id = String(evt.target.data('id') ?? '')
      if (!id) return
      highlightSimilar(id)
      onSelect?.(id)
    })

    cy.on('tap', (evt) => {
      if (evt.target !== cy) return
      cy.edges('.sim').removeClass('simActive')
      cy.nodes().unselect()
      onSelect?.('')
    })

    return () => {
      cyRef.current = null
      cy.destroy()
    }
  }, [elements, onSelect])

  return (
    <div
      ref={ref}
      style={{
        height,
        borderRadius: 16,
        border: '1px solid rgba(255,255,255,0.1)',
        background: 'rgba(0,0,0,0.08)',
        boxShadow: 'inset 0 0 0 1px rgba(255,255,255,0.02)',
      }}
    />
  )
}
