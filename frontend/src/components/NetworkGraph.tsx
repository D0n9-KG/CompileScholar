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

export default function NetworkGraph({
  nodes,
  edges,
  onSelectNode,
  onToggleSelect,
  selectedIds,
  height = 620,
}: {
  nodes: NetworkNode[]
  edges: NetworkEdge[]
  onSelectNode?: (paperId: string) => void
  onToggleSelect?: (paperId: string) => void
  selectedIds?: string[]
  height?: number
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const cyRef = useRef<cytoscape.Core | null>(null)
  const elements = useMemo(() => {
    const nodeMap = new Map<string, NetworkNode>()
    for (const n of nodes) nodeMap.set(n.id, n)
    for (const e of edges) {
      if (!nodeMap.has(e.source)) nodeMap.set(e.source, { id: e.source })
      if (!nodeMap.has(e.target)) nodeMap.set(e.target, { id: e.target })
    }

    const nodeEls = Array.from(nodeMap.values()).map((n) => ({
      data: {
        id: n.id,
        label: n.paper_source ?? n.doi ?? (n.id.startsWith('doi:') ? n.id.slice(4, 14) : n.id.slice(0, 8)),
        title: n.title,
        year: n.year,
        ingested: n.ingested ? 'true' : 'false',
        in_scope: n.in_scope === false ? 'false' : 'true',
      },
    }))

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
            'text-wrap': 'wrap',
            'text-max-width': '120px',
            'text-background-color': 'rgba(0,0,0,0.55)',
            'text-background-opacity': 1,
            'text-background-padding': '3px',
            'text-background-shape': 'roundrectangle',
            'text-border-color': 'rgba(255,255,255,0.08)',
            'text-border-width': 1,
            'text-border-opacity': 0.65,
            'background-color': '#7cffcb',
            'background-opacity': 0.88,
            'border-width': 1,
            'border-color': 'rgba(255,255,255,0.35)',
            width: '24px',
            height: '24px',
          },
        },
        {
          selector: 'node[ingested = "false"]',
          style: {
            'background-color': 'rgba(255,255,255,0.08)',
            'background-opacity': 1,
            'border-color': 'rgba(255,255,255,0.28)',
            'border-width': 1.5,
            'border-style': 'dashed',
            width: '20px',
            height: '20px',
            color: 'rgba(255,255,255,0.78)',
          },
        },
        {
          selector: 'node[in_scope = "false"]',
          style: {
            'background-color': 'rgba(255,255,255,0.06)',
            'background-opacity': 1,
            'border-color': 'rgba(255,255,255,0.22)',
            'border-width': 1.5,
            'border-style': 'dashed',
            width: '20px',
            height: '20px',
            color: 'rgba(255,255,255,0.78)',
          },
        },
        {
          selector: 'edge',
          style: {
            width: 1.5,
            'line-color': 'rgba(255,255,255,0.22)',
            'line-opacity': 0.9,
            'target-arrow-color': 'rgba(255,255,255,0.22)',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
          },
        },
        {
          selector: 'edge[total_mentions > 3]',
          style: {
            width: 2.5,
            'line-color': 'rgba(110,115,255,0.65)',
            'target-arrow-color': 'rgba(110,115,255,0.65)',
          },
        },
        {
          selector: 'node:selected',
          style: {
            'background-color': '#6e73ff',
            'border-color': 'rgba(124,255,203,0.95)',
            'border-width': 3,
          },
        },
        {
          selector: '.faded',
          style: {
            opacity: 0.16,
            'text-opacity': 0.1,
            'line-opacity': 0.12,
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
          selector: 'node.highlighted',
          style: {
            'border-color': 'rgba(110,115,255,0.95)',
            'border-width': 2.5,
          },
        },
      ],
      layout: {
        name: 'breadthfirst',
        directed: true,
        padding: 20,
        spacingFactor: 1.2,
        animate: false,
      } as unknown as cytoscape.LayoutOptions,
      wheelSensitivity: 0.16,
    })
    cyRef.current = cy

    cy.on('tap', 'node', (evt) => {
      const n = evt.target
      const id = n.data('id') as string
      const oe = (evt as unknown as { originalEvent?: MouseEvent | PointerEvent }).originalEvent
      const multi = Boolean(oe && ('shiftKey' in oe ? (oe as MouseEvent).shiftKey : false))

      cy.elements().removeClass('faded highlighted')
      cy.elements().addClass('faded')
      n.closedNeighborhood().removeClass('faded').addClass('highlighted')
      n.removeClass('faded').addClass('highlighted')

      if (multi) {
        onToggleSelect?.(id)
      } else {
        n.select()
      }
      onSelectNode?.(id)
    })

    cy.on('tap', (evt) => {
      if (evt.target !== cy) return
      cy.elements().removeClass('faded highlighted')
    })

    return () => {
      cyRef.current = null
      cy.destroy()
    }
  }, [elements, onSelectNode, onToggleSelect])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    const set = new Set<string>((selectedIds ?? []).map(String))
    cy.nodes().removeClass('highlighted')
    for (const id of set) {
      const n = cy.$id(id)
      if (n && n.length) n.addClass('highlighted')
    }
  }, [selectedIds])

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
