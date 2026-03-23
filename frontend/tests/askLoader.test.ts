import { describe, expect, test } from 'vitest'

import { buildAskGraph, resolveAskGraph, type AskApiResponse } from '../src/loaders/ask'

describe('ask loader graph builder', () => {
  test('buildAskGraph keeps evidence nodes even when paper_source is missing', () => {
    const graph = buildAskGraph({
      answer: 'ok',
      evidence: [
        {
          md_path: 'C:\\papers\\02_1050\\paper.md',
          start_line: 10,
          end_line: 20,
          score: 0.9,
          snippet: 'evidence snippet',
        },
      ],
      graph_context: [],
      structured_knowledge: null,
    })

    const nodes = graph.filter((item) => item.group === 'nodes')
    expect(nodes.length).toBeGreaterThan(0)
    expect(nodes.some((item) => String(item.data.id).startsWith('paper_source:') || String(item.data.id).startsWith('paper_file:'))).toBe(
      true,
    )
  })

  test('resolveAskGraph falls back to overview graph when ask graph is empty', () => {
    const fallbackGraph = [
      {
        group: 'nodes' as const,
        data: { id: 'paper:seed', label: 'Seed Paper', kind: 'paper' },
      },
    ]

    const graph = resolveAskGraph(
      {
        answer: '',
        evidence: [],
        graph_context: [],
        structured_knowledge: null,
      },
      fallbackGraph,
    )

    expect(graph).toBe(fallbackGraph)
    expect(graph).toHaveLength(1)
  })

  test('buildAskGraph adds evidence nodes linked to paper nodes', () => {
    const graph = buildAskGraph({
      answer: 'ok',
      evidence: [
        {
          paper_source: 'paper-A',
          md_path: 'C:\\papers\\paper-A\\paper.md',
          start_line: 12,
          end_line: 18,
          snippet: 'snippet A',
        },
      ],
      graph_context: [],
      structured_knowledge: null,
    })

    const nodes = graph.filter((item) => item.group === 'nodes')
    const edges = graph.filter((item) => item.group === 'edges')
    expect(nodes.length).toBeGreaterThanOrEqual(2)
    expect(nodes.some((item) => String(item.data.id).startsWith('evidence:'))).toBe(true)
    expect(edges.some((item) => item.data.kind === 'evidenced_by')).toBe(true)
  })

  test('resolveAskGraph prefers a community-first ask graph over the fallback graph', () => {
    const fallbackGraph = [
      {
        group: 'nodes' as const,
        data: { id: 'paper:seed', label: 'Seed Paper', kind: 'paper' },
      },
    ]

    const graph = resolveAskGraph(
      {
        answer: 'ok',
        evidence: [],
        graph_context: [],
        fusion_evidence: [],
        structured_knowledge: null,
        grounding: [],
        structured_evidence: [
          {
            kind: 'community',
            source_id: 'gc:demo',
            community_id: 'gc:demo',
            text: 'Finite element stability community.',
            member_ids: ['cl-1', 'ke-1'],
            member_kinds: ['anchor', 'entity'],
            keyword_texts: ['finite element', 'stability'],
            score: 0.82,
          },
        ],
      } satisfies AskApiResponse,
      fallbackGraph,
    )

    expect(graph).not.toBe(fallbackGraph)
    expect(graph.some((item) => item.group === 'nodes' && item.data.kind === 'community')).toBe(true)
  })

  test('buildAskGraph keeps full description text for move/anchor/citation/entity node details', () => {
    const longAnchorText =
      'Mixing performance strongly correlates with impeller speed and fill level, and the effect remains stable across repeated trials without significant drift.'
    const longMoveText =
      'We first establish the baseline flow regime, then compare perturbation cases under controlled boundary conditions to isolate the dominant mixing factors.'
    const longCitation =
      'A comprehensive review on granular mixing mechanisms in rotating drum and ribbon systems'
    const longEvidence =
      'This evidence snippet includes detailed context about measurement setup, sampling interval, and confidence calibration to support the finding.'

    const graph = buildAskGraph({
      answer: 'ok',
      evidence: [
        {
          paper_source: 'paper-A',
          md_path: 'C:\\papers\\paper-A\\paper.md',
          start_line: 12,
          end_line: 18,
          snippet: longEvidence,
        },
      ],
      graph_context: [
        {
          paper_source: 'paper-A',
          cited_title: longCitation,
        },
      ],
      structured_knowledge: {
        research_moves: [{ paper_source: 'paper-A', move_id: 'mv-1', role: 'method', summary: longMoveText }],
        evidence_anchors: [{ paper_source: 'paper-A', anchor_id: 'ea-1', role: 'result', text: longAnchorText, confidence: 0.9 }],
      },
    })

    const nodes = graph.filter((item) => item.group === 'nodes').map((item) => item.data)
    const anchorNode = nodes.find((node) => node.kind === 'anchor')
    const moveNode = nodes.find((node) => node.kind === 'move')
    const citationNode = nodes.find((node) => node.kind === 'citation')
    const evidenceNode = nodes.find((node) => node.kind === 'entity' && String(node.id).startsWith('evidence:'))

    expect(anchorNode?.description).toContain('Mixing performance strongly correlates')
    expect(moveNode?.description).toContain('baseline flow regime')
    expect(citationNode?.description).toContain('granular mixing mechanisms')
    expect(evidenceNode?.description).toContain('measurement setup')
    expect(nodes.some((node) => node.kind === 'logic' || node.kind === 'claim')).toBe(false)
  })

  test('buildAskGraph prefers paper_title for paper node labels and descriptions', () => {
    const graph = buildAskGraph({
      answer: 'ok',
      evidence: [
        {
          paper_id: 'doi:10.1000/example',
          paper_source: 'paper-A',
          paper_title: 'A Unified Framework for Granular Mixing',
          md_path: 'C:\\papers\\paper-A\\paper.md',
          start_line: 8,
          end_line: 20,
          snippet: 'evidence snippet',
        },
      ],
      graph_context: [
        {
          paper_source: 'paper-A',
          cited_title: 'Related Prior Work',
        },
      ],
      structured_knowledge: {
        research_moves: [{ paper_source: 'paper-A', move_id: 'mv-1', role: 'method', summary: 'Method summary' }],
        evidence_anchors: [{ paper_source: 'paper-A', anchor_id: 'ea-1', role: 'result', text: 'Result finding', confidence: 0.9 }],
      },
    })

    const nodes = graph.filter((item) => item.group === 'nodes').map((item) => item.data)
    const paperNode = nodes.find((node) => node.id === 'paper:doi:10.1000/example')

    expect(paperNode?.kind).toBe('paper')
    expect(paperNode?.label).toBe('A Unified Framework for Granular Mixing')
    expect(paperNode?.description).toBe('A Unified Framework for Granular Mixing')
  })
})
