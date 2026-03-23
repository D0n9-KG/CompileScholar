import { beforeEach, describe, expect, test, vi } from 'vitest'

const { apiGetMock } = vi.hoisted(() => ({
  apiGetMock: vi.fn(),
}))

vi.mock('../src/api', () => ({
  apiGet: apiGetMock,
}))

import {
  invalidateOverviewCommunity3DGraphCache,
  loadOverviewCommunitySubgraph,
  loadOverviewCommunity3DGraph,
  resolveOverviewExpandedCommunityId,
} from '../src/loaders/overview'

describe('overviewCommunity3dLoader', () => {
  beforeEach(() => {
    apiGetMock.mockReset()
    invalidateOverviewCommunity3DGraphCache()
  })

  test('maps the capped community overview payload into clustered graph elements', async () => {
    apiGetMock.mockResolvedValue({
      nodes: [
        {
          id: 'community:gc:alpha',
          label: 'Alpha stability',
          kind: 'community',
          description: 'Top keywords: alpha, fem',
          cluster_key: 'community:gc:alpha',
          community_id: 'gc:alpha',
          keywords: ['alpha', 'fem', 'stability'],
        },
        {
          id: 'community:gc:beta',
          label: 'Beta transfer',
          kind: 'community',
          description: 'Top keywords: beta, transfer',
          cluster_key: 'community:gc:beta',
          community_id: 'gc:beta',
          keywords: ['beta', 'transfer'],
        },
      ],
      edges: [
        {
          id: 'similar:community:gc:alpha->community:gc:beta',
          source: 'community:gc:alpha',
          target: 'community:gc:beta',
          kind: 'similar',
          weight: 0.68,
        },
      ],
      stats: {
        community_total: 2,
        visible_communities: 2,
        visible_members: 0,
        truncated: false,
      },
    })

    const graph = await loadOverviewCommunity3DGraph({
      communityLimit: 12,
      maxNodes: 120,
      maxEdges: 180,
    })

    const nodes = graph.filter((item) => item.group === 'nodes').map((item) => item.data)
    const edges = graph.filter((item) => item.group === 'edges').map((item) => item.data)

    expect(apiGetMock).toHaveBeenCalledWith(
      '/community/overview-graph?community_limit=12&max_nodes=120&max_edges=180&include_members=false',
    )
    expect(nodes).toEqual([
      expect.objectContaining({
        id: 'community:gc:alpha',
        kind: 'community',
        clusterKey: 'community:gc:alpha',
        communityId: 'gc:alpha',
        keywords: ['alpha', 'fem', 'stability'],
      }),
      expect.objectContaining({
        id: 'community:gc:beta',
        kind: 'community',
        clusterKey: 'community:gc:beta',
        communityId: 'gc:beta',
        keywords: ['beta', 'transfer'],
      }),
    ])
    expect(edges).toEqual([
      expect.objectContaining({
        id: 'similar:community:gc:alpha->community:gc:beta',
        source: 'community:gc:alpha',
        target: 'community:gc:beta',
        kind: 'similar',
      }),
    ])
  })

  test('uses the all-community overview 3D defaults', async () => {
    apiGetMock.mockResolvedValue({ nodes: [], edges: [] })

    await loadOverviewCommunity3DGraph()

    expect(apiGetMock).toHaveBeenCalledWith(
      '/community/overview-graph?community_limit=400&max_nodes=400&max_edges=620&include_members=false',
    )
  })

  test('maps a community detail payload into an expandable community subgraph', async () => {
    apiGetMock.mockResolvedValue({
      community_id: 'gc:alpha',
      title: 'Alpha stability',
      summary: 'Focus on alpha pathways',
      keywords: ['alpha', 'fem'],
      member_count: 3,
      members: [
        {
          member_id: 'anchor-1',
          member_kind: 'EvidenceAnchor',
          text: 'Alpha anchor with the strongest signal.',
          paper_id: 'paper-1',
          paper_source: 'P-001',
          paper_title: 'Alpha Study',
          role: 'method',
        },
        {
          member_id: 'move-1',
          member_kind: 'ResearchMove',
          text: 'Logic chain that explains the alpha workflow.',
          paper_id: 'paper-1',
          paper_source: 'P-001',
          paper_title: 'Alpha Study',
          role: 'method',
        },
        {
          member_id: 'entity-1',
          member_kind: 'KnowledgeEntity',
          text: 'Graphene',
          source_chapter_id: 'chapter-9',
        },
      ],
    })

    const graph = await loadOverviewCommunitySubgraph('gc:alpha', { memberLimit: 500 })

    const nodes = graph.filter((item) => item.group === 'nodes').map((item) => item.data)
    const edges = graph.filter((item) => item.group === 'edges').map((item) => item.data)

    expect(apiGetMock).toHaveBeenCalledWith('/community/gc%3Aalpha?member_limit=500')
    expect(nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: 'community:gc:alpha',
          kind: 'community',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
        }),
        expect.objectContaining({
          id: 'anchor:anchor-1',
          kind: 'anchor',
          communityId: 'gc:alpha',
          paperId: 'paper-1',
          paperSource: 'P-001',
          paperTitle: 'Alpha Study',
          role: 'method',
        }),
        expect.objectContaining({
          id: 'move:move-1',
          kind: 'move',
          communityId: 'gc:alpha',
          paperId: 'paper-1',
        }),
        expect.objectContaining({
          id: 'entity:entity-1',
          kind: 'entity',
          communityId: 'gc:alpha',
          chapterId: 'chapter-9',
        }),
      ]),
    )
    expect(edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: 'contains:community:gc:alpha->anchor:anchor-1',
          source: 'community:gc:alpha',
          target: 'anchor:anchor-1',
          kind: 'contains',
        }),
        expect.objectContaining({
          id: 'contains:community:gc:alpha->move:move-1',
          source: 'community:gc:alpha',
          target: 'move:move-1',
          kind: 'contains',
        }),
        expect.objectContaining({
          id: 'contains:community:gc:alpha->entity:entity-1',
          source: 'community:gc:alpha',
          target: 'entity:entity-1',
          kind: 'contains',
        }),
      ]),
    )
  })

  test('detects when the current overview graph has been expanded into a single community subgraph', () => {
    const expandedCommunityId = resolveOverviewExpandedCommunityId([
      {
        group: 'nodes',
        data: {
          id: 'community:gc:alpha',
          label: 'Alpha stability',
          kind: 'community',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
        },
      },
      {
        group: 'nodes',
        data: {
          id: 'anchor:anchor-1',
          label: 'Alpha anchor',
          kind: 'anchor',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
        },
      },
      {
        group: 'edges',
        data: {
          id: 'contains:community:gc:alpha->anchor:anchor-1',
          source: 'community:gc:alpha',
          target: 'anchor:anchor-1',
          kind: 'contains',
          weight: 1,
        },
      },
    ])

    const overviewCommunityId = resolveOverviewExpandedCommunityId([
      {
        group: 'nodes',
        data: {
          id: 'community:gc:alpha',
          label: 'Alpha stability',
          kind: 'community',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
        },
      },
      {
        group: 'nodes',
        data: {
          id: 'community:gc:beta',
          label: 'Beta design',
          kind: 'community',
          communityId: 'gc:beta',
          clusterKey: 'community:gc:beta',
        },
      },
    ])

    expect(expandedCommunityId).toBe('gc:alpha')
    expect(overviewCommunityId).toBeNull()
  })
})
