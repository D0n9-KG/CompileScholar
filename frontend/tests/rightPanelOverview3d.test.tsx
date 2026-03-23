import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, test, vi } from 'vitest'

import type { GlobalState } from '../src/state/types'
import { INITIAL_STATE } from '../src/state/store'

const {
  apiGetMock,
  dispatchMock,
  loadOverviewCommunity3DGraphMock,
  loadOverviewCommunitySubgraphMock,
  loadOverviewGraphMock,
} = vi.hoisted(() => ({
  apiGetMock: vi.fn(),
  dispatchMock: vi.fn(),
  loadOverviewCommunity3DGraphMock: vi.fn(),
  loadOverviewCommunitySubgraphMock: vi.fn(),
  loadOverviewGraphMock: vi.fn(),
}))

let mockedState: GlobalState = INITIAL_STATE

vi.mock('../src/api', () => ({
  apiGet: apiGetMock,
}))

vi.mock('../src/loaders/overview', async () => {
  const actual = await vi.importActual<typeof import('../src/loaders/overview')>('../src/loaders/overview')
  return {
    ...actual,
    loadOverviewCommunity3DGraph: loadOverviewCommunity3DGraphMock,
    loadOverviewCommunitySubgraph: loadOverviewCommunitySubgraphMock,
    loadOverviewGraph: loadOverviewGraphMock,
  }
})

vi.mock('../src/state/store', async () => {
  const actual = await vi.importActual<typeof import('../src/state/store')>('../src/state/store')
  return {
    ...actual,
    useGlobalState: () => ({
      state: mockedState,
      dispatch: dispatchMock,
      switchModule: vi.fn(),
    }),
  }
})

vi.mock('../src/i18n', async () => {
  const actual = await vi.importActual<typeof import('../src/i18n')>('../src/i18n')
  return {
    ...actual,
    useI18n: () => ({
      locale: 'en-US',
      setLocale: vi.fn(),
      t: (_zh: string, en: string) => en,
    }),
  }
})

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return {
    ...actual,
    useNavigate: () => vi.fn(),
  }
})

import RightPanel from '../src/components/RightPanel'

describe('RightPanel overview 3D node details', () => {
  beforeEach(() => {
    apiGetMock.mockReset()
    dispatchMock.mockReset()
    loadOverviewCommunity3DGraphMock.mockReset()
    loadOverviewCommunitySubgraphMock.mockReset()
    loadOverviewGraphMock.mockReset()
    apiGetMock.mockResolvedValue({
      trace_id: 'trace:paper-1',
      schema_version: 'v2',
      built_at: '2026-03-22T00:00:00Z',
      paper_source: 'P-001',
      paper_metadata: {
        paper_id: 'paper-1',
        title: 'Alpha Study',
        paper_type: 'empirical',
        source_refs: ['chunk:1'],
      },
      canonical_core: {
        evidence_anchors: [
          {
            anchor_id: 'anchor-1',
            paper_id: 'paper-1',
            source_ref: 'chunk:1',
            modality: 'text',
            section_path: ['Method'],
            locator: { start_line: 10, end_line: 16 },
            quote: 'Anchor summary',
            citation_ids: [],
            support_type: 'direct',
            weak: false,
          },
        ],
        moves: [
          {
            move_id: 'move-1',
            sequence_no: 1,
            role: 'method',
            act_type: 'propose_method',
            summary: 'Method summary',
            anchor_ids: ['anchor-1'],
            confidence: 0.8,
          },
        ],
        move_relations: [],
        citation_acts: [],
        figure_refs: [],
        table_refs: [],
      },
      derived_views: {},
      quality: { quality_tier: 'green' },
    })
    mockedState = {
      ...INITIAL_STATE,
      activeModule: 'overview',
      graphElements: [
        {
          group: 'nodes',
          data: {
            id: 'paper:paper-1',
            label: 'P-001',
            description: 'Alpha Study',
            kind: 'paper',
            paperId: 'paper-1',
          },
        },
      ],
      selectedNode: {
        id: 'anchor:anchor-1',
        kind: 'anchor',
        label: 'Alpha anchor with the strongest signal.',
      },
    }
  })

  test('loads 3D overview graph context when the selected node is absent from the main overview graph', async () => {
    loadOverviewCommunity3DGraphMock.mockResolvedValue([
      {
        group: 'nodes',
        data: {
          id: 'community:gc:alpha',
          label: 'Alpha stability',
          kind: 'community',
          description: 'Top keywords: alpha, fem',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
        },
      },
      {
        group: 'nodes',
        data: {
          id: 'anchor:anchor-1',
          label: 'Alpha anchor with the strongest signal.',
          kind: 'anchor',
          description: 'Alpha anchor with the strongest signal.',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
          paperId: 'paper-1',
          paperSource: 'P-001',
          paperTitle: 'Alpha Study',
          role: 'Method',
        },
      },
      {
        group: 'edges',
        data: {
          id: 'contains:community:gc:alpha->anchor:anchor-1',
          source: 'community:gc:alpha',
          target: 'anchor:anchor-1',
          kind: 'contains',
          weight: 0.92,
        },
      },
    ])

    render(<RightPanel collapsed={false} onToggle={() => {}} />)

    await waitFor(() => expect(loadOverviewCommunity3DGraphMock).toHaveBeenCalledTimes(1))
    expect(loadOverviewCommunity3DGraphMock).toHaveBeenCalledWith()
    await waitFor(() => expect(screen.getAllByText('P-001').length).toBeGreaterThan(0))
    expect(screen.getAllByText('Alpha Study').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: 'View Research Moves' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View Evidence Anchors' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'View Logic Steps' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'View Claims' })).not.toBeInTheDocument()
  })

  test('falls back to paper trace preview metadata when a 3D anchor node is missing paper source', async () => {
    loadOverviewCommunity3DGraphMock.mockResolvedValue([
      {
        group: 'nodes',
        data: {
          id: 'community:gc:alpha',
          label: 'Alpha stability',
          kind: 'community',
          description: 'Top keywords: alpha, fem',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
        },
      },
      {
        group: 'nodes',
        data: {
          id: 'anchor:anchor-1',
          label: 'Alpha anchor with the strongest signal.',
          kind: 'anchor',
          description: 'Alpha Study | Method | Alpha anchor with the strongest signal.',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
          paperId: 'paper-1',
          paperTitle: 'Alpha Study',
          role: 'Method',
        },
      },
      {
        group: 'edges',
        data: {
          id: 'contains:community:gc:alpha->anchor:anchor-1',
          source: 'community:gc:alpha',
          target: 'anchor:anchor-1',
          kind: 'contains',
          weight: 0.92,
        },
      },
    ])

    render(<RightPanel collapsed={false} onToggle={() => {}} />)

    await waitFor(() => expect(loadOverviewCommunity3DGraphMock).toHaveBeenCalledTimes(1))
    expect(loadOverviewCommunity3DGraphMock).toHaveBeenCalledWith()
    await waitFor(() => expect(screen.getAllByText('P-001').length).toBeGreaterThan(0))
  })

  test('expands the selected overview community into its own subgraph from the sidebar action', async () => {
    loadOverviewCommunity3DGraphMock.mockResolvedValue([
      {
        group: 'nodes',
        data: {
          id: 'community:gc:alpha',
          label: 'Alpha stability',
          kind: 'community',
          description: 'Top keywords: alpha, fem',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
        },
      },
      {
        group: 'nodes',
        data: {
          id: 'anchor:anchor-1',
          label: 'Alpha anchor with the strongest signal.',
          kind: 'anchor',
          description: 'Alpha Study | Method | Alpha anchor with the strongest signal.',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
          paperId: 'paper-1',
          paperSource: 'P-001',
          paperTitle: 'Alpha Study',
          role: 'Method',
        },
      },
      {
        group: 'edges',
        data: {
          id: 'contains:community:gc:alpha->anchor:anchor-1',
          source: 'community:gc:alpha',
          target: 'anchor:anchor-1',
          kind: 'contains',
          weight: 0.92,
        },
      },
    ])
    loadOverviewCommunitySubgraphMock.mockResolvedValue([
      {
        group: 'nodes',
        data: {
          id: 'community:gc:alpha',
          label: 'Alpha stability',
          kind: 'community',
          description: 'Focus on alpha pathways',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
        },
      },
      {
        group: 'nodes',
        data: {
          id: 'anchor:anchor-1',
          label: 'Alpha anchor with the strongest signal.',
          kind: 'anchor',
          description: 'Alpha Study | Method | Alpha anchor with the strongest signal.',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
          paperId: 'paper-1',
          paperSource: 'P-001',
          paperTitle: 'Alpha Study',
          role: 'Method',
        },
      },
      {
        group: 'nodes',
        data: {
          id: 'move:move-1',
          label: 'Method pathway that explains the alpha workflow.',
          kind: 'move',
          description: 'Alpha Study | Method pathway that explains the alpha workflow.',
          communityId: 'gc:alpha',
          clusterKey: 'community:gc:alpha',
          paperId: 'paper-1',
          paperSource: 'P-001',
          paperTitle: 'Alpha Study',
          role: 'Method',
        },
      },
      {
        group: 'edges',
        data: {
          id: 'contains:community:gc:alpha->anchor:anchor-1',
          source: 'community:gc:alpha',
          target: 'anchor:anchor-1',
          kind: 'contains',
          weight: 0.92,
        },
      },
      {
        group: 'edges',
        data: {
          id: 'contains:community:gc:alpha->move:move-1',
          source: 'community:gc:alpha',
          target: 'move:move-1',
          kind: 'contains',
          weight: 0.81,
        },
      },
    ])

    render(<RightPanel collapsed={false} onToggle={() => {}} />)

    const button = await screen.findByRole('button', { name: 'Expand Community' })
    fireEvent.click(button)

    await waitFor(() => expect(loadOverviewCommunitySubgraphMock).toHaveBeenCalledWith('gc:alpha'))
    await waitFor(() =>
      expect(dispatchMock).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'SET_GRAPH',
          layout: 'cose',
        }),
      ),
    )
    expect(dispatchMock).toHaveBeenCalledWith({
      type: 'SET_SELECTED',
      node: expect.objectContaining({
        id: 'community:gc:alpha',
        kind: 'community',
        label: 'Alpha stability',
      }),
    })
  })

  test('offers a return action when the overview is already focused on a single community subgraph', async () => {
    mockedState = {
      ...INITIAL_STATE,
      activeModule: 'overview',
      graphElements: [
        {
          group: 'nodes',
          data: {
            id: 'community:gc:alpha',
            label: 'Alpha stability',
            kind: 'community',
            description: 'Focus on alpha pathways',
            communityId: 'gc:alpha',
            clusterKey: 'community:gc:alpha',
          },
        },
        {
          group: 'nodes',
          data: {
            id: 'anchor:anchor-1',
            label: 'Alpha anchor with the strongest signal.',
            kind: 'anchor',
            description: 'Alpha Study | Method | Alpha anchor with the strongest signal.',
            communityId: 'gc:alpha',
            clusterKey: 'community:gc:alpha',
            paperId: 'paper-1',
            paperSource: 'P-001',
            paperTitle: 'Alpha Study',
            role: 'Method',
          },
        },
        {
          group: 'edges',
          data: {
            id: 'contains:community:gc:alpha->anchor:anchor-1',
            source: 'community:gc:alpha',
            target: 'anchor:anchor-1',
            kind: 'contains',
            weight: 0.92,
          },
        },
      ],
      selectedNode: {
        id: 'community:gc:alpha',
        kind: 'community',
        label: 'Alpha stability',
      },
    }
    loadOverviewGraphMock.mockResolvedValue([
      {
        group: 'nodes',
        data: {
          id: 'paper:paper-1',
          label: 'P-001',
          description: 'Alpha Study',
          kind: 'paper',
          paperId: 'paper-1',
        },
      },
    ])

    render(<RightPanel collapsed={false} onToggle={() => {}} />)

    const buttons = await screen.findAllByRole('button', { name: 'Return to Overview' })
    fireEvent.click(buttons[0])

    await waitFor(() => expect(loadOverviewGraphMock).toHaveBeenCalledTimes(1))
    await waitFor(() =>
      expect(dispatchMock).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'SET_GRAPH',
          layout: 'cose',
        }),
      ),
    )
  })
})
