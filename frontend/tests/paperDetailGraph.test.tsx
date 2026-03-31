import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, test, vi } from 'vitest'

import { I18nProvider, LOCALE_STORAGE_KEY } from '../src/i18n'

const { apiGetMock } = vi.hoisted(() => ({
  apiGetMock: vi.fn(),
}))

vi.mock('../src/api', () => ({
  apiBaseUrl: () => 'http://127.0.0.1:8080',
  apiGet: apiGetMock,
  apiPatch: vi.fn(),
  apiPost: vi.fn(),
}))

vi.mock('../src/components/MarkdownView', () => ({
  default: ({ markdown }: { markdown: string }) => <div>{markdown}</div>,
}))

vi.mock('../src/components/SignalGraph', () => ({
  default: ({
    nodes,
    onSelect,
  }: {
    nodes: Array<{ id: string; label: string }>
    onSelect?: (nodeId: string) => void
  }) => (
    <div data-testid="signal-graph-mock">
      {nodes.map((node) => (
        <button key={node.id} onClick={() => onSelect?.(node.id)}>
          {node.label}
        </button>
      ))}
    </div>
  ),
}))

import PaperDetailPage from '../src/pages/PaperDetailPage'

const trace = {
  trace_id: 'trace:doi:10.1000/example',
  schema_version: 'v2',
  built_at: '2026-03-22T12:00:00Z',
  paper_metadata: {
    paper_id: 'doi:10.1000/example',
    canonical_doi: '10.1000/example',
    title: 'A Unified Study of Granular Stability',
    year: 2024,
    authors: ['Ada Researcher', 'Bo Analyst'],
    venue: 'Journal of Test Cases',
    paper_type: 'empirical',
    source_refs: ['chunk-1', 'chunk-2'],
  },
  canonical_core: {
    evidence_anchors: [
      {
        anchor_id: 'anchor-1',
        paper_id: 'doi:10.1000/example',
        source_ref: 'chunk-1',
        modality: 'text',
        section_path: ['Method'],
        locator: { start_line: 11, end_line: 20 },
        quote: 'We propose a finite-element solver for granular stability analysis.',
        citation_ids: [],
        support_type: 'direct',
        weak: false,
      },
      {
        anchor_id: 'anchor-2',
        paper_id: 'doi:10.1000/example',
        source_ref: 'chunk-2',
        modality: 'text',
        section_path: ['Result'],
        locator: { start_line: 42, end_line: 50 },
        quote: 'The method improves stability prediction over the baseline.',
        citation_ids: ['cite-1'],
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
        summary: 'Propose a finite-element solver.',
        research_objects: [
          {
            surface: 'granular stability',
            normalized: 'granular stability',
            anchor_ids: ['anchor-1'],
            confidence: 0.88,
            inferred: false,
          },
        ],
        methods: [
          {
            surface: 'finite-element solver',
            normalized: 'finite element solver',
            type: 'computational',
            anchor_ids: ['anchor-1'],
            confidence: 0.9,
            inferred: false,
          },
        ],
        observed_variables: [],
        metrics: [],
        comparators: [],
        conditions: [],
        effects: [],
        limitation_types: [],
        resource_mentions: [],
        anchor_ids: ['anchor-1'],
        slot_provenance: [
          {
            field: 'methods',
            value_index: 0,
            anchor_ids: ['anchor-1'],
            extraction_mode: 'direct',
            support_strength: 'strong',
            confidence: 0.9,
            notes: 'Method mention from the core proposal sentence.',
          },
        ],
        confidence: 0.83,
        audit_state: 'hot_path',
      },
      {
        move_id: 'move-2',
        sequence_no: 2,
        role: 'result',
        act_type: 'report_effect',
        summary: 'Improve stability prediction over the baseline.',
        research_objects: [],
        methods: [],
        observed_variables: [],
        metrics: [
          {
            surface: 'stability prediction',
            normalized: 'stability prediction',
            anchor_ids: ['anchor-2'],
            confidence: 0.84,
            inferred: false,
          },
        ],
        comparators: [
          {
            surface: 'baseline',
            normalized: 'baseline',
            anchor_ids: ['anchor-2'],
            confidence: 0.8,
            inferred: false,
          },
        ],
        conditions: [],
        effects: [
          {
            direction: 'improve',
            comparator_surface: 'baseline',
            anchor_ids: ['anchor-2'],
            confidence: 0.86,
          },
        ],
        limitation_types: [],
        resource_mentions: [],
        anchor_ids: ['anchor-2'],
        slot_provenance: [],
        confidence: 0.89,
        audit_state: 'hot_path',
      },
    ],
    move_relations: [
      {
        relation_id: 'rel-1',
        source_move_id: 'move-1',
        target_move_id: 'move-2',
        relation_type: 'yields',
        anchor_ids: ['anchor-2'],
        confidence: 0.8,
      },
    ],
    citation_acts: [
      {
        citation_act_id: 'cite-1',
        source_move_id: 'move-1',
        target_paper_id: 'doi:10.1000/cited',
        purpose: 'background',
        polarity: 'positive',
        semantic_signal: 'method_transfer_hint',
        target_scope: 'method',
        anchor_ids: ['anchor-1'],
        confidence: 0.71,
      },
    ],
    figure_refs: [],
    table_refs: [],
  },
  derived_views: {
    l2_5_slot_inventory: {},
    l1_bridge_hints: {},
    community_signatures: [],
    route_feature_candidates: [],
    paper_summaries: {
      one_paragraph_summary: 'This paper proposes a finite-element solver and reports stronger stability prediction.',
      move_role_distribution: {
        method: 1,
        result: 1,
      },
      key_method_summary: 'Propose a finite-element solver.',
    },
  },
  quality: {
    quality_tier: 'green',
    quality_tier_score: 0.94,
    quality_flags: [],
    audit_status: 'not_needed',
    l2_completeness_audit: {
      paper_type: 'empirical',
      move_count: 2,
      expected_roles: ['problem', 'method', 'result'],
      missing_expected_roles: ['problem'],
      missing_expected_slot_fields: [],
      sparse_expected_slot_fields: ['comparators'],
      noise_move_ids: [],
      ready_for_community: true,
      ready_for_l3: true,
      ready_for_l4: false,
      completeness_score: 0.78,
    },
  },
}

function renderPaperDetail() {
  apiGetMock.mockImplementation(async (path: string) => {
    if (path === '/papers/doi%3A10.1000%2Fexample/logic-trace') return trace
    if (path === '/papers/doi%3A10.1000%2Fexample/content') return '# Source Content\n\nOriginal markdown body.'
    throw new Error(`unexpected apiGet path: ${path}`)
  })

  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={['/papers/doi%3A10.1000%2Fexample']}>
        <Routes>
          <Route path="/papers/:paperId" element={<PaperDetailPage />} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>,
  )
}

function byTextContent(text: string) {
  return (_content: string, element: Element | null) => element?.textContent?.includes(text) ?? false
}

describe('PaperDetailPage paper logic trace workbench', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'en-US')
  })

  test('renders research move graph and shows a move detail card when a node is selected', async () => {
    const { container } = renderPaperDetail()

    await waitFor(() => expect(screen.getByTestId('signal-graph-mock')).toBeInTheDocument())
    expect(container.querySelector('.paperTraceWorkbench')).not.toBeNull()
    expect(screen.getAllByText('Trace Version').length).toBeGreaterThan(0)
    expect(screen.getByText('Trace Workspace')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '1. Method' }))

    const detailCard = container.querySelector('.paperTraceDetailCard')
    expect(detailCard).not.toBeNull()
    const detail = within(detailCard as HTMLElement)

    expect(detail.getByText('Node Detail')).toBeInTheDocument()
    expect(detail.getByText('Research Move')).toBeInTheDocument()
    expect(detail.getByText('Propose a finite-element solver.')).toBeInTheDocument()
    expect(detail.getByText('Role')).toBeInTheDocument()
    expect(detail.getByText('Act Type')).toBeInTheDocument()
    expect(detail.getByText('Summary')).toBeInTheDocument()
    expect(detail.getAllByText('Evidence Anchors').length).toBeGreaterThan(0)
    expect(detail.getAllByText(/Lines 11-20/i).length).toBeGreaterThan(0)
  })

  test('shows complete move detail sections including ids, slot values, and provenance', async () => {
    const { container } = renderPaperDetail()

    await waitFor(() => expect(screen.getByTestId('signal-graph-mock')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: '1. Method' }))

    const detail = within(container.querySelector('.paperTraceDetailCard') as HTMLElement)
    expect(detail.getByText('Move ID')).toBeInTheDocument()
    expect(detail.getByText(byTextContent('Move ID: move-1'), { selector: '.metaLine' })).toBeInTheDocument()
    expect(detail.getByText('Audit State')).toBeInTheDocument()
    expect(detail.getByText(byTextContent('Audit State: hot_path'), { selector: '.metaLine' })).toBeInTheDocument()
    expect(detail.getByText('Research Objects')).toBeInTheDocument()
    expect(detail.getByText(/granular stability/i, { selector: '.metaLine' })).toBeInTheDocument()
    expect(detail.getByText('Methods')).toBeInTheDocument()
    expect(detail.getByText(/finite element solver/i, { selector: '.metaLine' })).toBeInTheDocument()
    expect(detail.getByText('Slot Provenance')).toBeInTheDocument()
    expect(detail.getByText(/Method mention from the core proposal sentence/i)).toBeInTheDocument()
  })

  test('shows complete anchor detail sections including source and citation metadata', async () => {
    const { container } = renderPaperDetail()

    await waitFor(() => expect(screen.getByTestId('signal-graph-mock')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: 'anchor-2' }))

    const detail = within(container.querySelector('.paperTraceDetailCard') as HTMLElement)
    expect(detail.getByText('Evidence Anchor')).toBeInTheDocument()
    expect(detail.getByText('anchor-2')).toBeInTheDocument()
    expect(detail.getByText('Source Ref')).toBeInTheDocument()
    expect(detail.getByText(byTextContent('Source Ref: chunk-2'), { selector: '.metaLine' })).toBeInTheDocument()
    expect(detail.getByText('Modality')).toBeInTheDocument()
    expect(detail.getByText(byTextContent('Modality: text'), { selector: '.metaLine' })).toBeInTheDocument()
    expect(detail.getByText('Citation IDs')).toBeInTheDocument()
    expect(detail.getByText('cite-1')).toBeInTheDocument()
  })

  test('shows rich citation detail with source move, target paper, and supporting anchors', async () => {
    const { container } = renderPaperDetail()

    await waitFor(() => expect(screen.getByTestId('signal-graph-mock')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: 'cite-1' }))

    const detail = within(container.querySelector('.paperTraceDetailCard') as HTMLElement)
    expect(detail.getByText('Citation Act')).toBeInTheDocument()
    expect(detail.getByText('cite-1')).toBeInTheDocument()
    expect(detail.getByText('Source Move')).toBeInTheDocument()
    expect(detail.getByText(byTextContent('Source Move: 1. Method | move-1'), { selector: '.metaLine' })).toBeInTheDocument()
    expect(detail.getByText('Target Paper')).toBeInTheDocument()
    expect(detail.getByText(byTextContent('Target Paper: doi:10.1000/cited'), { selector: '.metaLine' })).toBeInTheDocument()
    expect(detail.getByText('Purpose')).toBeInTheDocument()
    expect(detail.getByText(byTextContent('Purpose: background'), { selector: '.metaLine' })).toBeInTheDocument()
    expect(detail.getByText('Semantic Signal')).toBeInTheDocument()
    expect(detail.getByText(/method_transfer_hint/i)).toBeInTheDocument()
    expect(detail.getByText('Supporting Anchors')).toBeInTheDocument()
    expect(detail.getByText(/anchor-1 \| Lines 11-20 \| Method \| chunk-1/i)).toBeInTheDocument()
  })

  test('shows localized quality, completeness, and readiness summary for the current trace', async () => {
    renderPaperDetail()

    await waitFor(() => expect(screen.getByTestId('signal-graph-mock')).toBeInTheDocument())

    expect(screen.getByTestId('trace-quality-tier')).toHaveTextContent('High Confidence')
    expect(screen.getByTestId('trace-completeness-score')).toHaveTextContent('78%')
    expect(screen.getByTestId('trace-ready-l3')).toHaveTextContent(/ready/i)
    expect(screen.getByTestId('trace-ready-l4')).toHaveTextContent(/hold/i)
    expect(screen.getByTestId('trace-missing-roles')).toHaveTextContent(/problem/i)
    expect(screen.getByTestId('trace-sparse-slots')).toHaveTextContent(/comparators/i)
  })

  test('renders move, relation, evidence, and citation views without legacy logic-step or claim tabs', async () => {
    renderPaperDetail()

    await waitFor(() => expect(screen.getAllByText('Research Moves').length).toBeGreaterThan(0))
    expect(screen.getByText('Move Relations')).toBeInTheDocument()
    expect(screen.getByText('Evidence Anchors')).toBeInTheDocument()
    expect(screen.getByText('Citation Acts')).toBeInTheDocument()
    expect(screen.queryByText('Logic Steps')).not.toBeInTheDocument()
    expect(screen.queryByText('Claims')).not.toBeInTheDocument()

    expect(screen.getAllByText('1. Method').length).toBeGreaterThan(0)
    expect(screen.getAllByText('2. Result').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Propose a finite-element solver.').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('button', { name: 'Move Relations' }))
    expect(await screen.findByText('Yields')).toBeInTheDocument()
    expect(screen.getByText('1. Method -> 2. Result')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Evidence Anchors' }))
    await waitFor(() => expect(screen.getAllByText('anchor-1').length).toBeGreaterThan(0))
    expect(screen.getByText(/finite-element solver for granular stability analysis/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Citation Acts' }))
    await waitFor(() => expect(screen.getAllByText('cite-1').length).toBeGreaterThan(0))
    expect(screen.getByText(/method_transfer_hint/)).toBeInTheDocument()
  })
})
