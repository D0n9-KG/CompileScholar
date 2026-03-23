import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, test, vi } from 'vitest'

const { apiGetMock } = vi.hoisted(() => ({
  apiGetMock: vi.fn(),
}))

vi.mock('../src/api', () => ({
  apiGet: apiGetMock,
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

import TextbookDetailPage from '../src/pages/TextbookDetailPage'

describe('TextbookDetailPage paper trace adaptation', () => {
  beforeEach(() => {
    apiGetMock.mockReset()
    apiGetMock.mockImplementation(async (path: string) => {
      if (path === '/textbooks/tb-1') {
        return {
          textbook_id: 'tb-1',
          title: 'Continuum Mechanics',
          authors: ['A. Author'],
          year: 2020,
          edition: '2nd',
          doc_type: 'textbook',
          total_chapters: 1,
          chapters: [
            {
              chapter_id: 'ch-1',
              chapter_num: 1,
              title: 'Finite Element Foundations',
              entity_count: 2,
              relation_count: 1,
            },
          ],
        }
      }
      if (path === '/graph/papers?limit=180') {
        return {
          papers: [
            {
              paper_id: 'doi:10.1000/example',
              title: 'Finite Element Stability Study',
              paper_source: 'paper-A',
              year: 2024,
            },
          ],
        }
      }
      if (path === '/textbooks/tb-1/chapters/ch-1/entities') {
        return {
          entities: [
            {
              entity_id: 'ent-1',
              name: 'Finite Element',
              entity_type: 'method',
              description: 'A numerical method',
            },
          ],
          relations: [],
        }
      }
      if (path === '/papers/doi%3A10.1000%2Fexample/logic-trace') {
        return {
          trace_id: 'trace:doi:10.1000/example',
          schema_version: 'v2',
          built_at: '2026-03-22T00:00:00Z',
          paper_metadata: {
            paper_id: 'doi:10.1000/example',
            title: 'Finite Element Stability Study',
            paper_type: 'empirical',
            source_refs: ['chunk:1'],
          },
          canonical_core: {
            evidence_anchors: [
              {
                anchor_id: 'anchor-1',
                paper_id: 'doi:10.1000/example',
                source_ref: 'chunk:1',
                modality: 'text',
                section_path: ['Method'],
                locator: { start_line: 11, end_line: 16 },
                quote: 'We ground the solver in finite-element stability analysis.',
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
                summary: 'Propose a finite-element solver.',
                anchor_ids: ['anchor-1'],
                confidence: 0.86,
              },
            ],
            move_relations: [],
            citation_acts: [],
            figure_refs: [],
            table_refs: [],
          },
          derived_views: {},
          quality: { quality_tier: 'green' },
        }
      }
      throw new Error(`unexpected apiGet path: ${path}`)
    })
  })

  test('loads paper logic traces for linked papers and renders move/anchor language instead of logic/claim language', async () => {
    render(
      <MemoryRouter initialEntries={['/textbooks/tb-1']}>
        <Routes>
          <Route path="/textbooks/:textbookId" element={<TextbookDetailPage />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('章节到论文的智能桥接')).toBeInTheDocument()
    expect((await screen.findAllByText(/Finite Element Stability Study/)).length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('button', { name: /^Finite Element Stability Study$/ }))

    await waitFor(() =>
      expect(apiGetMock).toHaveBeenCalledWith('/papers/doi%3A10.1000%2Fexample/logic-trace'),
    )
    expect(apiGetMock).not.toHaveBeenCalledWith('/graph/paper/doi%3A10.1000%2Fexample')
    expect(await screen.findByText('Research Moves')).toBeInTheDocument()
    expect(screen.getByText('Evidence Anchors')).toBeInTheDocument()
    expect(screen.queryByText(/logic/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/claim/i)).not.toBeInTheDocument()
  })
})
