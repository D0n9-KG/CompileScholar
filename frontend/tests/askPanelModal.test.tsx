import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, test, vi } from 'vitest'
import AskPanel from '../src/panels/AskPanel'

const dispatch = vi.fn()

vi.mock('../src/api', () => ({
  apiBaseUrl: () => 'http://127.0.0.1:18000',
  apiGet: vi.fn(async () => ({ papers: [] })),
  apiPost: vi.fn(async () => ({})),
}))

vi.mock('../src/components/MarkdownView', () => ({
  default: ({ markdown, className }: { markdown: string; className?: string }) => (
    <div className={className}>{markdown}</div>
  ),
}))

vi.mock('../src/i18n', () => ({
  useI18n: () => ({
    locale: 'en-US',
    t: (_zh: string, en: string) => en,
  }),
}))

vi.mock('../src/loaders/ask', () => ({
  resolveAskGraph: vi.fn(() => []),
}))

vi.mock('../src/loaders/overview', () => ({
  loadOverviewGraph: vi.fn(async () => []),
}))

vi.mock('../src/scope', () => ({
  loadScope: () => ({ mode: 'all' }),
  saveScope: vi.fn(),
  scopeLabel: () => 'All Papers',
}))

vi.mock('../src/state/askSessions', () => ({
  ASK_STORE_EVENT: 'logickg:ask_store_changed',
  ASK_STORE_KEY: 'logickg:ask_store',
  getCurrentAskSession: (ask: { sessions?: unknown[] }) => ask.sessions?.[0] ?? null,
  isAskStatePristine: () => false,
  readAskModuleStateFromStorage: () => null,
  serializeAskModuleState: () => '{}',
}))

vi.mock('../src/state/store', () => ({
  useGlobalState: () => ({
    state: {
      ask: {
        currentId: null,
        currentSessionId: 'session-1',
        history: [],
        draftQuestion: '',
        draftK: 8,
        sessions: [
          {
            id: 'session-1',
            title: '',
            draftQuestion: '',
            updatedAt: Date.now(),
            currentId: null,
            history: [],
          },
        ],
      },
      graphElements: [],
    },
    dispatch,
  }),
}))

vi.mock('../src/panels/askPanelModel', () => ({
  buildConversationPayload: vi.fn(() => []),
  buildChatMessages: vi.fn(() => []),
  assistantTurnText: vi.fn(() => 'Thinking...'),
  buildScopePaperOptions: vi.fn(() => []),
  getScopePaperRenderState: vi.fn(() => ({ visible: [], remaining: 0, hasMore: false })),
  shouldAutoRetryWithAllScope: vi.fn(() => false),
  toConversationTurns: vi.fn(() => []),
  toggleScopePaperIds: vi.fn((paperIds: string[], paperId: string) =>
    paperIds.includes(paperId) ? paperIds.filter((id) => id !== paperId) : [...paperIds, paperId],
  ),
}))

describe('AskPanel modal chat window', () => {
  test('opens and closes the expanded chat modal', async () => {
    render(<AskPanel />)

    fireEvent.click(await screen.findByRole('button', { name: /open chat window/i }))

    expect(screen.getByRole('dialog', { name: /expanded chat workspace/i })).toBeInTheDocument()
    expect(screen.getByText('Chat workspace is open in a larger window.')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /close/i }))

    expect(screen.queryByRole('dialog', { name: /expanded chat workspace/i })).not.toBeInTheDocument()
  })
})
