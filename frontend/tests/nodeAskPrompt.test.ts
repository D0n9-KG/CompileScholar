import { describe, expect, test } from 'vitest'

import { buildNodeAskQuestion } from '../src/nodeAskPrompt'

describe('buildNodeAskQuestion', () => {
  test('paper prompt stays generic instead of appending raw paper labels', () => {
    expect(buildNodeAskQuestion('paper', '05_340', 'en-US')).toBe(
      "Summarize this paper's core methods, key findings, and verifiable evidence.",
    )
  })

  test('move and anchor prompts use new PaperLogicTrace terminology while keeping explicit labels', () => {
    expect(buildNodeAskQuestion('move', 'Move A', 'en-US')).toContain('research move')
    expect(buildNodeAskQuestion('move', 'Move A', 'en-US')).toContain('Move A')
    expect(buildNodeAskQuestion('anchor', 'Anchor B', 'en-US')).toContain('evidence anchor')
    expect(buildNodeAskQuestion('anchor', 'Anchor B', 'en-US')).toContain('Anchor B')
  })

  test('unknown kinds fall back to the generic node prompt', () => {
    expect(buildNodeAskQuestion('route', 'Route A', 'en-US')).toContain('Explain this node')
    expect(buildNodeAskQuestion('artifact', 'Artifact B', 'en-US')).toContain('Explain this node')
  })
})
