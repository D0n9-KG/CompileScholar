import { describe, expect, test } from 'vitest'

import { graphKindLabel, isAnchorGraphKind, isMoveGraphKind, normalizeGraphKind } from '../src/graphKinds'

describe('graphKinds', () => {
  test('treats move and anchor as first-class PaperLogicTrace graph kinds', () => {
    expect(isMoveGraphKind('move')).toBe(true)
    expect(isMoveGraphKind('research_move')).toBe(true)
    expect(isAnchorGraphKind('anchor')).toBe(true)
    expect(isAnchorGraphKind('evidence_anchor')).toBe(true)
  })

  test('leaves unknown graph kinds untouched', () => {
    expect(normalizeGraphKind('route')).toBe('route')
    expect(normalizeGraphKind('artifact')).toBe('artifact')
    expect(normalizeGraphKind('custom_node')).toBe('custom_node')
  })

  test('labels move and anchor with new L2 terminology', () => {
    expect(graphKindLabel('move', 'en-US')).toBe('Research Move')
    expect(graphKindLabel('anchor', 'en-US')).toBe('Evidence Anchor')
    expect(graphKindLabel('route', 'en-US')).toBe('route')
    expect(graphKindLabel('artifact', 'en-US')).toBe('artifact')
  })
})
