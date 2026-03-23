import { describe, expect, test } from 'vitest'

import { paperRefForAskScope } from '../src/paperRefs'

describe('paperRefForAskScope', () => {
  test('prefers explicit paperId metadata', () => {
    expect(paperRefForAskScope({ id: '569', kind: 'paper', paperId: 'doi:10.1000/test' })).toBe('doi:10.1000/test')
  })

  test('normalizes paper, move, and anchor prefixes', () => {
    expect(paperRefForAskScope({ id: 'paper:doi:10.1000/test', kind: 'paper' })).toBe('doi:10.1000/test')
    expect(paperRefForAskScope({ id: 'paper_source:07_1605', kind: 'paper' })).toBe('07_1605')
    expect(
      paperRefForAskScope({
        id: 'move:bc082d21ddcde94212aab4ab474d9e32097a34ab90995a8bd181b29b1ed29026:0',
        kind: 'move',
      }),
    ).toBe('bc082d21ddcde94212aab4ab474d9e32097a34ab90995a8bd181b29b1ed29026')
    expect(
      paperRefForAskScope({
        id: 'anchor:bc082d21ddcde94212aab4ab474d9e32097a34ab90995a8bd181b29b1ed29026:1',
        kind: 'anchor',
      }),
    ).toBe('bc082d21ddcde94212aab4ab474d9e32097a34ab90995a8bd181b29b1ed29026')
  })

  test('accepts likely paper ids and rejects graph-only numeric ids', () => {
    expect(
      paperRefForAskScope({
        id: 'bc082d21ddcde94212aab4ab474d9e32097a34ab90995a8bd181b29b1ed29026',
        kind: 'paper',
      }),
    ).toBe('bc082d21ddcde94212aab4ab474d9e32097a34ab90995a8bd181b29b1ed29026')
    expect(paperRefForAskScope({ id: '569', kind: 'paper' })).toBe('')
  })
})
