import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, test } from 'vitest'

describe('ask layout css', () => {
  test('allocates more height to the ask conversation area on desktop', () => {
    const css = readFileSync(resolve(process.cwd(), 'src/components/layout.css'), 'utf8')

    expect(css).toMatch(/\.kgAskHistoryList\s*\{[\s\S]*max-height:\s*min\(16vh,\s*140px\);[\s\S]*overflow:\s*auto;/)
    expect(css).toMatch(/\.kgAskConversationSection\s*\{[\s\S]*flex:\s*1 1 420px;[\s\S]*min-height:\s*420px;/)
    expect(css).toMatch(/@media\s*\(max-width:\s*920px\)\s*\{[\s\S]*\.kgAskConversationSection\s*\{[\s\S]*min-height:\s*260px;/)
  })
})
