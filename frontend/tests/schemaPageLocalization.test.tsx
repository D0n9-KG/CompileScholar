import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, test } from 'vitest'

import { I18nProvider } from '../src/i18n'
import SchemaPage from '../src/pages/SchemaPage'

describe('SchemaPage localization', () => {
  test('uses PaperLogicTrace terminology instead of the retired logic-claim schema editor', () => {
    render(
      <I18nProvider>
        <MemoryRouter initialEntries={['/schema']}>
          <Routes>
            <Route path="/schema" element={<SchemaPage />} />
          </Routes>
        </MemoryRouter>
      </I18nProvider>,
    )

    expect(screen.getByText('PaperLogicTrace Compiler Policy')).toBeInTheDocument()
    expect(screen.getAllByText(/ResearchMove/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/EvidenceAnchor/i).length).toBeGreaterThan(0)
    expect(screen.queryByText(/Logic Step/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Claim/i)).not.toBeInTheDocument()
  })
})
