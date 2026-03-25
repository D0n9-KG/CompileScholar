import { render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, test, vi } from 'vitest'

vi.mock('../src/components/TopBar', () => ({
  default: () => <div>TopBar</div>,
}))

vi.mock('../src/components/StatusBar', () => ({
  default: () => <div>StatusBar</div>,
}))

vi.mock('../src/components/LeftPanel', () => ({
  default: () => <div>LeftPanel</div>,
}))

vi.mock('../src/components/RightPanel', () => ({
  default: () => <div>RightPanel</div>,
}))

vi.mock('../src/components/GraphCanvas', () => ({
  default: () => <div>GraphCanvas</div>,
}))

vi.mock('../src/components/PageWorkbench', () => ({
  default: ({ title, children }: { title: string; children: ReactNode }) => (
    <div>
      <div>{title}</div>
      <div>{children}</div>
    </div>
  ),
}))

vi.mock('../src/pages/OpsWorkbench', () => ({
  default: () => <div>Ops Workbench Page</div>,
}))

vi.mock('../src/pages/TasksPage', () => ({
  default: () => <div>Tasks Page</div>,
}))

vi.mock('../src/pages/ConfigCenterPage', () => ({
  default: () => <div>Config Center Page</div>,
}))

vi.mock('../src/pages/UnresolvedPage', () => ({
  default: () => <div>Unresolved Page</div>,
}))

vi.mock('../src/pages/IngestPage', () => ({
  default: () => <div>Ingest Page</div>,
}))

vi.mock('../src/pages/DiscoveryPage', () => ({
  default: () => <div>Discovery Page</div>,
}))

vi.mock('../src/pages/PaperDetailPage', () => ({
  default: () => <div>Paper Detail</div>,
}))

vi.mock('../src/pages/TextbookDetailPage', () => ({
  default: () => <div>Textbook Detail</div>,
}))

import App from '../src/App'

describe('App discovery route retirement', () => {
  beforeEach(() => {
    window.history.pushState({}, '', '/discovery')
  })

  test('redirects /discovery to /ops', async () => {
    render(<App />)

    expect(await screen.findByText('Ops Workbench Page')).toBeInTheDocument()
    expect(screen.queryByText('Discovery Page')).not.toBeInTheDocument()
  })

  test('opens task list route directly', async () => {
    window.history.pushState({}, '', '/tasks')

    render(<App />)

    expect(await screen.findByText('Tasks Page')).toBeInTheDocument()
  })

  test('redirects imported sources route to ingest center', async () => {
    window.history.pushState({}, '', '/imported-sources')

    render(<App />)

    expect(await screen.findByText('Ingest Page')).toBeInTheDocument()
  })
})
