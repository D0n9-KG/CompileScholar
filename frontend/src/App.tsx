import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import AskPage from './pages/AskPage'
import ContentPage from './pages/ContentPage'
import GraphPage from './pages/GraphPage'
import HomePage from './pages/HomePage'
import IngestPage from './pages/IngestPage'
import PaperDetailPage from './pages/PaperDetailPage'
import PapersPage from './pages/PapersPage'
import EvolutionPage from './pages/EvolutionPage'
import SchemaPage from './pages/SchemaPage'
import TasksPage from './pages/TasksPage'
import UnresolvedPage from './pages/UnresolvedPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/ingest" element={<IngestPage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/evolution" element={<EvolutionPage />} />
          <Route path="/content" element={<ContentPage />} />
          <Route path="/papers" element={<PapersPage />} />
          <Route path="/paper/:paperId" element={<PaperDetailPage />} />
          <Route path="/unresolved" element={<UnresolvedPage />} />
          <Route path="/ask" element={<AskPage />} />
          <Route path="/tasks" element={<TasksPage />} />
          <Route path="/schema" element={<SchemaPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
