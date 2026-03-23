export type PaperType =
  | 'empirical'
  | 'theoretical'
  | 'review'
  | 'software'
  | 'benchmark'
  | 'case_study'
  | 'unknown'

export type MoveRole =
  | 'problem'
  | 'background'
  | 'hypothesis'
  | 'method'
  | 'experiment'
  | 'result'
  | 'interpretation'
  | 'limitation'
  | 'future_work'

export type MoveActType =
  | 'identify_gap'
  | 'define_task'
  | 'formulate_hypothesis'
  | 'propose_method'
  | 'adapt_method'
  | 'build_resource'
  | 'set_condition'
  | 'run_experiment'
  | 'measure_outcome'
  | 'compare_baseline'
  | 'report_effect'
  | 'explain_mechanism'
  | 'diagnose_failure'
  | 'state_limitation'
  | 'suggest_extension'

export type MentionValue = {
  surface: string
  normalized?: string | null
  type?: string | null
  anchor_ids?: string[]
  confidence?: number | null
  inferred?: boolean
}

export type EffectValue = {
  direction: 'increase' | 'decrease' | 'improve' | 'worsen' | 'mixed' | 'none' | 'unknown'
  magnitude_text?: string | null
  magnitude_numeric?: number | null
  unit?: string | null
  comparator_surface?: string | null
  anchor_ids?: string[]
  confidence?: number | null
}

export type SlotProvenance = {
  field: string
  value_index?: number | null
  anchor_ids?: string[]
  extraction_mode: 'direct' | 'normalized' | 'inferred'
  support_strength: 'exact' | 'strong' | 'weak'
  confidence?: number | null
  notes?: string | null
}

export type ResearchMove = {
  move_id: string
  sequence_no: number
  role: MoveRole
  act_type: MoveActType
  summary: string
  research_objects?: MentionValue[]
  methods?: MentionValue[]
  observed_variables?: MentionValue[]
  metrics?: MentionValue[]
  comparators?: MentionValue[]
  conditions?: MentionValue[]
  effects?: EffectValue[]
  limitation_types?: MentionValue[]
  resource_mentions?: MentionValue[]
  anchor_ids?: string[]
  slot_provenance?: SlotProvenance[]
  confidence?: number | null
  audit_state?: 'hot_path' | 'audited'
}

export type EvidenceAnchor = {
  anchor_id: string
  paper_id: string
  source_ref: string
  modality: 'text' | 'figure' | 'table' | 'citation'
  section_path?: string[]
  locator?: Record<string, unknown>
  quote: string
  citation_ids?: string[]
  support_type: 'direct' | 'contextual' | 'indirect'
  weak?: boolean
}

export type MoveRelation = {
  relation_id: string
  source_move_id: string
  target_move_id: string
  relation_type: 'motivates' | 'addresses' | 'implements' | 'evaluates' | 'yields' | 'explains' | 'limits' | 'extends'
  anchor_ids?: string[]
  confidence?: number | null
}

export type CitationAct = {
  citation_act_id: string
  source_move_id?: string | null
  target_paper_id?: string | null
  purpose?: string | null
  polarity?: string | null
  semantic_signal?: string | null
  target_scope?: string | null
  anchor_ids?: string[]
  confidence?: number | null
}

export type FigureRef = {
  figure_id: string
  caption?: string | null
  anchor_ids?: string[]
}

export type TableRef = {
  table_id: string
  caption?: string | null
  anchor_ids?: string[]
}

export type PaperMetadata = {
  paper_id: string
  canonical_doi?: string | null
  title: string
  year?: number | null
  authors?: string[]
  venue?: string | null
  paper_type: PaperType
  source_refs?: string[]
}

export type PaperLogicTrace = {
  trace_id: string
  schema_version: string
  built_at: string
  paper_metadata: PaperMetadata
  canonical_core: {
    evidence_anchors: EvidenceAnchor[]
    moves: ResearchMove[]
    move_relations: MoveRelation[]
    citation_acts: CitationAct[]
    figure_refs: FigureRef[]
    table_refs: TableRef[]
  }
  derived_views?: Record<string, unknown>
  quality?: Record<string, unknown>
}

export function normalizeText(value: unknown): string {
  return String(value ?? '')
    .replace(/\s+/g, ' ')
    .trim()
}

export function moveRoleLabel(role: MoveRole | string): string {
  const key = normalizeText(role)
  const labels: Record<string, string> = {
    problem: 'Problem',
    background: 'Background',
    hypothesis: 'Hypothesis',
    method: 'Method',
    experiment: 'Experiment',
    result: 'Result',
    interpretation: 'Interpretation',
    limitation: 'Limitation',
    future_work: 'Future Work',
  }
  return labels[key] ?? (key || 'Move')
}

export function moveActLabel(actType: MoveActType | string): string {
  const key = normalizeText(actType)
  const labels: Record<string, string> = {
    identify_gap: 'Identify Gap',
    define_task: 'Define Task',
    formulate_hypothesis: 'Formulate Hypothesis',
    propose_method: 'Propose Method',
    adapt_method: 'Adapt Method',
    build_resource: 'Build Resource',
    set_condition: 'Set Condition',
    run_experiment: 'Run Experiment',
    measure_outcome: 'Measure Outcome',
    compare_baseline: 'Compare Baseline',
    report_effect: 'Report Effect',
    explain_mechanism: 'Explain Mechanism',
    diagnose_failure: 'Diagnose Failure',
    state_limitation: 'State Limitation',
    suggest_extension: 'Suggest Extension',
  }
  return labels[key] ?? (key || 'Unknown')
}

export function formatMoveLabel(move: ResearchMove): string {
  return `${move.sequence_no}. ${moveRoleLabel(move.role)}`
}

export function mentionTokens(mentions: MentionValue[] | null | undefined): string[] {
  const tokens = (mentions ?? [])
    .map((item) => normalizeText(item.normalized || item.surface))
    .filter(Boolean)
  return Array.from(new Set(tokens))
}

export function tracePreviewMoves(trace: PaperLogicTrace, limit = 3): string[] {
  return (trace.canonical_core.moves ?? [])
    .slice()
    .sort((a, b) => a.sequence_no - b.sequence_no)
    .slice(0, limit)
    .map((move) => `${formatMoveLabel(move)} ${normalizeText(move.summary)}`.trim())
    .filter(Boolean)
}

export function tracePreviewRelations(trace: PaperLogicTrace, limit = 3): string[] {
  return (trace.canonical_core.move_relations ?? [])
    .slice(0, limit)
    .map((relation) => `${normalizeText(relation.relation_type)} ${normalizeText(relation.source_move_id)} -> ${normalizeText(relation.target_move_id)}`.trim())
    .filter(Boolean)
}
