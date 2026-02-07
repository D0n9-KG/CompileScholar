import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { apiGet, apiPost } from '../api'

type PaperType = 'research' | 'review'

type Schema = {
  paper_type: PaperType
  version: number
  steps: Array<{ id: string; label_zh?: string; label_en?: string; enabled?: boolean; order?: number }>
  claim_kinds: Array<{ id: string; label_zh?: string; label_en?: string; enabled?: boolean }>
  rules: Record<string, unknown>
  prompts?: Record<string, string>
}

function clone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T
}

const ID_RE = /^[A-Za-z][A-Za-z0-9_-]{0,47}$/

const DEFAULT_PROMPTS: Record<string, string> = {
  logic_claims_system:
    "You extract a paper's reasoning structure for a research knowledge graph.\nReturn STRICT JSON only (no prose, no Markdown).\n\nGROUNDING / FAITHFULNESS:\n- Be strictly faithful to the provided paper text.\n- Do NOT invent details, numbers, conditions, or causal claims.\n- If something is not explicitly supported, omit it (preferred) or lower confidence.\n\nLANGUAGE / STYLE:\n- Use the same language as the paper text.\n- Write COMPLETE sentences only (no fragments, no missing subjects/verbs).\n- Keep technical symbols/variables exactly as in the paper.\n\nDIFFERENT OUTPUT GRANULARITIES:\n1) logic: for EACH allowed step_type, write a DETAILED mini-paragraph summary.\n   - 2–6 complete sentences (NOT a single sentence).\n   - Include key entities, methods, assumptions/conditions, and important numbers/definitions if present.\n2) claims: write concise, atomic KEY POINTS.\n   - 1–2 complete sentences each.\n   - Each claim must be specific and directly supported by the text.\n   - Avoid duplicating the logic summaries verbatim.\n\nSCHEMA RULES:\n- Each claim MUST belong to exactly ONE step_type (from the allowed list).\n- Each claim MUST have claim_kinds as a LIST (multi-select) chosen from allowed kinds (prefer 1–3 kinds).\n- Confidence values must be in [0,1].\n",
  logic_claims_user_template:
    "Paper metadata:\nTitle: {{title}}\nAuthors: {{authors}}\nYear: {{year}}\nDOI: {{doi}}\n\nAllowed step types: {{step_ids}}\nAllowed claim kinds: {{kind_ids}}\nTarget number of claims: {{cmin}}-{{cmax}}\n\nPaper text (extracted from Markdown):\n{{body}}\n\nOutput JSON schema (STRICT):\n{\n  \"logic\": {\n    \"<StepType>\": {\"summary\": \"2-6 full sentences...\", \"confidence\": 0.0}\n  },\n  \"claims\": [\n    {\"text\":\"1-2 full sentences...\",\"confidence\":0.0,\"step_type\":\"<StepType>\",\"claim_kinds\":[\"KindA\",\"KindB\"]}\n  ]\n}\n",
  evidence_pick_system:
    "Pick evidence chunks for claims. Return STRICT JSON only (no prose).\n- Pick chunks that DIRECTLY support the claim wording.\n- Prefer chunks containing the key definition/number/equation mentioned.\n- If evidence is weak/indirect, still pick the best available and set weak=true.\n",
  evidence_pick_user_template:
    "Pick {{emin}}-{{emax}} chunk_id(s) per claim from its candidates.\nIf none strongly supports it, still pick best 1 and set weak=true.\n\nInput claims JSON:\n{{payload_json}}\n\nOutput JSON schema:\n{ \"items\": [ {\"claim_key\":\"...\",\"evidence_chunk_ids\":[\"...\"],\"weak\":false} ] }\n",
  citation_purpose_batch_system:
    "You classify the PURPOSE of citations in a mechanics paper.\nReturn STRICT JSON only.\nFor each cited_paper_id, output 1-3 labels from the allowed list and scores in [0,1].\nBe conservative: if evidence is weak, use Background/Summary with low confidence.\nAllowed labels: {{allowed_labels}}",
  citation_purpose_batch_user_template:
    "Citing paper title: {{citing_title}}\n\nFor each citation, you are given the cited paper metadata (may be empty) and context snippets.\nInput JSON:\n{{cites_json}}\n\nOutput JSON schema:\n{\n  \"cites\": [\n    {\"cited_paper_id\": \"doi:10....\", \"labels\": [\"MethodUse\"], \"scores\":[0.72]}\n  ]\n}\n",
}

function intOr(v: unknown, fallback: number) {
  const n = Number(v)
  return Number.isFinite(n) ? Math.trunc(n) : fallback
}

export default function SchemaPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [paperType, setPaperType] = useState<PaperType>('research')
  const [schema, setSchema] = useState<Schema | null>(null)
  const [baselineJson, setBaselineJson] = useState<string>('')
  const [versions, setVersions] = useState<number[]>([])
  const [error, setError] = useState<string>('')
  const [info, setInfo] = useState<string>('')
  const [busy, setBusy] = useState<boolean>(false)
  const [activateVersion, setActivateVersion] = useState<string>('')

  const [newStepId, setNewStepId] = useState<string>('')
  const [newStepZh, setNewStepZh] = useState<string>('')
  const [newStepEn, setNewStepEn] = useState<string>('')
  const [newStepOrder, setNewStepOrder] = useState<number>(0)
  const [newKindId, setNewKindId] = useState<string>('')
  const [newKindZh, setNewKindZh] = useState<string>('')
  const [newKindEn, setNewKindEn] = useState<string>('')

  async function refresh(pt: PaperType) {
    setBusy(true)
    setError('')
    setInfo('')
    try {
      const s = await apiGet<{ schema: Schema }>(`/schema/active?paper_type=${encodeURIComponent(pt)}`)
      setSchema(s.schema)
      setBaselineJson(JSON.stringify(s.schema))
      const vs = await apiGet<{ versions: Array<{ version: number }> }>(`/schema/versions?paper_type=${encodeURIComponent(pt)}`)
      setVersions((vs.versions ?? []).map((x) => x.version))
      setActivateVersion('')
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    refresh(paperType).catch(() => {})
  }, [paperType])

  const dirty = useMemo(() => {
    if (!schema || !baselineJson) return false
    try {
      return JSON.stringify(schema) !== baselineJson
    } catch {
      return false
    }
  }, [baselineJson, schema])

  type Tab = 'steps' | 'points' | 'rules' | 'prompts'
  const tab = useMemo(() => {
    const t = String(searchParams.get('tab') ?? '').trim()
    if (t === 'points' || t === 'rules' || t === 'prompts' || t === 'steps') return t
    return 'steps' as const
  }, [searchParams])

  function selectTab(t: Tab) {
    const next = new URLSearchParams(searchParams)
    next.set('tab', t)
    setSearchParams(next, { replace: true })
  }

  const steps = useMemo(() => (schema?.steps ?? []).slice().sort((a, b) => intOr(a.order, 0) - intOr(b.order, 0)), [schema])
  const kinds = useMemo(() => (schema?.claim_kinds ?? []).slice(), [schema])

  function updateStep(id: string, patch: Partial<Schema['steps'][number]>) {
    if (!schema) return
    const next = clone(schema)
    next.steps = next.steps.map((s) => (s.id === id ? { ...s, ...patch } : s))
    setSchema(next)
  }

  function updateKind(id: string, patch: Partial<Schema['claim_kinds'][number]>) {
    if (!schema) return
    const next = clone(schema)
    next.claim_kinds = next.claim_kinds.map((k) => (k.id === id ? { ...k, ...patch } : k))
    setSchema(next)
  }

  function updateRule(key: string, value: unknown) {
    if (!schema) return
    const next = clone(schema)
    next.rules = { ...(next.rules ?? {}), [key]: value }
    setSchema(next)
  }

  function updatePrompt(key: string, value: string) {
    if (!schema) return
    const next = clone(schema)
    const prompts: Record<string, string> = { ...(next.prompts ?? {}) }
    // Treat empty as "no override" (fall back to default prompts on backend).
    if (!value.trim()) delete prompts[key]
    else prompts[key] = value
    next.prompts = Object.keys(prompts).length ? prompts : undefined
    setSchema(next)
  }

  function clearPrompt(key: string) {
    if (!schema) return
    const next = clone(schema)
    const prompts: Record<string, string> = { ...(next.prompts ?? {}) }
    delete prompts[key]
    next.prompts = Object.keys(prompts).length ? prompts : undefined
    setSchema(next)
  }

  function fillDefaultPrompts() {
    if (!schema) return
    const next = clone(schema)
    next.prompts = { ...(DEFAULT_PROMPTS ?? {}), ...(next.prompts ?? {}) }
    setSchema(next)
    setInfo('已填入默认提示词（未覆盖你已编辑的项）。')
  }

  function promptValue(key: string): string {
    if (!schema) return ''
    const v = (schema.prompts ?? {})[key]
    if (typeof v === 'string' && v.length > 0) return v
    return String((DEFAULT_PROMPTS as Record<string, string>)[key] ?? '')
  }

  function removeStep(id: string) {
    if (!schema) return
    const next = clone(schema)
    next.steps = next.steps.filter((s) => s.id !== id)
    if (next.steps.length < 1) return
    setSchema(next)
  }

  function removeKind(id: string) {
    if (!schema) return
    const next = clone(schema)
    next.claim_kinds = next.claim_kinds.filter((k) => k.id !== id)
    if (next.claim_kinds.length < 1) return
    setSchema(next)
  }

  const canAddStep = useMemo(() => {
    if (!schema) return false
    const id = newStepId.trim()
    if (!ID_RE.test(id)) return false
    const exists = (schema.steps ?? []).some((s) => s.id === id)
    return !exists
  }, [newStepId, schema])

  const canAddKind = useMemo(() => {
    if (!schema) return false
    const id = newKindId.trim()
    if (!ID_RE.test(id)) return false
    const exists = (schema.claim_kinds ?? []).some((k) => k.id === id)
    return !exists
  }, [newKindId, schema])

  function addStep() {
    if (!schema || !canAddStep) return
    const next = clone(schema)
    next.steps = [
      ...(next.steps ?? []),
      { id: newStepId.trim(), label_zh: newStepZh.trim(), label_en: newStepEn.trim(), enabled: true, order: intOr(newStepOrder, 0) },
    ]
    setSchema(next)
    setNewStepId('')
    setNewStepZh('')
    setNewStepEn('')
    setNewStepOrder(0)
  }

  function addKind() {
    if (!schema || !canAddKind) return
    const next = clone(schema)
    next.claim_kinds = [
      ...(next.claim_kinds ?? []),
      { id: newKindId.trim(), label_zh: newKindZh.trim(), label_en: newKindEn.trim(), enabled: true },
    ]
    setSchema(next)
    setNewKindId('')
    setNewKindZh('')
    setNewKindEn('')
  }

  async function saveNewVersion() {
    if (!schema) return
    setBusy(true)
    setError('')
    setInfo('')
    try {
      const schemaForSave: Schema = clone(schema)
      if (schemaForSave.prompts) {
        const p: Record<string, string> = {}
        for (const [k, v] of Object.entries(schemaForSave.prompts ?? {})) {
          const s = String(v ?? '')
          if (s.trim()) p[k] = s
        }
        schemaForSave.prompts = Object.keys(p).length ? p : undefined
      }
      const res = await apiPost<{ schema: Schema }>('/schema/new', {
        paper_type: paperType,
        schema: schemaForSave,
        activate: true,
      })
      setSchema(res.schema)
      setInfo(`已保存为新版本并激活：v${res.schema.version}`)
      await refresh(paperType)
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setBusy(false)
    }
  }

  async function activate() {
    const v = Number(activateVersion)
    if (!Number.isFinite(v) || v <= 0) return
    if (dirty && !window.confirm('当前有未保存改动，切换版本会丢失这些改动。确定继续吗？')) return
    setBusy(true)
    setError('')
    setInfo('')
    try {
      const res = await apiPost<{ schema: Schema }>('/schema/activate', { paper_type: paperType, version: Math.trunc(v) })
      setInfo(`已切换到版本：v${res.schema.version}`)
      await refresh(paperType)
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">Schema（可配置）</h2>
          <div className="pageSubtitle">在默认配置基础上自定义：逻辑链步骤、要点(Claim)类型、数量区间与规则（对未来重建生效）</div>
        </div>
        <div className="pageActions">
          <select
            className="select"
            style={{ width: 200 }}
            value={paperType}
            onChange={(e) => {
              const next = e.target.value as PaperType
              if (next === paperType) return
              if (dirty && !window.confirm('当前有未保存改动，切换论文类型会丢失这些改动。确定继续吗？')) return
              setPaperType(next)
            }}
          >
            <option value="research">研究型(Research)</option>
            <option value="review">综述型(Review)</option>
          </select>
          <button
            className="btn"
            disabled={busy}
            onClick={() => {
              if (dirty && !window.confirm('当前有未保存改动，刷新会丢失这些改动。确定继续吗？')) return
              refresh(paperType).catch(() => {})
            }}
          >
            {busy ? '加载中…' : '刷新'}
          </button>
        </div>
      </div>

      {error && <div className="errorBox">{error}</div>}
      {info && <div className="infoBox">{info}</div>}

      {!schema ? (
        <div className="panel">
          <div className="panelBody">加载中…</div>
        </div>
      ) : (
        <div className="stack">
          <div className="panel">
            <div className="panelHeader">
              <div className="split">
                <div className="panelTitle">版本</div>
                <div className="row">
                  {dirty && (
                    <span className="pill">
                      <span className="kicker">未保存</span> 有改动
                    </span>
                  )}
                  <span className="pill">
                    <span className="kicker">当前</span> v{schema.version}
                  </span>
                </div>
              </div>
            </div>
            <div className="panelBody">
              <div className="row">
                <button className="btn btnPrimary" disabled={busy} onClick={saveNewVersion}>
                  保存为新版本并激活
                </button>
                <span className="kicker">切换版本</span>
                <select className="select" style={{ width: 140 }} value={activateVersion} onChange={(e) => setActivateVersion(e.target.value)}>
                  <option value="">选择版本…</option>
                  {versions.map((v) => (
                    <option key={v} value={String(v)}>
                      v{v}
                    </option>
                  ))}
                </select>
                <button className="btn" disabled={busy || !activateVersion} onClick={activate}>
                  激活
                </button>
              </div>
              <div className="hint" style={{ marginTop: 10 }}>
                提示：Schema 变更只影响未来的“重建/替换/新导入”。已入库论文的展示会按其记录的 schema_version 渲染。
              </div>
            </div>
          </div>

          <div className="row" style={{ marginBottom: 6 }}>
            <span className="kicker">页面</span>
            <button className={`chip ${tab === 'steps' ? 'chipActive' : ''}`} onClick={() => selectTab('steps')}>
              逻辑节点
            </button>
            <button className={`chip ${tab === 'points' ? 'chipActive' : ''}`} onClick={() => selectTab('points')}>
              要点类型
            </button>
            <button className={`chip ${tab === 'rules' ? 'chipActive' : ''}`} onClick={() => selectTab('rules')}>
              规则
            </button>
            <button className={`chip ${tab === 'prompts' ? 'chipActive' : ''}`} onClick={() => selectTab('prompts')}>
              提示词
            </button>
          </div>

          {tab === 'steps' && (
          <div className="panel">
            <div className="panelHeader">
              <div className="panelTitle">逻辑节点(Logic Chain)</div>
            </div>
            <div className="panelBody">
              <div className="itemCard" style={{ marginBottom: 14 }}>
                <div className="itemTitle">新增步骤</div>
                <div className="hint" style={{ marginTop: 6 }}>
                  Step ID 需为 ASCII slug：以字母开头，允许字母/数字/下划线/短横线，最长 48 字符。
                </div>
                <div className="row" style={{ marginTop: 12, alignItems: 'center' }}>
                  <div className="kicker" style={{ width: 110 }}>
                    id
                  </div>
                  <input className="input" value={newStepId} onChange={(e) => setNewStepId(e.target.value)} placeholder="例如 Background2" />
                  <div className="kicker" style={{ width: 110, marginLeft: 10 }}>
                    order
                  </div>
                  <input className="input" style={{ width: 120 }} type="number" value={newStepOrder} onChange={(e) => setNewStepOrder(intOr(e.target.value, 0))} />
                </div>
                <div className="row" style={{ marginTop: 10 }}>
                  <div className="kicker" style={{ width: 110 }}>
                    中文标签
                  </div>
                  <input className="input" value={newStepZh} onChange={(e) => setNewStepZh(e.target.value)} placeholder="例如 背景补充" />
                </div>
                <div className="row" style={{ marginTop: 10 }}>
                  <div className="kicker" style={{ width: 110 }}>
                    英文标签
                  </div>
                  <input className="input" value={newStepEn} onChange={(e) => setNewStepEn(e.target.value)} placeholder="例如 Background (Alt)" />
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <button className="btn btnPrimary" disabled={busy || !canAddStep} onClick={addStep}>
                    新增步骤
                  </button>
                  {!canAddStep && newStepId.trim() && <span className="kicker">ID 无效或已存在</span>}
                </div>
              </div>

              <div className="list">
                {steps.map((s) => (
                  <div key={s.id} className="itemCard">
                    <div className="split">
                      <div className="itemTitle">
                        <code>{s.id}</code>
                      </div>
                      <div className="row" style={{ gap: 10 }}>
                        <label className="row" style={{ gap: 8 }}>
                          <input type="checkbox" checked={!!s.enabled} onChange={(e) => updateStep(s.id, { enabled: e.target.checked })} />
                          <span className="kicker">启用</span>
                        </label>
                        <button className="btn btnSmall" disabled={busy || (schema.steps?.length ?? 0) <= 1} onClick={() => removeStep(s.id)}>
                          删除
                        </button>
                      </div>
                    </div>
                    <div className="row" style={{ marginTop: 10 }}>
                      <div style={{ width: 110 }} className="kicker">
                        顺序(order)
                      </div>
                      <input className="input" style={{ width: 120 }} type="number" value={intOr(s.order, 0)} onChange={(e) => updateStep(s.id, { order: intOr(e.target.value, 0) })} />
                    </div>
                    <div className="row" style={{ marginTop: 10 }}>
                      <div style={{ width: 110 }} className="kicker">
                        中文标签
                      </div>
                      <input className="input" value={String(s.label_zh ?? '')} onChange={(e) => updateStep(s.id, { label_zh: e.target.value })} />
                    </div>
                    <div className="row" style={{ marginTop: 10 }}>
                      <div style={{ width: 110 }} className="kicker">
                        英文标签
                      </div>
                      <input className="input" value={String(s.label_en ?? '')} onChange={(e) => updateStep(s.id, { label_en: e.target.value })} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          )}

          {tab === 'points' && (
          <div className="panel">
            <div className="panelHeader">
              <div className="panelTitle">要点类型(Claim Kinds)</div>
            </div>
            <div className="panelBody">
              <div className="itemCard" style={{ marginBottom: 14 }}>
                <div className="itemTitle">新增类型</div>
                <div className="hint" style={{ marginTop: 6 }}>
                  Kind ID 需为 ASCII slug（同上规则）。新增后需“保存为新版本并激活”，并在论文上执行“重建”才会生效。
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <div className="kicker" style={{ width: 110 }}>
                    id
                  </div>
                  <input className="input" value={newKindId} onChange={(e) => setNewKindId(e.target.value)} placeholder="例如 Hypothesis" />
                </div>
                <div className="row" style={{ marginTop: 10 }}>
                  <div className="kicker" style={{ width: 110 }}>
                    中文标签
                  </div>
                  <input className="input" value={newKindZh} onChange={(e) => setNewKindZh(e.target.value)} placeholder="例如 假说" />
                </div>
                <div className="row" style={{ marginTop: 10 }}>
                  <div className="kicker" style={{ width: 110 }}>
                    英文标签
                  </div>
                  <input className="input" value={newKindEn} onChange={(e) => setNewKindEn(e.target.value)} placeholder="例如 Hypothesis" />
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <button className="btn btnPrimary" disabled={busy || !canAddKind} onClick={addKind}>
                    新增类型
                  </button>
                  {!canAddKind && newKindId.trim() && <span className="kicker">ID 无效或已存在</span>}
                </div>
              </div>

              <div className="list">
                {kinds.map((k) => (
                  <div key={k.id} className="itemCard">
                    <div className="split">
                      <div className="itemTitle">
                        <code>{k.id}</code>
                      </div>
                      <div className="row" style={{ gap: 10 }}>
                        <label className="row" style={{ gap: 8 }}>
                          <input type="checkbox" checked={!!k.enabled} onChange={(e) => updateKind(k.id, { enabled: e.target.checked })} />
                          <span className="kicker">启用</span>
                        </label>
                        <button className="btn btnSmall" disabled={busy || (schema.claim_kinds?.length ?? 0) <= 1} onClick={() => removeKind(k.id)}>
                          删除
                        </button>
                      </div>
                    </div>
                    <div className="row" style={{ marginTop: 10 }}>
                      <div style={{ width: 110 }} className="kicker">
                        中文标签
                      </div>
                      <input className="input" value={String(k.label_zh ?? '')} onChange={(e) => updateKind(k.id, { label_zh: e.target.value })} />
                    </div>
                    <div className="row" style={{ marginTop: 10 }}>
                      <div style={{ width: 110 }} className="kicker">
                        英文标签
                      </div>
                      <input className="input" value={String(k.label_en ?? '')} onChange={(e) => updateKind(k.id, { label_en: e.target.value })} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          )}

          {tab === 'rules' && (
          <div className="panel">
            <div className="panelHeader">
              <div className="panelTitle">规则(Rules)</div>
            </div>
            <div className="panelBody">
              <div className="row">
                <div className="pill" style={{ gap: 10 }}>
                  <span className="kicker">claims_min</span>
                  <input
                    className="input"
                    style={{ width: 110 }}
                    type="number"
                    value={intOr(schema.rules?.claims_per_paper_min, 24)}
                    onChange={(e) => updateRule('claims_per_paper_min', intOr(e.target.value, 24))}
                  />
                </div>
                <div className="pill" style={{ gap: 10 }}>
                  <span className="kicker">claims_max</span>
                  <input
                    className="input"
                    style={{ width: 110 }}
                    type="number"
                    value={intOr(schema.rules?.claims_per_paper_max, 48)}
                    onChange={(e) => updateRule('claims_per_paper_max', intOr(e.target.value, 48))}
                  />
                </div>
                <div className="pill" style={{ gap: 10 }}>
                  <span className="kicker">evidence_min</span>
                  <input
                    className="input"
                    style={{ width: 110 }}
                    type="number"
                    value={intOr(schema.rules?.machine_evidence_min, 1)}
                    onChange={(e) => updateRule('machine_evidence_min', intOr(e.target.value, 1))}
                  />
                </div>
                <div className="pill" style={{ gap: 10 }}>
                  <span className="kicker">evidence_max</span>
                  <input
                    className="input"
                    style={{ width: 110 }}
                    type="number"
                    value={intOr(schema.rules?.machine_evidence_max, 2)}
                    onChange={(e) => updateRule('machine_evidence_max', intOr(e.target.value, 2))}
                  />
                </div>
                <div className="pill" style={{ gap: 10 }}>
                  <span className="kicker">logic_evidence_min</span>
                  <input
                    className="input"
                    style={{ width: 110 }}
                    type="number"
                    value={intOr((schema.rules as Record<string, unknown>)?.logic_evidence_min, 1)}
                    onChange={(e) => updateRule('logic_evidence_min', intOr(e.target.value, 1))}
                  />
                </div>
                <div className="pill" style={{ gap: 10 }}>
                  <span className="kicker">logic_evidence_max</span>
                  <input
                    className="input"
                    style={{ width: 110 }}
                    type="number"
                    value={intOr((schema.rules as Record<string, unknown>)?.logic_evidence_max, 2)}
                    onChange={(e) => updateRule('logic_evidence_max', intOr(e.target.value, 2))}
                  />
                </div>
                <div className="pill" style={{ gap: 10 }}>
                  <span className="kicker">evidence_verify</span>
                  <select
                    className="select"
                    style={{ width: 150 }}
                    value={String((schema.rules as Record<string, unknown>)?.evidence_verification ?? 'llm')}
                    onChange={(e) => updateRule('evidence_verification', e.target.value)}
                  >
                    <option value="llm">大模型(LLM)</option>
                    <option value="off">关闭</option>
                  </select>
                </div>
              </div>
              <div className="hint" style={{ marginTop: 10 }}>
                保存后需要对论文执行“重建”才会按新规则重新生成逻辑链/要点/证据。
              </div>
            </div>
          </div>
          )}

          {tab === 'prompts' && (
          <div className="panel">
            <div className="panelHeader">
              <div className="panelTitle">提示词(Prompts)</div>
            </div>
            <div className="panelBody">
              <div className="hint">
                这是“高级配置”：可覆盖默认提示词。支持简单变量替换（形如 <code>{'{{title}}'}</code>）。推荐先只改 system，再逐步调 user_template。
                <br />
                <b>留空表示使用默认提示词</b>（不会清空；也不会影响已入库论文，需重建才生效）。
                <br />
                说明：提示词里仍使用 <code>claims</code> 字段名（系统内部 JSON key），但前端展示称为“要点”。
              </div>

              <div className="row" style={{ marginTop: 10 }}>
                <button className="btn" disabled={busy} onClick={fillDefaultPrompts}>
                  填入默认提示词
                </button>
              </div>

              <div className="list" style={{ marginTop: 12 }}>
                <div className="itemCard">
                  <div className="split">
                    <div className="itemTitle">Logic/要点 提取：system</div>
                    <div className="row" style={{ gap: 8 }}>
                      <span className={schema.prompts?.logic_claims_system ? 'badge badgeOk' : 'badge'}>
                        {schema.prompts?.logic_claims_system ? '已覆盖' : '默认'}
                      </span>
                      <button className="btn btnSmall" disabled={busy || !schema.prompts?.logic_claims_system} onClick={() => clearPrompt('logic_claims_system')}>
                        清除覆盖
                      </button>
                    </div>
                  </div>
                  <textarea className="textarea" value={promptValue('logic_claims_system')} onChange={(e) => updatePrompt('logic_claims_system', e.target.value)} />
                </div>
                <div className="itemCard">
                  <div className="split">
                    <div className="itemTitle">Logic/要点 提取：user_template</div>
                    <div className="row" style={{ gap: 8 }}>
                      <span className={schema.prompts?.logic_claims_user_template ? 'badge badgeOk' : 'badge'}>
                        {schema.prompts?.logic_claims_user_template ? '已覆盖' : '默认'}
                      </span>
                      <button className="btn btnSmall" disabled={busy || !schema.prompts?.logic_claims_user_template} onClick={() => clearPrompt('logic_claims_user_template')}>
                        清除覆盖
                      </button>
                    </div>
                  </div>
                  <div className="hint" style={{ marginTop: 6 }}>
                    可用变量：title, authors, year, doi, step_ids, kind_ids, cmin, cmax, body
                  </div>
                  <textarea className="textarea" value={promptValue('logic_claims_user_template')} onChange={(e) => updatePrompt('logic_claims_user_template', e.target.value)} />
                </div>

                <div className="itemCard">
                  <div className="split">
                    <div className="itemTitle">Evidence 选择：system</div>
                    <div className="row" style={{ gap: 8 }}>
                      <span className={schema.prompts?.evidence_pick_system ? 'badge badgeOk' : 'badge'}>
                        {schema.prompts?.evidence_pick_system ? '已覆盖' : '默认'}
                      </span>
                      <button className="btn btnSmall" disabled={busy || !schema.prompts?.evidence_pick_system} onClick={() => clearPrompt('evidence_pick_system')}>
                        清除覆盖
                      </button>
                    </div>
                  </div>
                  <textarea className="textarea" value={promptValue('evidence_pick_system')} onChange={(e) => updatePrompt('evidence_pick_system', e.target.value)} />
                </div>
                <div className="itemCard">
                  <div className="split">
                    <div className="itemTitle">Evidence 选择：user_template</div>
                    <div className="row" style={{ gap: 8 }}>
                      <span className={schema.prompts?.evidence_pick_user_template ? 'badge badgeOk' : 'badge'}>
                        {schema.prompts?.evidence_pick_user_template ? '已覆盖' : '默认'}
                      </span>
                      <button className="btn btnSmall" disabled={busy || !schema.prompts?.evidence_pick_user_template} onClick={() => clearPrompt('evidence_pick_user_template')}>
                        清除覆盖
                      </button>
                    </div>
                  </div>
                  <div className="hint" style={{ marginTop: 6 }}>
                    可用变量：emin, emax, payload_json
                  </div>
                  <textarea className="textarea" value={promptValue('evidence_pick_user_template')} onChange={(e) => updatePrompt('evidence_pick_user_template', e.target.value)} />
                </div>

                <div className="itemCard">
                  <div className="split">
                    <div className="itemTitle">Citation Purpose（批量）：system</div>
                    <div className="row" style={{ gap: 8 }}>
                      <span className={schema.prompts?.citation_purpose_batch_system ? 'badge badgeOk' : 'badge'}>
                        {schema.prompts?.citation_purpose_batch_system ? '已覆盖' : '默认'}
                      </span>
                      <button className="btn btnSmall" disabled={busy || !schema.prompts?.citation_purpose_batch_system} onClick={() => clearPrompt('citation_purpose_batch_system')}>
                        清除覆盖
                      </button>
                    </div>
                  </div>
                  <textarea className="textarea" value={promptValue('citation_purpose_batch_system')} onChange={(e) => updatePrompt('citation_purpose_batch_system', e.target.value)} />
                </div>
                <div className="itemCard">
                  <div className="split">
                    <div className="itemTitle">Citation Purpose（批量）：user_template</div>
                    <div className="row" style={{ gap: 8 }}>
                      <span className={schema.prompts?.citation_purpose_batch_user_template ? 'badge badgeOk' : 'badge'}>
                        {schema.prompts?.citation_purpose_batch_user_template ? '已覆盖' : '默认'}
                      </span>
                      <button className="btn btnSmall" disabled={busy || !schema.prompts?.citation_purpose_batch_user_template} onClick={() => clearPrompt('citation_purpose_batch_user_template')}>
                        清除覆盖
                      </button>
                    </div>
                  </div>
                  <div className="hint" style={{ marginTop: 6 }}>
                    可用变量：citing_title, cites_json, allowed_labels
                  </div>
                  <textarea className="textarea" value={promptValue('citation_purpose_batch_user_template')} onChange={(e) => updatePrompt('citation_purpose_batch_user_template', e.target.value)} />
                </div>
              </div>

              <div className="hint" style={{ marginTop: 10 }}>
                保存后对论文执行“重建”才会用新提示词生成；提示词写错可能导致 JSON 解析失败。
              </div>
            </div>
          </div>
          )}
        </div>
      )}
    </div>
  )
}
