import { useEffect, useMemo, useState } from 'react'
import { apiGet, apiPost, apiPostForm } from '../api'
import { TERMS } from '../ui/terms'
import type { NetworkNode } from './NetworkGraph'

type TaskInfo = {
  task_id: string
  status?: string
  progress?: number
  stage?: string
  message?: string | null
  error?: string | null
} & Record<string, unknown>

type ScanUnit = {
  unit_id: string
  md_rel_path: string
  doi?: string | null
  title?: string | null
  year?: number | null
  paper_type?: string | null
  status: string
  error?: string | null
  existing_paper_id?: string | null
}

type UploadScan = {
  upload_id: string
  mode: string
  root: string
  units: ScanUnit[]
  errors?: unknown[]
}

type WebkitFile = File & { webkitRelativePath?: string }
type FolderFile = { path: string; file: File; size: number }

function taskStatusLabel(status: string | null | undefined) {
  const s = String(status ?? '')
  if (!s) return ''
  if (s === 'queued') return '排队中'
  if (s === 'running') return '进行中'
  if (s === 'succeeded') return '成功'
  if (s === 'failed') return '失败'
  if (s === 'canceled') return '已取消'
  return s
}

function taskStageLabel(stage: string | null | undefined) {
  const s = String(stage ?? '')
  if (!s) return ''
  if (s.includes('crossref')) return `${TERMS.crossref} 解析`
  if (s.includes('neo4j_clear')) return '清理 Neo4j'
  if (s.includes('neo4j_write')) return '写入 Neo4j'
  if (s.includes('llm')) return `${TERMS.llm} 抽取`
  if (s.includes('faiss')) return `${TERMS.faiss} 重建`
  if (s === 'done') return '完成'
  if (s === 'canceled') return '已取消'
  if (s === 'failed') return '失败'
  return s
}

export default function GraphNodeDrawer({
  node,
  onOpenDetail,
  onRefreshNetwork,
}: {
  node: NetworkNode
  onOpenDetail: (paperId: string) => void
  onRefreshNetwork: () => Promise<void>
}) {
  const [error, setError] = useState<string>('')
  const [info, setInfo] = useState<string>('')

  const [zipFile, setZipFile] = useState<File | null>(null)
  const [folderFiles, setFolderFiles] = useState<FolderFile[]>([])
  const [uploadBusy, setUploadBusy] = useState<boolean>(false)
  const [uploadId, setUploadId] = useState<string>('')
  const [uploadProgress, setUploadProgress] = useState<{ sent: number; total: number }>({ sent: 0, total: 0 })
  const [scan, setScan] = useState<UploadScan | null>(null)
  const [doiInput, setDoiInput] = useState<string>('')
  const [paperType, setPaperType] = useState<string>('research')

  const [taskId, setTaskId] = useState<string>('')
  const [task, setTask] = useState<TaskInfo | null>(null)

  const chunkBytes = 8 * 1024 * 1024

  const uploadPct = useMemo(() => {
    if (!uploadProgress.total) return 0
    return Math.round((uploadProgress.sent / uploadProgress.total) * 100)
  }, [uploadProgress.sent, uploadProgress.total])

  const paperDoi = useMemo(() => {
    if (node.doi) return node.doi
    if (node.id.startsWith('doi:')) return node.id.slice(4)
    return ''
  }, [node.doi, node.id])

  useEffect(() => {
    // Reset per-node UI state when selection changes
    setError('')
    setInfo('')
    setZipFile(null)
    setFolderFiles([])
    setUploadBusy(false)
    setUploadId('')
    setUploadProgress({ sent: 0, total: 0 })
    setScan(null)
    setTaskId('')
    setTask(null)
    setDoiInput(paperDoi)
    setPaperType('research')
  }, [node.id, paperDoi])

  async function refreshScan(id: string) {
    const s = await apiGet<UploadScan>(`/ingest/upload/scan?upload_id=${encodeURIComponent(id)}`)
    setScan(s)
    return s
  }

  async function uploadChunksZip(id: string, file: File) {
    const totalChunks = Math.ceil(file.size / chunkBytes)
    const status = await apiGet<{ received?: number[] }>(`/ingest/upload/status?upload_id=${encodeURIComponent(id)}`)
    const received = new Set<number>((status?.received ?? []) as number[])
    setUploadProgress({ sent: 0, total: file.size })
    for (let idx = 0; idx < totalChunks; idx++) {
      if (received.has(idx)) continue
      const start = idx * chunkBytes
      const end = Math.min(file.size, start + chunkBytes)
      const blob = file.slice(start, end)
      const form = new FormData()
      form.append('upload_id', id)
      form.append('index', String(idx))
      form.append('blob', blob, `chunk-${idx}.bin`)
      await apiPostForm('/ingest/upload/chunk', form)
      setUploadProgress((p) => ({ sent: Math.min(p.total, p.sent + (end - start)), total: p.total }))
    }
  }

  async function uploadChunksFolder(id: string, files: FolderFile[]) {
    const total = files.reduce((a, b) => a + b.size, 0)
    setUploadProgress({ sent: 0, total })
    for (const f of files) {
      const totalChunks = Math.max(1, Math.ceil(f.size / chunkBytes))
      const st = await apiGet<{ received?: number[] }>(
        `/ingest/upload/status?upload_id=${encodeURIComponent(id)}&file_path=${encodeURIComponent(f.path)}`,
      )
      const received = new Set<number>((st?.received ?? []) as number[])
      for (let idx = 0; idx < totalChunks; idx++) {
        if (received.has(idx)) continue
        const start = idx * chunkBytes
        const end = Math.min(f.size, start + chunkBytes)
        const blob = f.file.slice(start, end)
        const form = new FormData()
        form.append('upload_id', id)
        form.append('index', String(idx))
        form.append('file_path', f.path)
        form.append('blob', blob, `chunk-${idx}.bin`)
        await apiPostForm('/ingest/upload/chunk', form)
        setUploadProgress((p) => ({ sent: Math.min(p.total, p.sent + (end - start)), total: p.total }))
      }
    }
  }

  async function startUpload(mode: 'zip' | 'folder') {
    setUploadBusy(true)
    setError('')
    setInfo('')
    setTask(null)
    setTaskId('')
    setScan(null)
    try {
      if (mode === 'zip') {
        if (!zipFile) throw new Error('请先选择一个 .zip 文件。')
        const start = await apiPost<{ upload_id: string }>('/ingest/upload/start', {
          mode: 'zip',
          chunk_bytes: chunkBytes,
          total_bytes: zipFile.size,
          filename: zipFile.name,
        })
        const id = String(start.upload_id ?? '')
        setUploadId(id)
        await uploadChunksZip(id, zipFile)
        const s = await apiPost<UploadScan>(`/ingest/upload/finish?upload_id=${encodeURIComponent(id)}`, {})
        setScan(s)
        setInfo(`上传完成：${id}`)
        await maybeAutoReplace(id, s)
      } else {
        if (folderFiles.length === 0) throw new Error('请先选择一个文件夹。')
        const start = await apiPost<{ upload_id: string }>('/ingest/upload/start', {
          mode: 'folder',
          chunk_bytes: chunkBytes,
          files: folderFiles.map((f) => ({ path: f.path, size: f.size })),
        })
        const id = String(start.upload_id ?? '')
        setUploadId(id)
        await uploadChunksFolder(id, folderFiles)
        const s = await apiPost<UploadScan>(`/ingest/upload/finish?upload_id=${encodeURIComponent(id)}`, {})
        setScan(s)
        setInfo(`上传完成：${id}`)
        await maybeAutoReplace(id, s)
      }
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    } finally {
      setUploadBusy(false)
    }
  }

  async function setDoi(unitId: string, doi: string) {
    if (!uploadId) return
    const d = doi.trim()
    if (!d) return
    setError('')
    try {
      await apiPost<Record<string, unknown>>('/ingest/upload/set_doi', { upload_id: uploadId, unit_id: unitId, doi: d })
      const s = await refreshScan(uploadId)
      await maybeAutoReplace(uploadId, s)
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    }
  }

  async function setUnitPaperType(unitId: string, pt: string) {
    if (!uploadId) return
    const v = (pt ?? '').trim().toLowerCase()
    if (!v) return
    try {
      await apiPost<Record<string, unknown>>('/ingest/upload/set_paper_type', { upload_id: uploadId, unit_id: unitId, paper_type: v })
      await refreshScan(uploadId)
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    }
  }

  async function replaceWithNew(unitId: string) {
    if (!uploadId) return
    setError('')
    try {
      const res = await apiPost<{ task_id: string }>('/ingest/upload/replace_with_new', { upload_id: uploadId, unit_id: unitId })
      setTaskId(res.task_id ?? '')
      setInfo(`已提交补全导入任务：${res.task_id ?? ''}`)
    } catch (e: unknown) {
      setError(String((e as { message?: unknown } | null)?.message ?? e))
    }
  }

  async function maybeAutoReplace(id: string, s: UploadScan | null) {
    if (!id) return
    const scan0 = s ?? scan
    const units = scan0?.units ?? []
    if (units.length === 0) return

    const target = paperDoi.trim().toLowerCase()
    const pick = units.find((u) => String(u.doi ?? '').trim().toLowerCase() === target) ?? units[0]
    const status = String(pick.status ?? '')
    if (status === 'need_doi') {
      // wait for user to set DOI
      setInfo(`扫描完成：需要 DOI 后才能补全导入。`)
      return
    }

    // Ensure paper_type is set (best-effort; scan contains paper_type but backend reads from overrides/meta).
    await setUnitPaperType(pick.unit_id, paperType)
    // For stub补全：无论 ready/conflict，都走 replace
    await replaceWithNew(pick.unit_id)
  }

  async function pollTask(id: string) {
    const t = await apiGet<TaskInfo>(`/tasks/${encodeURIComponent(id)}`)
    setTask(t)
    const status = String(t?.status ?? '')
    if (status && !['queued', 'running'].includes(status)) {
      if (status === 'succeeded') {
        setInfo(`补全导入完成：${id}`)
        await onRefreshNetwork()
      } else if (status === 'failed') {
        setError(String(t?.error ?? t?.message ?? '任务失败'))
      }
    }
  }

  useEffect(() => {
    if (!taskId) return
    let alive = true
    const tick = async () => {
      if (!alive) return
      await pollTask(taskId)
    }
    tick().catch(() => {})
    const iv = setInterval(() => tick().catch(() => {}), 1200)
    return () => {
      alive = false
      clearInterval(iv)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId])

  const showUploader = !node.ingested

  return (
    <div className="stack">
      {error && <div className="errorBox">{error}</div>}
      {info && (
        <div className="infoBox">
          <div className="split">
            <div style={{ whiteSpace: 'pre-wrap' }}>{info}</div>
            <button className="btn btnSmall" onClick={() => setInfo('')}>
              清除
            </button>
          </div>
        </div>
      )}

      <div className="itemCard">
        <div className="itemTitle">{node.title ?? node.paper_source ?? node.doi ?? node.id}</div>
        <div className="itemMeta" style={{ marginTop: 8 }}>
          {node.year ? `年份 ${node.year}` : '年份未知'} · <code>{node.id}</code>
        </div>
        {paperDoi && (
          <div className="itemMeta" style={{ marginTop: 6 }}>
            DOI <code>{paperDoi}</code>
          </div>
        )}
      </div>

      {!node.ingested && (
        <div className="infoBox">
          <div style={{ fontWeight: 800, marginBottom: 6 }}>该节点仅包含元数据（{TERMS.stub}）。</div>
          <div className="metaLine">上传 MinerU 输出（含 .md 与 images/）后，可补全导入并生成逻辑链/要点/图片。</div>
        </div>
      )}

      <button className="btn btnPrimary" onClick={() => onOpenDetail(node.id)}>
        打开详情
      </button>

      {showUploader && (
        <div className="panel" style={{ borderRadius: 14 }}>
          <div className="panelHeader">
            <div className="split">
              <div className="panelTitle">补全导入（上传）</div>
              {uploadProgress.total > 0 && (
                <span className="pill">
                  <span className="kicker">进度</span> {uploadPct}% · {(uploadProgress.sent / 1e6).toFixed(1)}/{(uploadProgress.total / 1e6).toFixed(1)} MB
                </span>
              )}
            </div>
          </div>
          <div className="panelBody">
            <div className="stack">
              {uploadProgress.total > 0 && (
                <div className="progress">
                  <div className="progressBar" style={{ width: `${uploadPct}%` }} />
                </div>
              )}

              <div className="itemCard">
                <div className="itemTitle">ZIP（推荐）</div>
                <div className="row" style={{ marginTop: 8 }}>
                  <input type="file" accept=".zip" onChange={(e) => setZipFile(e.target.files?.[0] ?? null)} />
                  <button className="btn btnPrimary" disabled={uploadBusy} onClick={() => startUpload('zip')}>
                    {uploadBusy ? '上传中…' : '上传并补全'}
                  </button>
                </div>
              </div>

              <div className="itemCard">
                <div className="itemTitle">论文类型</div>
                <div className="row" style={{ marginTop: 8 }}>
                  <select className="select" style={{ width: 220 }} value={paperType} onChange={(e) => setPaperType(e.target.value)}>
                    <option value="research">研究型(Research)</option>
                    <option value="review">综述型(Review)</option>
                  </select>
                </div>
              </div>

              <div className="itemCard">
                <div className="itemTitle">文件夹</div>
                <div className="row" style={{ marginTop: 8 }}>
                  <input
                    type="file"
                    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                    // @ts-ignore
                    webkitdirectory="true"
                    multiple
                    onChange={(e) => {
                      const files = Array.from(e.target.files ?? [])
                      const mapped: FolderFile[] = files.map((f) => {
                        const wf = f as WebkitFile
                        return { path: String(wf.webkitRelativePath ?? f.name), file: f, size: f.size }
                      })
                      setFolderFiles(mapped)
                    }}
                  />
                  <button className="btn btnPrimary" disabled={uploadBusy} onClick={() => startUpload('folder')}>
                    {uploadBusy ? '上传中…' : `上传并补全（${folderFiles.length} 个文件）`}
                  </button>
                </div>
              </div>

              {scan && scan.units.some((u) => u.status === 'need_doi') && (
                <div className="itemCard">
                  <div className="itemTitle">缺失 DOI</div>
                  <div className="metaLine" style={{ marginTop: 6 }}>
                    扫描到的论文缺 DOI，需要先设置 DOI 才能补全导入。
                  </div>
                  <div className="row" style={{ marginTop: 10 }}>
                    <input className="input" value={doiInput} onChange={(e) => setDoiInput(e.target.value)} placeholder="DOI（10.xxxx/...）" />
                    <button
                      className="btn btnPrimary"
                      onClick={() => {
                        const u = scan.units.find((x) => x.status === 'need_doi')
                        if (!u) return
                        setDoi(u.unit_id, doiInput)
                      }}
                    >
                      设置 DOI 并继续
                    </button>
                  </div>
                </div>
              )}

              {taskId && (
                <div className="metaLine">
                  任务: <code>{taskId}</code> · {taskStatusLabel(task?.status)} · {(Number(task?.progress ?? 0) || 0).toFixed(2)} · {taskStageLabel(task?.stage)}
                </div>
              )}

              {uploadId && (
                <div className="metaLine">
                  upload_id: <code>{uploadId}</code>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
