from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from app.citations.aggregate import build_reference_and_cite_records
from app.crossref.client import CrossrefClient
from app.graph.neo4j_client import Neo4jClient
from app.graph.neo4j_client import paper_id_for_md_path
from app.ingest.figures import extract_figures_from_markdown
from app.ingest.paper_meta import load_canonical_meta
from app.ingest.models import Chunk, MdSpan
from app.ingest.parse_md import parse_mineru_markdown
from app.llm.citation_purpose import classify_citation_purposes_batch
from app.llm.logic_claims_v2 import add_evidence_and_targets, extract_logic_and_claims_v2
from app.schema_store import load_active
from app.settings import settings
from app.vector.faiss_store import build_faiss_for_chunks


ProgressFn = Callable[[str, float, str | None], None]
LogFn = Callable[[str], None]


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _storage_dir() -> Path:
    p = _backend_root() / settings.storage_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def rebuild_paper(
    paper_id: str,
    progress: ProgressFn | None = None,
    log: LogFn | None = None,
) -> dict[str, Any]:
    def notify(stage: str, p: float, msg: str | None = None) -> None:
        if progress:
            progress(stage, p, msg)

    def write_log(line: str) -> None:
        if log:
            log(line)

    notify("rebuild:load", 0.05, "Loading paper from Neo4j")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        paper = client.get_paper_basic(paper_id)

    md_path = str(paper.get("source_md_path") or "").strip()
    if not md_path:
        raise FileNotFoundError(f"Paper has no source_md_path: {paper_id}")
    md_file = Path(md_path)
    if not md_file.exists():
        raise FileNotFoundError(f"Markdown not found on disk: {md_path}")

    expected_doi = None
    if paper_id.startswith("doi:"):
        expected_doi = paper_id[4:]

    notify("rebuild:parse", 0.15, "Parsing markdown")
    doc = parse_mineru_markdown(str(md_file))
    expected_paper_id = paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi)
    if expected_paper_id != paper_id:
        raise RuntimeError(
            f"paper_id mismatch: requested={paper_id!r}, parsed={expected_paper_id!r}. "
            "Refuse to rebuild to avoid overwriting a different paper."
        )

    notify("rebuild:crossref", 0.30, "Resolving references via Crossref")
    crossref = CrossrefClient()
    cite_rec = build_reference_and_cite_records(doc, crossref=crossref)

    notify("rebuild:neo4j_clear", 0.42, "Clearing existing subgraph for this paper")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        client.ensure_schema()
        client.delete_paper_subgraph(paper_id)

    notify("rebuild:neo4j_write", 0.50, "Writing rebuilt data to Neo4j")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        client.upsert_paper_and_chunks(doc)
        try:
            meta = load_canonical_meta(doc.paper.md_path)
            paper_type = str(meta.get("paper_type") or "research").strip().lower()
            if paper_type not in {"research", "review"}:
                paper_type = "research"
            schema = load_active(paper_type)  # type: ignore[arg-type]
            client.update_paper_props(
                paper_id,
                {
                    "paper_type": paper_type,
                    "schema_paper_type": paper_type,
                    "schema_version": int(schema.get("version") or 1),
                },
            )
        except Exception:
            pass
        try:
            figs = extract_figures_from_markdown(paper_id=paper_id, md_path=doc.paper.md_path)
            client.upsert_figures(
                paper_id,
                [
                    {
                        "figure_id": f.figure_id,
                        "paper_id": paper_id,
                        "md_path": f.md_path,
                        "rel_path": f.rel_path,
                        "filename": f.filename,
                        "img_line": f.img_line,
                        "caption_text": f.caption_text,
                        "caption_start_line": f.caption_start_line,
                        "caption_end_line": f.caption_end_line,
                    }
                    for f in figs
                ],
            )
        except Exception:
            pass
        if cite_rec.get("paper_id"):
            client.upsert_references_and_citations(
                paper_id=cite_rec["paper_id"],
                refs=cite_rec["refs"],
                cited_papers=cite_rec["cited_papers"],
                cites_resolved=cite_rec["cites_resolved"],
                cites_unresolved=cite_rec["cites_unresolved"],
            )

    notify("rebuild:llm", 0.68, "Running LLM extraction (Logic/Claims/Citation Purposes)")
    meta = load_canonical_meta(doc.paper.md_path)
    paper_type = str(meta.get("paper_type") or "research").strip().lower()
    if paper_type not in {"research", "review"}:
        paper_type = "research"
    schema = load_active(paper_type)  # type: ignore[arg-type]
    steps_sorted = sorted(schema.get("steps") or [], key=lambda x: int((x or {}).get("order") or 0))
    step_order = [str(s.get("id") or "") for s in steps_sorted if bool((s or {}).get("enabled", True)) and str((s or {}).get("id") or "").strip()]
    if not step_order:
        step_order = [str(s.get("id") or "") for s in steps_sorted if str((s or {}).get("id") or "").strip()]

    logic_claims = extract_logic_and_claims_v2(doc, paper_id=paper_id, schema=schema)
    try:
        from app.llm.logic_claims_v2 import add_logic_step_evidence

        add_logic_step_evidence(doc, schema=schema, logic=logic_claims["logic"])
    except Exception:
        pass
    add_evidence_and_targets(doc, schema=schema, claims=logic_claims["claims"], cite_rec=cite_rec)

    purposes = []
    chunk_by_id = {c.chunk_id: c for c in doc.chunks}
    citing_title = doc.paper.title or doc.paper.title_alt or doc.paper.paper_source
    batch_in = []
    for cr in cite_rec.get("cites_resolved") or []:
        cited_paper_id = cr.get("cited_paper_id")
        cited_doi = None
        if cited_paper_id and str(cited_paper_id).startswith("doi:"):
            cited_doi = str(cited_paper_id)[4:]
        cited_title = None
        for cp in cite_rec.get("cited_papers") or []:
            if cp.get("paper_id") == cited_paper_id:
                cited_title = cp.get("title")
                break
        contexts = []
        for cid in cr.get("evidence_chunk_ids") or []:
            ch = chunk_by_id.get(cid)
            if ch and ch.text:
                contexts.append(ch.text)
        batch_in.append(
            {
                "cited_paper_id": cited_paper_id,
                "cited_title": cited_title,
                "cited_doi": cited_doi,
                "contexts": contexts,
            }
        )
    batch_out = classify_citation_purposes_batch(citing_title=citing_title, cites=batch_in, prompt_overrides=schema.get("prompts"))
    by_id = batch_out.get("by_id") or {}
    for cr in cite_rec.get("cites_resolved") or []:
        cited_paper_id = cr.get("cited_paper_id")
        if not cited_paper_id:
            continue
        x = by_id.get(str(cited_paper_id)) or {"labels": ["Background"], "scores": [0.4]}
        purposes.append({"cited_paper_id": cited_paper_id, "labels": x["labels"], "scores": x["scores"]})

    notify("rebuild:neo4j_llm", 0.78, "Writing LLM outputs to Neo4j")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        client.upsert_logic_steps_and_claims(paper_id=paper_id, logic=logic_claims["logic"], claims=logic_claims["claims"], step_order=step_order)
        # Re-apply human evidence overrides (if any) on top of the rebuilt machine graph.
        try:
            client.apply_human_claim_evidence_overrides(paper_id)
        except Exception:
            pass
        try:
            client.apply_human_logic_step_evidence_overrides(paper_id)
        except Exception:
            pass
        for p in purposes:
            if not p.get("cited_paper_id"):
                continue
            client.update_cites_purposes(
                citing_paper_id=paper_id,
                cited_paper_id=p["cited_paper_id"],
                labels=p["labels"],
                scores=p["scores"],
            )

    notify("rebuild:artifacts", 0.86, "Writing rebuilt artifacts to storage")
    out_dir = _storage_dir() / "derived" / "papers" / _safe_id(paper_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "document_ir.json").write_text(
        json.dumps(
            {
                "paper": doc.paper.__dict__,
                "chunks": [{**c.__dict__, "span": c.span.__dict__} for c in doc.chunks],
                "references": [r.__dict__ for r in doc.references],
                "citations": [{**ce.__dict__, "span": ce.span.__dict__} for ce in doc.citations],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "citations.json").write_text(json.dumps(cite_rec, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "llm_imrad.json").write_text(json.dumps(logic_claims, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "llm_citation_purposes.json").write_text(json.dumps(purposes, ensure_ascii=False, indent=2), encoding="utf-8")

    write_log(f"rebuilt artifacts in {out_dir}")
    notify("rebuild:paper_done", 0.92, "Paper rebuild done")
    return {
        "paper_id": paper_id,
        "source_md_path": md_path,
        "artifacts_dir": str(out_dir),
        "citations": {
            "refs": len(cite_rec.get("refs") or []),
            "cites_resolved": len(cite_rec.get("cites_resolved") or []),
            "cites_unresolved": len(cite_rec.get("cites_unresolved") or []),
        },
        "llm": {"purposes": len(purposes), "claims": len(logic_claims.get("claims") or [])},
    }


def replace_paper_from_md_path(
    paper_id: str,
    md_path: str,
    progress: ProgressFn | None = None,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """
    Replace a paper's subgraph using a specific markdown path (used for DOI-conflict 'Replace with new').
    """
    def notify(stage: str, p: float, msg: str | None = None) -> None:
        if progress:
            progress(stage, p, msg)

    def write_log(line: str) -> None:
        if log:
            log(line)

    md_file = Path(md_path)
    if not md_file.exists():
        raise FileNotFoundError(f"Markdown not found on disk: {md_path}")

    notify("replace:parse", 0.10, "Parsing markdown")
    doc = parse_mineru_markdown(str(md_file))
    expected_paper_id = paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi)
    if expected_paper_id != paper_id:
        raise RuntimeError(f"paper_id mismatch: requested={paper_id!r}, parsed={expected_paper_id!r}")

    notify("replace:crossref", 0.25, "Resolving references via Crossref")
    crossref = CrossrefClient()
    cite_rec = build_reference_and_cite_records(doc, crossref=crossref)

    notify("replace:neo4j_clear", 0.40, "Clearing existing subgraph for this paper")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        client.ensure_schema()
        client.delete_paper_subgraph(paper_id)

    notify("replace:neo4j_write", 0.52, "Writing rebuilt data to Neo4j")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        client.upsert_paper_and_chunks(doc)
        try:
            meta = load_canonical_meta(doc.paper.md_path)
            paper_type = str(meta.get("paper_type") or "research").strip().lower()
            if paper_type not in {"research", "review"}:
                paper_type = "research"
            schema = load_active(paper_type)  # type: ignore[arg-type]
            client.update_paper_props(
                paper_id,
                {
                    "paper_type": paper_type,
                    "schema_paper_type": paper_type,
                    "schema_version": int(schema.get("version") or 1),
                },
            )
        except Exception:
            pass
        try:
            figs = extract_figures_from_markdown(paper_id=paper_id, md_path=doc.paper.md_path)
            client.upsert_figures(
                paper_id,
                [
                    {
                        "figure_id": f.figure_id,
                        "paper_id": paper_id,
                        "md_path": f.md_path,
                        "rel_path": f.rel_path,
                        "filename": f.filename,
                        "img_line": f.img_line,
                        "caption_text": f.caption_text,
                        "caption_start_line": f.caption_start_line,
                        "caption_end_line": f.caption_end_line,
                    }
                    for f in figs
                ],
            )
        except Exception:
            pass
        if cite_rec.get("paper_id"):
            client.upsert_references_and_citations(
                paper_id=cite_rec["paper_id"],
                refs=cite_rec["refs"],
                cited_papers=cite_rec["cited_papers"],
                cites_resolved=cite_rec["cites_resolved"],
                cites_unresolved=cite_rec["cites_unresolved"],
            )

    notify("replace:llm", 0.70, "Running LLM extraction (Logic/Claims/Citation Purposes)")
    meta = load_canonical_meta(doc.paper.md_path)
    paper_type = str(meta.get("paper_type") or "research").strip().lower()
    if paper_type not in {"research", "review"}:
        paper_type = "research"
    schema = load_active(paper_type)  # type: ignore[arg-type]
    steps_sorted = sorted(schema.get("steps") or [], key=lambda x: int((x or {}).get("order") or 0))
    step_order = [str(s.get("id") or "") for s in steps_sorted if bool((s or {}).get("enabled", True)) and str((s or {}).get("id") or "").strip()]
    if not step_order:
        step_order = [str(s.get("id") or "") for s in steps_sorted if str((s or {}).get("id") or "").strip()]

    logic_claims = extract_logic_and_claims_v2(doc, paper_id=paper_id, schema=schema)
    try:
        from app.llm.logic_claims_v2 import add_logic_step_evidence

        add_logic_step_evidence(doc, schema=schema, logic=logic_claims["logic"])
    except Exception:
        pass
    add_evidence_and_targets(doc, schema=schema, claims=logic_claims["claims"], cite_rec=cite_rec)

    purposes = []
    chunk_by_id = {c.chunk_id: c for c in doc.chunks}
    citing_title = doc.paper.title or doc.paper.title_alt or doc.paper.paper_source
    batch_in = []
    for cr in cite_rec.get("cites_resolved") or []:
        cited_paper_id = cr.get("cited_paper_id")
        cited_doi = None
        if cited_paper_id and str(cited_paper_id).startswith("doi:"):
            cited_doi = str(cited_paper_id)[4:]
        cited_title = None
        for cp in cite_rec.get("cited_papers") or []:
            if cp.get("paper_id") == cited_paper_id:
                cited_title = cp.get("title")
                break
        contexts = []
        for cid in cr.get("evidence_chunk_ids") or []:
            ch = chunk_by_id.get(cid)
            if ch and ch.text:
                contexts.append(ch.text)
        batch_in.append(
            {
                "cited_paper_id": cited_paper_id,
                "cited_title": cited_title,
                "cited_doi": cited_doi,
                "contexts": contexts,
            }
        )
    batch_out = classify_citation_purposes_batch(citing_title=citing_title, cites=batch_in, prompt_overrides=schema.get("prompts"))
    by_id = batch_out.get("by_id") or {}
    for cr in cite_rec.get("cites_resolved") or []:
        cited_paper_id = cr.get("cited_paper_id")
        if not cited_paper_id:
            continue
        x = by_id.get(str(cited_paper_id)) or {"labels": ["Background"], "scores": [0.4]}
        purposes.append({"cited_paper_id": cited_paper_id, "labels": x["labels"], "scores": x["scores"]})

    notify("replace:neo4j_llm", 0.82, "Writing LLM outputs to Neo4j")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        client.upsert_logic_steps_and_claims(paper_id=paper_id, logic=logic_claims["logic"], claims=logic_claims["claims"], step_order=step_order)
        try:
            client.apply_human_claim_evidence_overrides(paper_id)
        except Exception:
            pass
        try:
            client.apply_human_logic_step_evidence_overrides(paper_id)
        except Exception:
            pass
        for p in purposes:
            if not p.get("cited_paper_id"):
                continue
            client.update_cites_purposes(
                citing_paper_id=paper_id,
                cited_paper_id=p["cited_paper_id"],
                labels=p["labels"],
                scores=p["scores"],
            )

    write_log(f"replaced {paper_id} from {md_path}")
    notify("replace:done", 1.0, "Done")
    return {"paper_id": paper_id, "source_md_path": md_path, "claims": len(logic_claims.get('claims') or [])}


def rebuild_global_faiss(progress: ProgressFn | None = None, log: LogFn | None = None) -> dict[str, Any]:
    def notify(stage: str, p: float, msg: str | None = None) -> None:
        if progress:
            progress(stage, p, msg)

    def write_log(line: str) -> None:
        if log:
            log(line)

    notify("rebuild:faiss_load", 0.10, "Loading chunks from Neo4j")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        rows = client.list_chunks_for_faiss(limit=200000)

    chunks: list[Chunk] = []
    for r in rows:
        span = MdSpan(start_line=int(r.get("start_line") or 0), end_line=int(r.get("end_line") or 0))
        chunks.append(
            Chunk(
                chunk_id=str(r.get("chunk_id")),
                paper_source=str(r.get("paper_source") or ""),
                md_path=str(r.get("md_path") or ""),
                span=span,
                section=r.get("section"),
                kind=str(r.get("kind") or "block"),
                text=str(r.get("text") or ""),
            )
        )

    if not chunks:
        raise FileNotFoundError("No chunks found in Neo4j (did you ingest anything?)")

    notify("rebuild:faiss_build", 0.55, f"Building FAISS over {len(chunks)} chunks")
    out_dir = _storage_dir() / "faiss"
    res = build_faiss_for_chunks(chunks, out_dir=str(out_dir))
    write_log(f"built global FAISS in {out_dir}")
    notify("rebuild:faiss_done", 1.0, "FAISS rebuild done")
    return {"faiss": res, "dir": str(out_dir)}
