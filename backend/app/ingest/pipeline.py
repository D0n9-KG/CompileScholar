from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.citations.aggregate import build_reference_and_cite_records
from app.citations.citation_event_recovery import recover_citation_events_from_references
from app.crossref.client import CrossrefClient
from app.extraction.orchestrator import run_phase1_extraction
from app.graph.neo4j_client import Neo4jClient
from app.graph.neo4j_client import paper_id_for_md_path
from app.ingest.figures import extract_figures_from_markdown
from app.ingest.models import DocumentIR
from app.ingest.paper_meta import load_canonical_meta
from app.ingest.parse_md import find_mineru_markdowns, parse_mineru_markdown
from app.llm.citation_purpose import classify_citation_purposes_batch
from app.llm.reference_recovery import recover_references_with_agent
from app.schema_store import load_active
from app.settings import settings
from app.vector.faiss_store import build_faiss_for_chunks


ProgressFn = Callable[[str, float, str | None], None]


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s or ""))


def _paper_type_for_md(md_path: str) -> str:
    try:
        meta = load_canonical_meta(md_path)
        paper_type = str(meta.get("paper_type") or "research").strip().lower()
        if paper_type not in {"research", "review"}:
            return "research"
        return paper_type
    except Exception:
        return "research"


def _schema_for_md(md_path: str) -> dict:
    paper_type = _paper_type_for_md(md_path)
    try:
        return load_active(paper_type)  # type: ignore[arg-type]
    except Exception:
        return load_active("research")  # type: ignore[arg-type]


def _write_document_ir(path: Path, doc: DocumentIR) -> None:
    path.write_text(
        json.dumps(
            {
                "paper": doc.paper.__dict__,
                "chunks": [
                    {
                        **c.__dict__,
                        "span": c.span.__dict__,
                    }
                    for c in doc.chunks
                ],
                "references": [r.__dict__ for r in doc.references],
                "citations": [
                    {
                        **ce.__dict__,
                        "span": ce.span.__dict__,
                    }
                    for ce in doc.citations
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def ingest_markdowns(md_files: list[str], progress: ProgressFn | None = None) -> dict:
    def notify(stage: str, p: float, msg: str | None = None) -> None:
        if progress:
            progress(stage, p, msg)

    if not md_files:
        raise FileNotFoundError("No markdown files provided")

    notify("ingest:init", 0.06, "Preparing run directory")
    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(__file__).resolve().parents[2] / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir.parent / "LATEST").write_text(run_id, encoding="utf-8")

    notify("ingest:parse", 0.12, f"Parsing {len(md_files)} markdown(s)")
    parsed = []
    for md in md_files:
        doc = parse_mineru_markdown(md)
        parsed.append(doc)
        out = run_dir / f"{doc.paper.paper_source}.document_ir.json"
        _write_document_ir(out, doc)

    notify("ingest:reference_recovery", 0.24, "Recovering references for papers with missing/low parsed refs")
    reference_recovery: list[dict] = []
    for idx, doc in enumerate(parsed):
        schema_for_recovery = _schema_for_md(doc.paper.md_path)
        before_refs = len(doc.references or [])
        recovered_doc, rr = recover_references_with_agent(
            doc,
            prompt_overrides=schema_for_recovery.get("prompts"),
            rules=schema_for_recovery.get("rules"),
        )
        parsed[idx] = recovered_doc
        rr["paper_source"] = doc.paper.paper_source
        rr["paper_id"] = paper_id_for_md_path(recovered_doc.paper.md_path, doi=recovered_doc.paper.doi)
        rr["schema_version"] = int(schema_for_recovery.get("version") or 1)
        rr["schema_paper_type"] = str(schema_for_recovery.get("paper_type") or "research")
        reference_recovery.append(rr)

        rr_path = run_dir / f"{doc.paper.paper_source}.reference_recovery.json"
        rr_path.write_text(json.dumps(rr, ensure_ascii=False, indent=2), encoding="utf-8")
        if int(rr.get("after_refs") or before_refs) != before_refs:
            out = run_dir / f"{doc.paper.paper_source}.document_ir.json"
            _write_document_ir(out, recovered_doc)

    notify("ingest:citation_event_recovery", 0.30, "Recovering citation events from references when needed")
    citation_event_recovery: list[dict] = []
    for idx, doc in enumerate(parsed):
        schema_for_recovery = _schema_for_md(doc.paper.md_path)
        recovered_doc, cer = recover_citation_events_from_references(
            doc,
            rules=schema_for_recovery.get("rules"),
        )
        parsed[idx] = recovered_doc
        cer["paper_source"] = doc.paper.paper_source
        cer["paper_id"] = paper_id_for_md_path(recovered_doc.paper.md_path, doi=recovered_doc.paper.doi)
        cer["schema_version"] = int(schema_for_recovery.get("version") or 1)
        cer["schema_paper_type"] = str(schema_for_recovery.get("paper_type") or "research")
        citation_event_recovery.append(cer)

        cer_path = run_dir / f"{doc.paper.paper_source}.citation_event_recovery.json"
        cer_path.write_text(json.dumps(cer, ensure_ascii=False, indent=2), encoding="utf-8")
        if int(cer.get("after_events") or 0) != int(cer.get("before_events") or 0):
            out = run_dir / f"{doc.paper.paper_source}.document_ir.json"
            _write_document_ir(out, recovered_doc)

    notify("ingest:crossref", 0.35, "Resolving references via Crossref")
    crossref = CrossrefClient()
    cite_records = []
    for doc in parsed:
        try:
            rec = build_reference_and_cite_records(doc, crossref=crossref)
            cite_records.append(rec)
            out = run_dir / f"{doc.paper.paper_source}.citations.json"
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            cite_records.append(
                {
                    "paper_id": None,
                    "error": str(exc),
                    "paper_source": doc.paper.paper_source,
                }
            )

    notify("ingest:neo4j", 0.52, "Writing to Neo4j (papers, chunks, cites)")
    neo4j_written = False
    neo4j_error = None
    try:
        with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
            client.ensure_schema()
            for doc in parsed:
                client.upsert_paper_and_chunks(doc)
                try:
                    paper_id = paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi)
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
                    paper_id = paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi)
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
                    # figures are optional; do not fail ingestion
                    pass
            for rec in cite_records:
                if not rec or not rec.get("paper_id"):
                    continue
                client.upsert_references_and_citations(
                    paper_id=rec["paper_id"],
                    refs=rec["refs"],
                    cited_papers=rec["cited_papers"],
                    cites_resolved=rec["cites_resolved"],
                    cites_unresolved=rec["cites_unresolved"],
                )
            neo4j_written = True
    except Exception as exc:  # noqa: BLE001
        neo4j_error = str(exc)

    notify("ingest:llm", 0.70, "Running LLM extraction (Logic/Claims/Citation Purposes)")
    llm_built = False
    llm_error = None
    llm_outputs = []
    try:
        # LLM extraction (DeepSeek) + write to Neo4j; if Neo4j isn't available, still write artifacts.
        for doc, rec in zip(parsed, cite_records):
            paper_id = rec.get("paper_id")
            if not paper_id:
                continue

            meta = load_canonical_meta(doc.paper.md_path)
            paper_type = str(meta.get("paper_type") or "research").strip().lower()
            if paper_type not in {"research", "review"}:
                paper_type = "research"
            schema = load_active(paper_type)  # type: ignore[arg-type]
            phase1_artifacts_dir = run_dir / "raw_pool" / _safe_id(paper_id)
            phase1 = run_phase1_extraction(
                doc=doc,
                paper_id=str(paper_id),
                cite_rec=rec,
                schema=schema,
                artifacts_dir=phase1_artifacts_dir,
                allow_weak=bool(getattr(settings, "phase1_gate_allow_weak", False)),
            )
            step_order = list(phase1.get("step_order") or [])
            logic_claims = {
                "logic": phase1.get("logic") or {},
                "claims": phase1.get("validated_claims") or [],
                "quality_report": phase1.get("quality_report") or {},
                "raw_claim_candidates": len(phase1.get("claim_candidates") or []),
                "raw_claims_merged": len(phase1.get("claims_merged") or []),
                "rejected_claims": len(phase1.get("rejected_claims") or []),
            }
            llm_out = {"paper_id": paper_id, "schema": {"paper_type": paper_type, "version": schema.get("version")}, **logic_claims}
            out = run_dir / f"{doc.paper.paper_source}.llm_imrad.json"
            out.write_text(json.dumps(logic_claims, ensure_ascii=False, indent=2), encoding="utf-8")

            # Citation purpose classification for resolved cites
            purposes = []
            chunk_by_id = {c.chunk_id: c for c in doc.chunks}
            citing_title = doc.paper.title or doc.paper.title_alt or doc.paper.paper_source
            batch_in = []
            for cr in rec.get("cites_resolved") or []:
                cited_paper_id = cr.get("cited_paper_id")
                cited_doi = None
                if cited_paper_id and str(cited_paper_id).startswith("doi:"):
                    cited_doi = str(cited_paper_id)[4:]
                cited_title = None
                # Try to locate metadata from cited_papers list
                for cp in rec.get("cited_papers") or []:
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
            batch_out = classify_citation_purposes_batch(
                citing_title=citing_title,
                cites=batch_in,
                prompt_overrides=schema.get("prompts"),
                rules=schema.get("rules"),
            )
            by_id = batch_out.get("by_id") or {}
            for cr in rec.get("cites_resolved") or []:
                cited_paper_id = cr.get("cited_paper_id")
                if not cited_paper_id:
                    continue
                x = by_id.get(str(cited_paper_id)) or {"labels": ["Background"], "scores": [0.4]}
                purposes.append({"cited_paper_id": cited_paper_id, "labels": x["labels"], "scores": x["scores"]})
            out2 = run_dir / f"{doc.paper.paper_source}.llm_citation_purposes.json"
            out2.write_text(json.dumps(purposes, ensure_ascii=False, indent=2), encoding="utf-8")

            llm_out["citation_purposes"] = purposes
            llm_outputs.append(llm_out)

            # Write into Neo4j if available
            if neo4j_written:
                with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
                    client.upsert_logic_steps_and_claims(paper_id=paper_id, logic=logic_claims["logic"], claims=logic_claims["claims"], step_order=step_order)
                    try:
                        quality_report = logic_claims.get("quality_report") or {}
                        client.update_paper_props(
                            paper_id,
                            {
                                "phase1_quality_json": json.dumps(quality_report, ensure_ascii=False),
                                "phase1_gate_passed": bool(quality_report.get("gate_passed")),
                                "phase1_quality_tier": str(quality_report.get("quality_tier") or ""),
                                "phase1_quality_tier_score": float(quality_report.get("quality_tier_score") or 0.0),
                            },
                        )
                    except Exception:
                        pass
                    try:
                        client.upsert_proposition_mentions_for_claims(
                            paper_id=paper_id,
                            claims=logic_claims["claims"],
                            paper_year=doc.paper.year,
                        )
                    except Exception:
                        pass
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
        llm_built = True
    except Exception as exc:  # noqa: BLE001
        llm_error = str(exc)

    notify("ingest:faiss", 0.90, "Building FAISS index")
    faiss_built = False
    faiss_error = None
    faiss_dir = str(run_dir / "faiss")
    try:
        # Build one index over all chunks across all ingested papers.
        all_chunks = [c for d in parsed for c in d.chunks if c.kind != "heading"]
        build_faiss_for_chunks(all_chunks, out_dir=faiss_dir)
        faiss_built = True
    except Exception as exc:  # noqa: BLE001
        faiss_error = str(exc)

    notify("ingest:done", 1.0, "Done")
    return {
        "run_id": run_id,
        "md_files": md_files,
        "papers": [
            {
                "paper_source": d.paper.paper_source,
                "md_path": d.paper.md_path,
                "chunks": len(d.chunks),
                "references": len(d.references),
                "citation_events": len(d.citations),
                "doi": d.paper.doi,
                "title": d.paper.title,
                "year": d.paper.year,
            }
            for d in parsed
        ],
        "citations_built": [
            {
                "paper_id": r.get("paper_id"),
                "refs": len(r.get("refs") or []),
                "cites_resolved": len(r.get("cites_resolved") or []),
                "cites_unresolved": len(r.get("cites_unresolved") or []),
                "error": r.get("error"),
                "paper_source": r.get("paper_source"),
            }
            for r in cite_records
        ],
        "reference_recovery": reference_recovery,
        "citation_event_recovery": citation_event_recovery,
        "neo4j_written": neo4j_written,
        "neo4j_error": neo4j_error,
        "llm_built": llm_built,
        "llm_error": llm_error,
        "phase1_quality": [
            {
                "paper_id": o.get("paper_id"),
                "gate_passed": bool((o.get("quality_report") or {}).get("gate_passed")),
                "quality_tier": str((o.get("quality_report") or {}).get("quality_tier") or ""),
                "quality_tier_score": (o.get("quality_report") or {}).get("quality_tier_score"),
                "supported_claim_ratio": (o.get("quality_report") or {}).get("supported_claim_ratio"),
                "step_coverage_ratio": (o.get("quality_report") or {}).get("step_coverage_ratio"),
                "validated_claims": len(o.get("claims") or []),
            }
            for o in llm_outputs
        ],
        "faiss_built": faiss_built,
        "faiss_error": faiss_error,
        "faiss_dir": faiss_dir,
        "artifacts_dir": str(run_dir),
    }


def ingest_path(root_path: str, progress: ProgressFn | None = None) -> dict:
    def notify(stage: str, p: float, msg: str | None = None) -> None:
        if progress:
            progress(stage, p, msg)

    notify("ingest:scan", 0.02, f"Scanning markdowns under {root_path}")
    md_files = find_mineru_markdowns(root_path)
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {root_path}")
    return ingest_markdowns(md_files, progress=progress)
