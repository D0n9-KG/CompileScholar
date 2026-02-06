from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.citations.aggregate import build_reference_and_cite_records
from app.crossref.client import CrossrefClient
from app.graph.neo4j_client import Neo4jClient
from app.graph.neo4j_client import paper_id_for_md_path
from app.ingest.figures import extract_figures_from_markdown
from app.ingest.paper_meta import load_canonical_meta
from app.ingest.parse_md import find_mineru_markdowns, parse_mineru_markdown
from app.llm.citation_purpose import classify_citation_purposes_batch
from app.llm.logic_claims_v2 import add_evidence_and_targets, extract_logic_and_claims_v2
from app.schema_store import load_active
from app.settings import settings
from app.vector.faiss_store import build_faiss_for_chunks


ProgressFn = Callable[[str, float, str | None], None]


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
        out.write_text(
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
            add_evidence_and_targets(doc, schema=schema, claims=logic_claims["claims"], cite_rec=rec)
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
            batch_out = classify_citation_purposes_batch(citing_title=citing_title, cites=batch_in, prompt_overrides=schema.get("prompts"))
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
        "neo4j_written": neo4j_written,
        "neo4j_error": neo4j_error,
        "llm_built": llm_built,
        "llm_error": llm_error,
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
