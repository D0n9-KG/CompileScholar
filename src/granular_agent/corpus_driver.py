"""Corpus driver: uniform paper data access for LogicKG (Side B2).

Dual-track block loading:
  1. local MINERU_BASE/{paper_id}/content_list.json (the 2355-paper granular
     corpus) — fast, no server, unchanged behavior.
  2. sci-evo-extract API fallback (paper_registry client) for papers that live
     in the sci-evo library (the 219-paper set with references + mineru
     artifacts). Same content_list format, same block parsing.

Instance persistence keyed by paper_id:
  save_instance / load_instance / has_instance persist InstanceHypergraph as
  JSON under instance_root (default .research_tmp/instances/{paper_id}.json).
  This is the "超图按 paper_id 存" store; step 6 promotes it into a sci-evo
  artifact. Avoids re-extracting papers already processed.

See .research_tmp/DECISION_corpus_driver.md for the design + verification plan.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .hypergraph_schema import InstanceHypergraph
from .paper_registry import (
    KIND_MINERU_CONTENT_LIST,
    PaperNotFound,
    PaperRegistryClient,
    PaperRegistryError,
)

# local granular corpus (unchanged from structure_mapper.load_paper_blocks)
MINERU_BASE = "C:/Users/D0n9/Desktop/science_evo/data/upstream/remote_mineru/mineru_2355/papers"

# local instance store root (kept under .research_tmp — not committed)
INSTANCE_ROOT = os.environ.get(
    "LOGICKG_INSTANCE_ROOT",
    "C:/Users/D0n9/Desktop/LogicKG/.research_tmp/instances",
)


class CorpusDriver:
    """Uniform access to paper blocks, references, and persisted instances."""

    def __init__(
        self,
        mineru_base: str = MINERU_BASE,
        instance_root: str = INSTANCE_ROOT,
        client: PaperRegistryClient | None = None,
    ):
        self.mineru_base = mineru_base
        self.instance_root = instance_root
        self.client = client or PaperRegistryClient()

    # -- blocks ------------------------------------------------------------

    def load_blocks(self, paper_id: str) -> list[dict]:
        """Load paper text as blocks {index,char_start,char_end,text}.

        Local-first: reads MINERU_BASE/{paper_id}/content_list.json. Falls back
        to the sci-evo API (mineru_content_list artifact) when the local file is
        absent. Both sources are parsed by the same rule as
        structure_mapper.load_paper_blocks (text-only, skip short/ref-like).
        """
        local_path = os.path.join(self.mineru_base, paper_id, "content_list.json")
        if os.path.isfile(local_path):
            with open(local_path, encoding="utf-8") as fh:
                content_list = json.load(fh)
            return _blocks_from_content_list(content_list)
        # API fallback
        try:
            content_list = self.client.get_content_list(paper_id)
        except PaperNotFound:
            return []
        except PaperRegistryError as exc:
            print(f"  [corpus] content_list fetch failed for {paper_id}: {exc}", flush=True)
            return []
        if not content_list:
            return []
        return _blocks_from_content_list(content_list)

    # -- references (step-3 structural signal source) ----------------------

    def load_references(
        self, paper_id: str, source: str = "openalex", fetch: bool = False
    ) -> list[dict]:
        """Outgoing references (papers this paper cites) from the sci-evo registry.

        fetch=True triggers a fresh OpenAlex lookup + upsert (needs the paper's
        DOI on the sci-evo side). Returns [] when the paper is unknown to the
        registry.
        """
        try:
            return self.client.get_references(paper_id, source=source, fetch=fetch)
        except PaperNotFound:
            return []
        except PaperRegistryError as exc:
            print(f"  [corpus] references fetch failed for {paper_id}: {exc}", flush=True)
            return []

    # -- instance persistence (keyed by paper_id) -------------------------

    def has_instance(self, paper_id: str) -> bool:
        return os.path.isfile(self._instance_path(paper_id))

    def save_instance(self, inst: InstanceHypergraph) -> str:
        os.makedirs(self.instance_root, exist_ok=True)
        path = self._instance_path(inst.paper_id)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(inst.to_dict(), fh, ensure_ascii=False)
        return path

    def load_instance(self, paper_id: str) -> InstanceHypergraph | None:
        path = self._instance_path(paper_id)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return InstanceHypergraph.from_dict(json.load(fh))

    def _instance_path(self, paper_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in paper_id)
        return os.path.join(self.instance_root, f"{safe}.json")


# ---------------------------------------------------------------------------
# block parsing (mirrors structure_mapper.load_paper_blocks, factored out so
# both local-file and API content_list share one rule)
# ---------------------------------------------------------------------------

def _blocks_from_content_list(content_list: list[dict]) -> list[dict]:
    """Parse mineru content_list items into blocks.

    Same rule as structure_mapper.load_paper_blocks: keep text items, skip short
    (<10 char) and reference-like ("[12] ...") blocks, rebuild a contiguous
    char range with a joining space.
    """
    blocks: list[dict] = []
    cursor = 0
    for it in content_list:
        if not isinstance(it, dict):
            continue
        if it.get("type") != "text" or not it.get("text"):
            continue
        t = it["text"].strip()
        if len(t) < 10:
            continue
        if t.startswith("[") and any(c.isdigit() for c in t[:5]):
            continue
        end = cursor + len(t)
        blocks.append({"index": len(blocks), "char_start": cursor, "char_end": end, "text": t})
        cursor = end + 1
    return blocks
