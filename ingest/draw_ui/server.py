"""Local review tool for redrawing segments MolScribe genuinely got wrong
-- RDKit-invalid or wildcard-present (a real unresolved substituent, not
just a Markush template placeholder -- those are correctly out of scope
and just get skipped here, same as any other not-fixable case). A human
draws the real structure in an embedded Ketcher editor and the corrected
SMILES gets written straight into that segment's existing
decimer_results.json entry.

Verified end-to-end before building this (2026-08-25): Ketcher's own
SMILES export silently drops dative bonds (confirmed directly via
Indigo's Python API -- every internal bond-order value tested still
exported as a plain, unmarked bond), which would have been a real problem
for this corpus's recurring NHC-metal complexes. The fix is to never let
Ketcher/Indigo generate the SMILES at all: export a Molfile instead
(Indigo preserves the dative bond as MDL bond type 9 there), and let
RDKit -- which already grounds every rdkit_valid check elsewhere in this
pipeline -- do the actual Molfile -> SMILES conversion. Confirmed RDKit
reads MDL bond type 9 as a genuine DATIVE bond and writes proper `->`
SMILES for it. See ingest/draw_ui/static/vendor/ketcher/ (a from-source
Vite build of ketcher-react + ketcher-standalone -- the release zips are
npm library bundles, not a servable static app, so there was no
zero-build vendoring option) for the drawing surface itself.

Run: uvicorn server:app --port 8430 (from this directory, in the
`paper_extraction` conda env -- has PIL/numpy/rdkit already;
fastapi/uvicorn were added for split_ui and are already present).
"""

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PAPERS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "papers"
REVIEW_LOG_PATH = Path(__file__).resolve().parent / "review_log.json"

app = FastAPI()


def _load_json(path):
    if path.exists():
        return json.load(open(path))
    return {}


def _save_json(path, data):
    json.dump(data, open(path, "w"), indent=2)


def _key(paper, segment_path):
    return f"{paper}::{segment_path}"


def _build_queue():
    """Every segment that's RDKit-invalid or still carries a wildcard
    (MolScribe's '*' for an unresolved substituent), not yet reviewed.

    Cross-checks segment_manifest.json and skips any segment_path no
    longer present there -- decimer_results.json isn't touched when a
    segment gets superseded by the fragment-splitting tool (a real
    too_many_fragments merge, once split, leaves its own now-orphaned
    entry behind since only the manifest gets updated), so without this
    check the draw queue would include segments that aren't part of the
    active pipeline anymore at all. Confirmed real, not hypothetical: 8
    such orphaned entries found across 4 papers, 2026-08-25."""
    log = _load_json(REVIEW_LOG_PATH)
    items = []
    for paper_dir in sorted(PAPERS_DIR.iterdir()):
        results_path = paper_dir / "decimer_results.json"
        manifest_path = paper_dir / "segment_manifest.json"
        if not results_path.exists() or not manifest_path.exists():
            continue
        results = json.load(open(results_path))
        manifest = json.load(open(manifest_path))
        current_paths = {e["path"] for entries in manifest.values() for e in entries}
        for parent_image, entries in results.items():
            for e in entries:
                if e["segment_path"] not in current_paths:
                    continue
                smiles = e.get("smiles", "")
                needs_review = (not e.get("rdkit_valid")) or ("*" in smiles)
                if not needs_review:
                    continue
                k = _key(paper_dir.name, e["segment_path"])
                if k in log:
                    continue
                items.append({
                    "paper": paper_dir.name,
                    "parent_image": parent_image,
                    "segment_path": e["segment_path"],
                    "smiles": smiles,
                    "rdkit_valid": e["rdkit_valid"],
                    "mean_confidence": e["mean_confidence"],
                    "has_wildcard": "*" in smiles,
                })
    # Invalid-and-wildcard-and-unparseable first -- the clearest "the
    # model has no idea" cases -- then invalid-only, then wildcard-only
    # (often just one generic substituent in an otherwise-correct read).
    items.sort(key=lambda x: (not (not x["rdkit_valid"] and x["has_wildcard"]), x["rdkit_valid"]))
    return items


@app.get("/api/queue")
def get_queue():
    items = _build_queue()
    return {"total_remaining": len(items), "items": items}


@app.get("/api/image/{paper}/{segment_path:path}")
def get_image(paper: str, segment_path: str):
    path = PAPERS_DIR / paper / segment_path
    if not path.exists() or PAPERS_DIR not in path.resolve().parents:
        raise HTTPException(404, "not found")
    return FileResponse(path)


class SaveRequest(BaseModel):
    paper: str
    segment_path: str
    molfile: str


def _find_result_entry(results, segment_path):
    for parent_image, entries in results.items():
        for e in entries:
            if e["segment_path"] == segment_path:
                return e
    return None


@app.post("/api/save")
def save_drawing(req: SaveRequest):
    mol = Chem.MolFromMolBlock(req.molfile)
    if mol is None:
        raise HTTPException(400, "RDKit could not parse this structure -- check for open valences or a disconnected fragment")

    smiles = Chem.MolToSmiles(mol)
    # Re-parse the canonical SMILES itself, same as decimer_extract.py's
    # own rdkit_valid check -- MolToSmiles on an already-valid Mol should
    # always round-trip, but this is the one true validity gate the rest
    # of the pipeline relies on, so it's worth checking directly rather
    # than assuming MolFromMolBlock succeeding is sufficient.
    valid = Chem.MolFromSmiles(smiles) is not None

    results_path = PAPERS_DIR / req.paper / "decimer_results.json"
    results = json.load(open(results_path))
    entry = _find_result_entry(results, req.segment_path)
    if entry is None:
        raise HTTPException(404, "segment not found in decimer_results.json")

    entry["smiles"] = smiles
    entry["rdkit_valid"] = valid
    entry["need_human_review"] = False
    entry["source"] = "human_drawn"
    _save_json(results_path, results)

    log = _load_json(REVIEW_LOG_PATH)
    log[_key(req.paper, req.segment_path)] = {
        "action": "drawn", "smiles": smiles, "timestamp": time.time(),
    }
    _save_json(REVIEW_LOG_PATH, log)

    return {"ok": True, "smiles": smiles, "valid": valid}


class SkipRequest(BaseModel):
    paper: str
    segment_path: str


@app.post("/api/skip")
def skip_segment(req: SkipRequest):
    log = _load_json(REVIEW_LOG_PATH)
    log[_key(req.paper, req.segment_path)] = {"action": "skip", "timestamp": time.time()}
    _save_json(REVIEW_LOG_PATH, log)
    return {"ok": True}


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent / "static", html=True), name="static")
