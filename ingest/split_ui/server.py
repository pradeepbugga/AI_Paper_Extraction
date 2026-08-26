"""Local review tool for splitting `too_many_fragments=True` segments --
crops where Stage 1 segmentation under-merged multiple real compounds
into one image. Not every flagged segment is a real merge (confirmed via
manual review, 2026-08-25: NMR-spectrum junk, metal-complex decoder
failures, and (CH2)N chain-notation connection failures all also trip
this flag) -- a human decides per-image whether/how to split it, rather
than trusting the flag as a clean "needs splitting" signal.

Split interaction is a Windows-Snipping-Tool-style freeform lasso, not a
rectangle: a rectangle can't cleanly separate two structures whose ink
overlaps or sits close together, which is exactly the case this tool
exists for. Each lasso becomes its own new segment: cropped to the
lasso's bounding box, with every pixel inside that box but outside the
drawn polygon painted white -- isolating just the one structure's ink,
discarding whatever of a neighboring structure's ink also fell in the
box.

Two save actions, not one -- "extract & continue" vs "extract & finish"
-- because the natural human workflow (confirmed directly: every one of
the first 9 real splits done with the single-save version only ever drew
ONE shape per save, expecting the rest of the crop to come back for
another pass) is to peel structures off one at a time, not draw every
shape in a 3-structure crop before saving once. "Continue" creates the
drawn shape(s) as new segments AND a "remainder" segment (the source
image with those shapes' own ink painted white, everything else
untouched) that goes back into the review queue immediately; "finish"
creates only the drawn shape(s), no remainder, same as the original
single-save behavior -- for when this save genuinely captures everything
left worth keeping.

Writes real files: new split-segment PNGs alongside the original in
`segments/`, and updates `segment_manifest.json` (replaces the single
oversized entry with one entry per drawn shape, plus one remainder entry
on "continue", each carrying its own bbox translated into the PARENT
image's coordinate space -- same grounding contract batch_segment.py
already established, since Stage 5's Set-of-Mark relies on it). Does NOT
delete the original oversized crop file (non-destructive) and does NOT
touch `decimer_results.json` -- new segments (splits or remainders) have
no OCSR result yet; re-running OCSR on them is a separate, later step
(needs the pod's GPU). Because a remainder segment therefore has no
too_many_fragments flag of its own to be found by, `pending_queue.json`
tracks it directly so it still shows up in /api/queue.

Run: uvicorn server:app --reload --port 8420 (from this directory, in the
`paper_extraction` conda env -- has PIL/numpy already; fastapi/uvicorn
were added on top).
"""

import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel

PAPERS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "papers"
REVIEW_LOG_PATH = Path(__file__).resolve().parent / "review_log.json"
PENDING_QUEUE_PATH = Path(__file__).resolve().parent / "pending_queue.json"

app = FastAPI()


def _load_json(path):
    if path.exists():
        return json.load(open(path))
    return {}


def _save_json(path, data):
    json.dump(data, open(path, "w"), indent=2)


def _key(paper, segment_path):
    return f"{paper}::{segment_path}"


def _already_split_paths(log, paper):
    """segment_paths this paper's split tool has already produced (as a
    prior action's new_segments) -- these are deliberate, completed split
    decisions, not fresh candidates. Needed because too_many_fragments can
    be TRUE on a split-tool output too (e.g. a re-run of the fragment-count
    fix re-flagged some), and the review_log only ever excludes the
    ORIGINAL segment_path a decision was made about, not the children that
    decision produced -- without this, a segment you already split could
    come right back into the queue under its own new filename. Confirmed
    real, not hypothetical: page26_raster4_seg0_split0.png and others,
    2026-08-26."""
    produced = set()
    for k, v in log.items():
        if not k.startswith(f"{paper}::") or v.get("action") != "split":
            continue
        produced.update(v.get("new_segments", []))
    return produced


def _build_queue():
    """Every too_many_fragments=True segment across the corpus, PLUS every
    still-unresolved remainder from a previous "extract & continue" save,
    minus anything already reviewed (split or skipped) in review_log.json,
    minus anything that IS the output of an already-completed split."""
    log = _load_json(REVIEW_LOG_PATH)
    pending = _load_json(PENDING_QUEUE_PATH)
    items = []
    for paper_dir in sorted(PAPERS_DIR.iterdir()):
        results_path = paper_dir / "decimer_results.json"
        if not results_path.exists():
            continue
        results = json.load(open(results_path))
        already_split = _already_split_paths(log, paper_dir.name)
        for parent_image, entries in results.items():
            for e in entries:
                if not e.get("too_many_fragments"):
                    continue
                if e["segment_path"] in already_split:
                    continue
                k = _key(paper_dir.name, e["segment_path"])
                if k in log:
                    continue
                items.append({
                    "paper": paper_dir.name,
                    "parent_image": parent_image,
                    "segment_path": e["segment_path"],
                    "smiles": e["smiles"],
                    "rdkit_valid": e["rdkit_valid"],
                    "mean_confidence": e["mean_confidence"],
                    "n_fragments": e["smiles"].count(".") + 1,
                    "is_remainder": False,
                })

    for k, p in pending.items():
        if k in log:
            continue
        items.append({
            "paper": p["paper"],
            "parent_image": p["parent_image"],
            "segment_path": p["segment_path"],
            "smiles": "(remainder from a previous split -- not yet OCSR'd)",
            "rdkit_valid": None,
            "mean_confidence": None,
            "n_fragments": None,
            "is_remainder": True,
        })

    items.sort(key=lambda x: (x["is_remainder"], -(x["n_fragments"] or 0)))
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


class Point(BaseModel):
    x: float
    y: float


class EraseStroke(BaseModel):
    radius: float
    points: list[Point]


class SplitRequest(BaseModel):
    paper: str
    segment_path: str
    polygons: list[list[Point]]  # coords in the segment crop's own pixel space
    erase_strokes: list[EraseStroke] = []  # applied to the source image before cropping
    continue_remainder: bool = False  # True = "extract & continue", False = "extract & finish"


def _manifest_path(paper):
    return PAPERS_DIR / paper / "segment_manifest.json"


def _unique_split_path(segments_dir, stem, suffix, i):
    """f"{stem}_{suffix}{i}.png", bumping i until the filename doesn't
    already exist. Needed because the SAME original stem can legitimately
    get split more than once (peel one structure off, "continue", peel
    another off the remainder later; or a segment gets re-split after a
    manual recovery) -- without this, a second split from the same stem
    silently overwrites the first one's actual file on disk even though
    the manifest keeps two separate entries pointing at (now identical,
    wrong) content. Confirmed as a real corruption, not hypothetical: hit
    on copper_iron_2025/page3_fig1_seg45.png, 2026-08-25."""
    while (segments_dir / f"{stem}_{suffix}{i}.png").exists():
        i += 1
    return i, f"{stem}_{suffix}{i}.png"


def _find_manifest_entry(manifest, segment_path):
    """Returns (parent_image_key, entry_index, entry) for the manifest
    entry whose "path" matches segment_path, searching every parent
    image's segment list."""
    for parent_image, entries in manifest.items():
        for i, e in enumerate(entries):
            if e["path"] == segment_path:
                return parent_image, i, e
    return None, None, None


@app.post("/api/split")
def split_segment(req: SplitRequest):
    if not req.polygons:
        raise HTTPException(400, "no shapes drawn")

    paper_dir = PAPERS_DIR / req.paper
    src_path = paper_dir / req.segment_path
    if not src_path.exists():
        raise HTTPException(404, f"segment not found: {req.segment_path}")

    manifest_path = _manifest_path(req.paper)
    manifest = json.load(open(manifest_path))
    parent_image, entry_idx, entry = _find_manifest_entry(manifest, req.segment_path)
    if entry is None:
        raise HTTPException(404, "segment not in segment_manifest.json")

    orig_img = Image.open(src_path).convert("RGB")
    img_w, img_h = orig_img.size
    orig_bbox = entry["bbox"]  # [x0, y0, x1, y1] in the PARENT image's own coordinate space
    parent_x0, parent_y0 = (orig_bbox[0], orig_bbox[1]) if orig_bbox else (0, 0)

    # Erase strokes paint white on the working image BEFORE any polygon
    # crop -- for junk (a stray annotation mark, bled-in neighbor ink)
    # that sits INSIDE a kept structure's own lasso, where the lasso
    # boundary itself can't exclude it. Applied as a sequence of filled
    # circles along each stroke's path, same brush-stroke approximation
    # any paint-style tool uses -- dense enough since the frontend appends
    # a point on every mousemove.
    if req.erase_strokes:
        draw = ImageDraw.Draw(orig_img)
        for stroke in req.erase_strokes:
            r = max(0.5, stroke.radius)
            for p in stroke.points:
                x, y = max(0, min(img_w, p.x)), max(0, min(img_h, p.y))
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))

    # remainder starts as the (erase-applied) source image; each drawn
    # polygon's own ink gets painted white out of it below, so whatever's
    # left is "the rest of the crop, minus what was just extracted."
    remainder_img = orig_img.copy()
    remainder_draw = ImageDraw.Draw(remainder_img)

    stem = Path(req.segment_path).stem  # e.g. "segments/page78_raster0_seg0" -> "page78_raster0_seg0"
    new_entries = []
    new_paths = []
    for i, polygon in enumerate(req.polygons):
        pts = [(max(0, min(img_w, p.x)), max(0, min(img_h, p.y))) for p in polygon]
        if len(pts) < 3:
            continue  # not a real shape

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bx0, by0 = int(max(0, min(xs))), int(max(0, min(ys)))
        bx1, by1 = int(min(img_w, max(xs) + 1)), int(min(img_h, max(ys) + 1))
        if bx1 <= bx0 or by1 <= by0:
            continue

        # Mask: white everywhere inside the bbox but outside the polygon,
        # so only this shape's own ink survives -- discards whatever of a
        # neighboring structure's ink also fell inside the crop box.
        mask = Image.new("L", (bx1 - bx0, by1 - by0), 0)
        local_pts = [(x - bx0, y - by0) for x, y in pts]
        ImageDraw.Draw(mask).polygon(local_pts, fill=255)
        mask_arr = np.array(mask)

        crop = orig_img.crop((bx0, by0, bx1, by1))
        crop_arr = np.array(crop)
        crop_arr[mask_arr == 0] = 255  # paint outside-polygon pixels white
        out_img = Image.fromarray(crop_arr)

        _, new_filename = _unique_split_path(paper_dir / "segments", stem, "split", i)
        out_path = paper_dir / "segments" / new_filename
        out_img.save(out_path)

        new_entries.append({
            "path": f"segments/{new_filename}",
            "bbox": [parent_x0 + bx0, parent_y0 + by0, parent_x0 + bx1, parent_y0 + by1],
            "truncated_sides": [],
        })
        new_paths.append(f"segments/{new_filename}")

        # Remove this shape's own ink from the remainder too, so a
        # structure already extracted doesn't visually duplicate in it.
        remainder_draw.polygon(pts, fill=(255, 255, 255))

    if not new_entries:
        raise HTTPException(400, "no valid shapes (need >=3 points each)")

    manifest_entries = list(new_entries)
    remainder_key = None
    if req.continue_remainder:
        _, remainder_filename = _unique_split_path(paper_dir / "segments", stem, "remainder", 0)
        remainder_path = paper_dir / "segments" / remainder_filename
        remainder_img.save(remainder_path)
        remainder_rel_path = f"segments/{remainder_filename}"
        manifest_entries.append({
            "path": remainder_rel_path,
            "bbox": list(orig_bbox) if orig_bbox else [parent_x0, parent_y0, parent_x0 + img_w, parent_y0 + img_h],
            "truncated_sides": [],
        })
        remainder_key = _key(req.paper, remainder_rel_path)
        pending = _load_json(PENDING_QUEUE_PATH)
        pending[remainder_key] = {
            "paper": req.paper,
            "parent_image": parent_image,
            "segment_path": remainder_rel_path,
        }
        _save_json(PENDING_QUEUE_PATH, pending)

    manifest[parent_image][entry_idx:entry_idx + 1] = manifest_entries
    json.dump(manifest, open(manifest_path, "w"), indent=2)

    log = _load_json(REVIEW_LOG_PATH)
    log[_key(req.paper, req.segment_path)] = {
        "action": "split",
        "new_segments": new_paths,
        "remainder": remainder_key,
        "erased": bool(req.erase_strokes),
        "timestamp": time.time(),
    }
    # If we just split a PENDING remainder itself, resolve it out of the
    # pending queue too (it's already excluded via review_log by the same
    # key, but keep pending_queue.json tidy rather than accumulating
    # resolved entries forever).
    pending = _load_json(PENDING_QUEUE_PATH)
    pending.pop(_key(req.paper, req.segment_path), None)
    _save_json(PENDING_QUEUE_PATH, pending)
    _save_json(REVIEW_LOG_PATH, log)

    return {"ok": True, "new_segments": new_paths, "remainder": remainder_key}


class SkipRequest(BaseModel):
    paper: str
    segment_path: str


@app.post("/api/skip")
def skip_segment(req: SkipRequest):
    log = _load_json(REVIEW_LOG_PATH)
    log[_key(req.paper, req.segment_path)] = {"action": "skip", "timestamp": time.time()}
    _save_json(REVIEW_LOG_PATH, log)

    # A skipped item can itself be a pending remainder from an earlier
    # "extract & continue" -- resolve it out of pending_queue.json too,
    # same as /api/split already does, so the file doesn't accumulate
    # entries review_log has already made redundant.
    pending = _load_json(PENDING_QUEUE_PATH)
    if _key(req.paper, req.segment_path) in pending:
        del pending[_key(req.paper, req.segment_path)]
        _save_json(PENDING_QUEUE_PATH, pending)

    return {"ok": True}


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent / "static", html=True), name="static")
