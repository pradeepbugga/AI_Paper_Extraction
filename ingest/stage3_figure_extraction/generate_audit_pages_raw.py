"""Same audit page as generate_audit_pages.py, but the swatch image for each
card is DECIMER's own RAW mask crop -- the tight-image mask bbox at that
segment's seg_idx, straight from get_expanded_masks, before extend_bbox's
ink-walk, before two-pass's wide-crop recovery, before any of this
pipeline's own fix-up logic runs. Lets you judge DECIMER's own segmentation
quality in isolation from everything built on top of it (same idea as the
page383_raster0 spot-check: is the caption/table already excluded by
DECIMER itself, or only pulled in by our own extension code).

Card status/fields (compound_id, yield, flag, note, source) still come from
the real, current reaction_links.json -- only the pictured crop changes.

Must run in the decimer_seg env (needs get_expanded_masks).
Usage: python3 ingest/stage3_figure_extraction/generate_audit_pages_raw.py <output_dir_name>
e.g.:  python3 ingest/stage3_figure_extraction/generate_audit_pages_raw.py audit_pages_v4_raw
"""
import base64
import html
import json
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decimer_segmentation import get_expanded_masks
from batch_segment import _mask_bboxes
from generate_audit_pages import STYLE, SCRIPT, _status, PAPERS_DIR, ROOT

_SEG_IDX_RE = re.compile(r"_seg(\d+)\.png$")


class RawMaskCache:
    """One get_expanded_masks call per unique parent (tight) image, reused
    across every segment that came from it."""

    def __init__(self, paper_dir):
        self.paper_dir = paper_dir
        self._cache = {}  # image_path -> {seg_idx: bbox}
        self._img_cache = {}  # image_path -> cv2 image

    def _ensure(self, image_path):
        if image_path in self._cache:
            return
        full_path = self.paper_dir / image_path
        img = cv2.imread(str(full_path))
        if img is None:
            self._cache[image_path] = {}
            self._img_cache[image_path] = None
            return
        masks = get_expanded_masks(img)
        self._cache[image_path] = dict(_mask_bboxes(masks))
        self._img_cache[image_path] = img

    def raw_crop_b64(self, image_path, segment_path):
        self._ensure(image_path)
        img = self._img_cache.get(image_path)
        bboxes = self._cache.get(image_path, {})
        if img is None:
            return None
        m = _SEG_IDX_RE.search(segment_path)
        if not m:
            return None
        seg_idx = int(m.group(1))
        bbox = bboxes.get(seg_idx)
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        crop = img[y0:y1, x0:x1]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return None
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")


def _card(paper_dir, image_path, rec, cache):
    status = _status(rec)
    compound_id = rec.get("compound_id")
    yield_percent = rec.get("yield_percent")
    panel_label = rec.get("panel_label")
    segment_path = rec["segment_path"]
    note = rec.get("note")
    flag = rec.get("flag")
    source = rec.get("source", "")

    b64 = cache.raw_crop_b64(image_path, segment_path)
    if b64 is None:
        return ""

    search_parts = []
    if compound_id:
        search_parts.append(compound_id)
    search_parts.append(segment_path)
    search_parts.append(image_path)
    if note:
        search_parts.append(note)
    if flag:
        search_parts.append(flag)
    data_search = html.escape(" ".join(search_parts).lower(), quote=True)

    cid_html = (
        f'<span class="cid">{html.escape(compound_id)}</span>'
        if compound_id else '<span class="cid empty">null</span>'
    )
    yield_bits = []
    if yield_percent:
        yield_bits.append(f"yield {html.escape(str(yield_percent))}")
    if panel_label:
        yield_bits.append(f"({html.escape(str(panel_label))})")
    yield_html = f'<div class="yield">{" ".join(yield_bits)}</div>' if yield_bits else ""

    note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
    flag_html = f'<div class="flag">⚑ {html.escape(flag)}</div>' if flag else ""

    return f"""<article class="card" data-status="{status}" data-search="{data_search}" tabindex="0">
  <div class="swatch">
    <span class="pill {status}">{status}</span>
    <img src="data:image/png;base64,{b64}" alt="">
  </div>
  <div class="body">
    <div class="idrow">{cid_html}{yield_html}</div>
    <div class="path">{html.escape(segment_path)} <span style="opacity:.6">(raw mask)</span></div>
    {note_html}
    {flag_html}
    <div class="source">{html.escape(source)}</div>
  </div>
</article>
"""


def generate_paper(paper_dir, out_dir):
    links_path = paper_dir / "reaction_links.json"
    if not links_path.exists():
        return None
    links = json.loads(links_path.read_text())
    cache = RawMaskCache(paper_dir)

    cards_html = []
    total = has_id = no_id = flagged = 0
    for image_path, records in links.items():
        for rec in records:
            card = _card(paper_dir, image_path, rec, cache)
            if not card:
                continue
            cards_html.append(card)
            total += 1
            status = _status(rec)
            if status == "warn":
                flagged += 1
            elif status == "ok":
                has_id += 1
            else:
                no_id += 1

    name = paper_dir.name
    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Audit (raw DECIMER masks) — {html.escape(name)}</title>
{STYLE}
</head>
<body>
<header class="top">
  <h1>{html.escape(name)} <span style="opacity:.6">— raw DECIMER masks</span></h1>
  <p class="subtitle">Pictured crop is DECIMER's own raw mask bbox (pre extend_bbox, pre two-pass recovery) — judges segmentation quality in isolation. ID/yield/flag fields still come from the current pipeline.</p>
  <div class="stats">
<div class="stat"><span class="n">{total}</span><span class="label">total</span></div>
<div class="stat"><span class="n">{has_id}</span><span class="label">has id</span></div>
<div class="stat"><span class="n">{no_id}</span><span class="label">no id</span></div>
<div class="stat"><span class="n">{flagged}</span><span class="label">flagged</span></div>
</div>
  <div class="controls">
    <button class="filter-btn" data-filter="all" aria-pressed="true">All</button>
    <button class="filter-btn" data-filter="ok" aria-pressed="false">Has ID</button>
    <button class="filter-btn" data-filter="null" aria-pressed="false">No ID</button>
    <button class="filter-btn" data-filter="warn" aria-pressed="false">Flagged</button>
    <input id="search" type="text" placeholder="search id / path / note…">
    <span id="count"></span>
  </div>
</header>
<main>
{"".join(cards_html)}</main>
<div id="lightbox"><img src="" alt=""></div>
{SCRIPT}
</body>
</html>
"""
    out_path = out_dir / f"audit_{name}.html"
    out_path.write_text(page)
    return {"total": total, "has_id": has_id, "no_id": no_id, "flagged": flagged, "path": out_path}


def main():
    out_dirname = sys.argv[1] if len(sys.argv) > 1 else "audit_pages_v4_raw"
    out_dir = ROOT / out_dirname
    out_dir.mkdir(exist_ok=True)

    for paper_dir in sorted(p for p in PAPERS_DIR.iterdir() if p.is_dir()):
        result = generate_paper(paper_dir, out_dir)
        if result is None:
            print(f"{paper_dir.name}: no reaction_links.json, skipped")
            continue
        print(
            f"{paper_dir.name}: {result['total']} total, {result['has_id']} has id, "
            f"{result['no_id']} no id, {result['flagged']} flagged -> {result['path']}"
        )


if __name__ == "__main__":
    main()
