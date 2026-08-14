"""Runs DECIMER Segmentation (Mask R-CNN, trained specifically to locate
chemical structure depictions on journal pages) across every has_structures=1
figure in every paper's figure_tags.json, replacing the hand-built geometric
crop in structure_crop.py.

Must run in the `decimer_seg` conda env (Python 3.10) -- decimer-segmentation
pins an older TensorFlow/Mask-RCNN stack incompatible with the `decimer`
OCSR package's env. Writes one segment_manifest.json per paper:
{image_path: [segment_file_path, ...]} -- usually one segment, but a
multi-compound figure can legitimately produce several, and an image with
no detected structure produces an empty list (batch_decimer_extract.py
falls back to the original image in that case). Segment crops are written
alongside the originals under a "segments/" subfolder.

apply_mask (in the decimer_segmentation library) crops to the *exact* tight
bounding box of its (already mask-expanded) detection -- `image[y:y+h,
x:x+w]`, zero margin beyond whatever the mask itself covers. Checked
corpus-wide before trusting this was a real problem, not a one-off: 45.6%
of all segments had zero margin on the right edge specifically, and a
20-image visual sample against each one's Stage 1 source found roughly
70-80% of those were genuine content loss, not just tight-but-complete
crops -- severity ranged from a clipped subscript (NH2 -> NH) to a
substituent label disappearing entirely, leaving a dangling unlabeled bond
(OMe, MeO, F), to a case that reads as a completely different, wrong
molecule if OCSR'd as-is (a cyano/chlorine-substituted compound whose
segment showed plain unlabeled methyl stubs instead). None of this is
visible downstream -- a mis-cropped molecule still parses as valid
chemistry and can score high OCSR confidence, since the model is reading
exactly what it was shown.

Fixed by re-cropping ourselves: segment_chemical_structures(...,
return_bboxes=True) also returns each detection's bounding box in the
*original* image's coordinate space, so PADDED_BBOX_MARGIN_PX is added on
every side (clipped to the source image's own bounds -- this can only
recover real nearby content that Stage 1 already captured, never invent
new canvas) and the segment is re-cropped from the original image
ourselves rather than trusting the library's tight crop. Confirmed
directly on two real truncation cases before picking the margin size: 20px
only partially recovered a label ("MeO" missing its "M"), 45px fully
recovered both a missing "MeO" and a missing "Cl". Some cases still don't
fully recover even then, but not because the margin was too small --
found one where the expanded crop already exactly matched the full Stage
1 source image (nothing further to reveal) and the missing content was a
separate, Stage-1-level left-edge truncation, out of scope here.

A flat 45px margin on every detection is not free on a dense scope-table
page: two real compounds only ~33px apart (compound 48/49 on
copper_iron_2025's page3_fig2.png) would get expanded crops that overlap
by ~57px, bleeding a neighbor's stray fragment into each other's edges.
Confirmed visually this doesn't corrupt the core structure (it reads
cleanly; the bleed is peripheral text/ring fragments at the crop's edge),
but it's needless risk when it's avoidable. compute_safe_margin clamps
each of the 4 sides independently to at most half the gap to the nearest
*other* detection that overlaps this one on the perpendicular axis (a true
row/column neighbor, not a diagonal one a rectangular crop wouldn't
actually reach) -- so two adjacent detections' expanded crops can touch
but never overlap, while a detection with no nearby neighbor still gets
the full margin.

That neighbor list originally only ever contained other *detected
structures* -- real embedded PDF text (a neighboring compound's own label
in a dense grid, a page's own bold section header, reagent/step text next
to a reaction arrow) was invisible to it, so nothing stopped the margin
from padding straight into it. Confirmed on three real corpus cases before
trusting this was worth fixing (not a hypothetical): a scope-table grid
where compound 41a's own "MeO2C" label bled into neighboring compound
40a's crop; an SI NMR page where the bold "13C NMR 101 MHz, CDCl3" header
sitting directly above a structure got captured whole; and a mechanism
figure where multi-line reagent/step text next to a reaction arrow bled
into an adjacent structure's crop. All three produced real OCSR
hallucinations downstream. _nearby_text_pixel_boxes recovers each
figure's own page-space bounding box and pixel dimensions (already stored
by pdf_ingest.py in raw_extraction.json/SI_raw_extraction.json -- the same
numbers used to render the crop in the first place), maps a segment's
pixel bbox into that page's real point-space, and queries the PDF's own
embedded text (page.get_text, not OCR -- these are digitally-generated
PDFs, so the real text layer is authoritative and exact) for spans nearby.
Those spans are fed into compute_safe_margin as additional neighbors,
reusing its existing per-side clamp logic unchanged rather than writing a
second, parallel margin rule.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import fitz
from tqdm import tqdm
from decimer_segmentation import segment_chemical_structures

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"
PADDED_BBOX_MARGIN_PX = 45

_pdf_doc_cache = {}
_page_text_cache = {}


def has_structures(tags):
    tag = tags.get("has_structures")
    return bool(tag) and tag[0] == 1


def compute_safe_margin(index, bboxes, max_margin):
    """Returns (left, right, top, bottom) margins for bboxes[index], each
    independently clamped to at most half the gap to the nearest other
    detection that's a true row/column neighbor (overlaps this one on the
    perpendicular axis) in that direction -- see module docstring.

    A neighbor whose *raw* (pre-margin) box already overlaps this one --
    confirmed real on suzuki_nickel_2026/images/page6_fig1.png, where
    compound 40a's and 41a's own detections overlap by 10px before any
    padding -- fell through both `ox1 <= x0` (neighbor cleanly left) and
    `ox0 >= x1` (neighbor cleanly right) untouched, so neither condition
    ever fired and that side kept the full default margin instead of being
    clamped to zero -- the worst possible outcome, expanding straight
    through an already-touching neighbor (there, into 41a's own drawn
    "MeO2C" label) rather than away from it. Handled by falling back to
    the neighbor's *center* position when the boxes already overlap: don't
    add any further padding on the side facing that neighbor."""
    y0, x0, y1, x1 = bboxes[index]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = right = top = bottom = max_margin

    for j, (oy0, ox0, oy1, ox1) in enumerate(bboxes):
        if j == index:
            continue
        vertical_overlap = oy0 <= y1 and oy1 >= y0
        horizontal_overlap = ox0 <= x1 and ox1 >= x0
        ocx, ocy = (ox0 + ox1) / 2, (oy0 + oy1) / 2

        if vertical_overlap:
            if ox1 <= x0:
                left = min(left, (x0 - ox1) / 2)
            elif ox0 >= x1:
                right = min(right, (ox0 - x1) / 2)
            else:
                # neighbor's raw box already overlaps this one in x -- stop
                # expanding on whichever side faces it, rather than leaving
                # that side unclamped
                if ocx >= cx:
                    right = 0
                else:
                    left = 0
        if horizontal_overlap:
            if oy1 <= y0:
                top = min(top, (y0 - oy1) / 2)
            elif oy0 >= y1:
                bottom = min(bottom, (oy0 - y1) / 2)
            else:
                if ocy >= cy:
                    bottom = 0
                else:
                    top = 0

    return int(left), int(right), int(top), int(bottom)


def _load_page_geometry(paper_dir):
    """Returns {image_path: {source, page_number, page_bbox, pixel_w,
    pixel_h}} by reading raw_extraction.json/SI_raw_extraction.json --
    pdf_ingest.py already stores each figure's own page-space bbox (points)
    alongside its rendered pixel dimensions (the exact numbers used to
    render the crop), which is all that's needed to map a segment's pixel
    bbox back onto the real PDF page."""
    geometry = {}
    for fname, source in (("raw_extraction.json", "main"), ("SI_raw_extraction.json", "SI")):
        path = paper_dir / fname
        if not path.exists():
            continue
        data = json.load(open(path))
        for page in data["pages"]:
            for img in page["images"]:
                geometry[img["path"]] = {
                    "source": source,
                    "page_number": page["page_number"],
                    "page_bbox": img["bbox"],
                    "pixel_w": img["width"],
                    "pixel_h": img["height"],
                }
    return geometry


def _page_text_spans(paper_dir, source, page_number):
    """Real embedded PDF text spans (not OCR) for one page, as a list of
    (x0, y0, x1, y1) in the page's own point space. Cached per page since
    several figures can share one page."""
    key = (paper_dir, source, page_number)
    if key not in _page_text_cache:
        doc_key = (paper_dir, source)
        if doc_key not in _pdf_doc_cache:
            pdf_path = paper_dir / ("paper.pdf" if source == "main" else "SI.pdf")
            _pdf_doc_cache[doc_key] = fitz.open(pdf_path) if pdf_path.exists() else None
        doc = _pdf_doc_cache[doc_key]
        spans = []
        if doc is not None and 0 <= page_number - 1 < len(doc):
            page = doc[page_number - 1]
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block["lines"]:
                    for s in line["spans"]:
                        if s["text"].strip():
                            spans.append(tuple(s["bbox"]))
        _page_text_cache[key] = spans
    return _page_text_cache[key]


def _nearby_text_neighbor_boxes(paper_dir, image_path, geometry, img_w, img_h):
    """Real PDF text spans near this figure, converted into the figure
    crop's own pixel space and (y0, x0, y1, x1) order -- the same shape
    compute_safe_margin already expects for a detected-structure neighbor
    -- so they can be appended to the neighbor list and clamped against
    with zero changes to that function. See module docstring for the three
    confirmed real cases this fixes."""
    geo = geometry.get(image_path)
    if geo is None:
        return []
    px0, py0, px1, py1 = geo["page_bbox"]
    if px1 <= px0 or py1 <= py0:
        return []
    scale_x = geo["pixel_w"] / (px1 - px0)
    scale_y = geo["pixel_h"] / (py1 - py0)

    boxes = []
    for tx0, ty0, tx1, ty1 in _page_text_spans(paper_dir, geo["source"], geo["page_number"]):
        cx0 = (tx0 - px0) * scale_x
        cy0 = (ty0 - py0) * scale_y
        cx1 = (tx1 - px0) * scale_x
        cy1 = (ty1 - py0) * scale_y
        if cx1 < 0 or cx0 > img_w or cy1 < 0 or cy0 > img_h:
            continue  # not actually within this figure's own crop at all
        boxes.append((cy0, cx0, cy1, cx1))
    return boxes


def run_paper(paper_dir):
    tags_path = paper_dir / "figure_tags.json"
    if not tags_path.exists():
        return {}

    with open(tags_path) as f:
        all_tags = json.load(f)

    geometry = _load_page_geometry(paper_dir)

    targets = [path for path, tags in all_tags.items() if has_structures(tags)]
    segments_dir = paper_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    manifest = {}
    pbar = tqdm(targets, desc=paper_dir.name, unit="fig", mininterval=1.0)
    for image_path in pbar:
        full_path = paper_dir / image_path
        if not full_path.exists():
            tqdm.write(f"  MISSING {image_path}")
            continue
        pbar.set_postfix_str(image_path[-40:])
        start = time.time()
        img = cv2.imread(str(full_path))
        img_h, img_w = img.shape[:2]
        _, bboxes = segment_chemical_structures(img, expand=True, return_bboxes=True)
        elapsed = time.time() - start

        text_neighbors = _nearby_text_neighbor_boxes(paper_dir, image_path, geometry, img_w, img_h)
        neighbor_boxes = list(bboxes) + text_neighbors

        stem = Path(image_path).stem.replace("/", "_")
        seg_entries = []
        for seg_idx, (y0, x0, y1, x1) in enumerate(bboxes):
            m_left, m_right, m_top, m_bottom = compute_safe_margin(seg_idx, neighbor_boxes, PADDED_BBOX_MARGIN_PX)
            py0 = max(0, y0 - m_top)
            px0 = max(0, x0 - m_left)
            py1 = min(img_h, y1 + m_bottom)
            px1 = min(img_w, x1 + m_right)
            seg = img[py0:py1, px0:px1]
            if seg.shape[0] == 0 or seg.shape[1] == 0:
                continue
            seg_filename = f"{stem}_seg{seg_idx}.png"
            cv2.imwrite(str(segments_dir / seg_filename), seg)
            # bbox is in the *parent* image's own pixel coordinate space (the
            # actual crop bounds used above, post-margin/clamp) -- Stage 5
            # needs this to ground a vision-LLM call on where each segment's
            # structure sits within the original figure, not just that it
            # exists (see reaction_link.py).
            seg_entries.append({"path": f"segments/{seg_filename}", "bbox": [px0, py0, px1, py1]})

        manifest[image_path] = seg_entries
        tqdm.write(f"  {image_path} -> {len(seg_entries)} segment(s) ({elapsed:.1f}s)")

    return manifest


def main():
    paper_dirs = sorted(p for p in PAPERS_DIR.iterdir() if p.is_dir())
    for paper_dir in paper_dirs:
        manifest = run_paper(paper_dir)
        if not manifest:
            continue
        out_path = paper_dir / "segment_manifest.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        total_segments = sum(len(v) for v in manifest.values())
        zero_segment = sum(1 for v in manifest.values() if not v)
        print(
            f"{paper_dir.name}: {len(manifest)} figures, {total_segments} segments total, "
            f"{zero_segment} with no detected structure -> {out_path}"
        )


if __name__ == "__main__":
    main()
