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
"""

import json
import sys
import time
from pathlib import Path

import cv2
from tqdm import tqdm
from decimer_segmentation import segment_chemical_structures

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"
PADDED_BBOX_MARGIN_PX = 45


def has_structures(tags):
    tag = tags.get("has_structures")
    return bool(tag) and tag[0] == 1


def compute_safe_margin(index, bboxes, max_margin):
    """Returns (left, right, top, bottom) margins for bboxes[index], each
    independently clamped to at most half the gap to the nearest other
    detection that's a true row/column neighbor (overlaps this one on the
    perpendicular axis) in that direction -- see module docstring."""
    y0, x0, y1, x1 = bboxes[index]
    left = right = top = bottom = max_margin

    for j, (oy0, ox0, oy1, ox1) in enumerate(bboxes):
        if j == index:
            continue
        vertical_overlap = oy0 <= y1 and oy1 >= y0
        horizontal_overlap = ox0 <= x1 and ox1 >= x0

        if vertical_overlap:
            if ox1 <= x0:
                left = min(left, (x0 - ox1) / 2)
            if ox0 >= x1:
                right = min(right, (ox0 - x1) / 2)
        if horizontal_overlap:
            if oy1 <= y0:
                top = min(top, (y0 - oy1) / 2)
            if oy0 >= y1:
                bottom = min(bottom, (oy0 - y1) / 2)

    return int(left), int(right), int(top), int(bottom)


def run_paper(paper_dir):
    tags_path = paper_dir / "figure_tags.json"
    if not tags_path.exists():
        return {}

    with open(tags_path) as f:
        all_tags = json.load(f)

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

        stem = Path(image_path).stem.replace("/", "_")
        seg_entries = []
        for seg_idx, (y0, x0, y1, x1) in enumerate(bboxes):
            m_left, m_right, m_top, m_bottom = compute_safe_margin(seg_idx, bboxes, PADDED_BBOX_MARGIN_PX)
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
        tqdm.write(f"  {image_path} -> {len(seg_paths)} segment(s) ({elapsed:.1f}s)")

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
