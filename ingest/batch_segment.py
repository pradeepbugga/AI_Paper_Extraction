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

Crops to the *exact* tight bounding box of each detection's mask (via
get_expanded_masks -- `expand=True`'s dilation still pulls in nearby
disconnected ink, e.g. a separated substituent label, before the bbox is
taken), computed ourselves from the mask array rather than trusting
apply_mask's own crop, since apply_mask also whitens every pixel outside
the mask's own irregular outline and that's not needed here -- the raw
rectangular crop from the original image is enough once the bbox is tight.

An earlier version of this script added a flat pixel margin (with a
same-figure-neighbor clamp) around this tight bbox, to recover a small
number of confirmed cases where the tight mask bbox clipped a real
substituent label. That margin was reverted: visual review across several
dense figures showed the tight mask-derived bbox is consistently clean --
it naturally avoids a neighboring compound's own label or nearby
reaction-scheme text, which a padded rectangle (a shape with no relation
to where the actual ink is) could not reliably avoid clamping around in
every geometry. A tight crop occasionally clipping a label is a visible,
correctable-on-review failure; a padded crop silently pulling in a
neighbor's label is not. If truncation turns out to be a real recurring
problem on the full corpus re-run, revisit -- but don't reintroduce padding
speculatively.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from decimer_segmentation import get_expanded_masks

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"


def has_structures(tags):
    tag = tags.get("has_structures")
    return bool(tag) and tag[0] == 1


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
        masks = get_expanded_masks(img)
        elapsed = time.time() - start

        stem = Path(image_path).stem.replace("/", "_")
        seg_entries = []
        for seg_idx in range(masks.shape[2]):
            ys, xs = np.where(masks[:, :, seg_idx])
            if ys.size == 0:
                continue
            py0, py1 = int(ys.min()), int(ys.max()) + 1
            px0, px1 = int(xs.min()), int(xs.max()) + 1
            seg = img[py0:py1, px0:px1]
            if seg.shape[0] == 0 or seg.shape[1] == 0:
                continue
            seg_filename = f"{stem}_seg{seg_idx}.png"
            cv2.imwrite(str(segments_dir / seg_filename), seg)
            # bbox is in the *parent* image's own pixel coordinate space (the
            # actual crop bounds used above) -- Stage 5 needs this to ground
            # a vision-LLM call on where each segment's structure sits within
            # the original figure, not just that it exists (see
            # reaction_link.py).
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
