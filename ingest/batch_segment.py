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
"""

import json
import sys
import time
from pathlib import Path

import cv2
from decimer_segmentation import segment_chemical_structures

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
    for i, image_path in enumerate(targets, 1):
        full_path = paper_dir / image_path
        if not full_path.exists():
            print(f"  [{i}/{len(targets)}] MISSING {image_path}")
            continue
        start = time.time()
        img = cv2.imread(str(full_path))
        segments = segment_chemical_structures(img, expand=True)
        elapsed = time.time() - start

        stem = Path(image_path).stem.replace("/", "_")
        seg_paths = []
        for seg_idx, seg in enumerate(segments):
            seg_filename = f"{stem}_seg{seg_idx}.png"
            cv2.imwrite(str(segments_dir / seg_filename), seg)
            seg_paths.append(f"segments/{seg_filename}")

        manifest[image_path] = seg_paths
        print(f"  [{i}/{len(targets)}] {image_path} -> {len(seg_paths)} segment(s) ({elapsed:.1f}s)")

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
