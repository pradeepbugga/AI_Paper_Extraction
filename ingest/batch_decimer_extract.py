"""Runs decimer_extract.py across every segment produced by batch_segment.py
(DECIMER Segmentation's Mask R-CNN) and writes one decimer_results.json per
paper: {image_path: [{"smiles": str, "mean_confidence": float,
"need_human_review": bool, "detected_missing_abbreviations": [str],
"has_generic_substituent": bool, "rdkit_valid": bool, "segment_path": str},
...]}. See decimer_extract.py's docstring for what has_generic_substituent
catches (scope-table/scheme scaffolds with a placeholder R/X/Z group, not a
real compound -- kept as its own field, not folded silently into
need_human_review) and what rdkit_valid catches (syntactically/valence-
broken SMILES -- a floor on coherence, not a correctness check).

A figure normally produces exactly one segment (the isolated structure,
composite spectrum/labels dropped by the segmentation model), but a
multi-compound figure can legitimately produce several -- each gets its own
entry in the list rather than being merged, since each may be a distinct
real structure. A figure segment_manifest.json marks as having zero
segments (nothing detected) falls back to running DECIMER on the original,
uncropped image -- better to attempt extraction and let confidence/review
flagging catch a bad result than to silently skip the figure.

Requires the segment_manifest.json files batch_segment.py writes (run in the
separate decimer_seg conda env -- see that script's docstring for why).
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from decimer_extract import extract_structure

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"


def has_structures(tags):
    tag = tags.get("has_structures")
    return bool(tag) and tag[0] == 1


def run_paper(paper_dir):
    tags_path = paper_dir / "figure_tags.json"
    manifest_path = paper_dir / "segment_manifest.json"
    if not tags_path.exists() or not manifest_path.exists():
        return {}

    with open(tags_path) as f:
        all_tags = json.load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    targets = [path for path, tags in all_tags.items() if has_structures(tags)]
    results = {}
    pbar = tqdm(targets, desc=paper_dir.name, unit="fig", mininterval=1.0)
    for image_path in pbar:
        pbar.set_postfix_str(image_path[-40:])
        segment_paths = manifest.get(image_path, [])
        if not segment_paths:
            segment_paths = [image_path]  # nothing segmented -- fall back to the original

        entries = []
        for segment_path in segment_paths:
            full_path = paper_dir / segment_path
            if not full_path.exists():
                continue
            start = time.time()
            result = extract_structure(str(full_path))
            result["segment_path"] = segment_path
            elapsed = time.time() - start
            flag = " NEEDS REVIEW" if result["need_human_review"] else ""
            tqdm.write(
                f"  {image_path} ({segment_path}) "
                f"conf={result['mean_confidence']:.3f} ({elapsed:.1f}s){flag}"
            )
            entries.append(result)

        if entries:
            results[image_path] = entries
    return results


def main():
    paper_dirs = sorted(p for p in PAPERS_DIR.iterdir() if p.is_dir())
    grand_total = 0
    grand_flagged = 0
    for paper_dir in paper_dirs:
        results = run_paper(paper_dir)
        if not results:
            continue
        out_path = paper_dir / "decimer_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        all_entries = [e for entries in results.values() for e in entries]
        flagged = sum(1 for e in all_entries if e["need_human_review"])
        grand_total += len(all_entries)
        grand_flagged += flagged
        print(
            f"{paper_dir.name}: {len(all_entries)} structures extracted "
            f"(from {len(results)} figures), {flagged} flagged for review -> {out_path}"
        )

    print(f"\nTOTAL: {grand_total} extracted, {grand_flagged} flagged for review")


if __name__ == "__main__":
    main()
