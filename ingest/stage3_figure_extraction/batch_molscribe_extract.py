"""Batch runner for Stage 3's MolScribe OCSR pass (molscribe_extract.py).

Same shape as batch_decimer_extract.py -- same target selection
(has_structures=1 figures via figure_tags.json + segment_manifest.json,
same zero-segment fallback to the uncropped original, same truncated_sides
carry-through) -- and writes into the exact same decimer_results.json path
each paper already has, so this is a drop-in replacement for the OCSR
recognizer, not a new pipeline stage: nothing in Stage 4/5 needs to change
to consume its output. Kept as a separate script rather than folding into
batch_decimer_extract.py because it must run in the dedicated
`molscribe_venv`, not the `decimer_seg`/`paper_extraction` conda envs (see
molscribe_extract.py's docstring).

Loads the MolScribe model + easyocr Reader once for the whole run (both are
expensive -- model weights + GPU init) rather than per segment, and applies
molscribe_constants_patch.apply() once up front, matching
molscribe_ensemble_predict.py's own __main__ pattern.
"""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from molscribe_extract import extract_structure

PAPERS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "papers"


def has_structures(tags):
    tag = tags.get("has_structures")
    return bool(tag) and tag[0] == 1


def _load_model():
    import torch
    import easyocr
    from molscribe import MolScribe
    from huggingface_hub import hf_hub_download
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "fine_tune_data"))
    import molscribe_constants_patch
    molscribe_constants_patch.apply()

    ckpt = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MolScribe(ckpt, device=device)
    ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return model, ocr_reader


def run_paper(paper_dir, model, ocr_reader):
    tags_path = paper_dir / "figure_tags.json"
    manifest_path = paper_dir / "segment_manifest.json"
    if not tags_path.exists() or not manifest_path.exists():
        return {}

    with open(tags_path) as f:
        all_tags = json.load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    results_path = paper_dir / "decimer_results.json"
    existing_results = json.load(open(results_path)) if results_path.exists() else {}
    # segment_path -> its already-computed result, regardless of which
    # figure it's currently filed under -- a re-split segment can end up
    # under a different image_path than before, but its own crop content
    # (and therefore its OCSR result) doesn't change just because the
    # manifest entry that points to it moved.
    known_by_segment = {
        e["segment_path"]: e for entries in existing_results.values() for e in entries
    }

    targets = [path for path, tags in all_tags.items() if has_structures(tags)]
    results = {}
    skipped = 0
    pbar = tqdm(targets, desc=paper_dir.name, unit="fig", mininterval=1.0)
    for image_path in pbar:
        pbar.set_postfix_str(image_path[-40:])
        segments = manifest.get(image_path, [])
        if not segments:
            segments = [{"path": image_path, "bbox": None}]

        entries = []
        for segment in segments:
            segment_path, bbox = segment["path"], segment.get("bbox")
            full_path = paper_dir / segment_path
            if not full_path.exists():
                continue
            if segment_path in known_by_segment:
                entries.append(known_by_segment[segment_path])
                skipped += 1
                continue
            start = time.time()
            result = extract_structure(str(full_path), model, ocr_reader)
            result["segment_path"] = segment_path
            result["bbox"] = bbox
            truncated_sides = segment.get("truncated_sides", [])
            result["truncated_sides"] = truncated_sides
            result["need_human_review"] = result["need_human_review"] or bool(truncated_sides)
            elapsed = time.time() - start
            flag = " NEEDS REVIEW" if result["need_human_review"] else ""
            tqdm.write(
                f"  {image_path} ({segment_path}) "
                f"conf={result['mean_confidence']:.3f} ({elapsed:.1f}s){flag}"
            )
            entries.append(result)

        if entries:
            results[image_path] = entries
    if skipped:
        print(f"  ({paper_dir.name}: reused {skipped} already-computed results, unchanged)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", help="Only process this paper directory name")
    args = parser.parse_args()

    paper_dirs = sorted(p for p in PAPERS_DIR.iterdir() if p.is_dir())
    if args.paper:
        paper_dirs = [p for p in paper_dirs if p.name == args.paper]

    model, ocr_reader = _load_model()

    grand_total = 0
    grand_flagged = 0
    for paper_dir in paper_dirs:
        results = run_paper(paper_dir, model, ocr_reader)
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
    print(f"\nTOTAL: {grand_total} structures, {grand_flagged} flagged for review")


if __name__ == "__main__":
    main()
