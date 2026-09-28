"""Runs tag_figure.py across every figure in every paper and writes one
figure_tags.json per paper: {image_path: {tag: [label, source]}}.

Builds CLIP prototypes once (data/figure_examples/labels.csv) and reuses
them across all papers rather than rebuilding per paper -- the reference
set is independent of which paper is being tagged.
"""

import json
import sys
import time
from pathlib import Path

from figure_classify import build_prototypes
from tag_figure import tag_figure

PAPERS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "papers"


def tag_paper(paper_dir, prototypes):
    captions_path = paper_dir / "figure_captions.json"
    if not captions_path.exists():
        return {}

    with open(captions_path) as f:
        captions = json.load(f)

    results = {}
    for image_path, info in captions.items():
        full_path = paper_dir / image_path
        if not full_path.exists():
            continue
        tags = tag_figure(str(full_path), info["caption"], prototypes)
        results[image_path] = {tag: list(val) for tag, val in tags.items()}
    return results


def main():
    print("Building CLIP prototypes from reference set...")
    prototypes, _ = build_prototypes()

    paper_dirs = sorted(p for p in PAPERS_DIR.iterdir() if p.is_dir())
    for paper_dir in paper_dirs:
        start = time.time()
        results = tag_paper(paper_dir, prototypes)
        if not results:
            continue
        out_path = paper_dir / "figure_tags.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        elapsed = time.time() - start
        print(f"{paper_dir.name}: {len(results)} figures tagged -> {out_path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
