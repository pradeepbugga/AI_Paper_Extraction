"""Runs batch_segment's run_paper for exactly one paper, so several of these
can run in parallel (one per paper) to use idle CPU/GPU headroom -- the
committed batch_segment.py itself is untouched, this just calls its existing
run_paper() function directly. Mirrors run_one_paper_ocsr.py's pattern for
Phase 2."""
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from batch_segment import run_paper

paper_dir = Path(sys.argv[1])
manifest = run_paper(paper_dir)
out_path = paper_dir / "segment_manifest.json"
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)

total_segments = sum(len(v) for v in manifest.values())
zero_segment = sum(1 for v in manifest.values() if not v)
print(
    f"{paper_dir.name}: {len(manifest)} figures, {total_segments} segments total, "
    f"{zero_segment} with no detected structure -> {out_path}"
)
