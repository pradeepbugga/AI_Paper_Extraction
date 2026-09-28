"""Runs batch_decimer_extract's run_paper for exactly one paper, so several
can run in parallel (one per paper) -- mirrors run_one_paper_segment.py's
pattern for Phase 1. Must run in the paper_extraction conda env."""
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from batch_decimer_extract import run_paper

paper_dir = Path(sys.argv[1])
results = run_paper(paper_dir)
out_path = paper_dir / "decimer_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

all_entries = [e for entries in results.values() for e in entries]
flagged = sum(1 for e in all_entries if e["need_human_review"])
print(
    f"{paper_dir.name}: {len(all_entries)} structures extracted "
    f"(from {len(results)} figures), {flagged} flagged for review -> {out_path}"
)
