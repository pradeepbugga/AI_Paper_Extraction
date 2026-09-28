"""Merge a pod-pulled decimer_results.json tree into the local corpus.

Never scp a pod's decimer_results.json directly over the local copy: draw-tool
corrections (source == "human_drawn") only ever exist locally, since drawing
happens in the local draw_ui, not on the pod. A plain overwrite silently
clobbers them (this happened once already, see handoff_19.md and
project_stage3_handoff.md memory; the 7 lost corrections were hand-recovered
from review_log.json).

Usage:
    # 1. scp the pod's results into a staging dir, NOT over data/papers:
    scp -r pod:/workspace/.../data/papers /tmp/pod_pull

    # 2. dry-run to see what would change:
    python ingest/stage3_figure_extraction/sync_pod_results.py /tmp/pod_pull/papers

    # 3. apply once the diff looks right:
    python ingest/stage3_figure_extraction/sync_pod_results.py /tmp/pod_pull/papers --apply

Merge rule, per paper, keyed by segment_path across the whole file:
  - local segment has source == "human_drawn"  -> keep local, untouched
  - segment_path only in pod                   -> add (new segment from pod)
  - segment_path only in local                 -> keep local (e.g. locally split)
  - otherwise                                  -> take pod's version
"""

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def index_by_segment_path(results: dict) -> dict[str, tuple[str, dict]]:
    """Map segment_path -> (image_key, segment_dict)."""
    index = {}
    for image_key, segments in results.items():
        for seg in segments:
            index[seg["segment_path"]] = (image_key, seg)
    return index


def merge_paper(local: dict, pod: dict) -> tuple[dict, dict]:
    """Return (merged_results, stats)."""
    local_index = index_by_segment_path(local)
    pod_index = index_by_segment_path(pod)

    stats = {"kept_human_drawn": 0, "updated_from_pod": 0, "added_from_pod": 0, "kept_local_only": 0}

    merged: dict = {}

    for seg_path, (image_key, local_seg) in local_index.items():
        merged.setdefault(image_key, [])
        if local_seg.get("source") == "human_drawn":
            merged[image_key].append(local_seg)
            stats["kept_human_drawn"] += 1
        elif seg_path in pod_index:
            _, pod_seg = pod_index[seg_path]
            merged[image_key].append(pod_seg)
            stats["updated_from_pod"] += 1
        else:
            merged[image_key].append(local_seg)
            stats["kept_local_only"] += 1

    for seg_path, (image_key, pod_seg) in pod_index.items():
        if seg_path not in local_index:
            merged.setdefault(image_key, [])
            merged[image_key].append(pod_seg)
            stats["added_from_pod"] += 1

    return merged, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pod_papers_dir", type=Path, help="staging dir containing <paper>/decimer_results.json pulled from the pod")
    ap.add_argument("--local-papers-dir", type=Path, default=Path("data/papers"))
    ap.add_argument("--paper", help="only sync this one paper (default: all papers present in pod_papers_dir)")
    ap.add_argument("--apply", action="store_true", help="write the merged result (default: dry-run, print stats only)")
    args = ap.parse_args()

    paper_dirs = [args.pod_papers_dir / args.paper] if args.paper else sorted(args.pod_papers_dir.iterdir())

    total = {"kept_human_drawn": 0, "updated_from_pod": 0, "added_from_pod": 0, "kept_local_only": 0}
    any_changes = False

    for pod_paper_dir in paper_dirs:
        pod_file = pod_paper_dir / "decimer_results.json"
        if not pod_file.is_file():
            continue
        paper_name = pod_paper_dir.name
        local_file = args.local_papers_dir / paper_name / "decimer_results.json"
        if not local_file.is_file():
            print(f"[{paper_name}] no local decimer_results.json, skipping (not merging a brand-new paper)")
            continue

        pod = load(pod_file)
        local = load(local_file)
        merged, stats = merge_paper(local, pod)

        for k in total:
            total[k] += stats[k]

        changed = stats["updated_from_pod"] or stats["added_from_pod"]
        any_changes = any_changes or changed
        flag = " <- CHANGED" if changed else ""
        print(f"[{paper_name}] kept_human_drawn={stats['kept_human_drawn']} "
              f"updated_from_pod={stats['updated_from_pod']} "
              f"added_from_pod={stats['added_from_pod']} "
              f"kept_local_only={stats['kept_local_only']}{flag}")

        if args.apply and changed:
            with open(local_file, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"  wrote {local_file}")

    print()
    print(f"TOTAL: kept_human_drawn={total['kept_human_drawn']} "
          f"updated_from_pod={total['updated_from_pod']} "
          f"added_from_pod={total['added_from_pod']} "
          f"kept_local_only={total['kept_local_only']}")
    if not args.apply:
        print("\nDry run only — pass --apply to write changes.")
    elif not any_changes:
        print("\nNo changes to apply.")


if __name__ == "__main__":
    main()
