"""Batch runner for Stage 5's structure<->identifier linking, both tiers.

Routes every has_structures=1 figure by its Stage 3 segment count -- the
routing boundary confirmed by the multi-provider benchmark (see
si_structure_link.py's docstring): exactly 1 segment -> si_structure_link.py
(Gemini 3.7 Flash by default, the lightweight single-structure path), 2+
segments -> reaction_link.py, chunked Gemini 3.7 Flash by default when an
OpenRouter key is configured, single-call Claude Opus 5 otherwise. Single-
call Gemini 2.5 Pro was tried for this tier first (cheaper per call than
Opus) and rejected: it was consistently wrong, 3/3 runs, on one hard segment
of a real 35-segment grid, while Opus went 9/9 across two independently-hard
segments on the same figure. A larger 9-figure/157-structure comparison of
chunked Gemini 3.7 Flash (recursive spatial bisection into ~10-segment
chunks, see reaction_link.py's _cluster_segments) against single-call Opus
then came back at only 76.4% raw agreement -- but manual spot-checking found
that number was heavily confounded by two real bugs (Opus missing
non-numeric ligand names, one sampled figure being an out-of-scope DFT/
mechanism diagram) that have since been fixed. A re-run after both fixes
landed came back at 88.5% agreement, and every remaining disagreement
favored Flash or was a separate already-known Stage 3 bug -- see
reaction_link.py's GRID_CHUNKED_MODEL comment for the full history. A 3x
Claude Sonnet 5 majority-vote was also considered and rejected on cost
alone (a single Opus call is cheaper than three Sonnet calls at real
measured pricing) before any of this.

has_grid_layout is deliberately NOT the routing signal here, even though
reaction_link.py's own standalone candidate list uses it. Segment count is a
strictly more accurate proxy: real corpus figures exist with
has_grid_layout=0 but 2+ Stage-3 segments (an un-tagged multi-compound
figure, not a scope table) -- the tag alone would misroute these to the
lightweight single-structure path, which isn't built to disambiguate
multiple structures in one call.

Writes into the same per-paper reaction_links.json reaction_link.py already
produces (merged with whatever's already there), keyed by image_path -> list
of segment-level link records, all sharing the same field shape
reaction_link.py established. si_structure_link-sourced records leave the
grid-only fields (is_scoped_product, conditions, panel_label) null, since
those concepts don't apply to a standalone SI entry, and carry an extra
`note` field plus a `source` field recording which model produced the
record.

Concurrency: OpenRouter has no batch API for VLM calls (only some text
models), and a from-scratch Gemini Batch API integration (different auth,
JSONL submit/poll/parse, minutes-to-hours turnaround) was rejected as too
much new surface for a one-time corpus run costing a few dollars either way.
Runs everything concurrently instead, via asyncio + aiohttp/AsyncAnthropic --
these calls are almost entirely spent idle on network I/O, which is exactly
what asyncio is for (a thread pool would work too, but carries needless
OS-thread overhead for what's fundamentally a "wait on the network" workload).
A semaphore caps how many calls are in flight at once (--concurrency,
default 8) to stay well under provider rate limits.

Writes reaction_links.json after every completed figure, not once at the
end of each paper -- confirmed necessary after an earlier interrupted run
(killed a few figures into the first paper) would have discarded every
already-paid-for API call, since the old per-paper-only write meant nothing
was persisted until the whole paper's figure list finished.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import aiohttp
import anthropic
from tqdm import tqdm

import reaction_link
import si_structure_link

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"
DEFAULT_CONCURRENCY = 8


def _tag_fires(tags, name):
    v = tags.get(name)
    return bool(v) and v[0] == 1


def _candidates(paper_dir, decimer):
    """has_structures=1 image_paths that Stage 3 found at least one segment
    for -- an image with zero segments has nothing to link.

    Also excludes figures tagged BOTH is_computed AND has_plotted_data --
    confirmed via a real audit case (suzuki_iron_2024/images/page8_fig0.png,
    a DFT reaction-coordinate diagram) that this class of figure's
    "structures" are mechanistic placeholders (transition states, computed
    intermediates, catalytic-cycle scheme letters like A/B/C), not real
    isolable compounds -- there is nothing in them Stage 6 should ever want
    a compound_id for, so they're excluded before reaching the LLM at all
    rather than relying on per-segment flagging alone. is_computed ALONE is
    too noisy to filter on -- confirmed a false positive on a real substrate
    figure (nickelocene_2025/images/page2_fig0.png, caption mentions
    "idealized" and triggered the caption-keyword tag despite being a
    genuine natural-product substrate scope, not computed chemistry) that
    would have wrongly dropped real drug-name substrates (Ketamine,
    Ampicillin, etc.) it contains. Requiring has_plotted_data too (a DFT/
    mechanism figure almost always pairs its structures with an energy-
    profile plot; a real substrate scope doesn't) cut corpus-wide matches
    from 21/197 grid candidates down to 4/197, all confirmed genuine via
    caption ("Mechanistic investigations", "Linear free energy
    investigation", "Mechanistic Studies") with zero false positives found."""
    tags_path = paper_dir / "figure_tags.json"
    if not tags_path.exists():
        return []
    tags = json.load(open(tags_path))
    return [
        image_path
        for image_path, t in tags.items()
        if _tag_fires(t, "has_structures") and decimer.get(image_path)
        and not (_tag_fires(t, "is_computed") and _tag_fires(t, "has_plotted_data"))
    ]


async def _single_structure_record(session, paper_dir, image_path, segment, async_claude_client, openrouter_key):
    if openrouter_key:
        result = await si_structure_link.link_structure_gemini_async(session, openrouter_key, paper_dir, image_path)
        source = f"gemini:{si_structure_link.GEMINI_MODEL}"
    else:
        result = await si_structure_link.link_structure_claude_async(async_claude_client, paper_dir, image_path)
        source = f"claude:{si_structure_link.CLAUDE_LIGHTWEIGHT_MODEL}"
    return {
        "segment_path": segment["segment_path"],
        "is_scoped_product": None,
        "compound_id": result["compound_id"],
        "yield_percent": result["yield_percent"],
        "conditions": None,
        "panel_label": None,
        "flag": None,
        "note": result.get("note"),
        "source": source,
    }


async def _multi_structure_records(session, paper_dir, image_path, async_claude_client, openrouter_key):
    """Routes to chunked Gemini 3.7 Flash (GRID_CHUNKED_MODEL) when an
    OpenRouter key is configured -- confirmed cheaper AND at least as
    accurate as single-call Opus once the compound_id scoping confounds
    were fixed (see reaction_link.py's GRID_CHUNKED_MODEL comment for the
    validation history). Falls back to single-call Opus
    (GRID_RECOMMENDED_MODEL) when no OpenRouter key is configured, matching
    the single-structure tier's fallback pattern below."""
    segments = reaction_link._segments_for(paper_dir, image_path)
    if not segments:
        return []
    caption = reaction_link._caption_for(paper_dir, image_path)
    image_bytes = (paper_dir / image_path).read_bytes()

    if openrouter_key:
        links = await reaction_link.link_figure_gemini_chunked_async(
            session, openrouter_key, image_bytes, "image/png", caption, segments,
        )
        source = f"gemini-chunked:{reaction_link.GRID_CHUNKED_MODEL}"
    else:
        links = await reaction_link.link_figure_claude_async(
            async_claude_client, image_bytes, "image/png", caption, segments, model=reaction_link.GRID_RECOMMENDED_MODEL,
        )
        source = f"claude:{reaction_link.GRID_RECOMMENDED_MODEL}"
    if not links:
        return []
    for link in links:
        link["note"] = None
        link["source"] = source
    return links


async def _link_one(session, sem, paper_dir, image_path, segments, async_claude_client, openrouter_key):
    async with sem:
        if len(segments) == 1:
            record = await _single_structure_record(session, paper_dir, image_path, segments[0], async_claude_client, openrouter_key)
            return image_path, [record]
        return image_path, await _multi_structure_records(session, paper_dir, image_path, async_claude_client, openrouter_key)


async def link_paper(paper_dir, async_claude_client, openrouter_key, only_image=None, force=False, concurrency=DEFAULT_CONCURRENCY):
    decimer_path = paper_dir / "decimer_results.json"
    if not decimer_path.exists():
        return {}
    decimer = json.load(open(decimer_path))

    out_path = paper_dir / "reaction_links.json"
    out = json.load(open(out_path)) if out_path.exists() else {}

    targets = _candidates(paper_dir, decimer)
    if only_image:
        targets = [t for t in targets if t == only_image]
    elif not force:
        # Skip figures already linked by a prior run -- avoids re-billing the
        # same API call on every rerun for figures that didn't change.
        targets = [t for t in targets if t not in out]
    if not targets:
        return out

    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(_link_one(session, sem, paper_dir, image_path, decimer[image_path], async_claude_client, openrouter_key))
            for image_path in targets
        ]
        pbar = tqdm(total=len(tasks), desc=paper_dir.name, unit="fig", mininterval=1.0)
        for coro in asyncio.as_completed(tasks):
            try:
                image_path, records = await coro
            except Exception as e:
                tqdm.write(f"  FAILED: {e}")
                pbar.update(1)
                continue
            if records:
                out[image_path] = records
                json.dump(out, open(out_path, "w"), indent=2)  # incremental -- see module docstring
                ids = [r["compound_id"] for r in records if r["compound_id"]]
                tqdm.write(f"  {image_path} ({len(decimer[image_path])} seg): {ids or 'no IDs found'}")
            pbar.update(1)
        pbar.close()

    return out


async def _main_async(args):
    async_claude_client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("OPENROUTER_API_KEY not set -- falling back to Claude Haiku for single-structure figures and Claude Opus for grid figures")

    paper_dirs = sorted(p for p in PAPERS_DIR.iterdir() if p.is_dir())
    if args.paper:
        paper_dirs = [p for p in paper_dirs if p.name == args.paper]

    grand_total = 0
    for paper_dir in paper_dirs:
        result = await link_paper(
            paper_dir, async_claude_client, openrouter_key,
            only_image=args.image, force=args.force, concurrency=args.concurrency,
        )
        if not result:
            continue
        total = sum(len(v) for v in result.values())
        grand_total += total
        print(f"{paper_dir.name}: linked {total} segments across {len(result)} figures -> {paper_dir / 'reaction_links.json'}")

    print(f"\nTOTAL: {grand_total} segments linked")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", help="Only process this paper directory name")
    parser.add_argument("--image", help="Only process this one image_path (requires --paper)")
    parser.add_argument("--force", action="store_true", help="Re-link figures already present in reaction_links.json")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max in-flight API calls at once")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
