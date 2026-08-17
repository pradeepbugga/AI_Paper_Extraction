"""Stage 5, first slice: compound-ID/yield/condition linkage for scope-table
figures.

Scope-table/scheme figures (a grid of drawn structures, one per product
variant) carry their per-structure data -- an entry/compound-ID label, a
yield, sometimes a condition-variant annotation like "X = Cl" -- as small
text printed directly next to each structure in the image itself, not as a
separate Stage 4 table (`decimer_extract.py`'s `_strip_annotation_text`
already OCRs and paints over exactly this text before OCSR, so DECIMER
doesn't hallucinate on it -- the annotations were always there, just
discarded before this stage). Stage 3 already extracts each structure's
SMILES; this module's job is narrower than "read the figure" -- link each
already-extracted structure to its own label/yield/conditions by reading
the original (un-annotation-stripped) image, using Stage 3's SMILES + each
segment's bbox as grounding rather than re-deriving structures from pixels.

Candidate figures: has_structures=1 AND has_grid_layout=1 in figure_tags.json
(the latter fires on "substrate scope"/"scope" in the caption -- see
figure_caption_rules.py's SCOPE_RE). Confirmed 55 candidates across the
7-paper corpus.

Provider-agnostic by design: `link_figure_claude` is the only
provider-specific piece (image + caption + segment list in, a list of typed
link records out). A second implementation for a different model (Haiku 4.5,
Qwen3-VL) is a new function with the same signature, not a rewrite -- this
project's plan is to benchmark several models against the same 55-figure set
once the Claude Sonnet 5 path is validated.

Known adjacent failure modes to watch for when validating, both already
documented elsewhere in this pipeline: a `too_many_fragments=True` segment is
DECIMER Segmentation under-merging several real compounds into one crop (the
SMILES itself is unreliable) -- the model should flag rather than force a
confident single compound-ID match. A `has_generic_substituent=True` segment
is a scope-table scaffold with a placeholder R/X/Z group, not a real product
-- not every segment needs a link.

KNOWN GAP, found during manual audit of a real run (not yet fixed --
deliberately deferred): `yield_percent`/`conditions` are scalar fields, one
pair per segment, but a single scope-table entry can report more than one
condition-variant yield for the *same* product -- e.g. one compound made from
two different starting materials (X = Cl giving 92%, X = Br giving 93% of
the identical product) drawn once with two yields printed beneath it. The
current schema/prompt only captures one pair and silently drops the other
(confirmed on copper_iron_2025/images/page3_fig1.png, compound "1": kept
"92% (X = Cl)", dropped "93% (X = Br)"). This is NOT a hallucination or a
prompt-following failure -- there's structurally nowhere for the second pair
to go. Treat this module's yield_percent as best-effort/incomplete for any
entry with condition-variant yields, not authoritative. Real fix is Stage 5's
third slice (ID<->reaction conditions/screening, one-to-many by design) doing
proper multi-variant capture from scratch, superseding this field for those
cases -- not a schema patch here.
"""

import argparse
import asyncio
import base64
import io
import json
from pathlib import Path

import aiohttp
import anthropic
import requests
from PIL import Image, ImageDraw

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"

CLAUDE_MODEL = "claude-sonnet-5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# First candidate default for this tier, from a benchmark run on a real dense
# grid (page3_fig0.png, miyaura_iron_2025): on cost alone, a single Gemini
# 2.5 Pro call ($0.078) beat both 3x Claude Sonnet 5 majority-vote ($0.136 at
# intro pricing, $0.204 after) and a single Claude Opus 5 call ($0.125), and
# matched Opus's 3/3 on the one hard segment (seg14) tested first. REJECTED
# after checking a second hard segment on the same figure (seg18): Gemini
# was wrong 3/3 runs there (a consistent misread, not noise -- it read a
# different, real compound_id every time), while Opus went 3/3 on seg18 too
# -- 9/9 across both segments. A systematic per-segment miss isn't something
# a majority vote or a cheaper price fixes, so accuracy took priority over
# the ~60% cost premium for a linking stage everything downstream depends
# on. Left defined (and still used by si_structure_link.py's own,
# separately-validated single-structure tier) in case a future benchmark on
# more figures reverses this again -- see GRID_RECOMMENDED_MODEL below for
# what's actually used.
GRID_GEMINI_MODEL = "google/gemini-2.5-pro"

# No-OpenRouter-key fallback for this tier (see GRID_CHUNKED_MODEL below for
# the actual default). Opus costs more per call than a single Sonnet or
# Gemini call, but went 9/9 across two independently-hard segments on the
# densest tested figure, vs single-call Gemini's 6/9 (one segment,
# consistently wrong) and Sonnet's flaky 2/3 on the other -- and it's still
# cheaper than the 3x-Sonnet-vote alternative that was the original
# motivation for looking at cheaper models in the first place (see
# batch_link_structures.py for where this is wired in).
GRID_RECOMMENDED_MODEL = "claude-opus-5"

# Actual default for this tier as of the confound-corrected re-validation.
# Single-call Opus was the recommended default for a while on the strength
# of the 9/9-vs-6/9 result above, but that result was against single-call
# Gemini 2.5 Pro on ONE figure -- a much larger 9-figure/157-structure
# stratified comparison (chunked Gemini 3.7 Flash, recursive KD-tree-style
# bisection into ~10-segment chunks so each call sees a less crowded image)
# came back at only 76.4% raw agreement with Opus, cheap ($0.126 vs Opus's
# $0.677 total) but seemingly less accurate. Manual spot-checking of the
# disagreements found the raw number was heavily confounded by two real bugs
# that have since been fixed (see the compound_id field's docstring/prompt
# history): Opus was blindly missing non-numeric ligand names entirely
# separate from the model-choice question, and one whole figure in the
# sample was a DFT/mechanism figure that should never have been in scope at
# all. A re-run of the same methodology after both fixes landed (8 figures,
# 131 structures, the confirmed-out-of-scope mechanism figure dropped) came
# back at 88.5% agreement, and manual spot-checking of the *remaining*
# disagreements found they split into two groups, NEITHER of which favors
# Opus: ~8 were a separate, already-known Stage 3 bug
# (too_many_fragments under-detection on merged multi-structure crops,
# copper_iron_2025/images/page3_fig1.png) where neither model's answer is
# well-defined, and the other ~6 were Opus returning compound_id=null on
# real, physically real drug/natural-product trivial names that Flash read
# correctly -- confirmed by direct image inspection, e.g.
# suzuki_nickel_2026/images/page6_fig1.png literally labels two structures
# "(+)-sertraline" and "(+)-indatraline" with no numeric ID at all; Opus
# missed both, Flash got both. This is the same failure family as the
# ligand-name fix (Opus under-recognizing non-numeric identifiers) but for a
# category the earlier prompt broadening didn't cover. Once those two
# confounds are excluded, chunked Flash had zero remaining errors in the
# sample at roughly 1/6th Opus's cost -- so it's now the default, with Opus
# (GRID_RECOMMENDED_MODEL above) kept only as the fallback when no
# OpenRouter key is configured.
GRID_CHUNKED_MODEL = "google/gemini-3.7-flash"
GRID_CHUNK_TARGET_SIZE = 10
GRID_CHUNK_PAD_PX = 130

RECORD_LINKS_TOOL = {
    "name": "record_links",
    "description": (
        "Record the compound-ID label, yield, condition-variant annotation, "
        "and (if the figure has lettered sub-panels) panel label for each "
        "segment, read directly from the image next to each structure."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_path": {
                            "type": "string",
                            "description": "Must exactly match one of the segment_path values given in the prompt.",
                        },
                        "is_scoped_product": {
                            "type": "boolean",
                            "description": (
                                "True if this segment's drawing is one specific, individually-labeled entry in the "
                                "table/scope with its OWN reported yield in this figure (even if its SMILES was "
                                "misread as generic -- judge this from what the image actually shows, not from the "
                                "has_generic_substituent hint). False if the drawing is a reaction-scheme template, "
                                "or a reagent/catalyst/starting-material shown for context rather than as a "
                                "yield-bearing product of this scheme -- e.g. the general 'R-CF3' arrow-diagram "
                                "template drawn once at the top of a panel is False even though a specific product "
                                "below it sharing a similar structure is True. This field is about whether a YIELD "
                                "applies here, not about whether the structure has a real identifier -- a starting "
                                "material can be False here while still carrying its own real compound_id (see "
                                "below)."
                            ),
                        },
                        "compound_id": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": (
                                "The bold identifier printed next to THIS structure, in WHATEVER form the paper "
                                "uses -- a numbered/lettered compound label (e.g. '3a'), OR a mnemonic name with no "
                                "digits at all (e.g. a ligand abbreviation like 'SciOPP', 'dcype', 'tBu-Xantphos', "
                                "'1,8-dppn' -- these are just as much a real identifier as a numbered compound, "
                                "common in ligand/catalyst screening figures). Do not require a digit or a "
                                "particular format -- if there's a short bold/labeled name printed directly under "
                                "or beside this exact structure, report it verbatim, numeric or not. Report this "
                                "even when is_scoped_product is false -- papers often give a real, permanent "
                                "identifier to a starting material, ligand, or reagent (e.g. '1a', or a ligand name "
                                "like 'dppe') even though it's not this scheme's own numbered product and carries "
                                "no yield here; that identifier still matters downstream. Only leave this null when "
                                "no identifier is printed near this specific structure at all (a true unlabeled "
                                "template/reagent) -- never invent one, and never reuse a label that belongs to a "
                                "different, neighboring structure.\n"
                                "EXCEPTION -- do NOT report a label here if it's a mechanistic or computational "
                                "placeholder rather than a real, physically isolable compound: a DFT/computed "
                                "reaction-coordinate intermediate (e.g. 'I_A', 'I_B'), a transition state (e.g. "
                                "'TS_OA', 'TS_RE'), or a bare scheme letter used only to narrate a catalytic cycle "
                                "or mechanism figure (e.g. 'A', 'B', 'C', 'D' labeling boxes in an arrow diagram of "
                                "a proposed pathway, not a substrate scope). These only mean something within this "
                                "one mechanism figure and were never synthesized or characterized as standalone "
                                "compounds -- leave compound_id null for them even though a label is visibly "
                                "printed next to them. The distinguishing question: could this identifier ever be "
                                "looked up elsewhere (in the paper's text, SI, or a database) as a real, isolable "
                                "substance? If yes (a numbered product, a named ligand, a starting material), "
                                "report it. If it only exists to label a box in this one mechanism/DFT diagram, "
                                "leave it null."
                            ),
                        },
                        "yield_percent": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "The yield text as printed (e.g. '87%', 'trace', 'NR'), or null if none is shown for this structure. Must be null when is_scoped_product is false -- a non-product structure was never the subject of a yield measurement in this figure.",
                        },
                        "conditions": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Any condition-variant annotation specific to this structure (e.g. 'X = Cl', a footnote marker's condition), or null. Must be null when is_scoped_product is false.",
                        },
                        "panel_label": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "The lettered sub-panel this structure belongs to (e.g. 'b'), or null if the figure has no lettered panels.",
                        },
                        "flag": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Set if this segment is unreliable to link: e.g. 'multi_compound_merge' if the segment's SMILES looks like it covers more than one real structure, 'generic_scaffold' if is_scoped_product is false, 'smiles_misread' if is_scoped_product is true but the given SMILES looks wrong for what the image shows, or a free-text note for anything else worth flagging. Null if the link is straightforward.",
                        },
                    },
                    "required": ["segment_path", "is_scoped_product", "compound_id", "yield_percent", "conditions", "panel_label", "flag"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["links"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _tag_fires(tags, name):
    v = tags.get(name)
    return bool(v) and v[0] == 1


def load_candidates(paper_dir):
    """Figure image paths where has_structures=1 AND has_grid_layout=1."""
    tags_path = paper_dir / "figure_tags.json"
    if not tags_path.exists():
        return []
    all_tags = json.load(open(tags_path))
    return [
        image_path
        for image_path, tags in all_tags.items()
        if _tag_fires(tags, "has_structures") and _tag_fires(tags, "has_grid_layout")
    ]


def _caption_for(paper_dir, image_path):
    path = paper_dir / "figure_captions.json"
    if not path.exists():
        return None
    captions = json.load(open(path))
    entry = captions.get(image_path)
    return entry["caption"] if entry else None


def _segments_for(paper_dir, image_path):
    path = paper_dir / "decimer_results.json"
    if not path.exists():
        return []
    results = json.load(open(path))
    return results.get(image_path, [])


# Shared between _build_prompt and _build_prompt_marked so the two never
# drift out of sync on the actual linking rules -- only how each segment is
# described to the model (bbox text vs. a visible drawn mark) differs.
_LINKING_INSTRUCTIONS = """For every segment, first decide is_scoped_product from what the image
actually shows at that segment's location -- not from the has_generic_
substituent hint, which describes Stage 3's OCSR reading of the structure
and can be wrong in either direction:
- A segment can be a real, specific, individually-labeled product even
  though has_generic_substituent=True (Stage 3's OCSR misread part of a
  normal structure as a placeholder). In this case is_scoped_product is
  true -- read its own real label/yield/conditions normally, and set flag
  to note the SMILES looks wrong for what's drawn.
- A segment can be a reaction-scheme template, a reagent/catalyst box, or
  any other non-product component even when its own SMILES looks
  ordinary. In this case is_scoped_product is false, and yield_percent/
  conditions MUST both be null -- never reuse or borrow a nearby real
  product's yield/conditions for a non-product segment, even if they
  happen to sit right next to each other or share a similar structure.
  compound_id is DIFFERENT: still report it if a real identifier is
  printed specifically next to THIS structure, even when is_scoped_product
  is false -- papers often give a starting material or reagent its own
  real, permanent compound number (e.g. '1a') that isn't this scheme's own
  product but is still a real identifier worth capturing. Never invent one
  and never borrow a neighboring structure's identifier, but a genuine
  identifier belonging to this exact structure should never be discarded
  just because it's not a yield-bearing product.
  EXCEPTION: this only applies to real, physically isolable compounds
  (numbered products, named ligands/reagents, starting materials) -- not
  to mechanistic or computational placeholders. If this figure is a
  proposed-mechanism or DFT/computed reaction-coordinate diagram, labels
  like 'I_A', 'TS_OA', or a bare scheme letter ('A', 'B', 'C') marking a
  box in a catalytic-cycle arrow diagram are NOT real compound identifiers
  -- they only mean something inside this one figure and were never
  synthesized or characterized on their own. Leave compound_id null for
  those even though a label is visibly printed next to them."""


def _build_prompt(caption, segments):
    segment_lines = []
    for s in segments:
        bbox = s.get("bbox")
        bbox_desc = f"at pixel bbox {bbox} in this image" if bbox else "(no crop -- covers the whole image)"
        flags = []
        if s.get("too_many_fragments"):
            flags.append("too_many_fragments=True (may be several compounds merged into one crop)")
        if s.get("has_generic_substituent"):
            flags.append("has_generic_substituent=True (may be a placeholder scaffold, not a real product)")
        if s.get("truncated_sides"):
            flags.append(
                f"truncated_sides={s['truncated_sides']} (crop may still cut off a substituent label on this "
                "side even after automatic recovery -- SMILES could be wrong; read the label directly from the "
                "image rather than trusting it)"
            )
        flag_desc = f" [{', '.join(flags)}]" if flags else ""
        segment_lines.append(f"- segment_path={s['segment_path']!r} {bbox_desc}, SMILES={s['smiles']!r}{flag_desc}")

    caption_desc = f'Figure caption: "{caption}"' if caption else "No caption text was found for this figure."

    return f"""This image is a scope-table/scheme figure from a chemistry paper. Each
segment below is a structure Stage 3 already extracted and located within
this image (by pixel bbox). Your job is to link each one to the data printed
next to it in the image: its compound/entry-ID label, yield, any
condition-variant annotation specific to it, and (if the figure has lettered
sub-panels like (a)/(b)/(c)) which panel it belongs to.

{_LINKING_INSTRUCTIONS}

{caption_desc}

Segments already extracted from this figure:
{chr(10).join(segment_lines)}

Use the segment bboxes to find each structure in the image, then read the
label/yield/conditions text printed next to it. Call record_links with one
entry per segment_path listed above (all of them, even if you can't find a
label for one -- use null fields and set flag to explain why). Do not invent
data that isn't visibly printed in the image."""


def _letter_code(index):
    """0-indexed -> 'A', 'B', ..., 'Z', 'AA', 'AB', ... (spreadsheet-column
    style), i.e. digit-free no matter how many segments a chunk has."""
    index += 1
    code = ""
    while index > 0:
        index, rem = divmod(index - 1, 26)
        code = chr(65 + rem) + code
    return code


def _mark_for(index):
    """Short tag drawn on the image for a segment, e.g. '@C' for the 3rd
    segment in a chunk (order matches the chunk's own local_segments list,
    so drawing and prompt-building stay consistent) -- '@' + letters-ONLY,
    deliberately containing no digits at all, unlike the previous
    'seg42'-style tag. That 'seg'-prefixed scheme was ALSO supposed to be
    collision-proof (rejected an earlier bare-numeric-mark attempt for
    exactly this reason) and was validated on this exact failure mode on
    redox_neutral_2024/page5_fig0.png/page8_fig0.png (fixed 13/14 wrong/
    duplicated segments) -- but a later corpus re-run resurfaced the same
    kind of misattribution on the same figure (compound 10's identifier/
    yield attached to the segment showing reagent 8, compound 12's attached
    to a generic R/R' scheme-template segment), so 'seg'+digits evidently
    wasn't orthogonal enough: a tag like 'seg30' still contains a bare
    number the model can latch onto or blend with a real printed compound
    number/yield elsewhere on the page. A tag with NO digits at all removes
    that failure mode structurally rather than relying on a prefix string
    the model might silently strip. Letters are uppercase and '@'-prefixed
    specifically to avoid colliding with panel labels too (always lowercase,
    parenthesized, e.g. '(a)') and letter-prefixed ligand codes (e.g.
    'L11', which pairs a letter WITH digits, unlike this scheme)."""
    return "@" + _letter_code(index)


def _draw_marks(crop, local_segments):
    """Draws a red box + collision-proof mark tag (see _mark_for) directly
    onto the chunk crop for each of its own member segments, in place.
    Returns the same crop for chaining."""
    draw = ImageDraw.Draw(crop)
    for s in local_segments:
        lx0, ly0, lx1, ly1 = s["bbox"]
        draw.rectangle([lx0, ly0, lx1, ly1], outline="red", width=3)
        mark = s["_mark"]
        tag_w = 8 * len(mark) + 6
        draw.rectangle([lx0, ly0, lx0 + tag_w, ly0 + 18], fill="red")
        draw.text((lx0 + 3, ly0 + 2), mark, fill="white")
    return crop


def _build_prompt_marked(caption, local_segments):
    """Set-of-Mark variant of _build_prompt for the chunked grid tier: each
    segment is described by its visible drawn mark (see _draw_marks) instead
    of only a numeric bbox, since chunked-Flash was found to sometimes
    misattribute a neighboring segment's real label/yield to the wrong
    segment_path when grounded on numeric coordinates alone -- confirmed
    fixed by switching to visible marks on the same confirmed-bad figures
    (see _mark_for's docstring)."""
    segment_lines = []
    for s in local_segments:
        flags = []
        if s.get("too_many_fragments"):
            flags.append("too_many_fragments=True (may be several compounds merged into one crop)")
        if s.get("has_generic_substituent"):
            flags.append("has_generic_substituent=True (may be a placeholder scaffold, not a real product)")
        if s.get("truncated_sides"):
            flags.append(
                f"truncated_sides={s['truncated_sides']} (crop may still cut off a substituent label on this "
                "side even after automatic recovery -- SMILES could be wrong; read the label directly from the "
                "image rather than trusting it)"
            )
        flag_desc = f" [{', '.join(flags)}]" if flags else ""
        segment_lines.append(f"- segment_path={s['segment_path']!r} marked with red box labeled {s['_mark']!r} in the image, SMILES={s['smiles']!r}{flag_desc}")

    caption_desc = f'Figure caption: "{caption}"' if caption else "No caption text was found for this figure."

    return f"""This image is a crop from a scope-table/scheme figure from a chemistry
paper. Each structure Stage 3 already extracted has been marked with a red
bounding box and a small red label tag (e.g. "@C") directly on the image,
placed just above/left of its box. These tags are ONLY box identifiers for
this task -- they are letters-only (never digits) and always start with "@"
specifically so you can never confuse one with a real printed compound/
entry-ID label, which is always a bare number or number+letter (e.g. "12",
"3a") and never "@"-prefixed. Your job is to link each marked segment to the
data printed next to it in the image: its compound/entry-ID label, yield,
any condition-variant annotation, and panel if applicable.

CRITICAL: use the VISIBLE red box for each segment to find exactly which
structure it refers to, then read the label/yield text printed immediately
next to THAT boxed structure -- not a neighboring one, even if a
neighboring structure's label is more complete-looking or closer to the
box's edge. Do not borrow a nearby unmarked or differently-marked
structure's label just because it looks similar or is close by. Match
strictly by which red box encloses which drawing. A segment's own tag
(e.g. "@C") is never itself a compound_id or yield -- read those from the
page's own printed text near the box, never from the tag.

{_LINKING_INSTRUCTIONS}

{caption_desc}

Marked segments in this image:
{chr(10).join(segment_lines)}

Call record_links with one entry per segment_path listed above (all of
them, even if you can't find a label for one -- use null fields and set
flag to explain why). Do not invent data that isn't visibly printed in the
image."""


def _link_figure_kwargs(image_bytes, media_type, caption, segments, model):
    image_b64 = base64.standard_b64encode(image_bytes).decode()
    prompt = _build_prompt(caption, segments)
    return dict(
        model=model,
        max_tokens=16000,
        tools=[RECORD_LINKS_TOOL],
        tool_choice={"type": "tool", "name": "record_links"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )


def _finish_link_response(response, segments):
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"record_links call hit max_tokens with {len(segments)} segments -- "
            "response truncated, raise max_tokens or split the figure's segments across calls"
        )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return _enforce_non_product_nulls(tool_use.input["links"])


def link_figure_claude(client, image_bytes, media_type, caption, segments, model=CLAUDE_MODEL):
    """Provider-specific implementation #1. Returns a list of link dicts, one
    per segment_path. See module docstring for the provider-agnostic contract
    a second model's implementation should match. `model` defaults to Sonnet
    5 but accepts any Claude model string -- used to benchmark Haiku 4.5
    against Sonnet 5 on the same grid figures, see handoff notes.

    A dense scope table can carry 50+ segments -- confirmed hitting the old
    4096-token cap mid-tool-call on a real 59-segment figure
    (redox_neutral_2024/images/page5_fig0.png, stop_reason="max_tokens"),
    truncating the JSON before the tool call completed. Streamed so a large
    response doesn't risk an HTTP timeout either."""
    with client.messages.stream(**_link_figure_kwargs(image_bytes, media_type, caption, segments, model)) as stream:
        response = stream.get_final_message()
    return _finish_link_response(response, segments)


async def link_figure_claude_async(async_client, image_bytes, media_type, caption, segments, model=CLAUDE_MODEL):
    """Same contract as link_figure_claude, via anthropic.AsyncAnthropic --
    for batch runs across many figures at once (see batch_link_structures.py)."""
    async with async_client.messages.stream(**_link_figure_kwargs(image_bytes, media_type, caption, segments, model)) as stream:
        response = await stream.get_final_message()
    return _finish_link_response(response, segments)


# OpenRouter uses OpenAI-style function-calling tools, not Anthropic's
# input_schema shape -- same fields, different wrapper. Built once from
# RECORD_LINKS_TOOL rather than duplicated, matching si_structure_link.py's
# pattern, so the two providers can never drift out of sync with each other.
_RECORD_LINKS_TOOL_OPENROUTER = {
    "type": "function",
    "function": {
        "name": RECORD_LINKS_TOOL["name"],
        "description": RECORD_LINKS_TOOL["description"],
        "parameters": RECORD_LINKS_TOOL["input_schema"],
    },
}


def _gemini_grid_request_body(image_bytes, media_type, caption, segments, model):
    image_b64 = base64.standard_b64encode(image_bytes).decode()
    prompt = _build_prompt(caption, segments)
    return {
        "model": model,
        "tools": [_RECORD_LINKS_TOOL_OPENROUTER],
        "tool_choice": {"type": "function", "function": {"name": "record_links"}},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }


def _parse_gemini_grid_response(data, model):
    if "error" in data:
        raise RuntimeError(f"OpenRouter error for {model}: {data['error']}")
    tool_calls = data["choices"][0]["message"].get("tool_calls")
    if not tool_calls:
        raise RuntimeError(f"{model} returned no tool call: {data['choices'][0]['message'].get('content')}")
    links = json.loads(tool_calls[0]["function"]["arguments"])["links"]
    return _enforce_non_product_nulls(links)


def link_figure_gemini(api_key, image_bytes, media_type, caption, segments, model=GRID_GEMINI_MODEL):
    """Second provider implementation, via OpenRouter -- see module
    docstring's provider-agnostic contract. Now the recommended default for
    this tier -- see GRID_GEMINI_MODEL's comment for the benchmark that
    motivated the switch from Sonnet."""
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=_gemini_grid_request_body(image_bytes, media_type, caption, segments, model),
        timeout=180,
    )
    return _parse_gemini_grid_response(resp.json(), model)


async def link_figure_gemini_async(session, api_key, image_bytes, media_type, caption, segments, model=GRID_GEMINI_MODEL):
    """Same contract as link_figure_gemini, via a shared aiohttp
    ClientSession -- for batch runs across many figures at once (see
    batch_link_structures.py)."""
    async with session.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=_gemini_grid_request_body(image_bytes, media_type, caption, segments, model),
        timeout=aiohttp.ClientTimeout(total=180),
    ) as resp:
        data = await resp.json()
    return _parse_gemini_grid_response(data, model)


async def _link_chunk_marked_async(session, api_key, image_bytes, media_type, prompt, model):
    """Sends an already-built Set-of-Mark prompt (see _build_prompt_marked)
    for one chunk -- kept separate from link_figure_gemini_async rather than
    threading a marked/unmarked flag through it, since only the chunked
    grid-tier path (link_figure_gemini_chunked_async) ever has marks drawn
    on its image; every other caller of link_figure_gemini_async still wants
    the plain numeric-bbox prompt."""
    image_b64 = base64.standard_b64encode(image_bytes).decode()
    body = {
        "model": model,
        "tools": [_RECORD_LINKS_TOOL_OPENROUTER],
        "tool_choice": {"type": "function", "function": {"name": "record_links"}},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    async with session.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=aiohttp.ClientTimeout(total=180),
    ) as resp:
        data = await resp.json()
    return _parse_gemini_grid_response(data, model)


def _cluster_segments(segments, max_size):
    """Recursive spatial bisection: split segments by whichever axis (x or y
    of each segment's bbox center) has the larger spread, recurse until every
    leaf group is at or under max_size. Produces balanced chunks (e.g. 35
    segments -> 8/9/9/9 at max_size=10) so each grid-tier call sees a less
    crowded image than the full figure -- see GRID_CHUNKED_MODEL's comment
    for why this mattered (dense pages are where single-call models
    systematically confuse numeric-coordinate grounding between
    neighbors)."""
    if len(segments) <= max_size:
        return [segments]
    centers_x = [(s["bbox"][0] + s["bbox"][2]) / 2 for s in segments]
    centers_y = [(s["bbox"][1] + s["bbox"][3]) / 2 for s in segments]
    x_range = max(centers_x) - min(centers_x)
    y_range = max(centers_y) - min(centers_y)
    centers = centers_x if x_range >= y_range else centers_y
    order = sorted(range(len(segments)), key=lambda i: centers[i])
    mid = len(order) // 2
    left = [segments[i] for i in order[:mid]]
    right = [segments[i] for i in order[mid:]]
    return _cluster_segments(left, max_size) + _cluster_segments(right, max_size)


def _partition_and_cluster(segments, max_size):
    """Like _cluster_segments, but keeps too_many_fragments=True segments
    (Stage 3's own signal that a crop is a merged/ambiguous multi-compound
    blob, not a clean single structure) out of every chunk that also
    contains clean segments -- clustered separately instead. Found necessary
    on a real case (copper_iron_2025/images/page3_fig1.png): a merged blob
    covering an entire under-segmented row got clustered as an ordinary
    chunk-mate of an unrelated clean segment two rows away, and the model
    misattributed the merged blob's own real, visible label text to that
    unrelated segment instead. Isolating messy segments into their own
    chunk(s) means their crop bounds are computed only from their own (also
    messy) neighbors, so a clean segment's chunk crop is never forced to
    include a merged blob's territory just because they happened to land in
    the same balanced-size group. See also _mask_foreign_messy_regions for
    the second half of this fix (padding can still cause bleed even without
    direct chunk membership)."""
    messy = [s for s in segments if s.get("too_many_fragments")]
    clean = [s for s in segments if not s.get("too_many_fragments")]
    clusters = []
    if clean:
        clusters.extend(_cluster_segments(clean, max_size))
    if messy:
        clusters.extend(_cluster_segments(messy, max_size))
    return clusters


def _local_segments_for_chunk(cluster, x0, y0):
    out = []
    for s in cluster:
        sx0, sy0, sx1, sy1 = s["bbox"]
        s2 = dict(s)
        s2["bbox"] = [sx0 - x0, sy0 - y0, sx1 - x0, sy1 - y0]
        out.append(s2)
    return out


def _mask_foreign_messy_regions(crop, x0, y0, cluster, all_segments):
    """White-fills the pixel region of any too_many_fragments=True segment
    that ISN'T a member of this chunk but whose bbox still falls inside this
    chunk's padded crop -- the residual bleed case _partition_and_cluster
    alone doesn't catch: chunk padding (generous, since a segment's own
    label often sits outside its tight Stage-3 bbox) can reach into a
    neighboring merged blob's territory even when that blob isn't a direct
    chunk-mate. Masks the pixels rather than just instructing the model to
    ignore it -- confirmed elsewhere this session that this tier's model
    doesn't reliably follow every instruction in the prompt, so removing the
    stray text outright is more robust than asking the model not to read
    it."""
    member_paths = {s["segment_path"] for s in cluster}
    cw, ch = crop.size
    draw = None
    for s in all_segments:
        if s["segment_path"] in member_paths or not s.get("too_many_fragments"):
            continue
        bx0, by0, bx1, by1 = s["bbox"]
        lx0, ly0 = max(0, bx0 - x0), max(0, by0 - y0)
        lx1, ly1 = min(cw, bx1 - x0), min(ch, by1 - y0)
        if lx1 > lx0 and ly1 > ly0:
            if draw is None:
                draw = ImageDraw.Draw(crop)
            draw.rectangle([lx0, ly0, lx1, ly1], fill="white")
    return crop


async def link_figure_gemini_chunked_async(
    session, api_key, image_bytes, media_type, caption, segments,
    model=GRID_CHUNKED_MODEL, target_chunk_size=GRID_CHUNK_TARGET_SIZE, pad_px=GRID_CHUNK_PAD_PX,
):
    """Grid-tier default -- see GRID_CHUNKED_MODEL's comment for the
    validation history. Splits segments into spatially-balanced chunks (see
    _cluster_segments and _partition_and_cluster), crops the image to each
    chunk's bounding region with generous padding (label text sits outside a
    structure's own Stage-3 bbox), masks out any other merged/ambiguous
    segment's territory that padding still reaches into (see
    _mask_foreign_messy_regions), draws a Set-of-Mark box+tag directly on the
    image for each of the chunk's own segments (see _draw_marks/_mark_for),
    remaps bboxes to chunk-local coordinates, and dispatches one marked
    request per chunk concurrently (see _link_chunk_marked_async).

    The Set-of-Mark step was added after a confirmed real failure mode even
    with chunking already in place: chunked-Flash correctly read a boxed-
    adjacent segment's own visible content but still sometimes reported a
    NEIGHBORING segment's compound_id/yield instead of its own, when
    grounded only on numeric bbox coordinates in the prompt text (e.g.
    redox_neutral_2024/images/page5_fig0.png segment 42 -- its own crop
    clearly shows compound 13, but it was reported as compound 14, the
    unmarked structure sitting immediately to its right). Drawing a real
    box+tag on the image, with a 'seg'-prefixed tag guaranteed never to
    collide with a real printed compound label (unlike an earlier,
    correctly-rejected attempt at bare-numeric marks on a full 35+-segment
    unchunked page), fixed 13 of 14 previously wrong/duplicated segments
    across two independently-confirmed-bad figures in the same paper.

    Falls back to a single uncropped, unmarked call when the figure is
    already small enough (<= target_chunk_size) or lacks bboxes (the
    single-whole-image-segment case, which shouldn't reach this function via
    batch_link_structures.py's routing but is handled safely regardless)."""
    if len(segments) <= target_chunk_size or not all(s.get("bbox") for s in segments):
        return await link_figure_gemini_async(session, api_key, image_bytes, media_type, caption, segments, model=model)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img.size
    clusters = _partition_and_cluster(segments, target_chunk_size)

    async def _one_chunk(cluster):
        x0 = max(0, min(s["bbox"][0] for s in cluster) - pad_px)
        y0 = max(0, min(s["bbox"][1] for s in cluster) - pad_px)
        x1 = min(W, max(s["bbox"][2] for s in cluster) + pad_px)
        y1 = min(H, max(s["bbox"][3] for s in cluster) + pad_px)
        crop = img.crop((x0, y0, x1, y1))
        crop = _mask_foreign_messy_regions(crop, x0, y0, cluster, segments)
        local_segs = _local_segments_for_chunk(cluster, x0, y0)
        for i, s in enumerate(local_segs):
            s["_mark"] = _mark_for(i)
        crop = _draw_marks(crop, local_segs)
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        prompt = _build_prompt_marked(caption, local_segs)
        return await _link_chunk_marked_async(session, api_key, buf.getvalue(), "image/png", prompt, model)

    chunk_results = await asyncio.gather(*[_one_chunk(c) for c in clusters])
    merged = []
    for r in chunk_results:
        merged.extend(r)
    return merged


def _enforce_non_product_nulls(links):
    """Hard guarantee, independent of the model's own compliance: a segment
    marked is_scoped_product=False can never carry yield_percent/conditions
    data. Confirmed necessary on a real case
    (redox_neutral_2024/images/page5_fig0.png segment 55, the panel's own
    'R'-CH2-CF3' scheme template) where the model set is_scoped_product
    correctly-ish but still copied a neighboring real product's yield onto
    it -- the prompt instruction alone wasn't sufficient, so this is
    enforced in code rather than trusted.

    compound_id is deliberately EXCLUDED from this guarantee (was included
    until a real audit case: nickelocene_2025, compound "1a", a starting
    material with its own real, permanently-numbered compound ID -- the
    model correctly read "1a" and said so in its own flag text, but this
    function was blanket-nulling compound_id right alongside yield/
    conditions whenever is_scoped_product was False, discarding a real,
    non-borrowed identifier for no reason connected to the actual failure
    mode above, which was specifically about yield being copied from a
    neighbor. A starting material genuinely has no yield in this scheme
    (that invariant holds universally), but it can very much have its own
    real compound number -- so only yield_percent/conditions get the
    code-level guarantee; compound_id is trusted to the model (and to
    si_structure_link.py-style downstream review) the same as for any
    other field."""
    for link in links:
        if not link.get("is_scoped_product", True):
            link["yield_percent"] = None
            link["conditions"] = None
    return links


def link_figure(paper_dir, image_path, client, link_fn=link_figure_claude):
    segments = _segments_for(paper_dir, image_path)
    if not segments:
        return None
    caption = _caption_for(paper_dir, image_path)
    full_path = paper_dir / image_path
    image_bytes = full_path.read_bytes()
    return link_fn(client, image_bytes, "image/png", caption, segments)


async def link_figure_async(paper_dir, image_path, async_client, link_fn=link_figure_claude_async):
    segments = _segments_for(paper_dir, image_path)
    if not segments:
        return None
    caption = _caption_for(paper_dir, image_path)
    full_path = paper_dir / image_path
    image_bytes = full_path.read_bytes()
    return await link_fn(async_client, image_bytes, "image/png", caption, segments)


def link_paper(paper_dir, client, link_fn=link_figure_claude):
    out = {}
    for image_path in load_candidates(paper_dir):
        links = link_figure(paper_dir, image_path, client, link_fn=link_fn)
        if links:
            out[image_path] = links
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", help="Only process this paper directory name")
    parser.add_argument("--image", help="Only process this one image_path (requires --paper)")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    paper_dirs = sorted(p for p in PAPERS_DIR.iterdir() if p.is_dir())
    if args.paper:
        paper_dirs = [p for p in paper_dirs if p.name == args.paper]

    for paper_dir in paper_dirs:
        if args.image:
            links = link_figure(paper_dir, args.image, client)
            result = {args.image: links} if links else {}
        else:
            result = link_paper(paper_dir, client)
        if not result:
            continue
        out_path = paper_dir / "reaction_links.json"
        json.dump(result, open(out_path, "w"), indent=2)
        total = sum(len(v) for v in result.values())
        print(f"{paper_dir.name}: linked {total} segments across {len(result)} figures -> {out_path}")


if __name__ == "__main__":
    main()
