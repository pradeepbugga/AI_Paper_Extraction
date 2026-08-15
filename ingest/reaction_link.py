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
"""

import argparse
import base64
import json
from pathlib import Path

import anthropic

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"

CLAUDE_MODEL = "claude-sonnet-5"

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
                                "table/scope (even if its SMILES was misread as generic -- judge this from what the "
                                "image actually shows, not from the has_generic_substituent hint). False if the "
                                "drawing itself is a reaction-scheme template, a reagent/catalyst/starting-material "
                                "structure, or any other component shown for context rather than a specific "
                                "numbered/lettered product -- e.g. the general 'R-CF3' arrow-diagram template drawn "
                                "once at the top of a panel is False even though a specific product below it "
                                "sharing a similar structure is True."
                            ),
                        },
                        "compound_id": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "The bold compound/entry label printed next to this structure (e.g. '3a'), or null if none is visible. Must be null when is_scoped_product is false.",
                        },
                        "yield_percent": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "The yield text as printed (e.g. '87%', 'trace', 'NR'), or null if none is shown for this structure. Must be null when is_scoped_product is false.",
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
        flag_desc = f" [{', '.join(flags)}]" if flags else ""
        segment_lines.append(f"- segment_path={s['segment_path']!r} {bbox_desc}, SMILES={s['smiles']!r}{flag_desc}")

    caption_desc = f'Figure caption: "{caption}"' if caption else "No caption text was found for this figure."

    return f"""This image is a scope-table/scheme figure from a chemistry paper. Each
segment below is a structure Stage 3 already extracted and located within
this image (by pixel bbox). Your job is to link each one to the data printed
next to it in the image: its compound/entry-ID label, yield, any
condition-variant annotation specific to it, and (if the figure has lettered
sub-panels like (a)/(b)/(c)) which panel it belongs to.

For every segment, first decide is_scoped_product from what the image
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
  ordinary. In this case is_scoped_product is false, and compound_id/
  yield_percent/conditions MUST all be null -- never reuse or borrow a
  nearby real product's label/yield/conditions for a non-product segment,
  even if they happen to sit right next to each other or share a similar
  structure.

{caption_desc}

Segments already extracted from this figure:
{chr(10).join(segment_lines)}

Use the segment bboxes to find each structure in the image, then read the
label/yield/conditions text printed next to it. Call record_links with one
entry per segment_path listed above (all of them, even if you can't find a
label for one -- use null fields and set flag to explain why). Do not invent
data that isn't visibly printed in the image."""


def link_figure_claude(client, image_bytes, media_type, caption, segments, model=CLAUDE_MODEL):
    """Provider-specific implementation #1. Returns a list of link dicts, one
    per segment_path. See module docstring for the provider-agnostic contract
    a second model's implementation should match. `model` defaults to Sonnet
    5 but accepts any Claude model string -- used to benchmark Haiku 4.5
    against Sonnet 5 on the same grid figures, see handoff notes."""
    image_b64 = base64.standard_b64encode(image_bytes).decode()
    prompt = _build_prompt(caption, segments)

    # A dense scope table can carry 50+ segments -- confirmed hitting the
    # old 4096-token cap mid-tool-call on a real 59-segment figure
    # (redox_neutral_2024/images/page5_fig0.png, stop_reason="max_tokens"),
    # truncating the JSON before the tool call completed. Streamed so a
    # large response doesn't risk an HTTP timeout either.
    with client.messages.stream(
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
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"record_links call hit max_tokens with {len(segments)} segments -- "
            "response truncated, raise max_tokens or split the figure's segments across calls"
        )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return _enforce_non_product_nulls(tool_use.input["links"])


def _enforce_non_product_nulls(links):
    """Hard guarantee, independent of the model's own compliance: a segment
    marked is_scoped_product=False can never carry compound_id/yield_percent/
    conditions data. Confirmed necessary on a real case
    (redox_neutral_2024/images/page5_fig0.png segment 55, the panel's own
    'R'-CH2-CF3' scheme template) where the model set is_scoped_product
    correctly-ish but still copied a neighboring real product's yield onto
    it -- the prompt instruction alone wasn't sufficient, so this is
    enforced in code rather than trusted."""
    for link in links:
        if not link.get("is_scoped_product", True):
            link["compound_id"] = None
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
