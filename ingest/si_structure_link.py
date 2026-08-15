"""Stage 5, second slice: structure-to-identifier linking for individually-
drawn SI structures (as opposed to reaction_link.py's scope-table/grid
case).

Widened scope from reaction_link.py's original has_grid_layout=1-only
candidate set: most has_structures=1 figures are single, individually-drawn
SI compound entries, not scope-table grids, and every one of them still
needs its alphanumeric identifier (if any) linked before Stage 6 (entity
normalization) can do anything useful. A deterministic PDF-text-layer
proximity match (nearest real text to the structure's own bbox) was tried
first and rejected: tested against 879 real has_structures=1,
has_grid_layout=0 candidates across 6 papers, it hit only 5.8% (51/879).
Failures split two ways: (1) many "single structure, non-grid" figures are
actually screening/optimization tables with an embedded structure icon, not
per-compound entries at all -- a candidate-selection problem, not a text-
proximity problem; (2) many genuine per-compound entries are literature
compounds named only by IUPAC name, with no numeric ID printed anywhere
near them at all (correctly nothing to find, per the three-tier model
below) -- proximity search can't resolve this even in principle, since
matching a bare IUPAC name to an ID established elsewhere requires
structure-based cross-referencing, not text search.

Three-tier model of what "identifier" even means here, confirmed against
real SI pages before designing anything: Set A (every drawn structure,
including unlabeled mechanism intermediates or common-name reagents) superset
Set B (structures with a numeric/alphanumeric ID -- e.g. isolated, numbered
products) superset Set C (structures with full characterization data in SI --
NMR/HRMS/etc, a strict subset of B, since commercial/known-literature
compounds routinely get an ID and a structure but no individual write-up).
Every stage here should treat "stopped at an earlier tier" as a correct,
expected terminal state, not a failure to flag -- an unlabeled mechanistic
intermediate correctly getting compound_id=null is not an error.

Context fed to the model: the *whole page's* real PDF text (not a fixed
pixel-margin clip, not the containing section/record's full text either --
see below) plus the structure crop. Two coarser alternatives were tried and
rejected first: a fixed-margin bbox clip (grabs neighboring prose/headers
unpredictably) and si_parse.py's record-level attach_figures() (respects
real section boundaries, but inherits Stage 2's heading-detection
reliability -- broken outright for one corpus paper -- and even where it
works, a record spanning many pages with no detected sub-headings can
attach 100+ unrelated figures to itself, e.g. one "NMR Spectroscopic Data"
record in suzuki_iron_2024 owns 113 figures across pages 326-401). Whole-
page text sidesteps both: it needs no heading detection to work at all, and
never crosses a page boundary into unrelated content. Validated at 24/24
correct on a hand-picked diverse sample (numbered products, unnumbered
reagents, mechanistic intermediates, screening-table entries, generic
R-group templates), each with correct null-vs-real-ID discrimination.

CRITICAL prompt-design lesson, found by benchmarking cheaper models against
this same task: the first version of RECORD_ID_TOOL used "e.g. '3d'" as the
compound_id field's example value. Every tested model that fabricated an ID
where the correct answer was null fabricated the *literal string* "3d" --
across four unrelated papers, in figures with no such compound anywhere on
the page (a mechanistic scheme's lettered intermediate, generic R-group
templates, unlabeled reagents). This wasn't scattered noise: Qwen3-VL-8B-
Instruct alone regressed from 90.9% to 66.7% agreement on the exact same
samples with the "3d"-example prompt vs. the null-example-free one below,
and Qwen3-VL-30B-A3B-Instruct (non-thinking) showed the identical pattern
(75.8% vs the thinking variant's 90.9% on the same underlying model family).
The schema's own example value was anchoring weaker instruction-following
models into treating it as a plausible default guess under uncertainty.
Fixed by removing the concrete example and adding an explicit
never-guess/never-placeholder instruction (both in the schema description
and the user-facing prompt) -- confirmed this fix alone took Qwen3-VL-8B
from disqualified (repeated fabrication) to competitive with every other
model tested. Do not reintroduce a concrete example value into this schema
without re-testing every candidate model against the null cases above.

Multi-provider benchmark (66 samples, hardened prompt, vs. Claude Sonnet 5
as reference): Claude Haiku 4.5 and Gemini 3.7 Flash tied for best accuracy
(95.5%); Llama 4 Maverick and Qwen3-VL-8B-Instruct tied at 90.9% agreement,
but Maverick was surprisingly slow (~5.4s/call, close to "thinking"-mode
latency despite doing no extended reasoning) while Qwen3-VL-8B matched
Haiku's speed (~1.2s/call) at roughly 1/9th the per-token cost.
Qwen3-VL-30B-A3B-Instruct (non-thinking) was the outlier failure at 75.8%
even with the hardened prompt -- worse than its own smaller 8B sibling,
confirming the earlier "3d" fabrication susceptibility was about the
thinking/non-thinking mode, not raw parameter count.

Also benchmarked on Stage 5's *other* tier (reaction_link.py's dense
multi-segment grid figures, not this module's single-structure case) to
find the real Haiku/Sonnet routing boundary: on a 38-segment scope-table
grid, Haiku dropped to 50% agreement with Sonnet (real crop-to-label
mismatching on dense multi-structure disambiguation, not just occasional
misses -- a different failure mode than anything seen on the single-
structure case), while Gemini 3.7 Flash held up much better at 76.3%.
Confirms the tier split is real and necessary: route by segment count
(1 segment -> lightweight model, 2+ -> Sonnet), don't assume a model that's
good at single-structure linking generalizes to multi-structure grids.

Current default: Gemini 3.7 Flash (via OpenRouter), not Haiku -- ties
Haiku's single-structure accuracy, is meaningfully better on the grid tier,
and costs less (~$0.375/$1.875 per 1M in/out vs Haiku's $1/$5). No
link_structure_gemini() implementation exists yet in this module (only the
Claude path is wired up) -- add one following link_structure_claude()'s
signature/contract before switching the pipeline default.

Two more real disagreements found by manual review after the 66-sample
benchmark, both fixed in the live prompt above (not just documented as
declined, unlike the "3d" example):
1. Letter-prefix IDs (e.g. a ligand code like "L11") were being missed --
   the original description only described the number-first pattern
   ("3d"-shaped). Broadening the description to cover both forms fixed it
   for every model tested, but also caused a new regression: models started
   accepting bare single-letter mechanistic-scheme labels (e.g. "A" in a
   catalytic cycle diagram) as if they were compound IDs too. Confirmed
   neither Haiku 4.5 nor Qwen3-VL-8B reliably followed an explicit
   prompt-level "a bare single letter is never an ID" rule even when stated
   outright -- so this is enforced by _looks_like_real_id() in code instead
   (a real ID is always <=8 chars, purely alphanumeric, and contains at
   least one digit -- catches bare letters, full IUPAC names, and
   "Compound 21"-style prefixed strings for free, same category of fix as
   #2 below).
2. When a page has multiple named compounds (e.g. "Synthesis of 2a" and
   "Synthesis of 2b" both present), a model can find the right structure
   visually distinguishable (confirmed on a real case: a drawn 4-methoxy-
   phenyl boronate correctly matches "2a"'s procedure text, not "2b"'s
   plain-phenyl one) but still return null out of caution rather than
   commit -- and separately, a model can mistake the drawn structure for
   the *starting material* used in a synthesis rather than trusting that a
   structure under a "Synthesis of X" heading is X's own product structure
   by convention (confirmed: a boronate ate-complex product whose NMR
   shows a diagnostic tert-butyl signal was still read as "just the
   starting material" without this hint, even though the structure sits
   directly under "Synthesis of 2a"). Section-level text scoping (finding
   the nearest preceding "Synthesis of <ID>"-style heading on the same
   page, via real per-line PDF text positions, and clipping to that span)
   fixes the first case, but the explicit product-vs-starting-material
   CONVENTION line added to PROMPT_TEMPLATE below was sufficient on its
   own to fix the confirmed case even with the *full*, unscoped page (both
   "2a" and "2b" present) -- shipped the simpler fix; the section-scoping
   approach works too but wasn't needed given the prompt fix alone
   resolved the concrete case, so it isn't implemented as code here.
"""

import base64
import json
import re
from pathlib import Path

import fitz
import requests

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"

# Best Claude option (95.5% single-structure accuracy, ~1.2s/call), but NOT
# the pipeline's overall recommended default -- Gemini 3.7 Flash ties this
# on accuracy, beats it on the grid tier, and costs less. See module
# docstring; this constant exists for when the Claude path specifically is
# wanted (e.g. no OpenRouter key configured).
CLAUDE_LIGHTWEIGHT_MODEL = "claude-haiku-4-5-20251001"

# Recommended pipeline default -- see module docstring's benchmark summary.
GEMINI_MODEL = "google/gemini-3.7-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

RECORD_ID_TOOL = {
    "name": "record_id",
    "description": (
        "Report the compound's alphanumeric identifier (if any) and "
        "associated data found on this page."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "compound_id": {
                "type": ["string", "null"],
                "description": (
                    "The exact bold alphanumeric identifier printed immediately next to THIS "
                    "structure on THIS page. Papers use different ID conventions for different "
                    "kinds of entities -- e.g. isolated products/entries are often a number "
                    "optionally followed by lowercase letters, while ligands, catalysts, or other "
                    "reagent series are often a letter prefix followed by a number. Either form "
                    "counts. Return null if no such identifier is printed anywhere near this exact "
                    "structure -- do not guess, infer, reuse an identifier from a different "
                    "compound, or invent one."
                ),
            },
            "yield_percent": {
                "type": ["string", "null"],
                "description": "The yield text as printed, or null if none is shown.",
            },
            "note": {
                "type": "string",
                "description": "One sentence: what this structure actually is on this page (e.g. 'named reagent, no ID' or 'one entry in a screening table').",
            },
        },
        "required": ["compound_id", "yield_percent", "note"],
    },
}

PROMPT_TEMPLATE = (
    "Here is a cropped chemical structure image, and below it the full text of the SI page it "
    "was cropped from. Find this exact structure's alphanumeric compound identifier and yield if "
    "printed on this page. IMPORTANT: many structures on SI pages have NO identifier at all (e.g. "
    "named reagents, mechanistic intermediates, generic templates) -- if you do not find a "
    "specific identifier clearly associated with THIS exact structure, you must return null. "
    "Never guess or supply a placeholder value. Identifiers can take different forms (e.g. a "
    "number followed by letters, or a letter prefix followed by a number) depending on what kind "
    "of compound this is -- do not assume only one format is valid. CONVENTION: by strong "
    "convention in chemistry SI documents, a structure drawn directly under/beside a heading like "
    "'Synthesis of 2a' depicts the PRODUCT of that synthesis (i.e. is compound 2a itself), not a "
    "starting material or intermediate -- trust this convention even if the drawn structure looks "
    "structurally simpler than you might expect from the reported NMR data.\n\nPage text:\n{page_text}"
)

# A real compound/entry/ligand ID in this corpus is always short, purely
# alphanumeric, and contains at least one digit (3d, 22a, L11, 1, 44, ...).
# Added as a deterministic backstop rather than more prompt rules, after
# finding that neither Haiku 4.5 nor Qwen3-VL-8B reliably followed an
# explicit "a bare single letter is never an ID" prompt instruction (both
# kept returning mechanistic-scheme intermediate letters like "A"/"B" even
# with the rule stated outright) -- matches this codebase's existing
# pattern (see reaction_link.py's is_scoped_product handling) of backing a
# prompt instruction with a hard code guarantee once it's confirmed
# insufficient alone. Also catches other observed fabrication shapes for
# free: full IUPAC names, "Compound 21"-style prefixed strings, and
# multi-word garbage all fail this same check (no digit, or contains
# whitespace/punctuation), without adding a single word to the prompt --
# deliberately not tightening the prompt further, since more constraints
# measurably hurt the weaker model's accuracy (confirmed: Qwen3-VL-8B
# dropped when the schema grew more rules, while Haiku still missed the
# same case anyway -- the rule wasn't reliably learnable at either tier
# tested, so code is the more robust fix regardless of model choice).
_PLAUSIBLE_ID_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")


def _looks_like_real_id(compound_id):
    if compound_id is None:
        return True  # null is always a valid answer, nothing to reject
    if not _PLAUSIBLE_ID_RE.match(compound_id):
        return False  # contains whitespace/punctuation, or too long
    return any(ch.isdigit() for ch in compound_id)


def _page_text_for_image(paper_dir, image_path):
    """Whole-page real PDF text (not OCR -- these are digitally-generated
    PDFs) for the SI page a given structure image was extracted from. See
    module docstring for why this is whole-page rather than a bbox clip or
    a section/record's full (possibly multi-page) span."""
    raw = json.load(open(paper_dir / "SI_raw_extraction.json"))
    page_number = next(
        page["page_number"]
        for page in raw["pages"]
        for img in page["images"]
        if img["path"] == image_path
    )
    doc = fitz.open(paper_dir / "SI.pdf")
    return doc[page_number - 1].get_text()


def link_structure_claude(client, paper_dir, image_path, model=CLAUDE_LIGHTWEIGHT_MODEL):
    """Links one individually-drawn SI structure to its alphanumeric
    identifier (and yield, if shown), using the whole SI page's real text
    as context. Returns {"compound_id": str|None, "yield_percent": str|None,
    "note": str}."""
    page_text = _page_text_for_image(paper_dir, image_path)
    image_bytes = (paper_dir / image_path).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode()

    resp = client.messages.create(
        model=model,
        max_tokens=500,
        tools=[RECORD_ID_TOOL],
        tool_choice={"type": "tool", "name": "record_id"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_TEMPLATE.format(page_text=page_text)},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
            ],
        }],
    )
    tool_use = next(b for b in resp.content if b.type == "tool_use")
    result = tool_use.input
    if not _looks_like_real_id(result.get("compound_id")):
        result["compound_id"] = None
    return result


# OpenRouter uses OpenAI-style function-calling tools, not Anthropic's
# input_schema shape -- same fields, different wrapper. Built once from
# RECORD_ID_TOOL rather than duplicated, so the two providers can never
# drift out of sync with each other.
_RECORD_ID_TOOL_OPENROUTER = {
    "type": "function",
    "function": {
        "name": RECORD_ID_TOOL["name"],
        "description": RECORD_ID_TOOL["description"],
        "parameters": RECORD_ID_TOOL["input_schema"],
    },
}


def link_structure_gemini(api_key, paper_dir, image_path, model=GEMINI_MODEL):
    """Same contract as link_structure_claude, via OpenRouter. Recommended
    pipeline default -- see module docstring's benchmark summary."""
    page_text = _page_text_for_image(paper_dir, image_path)
    image_bytes = (paper_dir / image_path).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode()

    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "tools": [_RECORD_ID_TOOL_OPENROUTER],
            "tool_choice": {"type": "function", "function": {"name": "record_id"}},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEMPLATE.format(page_text=page_text)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }],
        },
        timeout=90,
    )
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"OpenRouter error for {model}: {data['error']}")
    tool_calls = data["choices"][0]["message"].get("tool_calls")
    if not tool_calls:
        raise RuntimeError(f"{model} returned no tool call for {image_path}: {data['choices'][0]['message'].get('content')}")
    result = json.loads(tool_calls[0]["function"]["arguments"])
    if not _looks_like_real_id(result.get("compound_id")):
        result["compound_id"] = None
    return result
