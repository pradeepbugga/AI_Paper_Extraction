# AI_Paper_Extraction

Multimodal pipeline for extracting structured, provenance-linked scientific
knowledge (synthesis conditions, characterization data, reaction outcomes)
from chemistry/materials literature — text, figures, and tables.

## Status

Early build, in progress. Working through the pipeline one stage at a time:

1. **PDF ingestion** (done) — parse text blocks and figures out of raw PDFs,
   including Supporting Information.
2. **Section/layout parsing** (done) — group raw text blocks into
   headings/paragraphs/figure captions and assemble a document structure,
   for both the main text and SI.
3. Figure understanding (classification, structure recognition)
4. Table extraction
5. Schema-driven LLM extraction
6. Entity normalization (canonical IDs)
7. Provenance-linked knowledge graph

## Stage 1: PDF ingestion

`ingest/pdf_ingest.py` extracts, per page:
- Text blocks (position + content)
- Figures

The nontrivial part: journal figures (chemical structures, plots) are almost
always drawn as vector graphics — hundreds of individual path/line/fill
operations — not embedded raster images. Naively pulling embedded images
misses them entirely. The script instead clusters nearby vector-drawing
bounding boxes on each page into figure-sized regions, then rasterizes each
region to PNG.

Test paper: Rowsell et al., "The iron-catalysed Suzuki coupling of aryl
chlorides," *Nature Catalysis* 7, 1186-1198 (2024) — chosen for its reaction
scheme, large substrate-scope figures, and mechanistic plots.

```
pip install -r requirements.txt
python3 ingest/pdf_ingest.py data/papers/suzuki_iron_2024
```

Outputs `raw_extraction.json` (per-page text blocks + figure metadata) and an
`images/` directory of rasterized figures, inside the paper's data directory.

### Supporting Information

Materials chemistry papers lean heavily on their SI PDF — most of the actual
synthesis procedures and compound characterization data (NMR shifts, IR,
HRMS, melting points) lives there, not in the main text. If `SI.pdf` is
present alongside `paper.pdf` in a paper's directory, the same run also
ingests it as its own document (`SI_raw_extraction.json`, `images_SI/`),
with every page tagged `"source": "main"` or `"source": "SI"` so downstream
stages can tell which document a given span of text or figure came from.
Image IDs and output filenames are kept in per-source namespaces since both
documents restart page numbering at 1.

Tested on two SI PDFs of very different scale — 111 pages (`copper_iron_2025`,
mostly text: procedures, per-compound characterization data, and NMR
spectrum plots drawn as dense vector line traces rather than chemical
structures) and 406 pages (`suzuki_iron_2024`, which also includes a large
block of raw DFT-calculation Cartesian coordinates). The same vector-drawing
clustering from the main text handles NMR spectrum traces correctly despite
them being a structurally different kind of vector drawing (one continuous
line, thousands of tiny path segments, vs. discrete chemical-structure
bonds) — verified by rendering a crop and confirming it's a clean,
correctly-cropped spectrum.

```
python3 ingest/pdf_ingest.py data/papers/copper_iron_2025
```

## Stage 2: section/layout parsing

`ingest/section_parse.py` turns the flat per-page text blocks from Stage 1
into a document structure: headings, paragraphs, and figure captions, in
reading order.

The first version of this hardcoded literal font names and a page-position
band tuned to one journal (Nature Catalysis), and it silently produced an
empty structure on a paper from a different publisher (ACS *Org. Lett.*) —
none of the hardcoded strings matched. It's now driven by document-relative
statistics instead of per-publisher constants:

- **Heading detection**: PyMuPDF's `blocks` text mode strips font metadata,
  so a heading and the paragraph right after it often land in the same block
  with no visual gap. Stage 2 re-reads each page with `dict` mode for
  per-line font size/boldness, computes the document's own body-text size
  (the size with the most non-bold characters), and ranks heading tiers the
  way a human reader would: font size first, then a numbered-list prefix or
  ALL CAPS as a tie-breaker when two tiers share the same size (needed once
  SI documents entered the picture — see Stage 2b below — since not every
  document expresses every heading tier with a size difference). A style
  tier only becomes a real heading *level* if it recurs across ≥2 distinct
  blocks (a one-off large bold block, like a title or byline, isn't a
  heading — real section headings are a family of different short blocks
  sharing one style); same-as-body-size tiers need ≥4, since that's a
  weaker signal, and are only eligible at the very start of a block, so a
  coincidentally bold-and-short wrapped line mid-paragraph can't qualify. A
  candidate is also rejected if it reads like a sentence (ends in
  `.`/`?`/`!`, not all-caps, >3 words) or runs over ~20 words, since real
  headings are short phrases, not emphasized prose.
- **Figure-embedded text**: small blocks that are actually chemical-structure
  labels sitting inside a figure (not body text) are dropped by checking
  overlap against the figure bounding boxes found in Stage 1.
- **Running headers/footers**: filtered per-line (not per-block, since a
  running header can end up fused into a content block on the page with no
  gap between them) by page-position band, plus text that recurs across a
  large share of pages — matched both verbatim and with a trailing page
  number stripped, since footers often differ only by that number.
- **Two-column layout**: detected per page from the actual gap in block
  x-positions rather than assumed at the page midpoint, so it also degrades
  gracefully to single-column pages.
- **Cross-page/column paragraphs**: a paragraph cut off at a page or column
  boundary (no terminal punctuation at the break) is stitched back onto the
  next block instead of appearing as two separate paragraphs.
- **Figure captions**: `Fig./Figure/Scheme/Table/Chart N` (case-insensitive)
  are matched to the nearest figure image on the page — checking both above
  and below the caption (publishers differ on which side it's on), and
  restricted to images sharing the caption's column first, so a two-column
  page doesn't grab a same-height figure from the other column.
- **References**: parsed by [GROBID](https://github.com/kermitt2/grobid)
  (`ingest/grobid_client.py`, `/api/processReferences`) into structured
  entries (authors, journal, volume, pages, year, DOI) instead of a prose
  paragraph blob — see below. Falls back to the prose "References" section
  if GROBID isn't running.

Validated against two papers from different publishers with different
typography: Rowsell et al., *Nature Catalysis* 7, 1186-1198 (2024)
(`suzuki_iron_2024`) and Roy et al., *Org. Lett.* 28, 32-38 (2026)
(`copper_iron_2025`).

```
python3 ingest/section_parse.py data/papers/suzuki_iron_2024
python3 ingest/section_parse.py data/papers/copper_iron_2025
```

Outputs `sections.json`: front-matter paragraphs, a list of top-level
sections (each with paragraphs and nested subsections), a list of figures
with captions linked to their `image_id`/`image_path` from Stage 1, and a
list of structured references.

### References via GROBID

An evaluation of GROBID as a full replacement for the hand-rolled section
parsing above found it excellent for bibliographic metadata but inconsistent
at section/figure segmentation across publishers (it missed nearly all
structure on the ACS paper and found zero figures there), so it's used
narrowly for just the one thing it's clearly better at: parsing the
reference list. GROBID runs as a local Docker container:

```
docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
```

`fetch_references()` posts the PDF to GROBID's `processReferences` endpoint
and parses the returned TEI XML `biblStruct` entries. If GROBID isn't
reachable, `references` comes back empty and the prose "References" section
stays in the section tree instead of being silently dropped.

## Stage 2b: SI section parsing

`ingest/si_parse.py` parses `SI_raw_extraction.json` the same way, but SI
documents have a different shape than the main text: instead of a handful of
prose sections, they're organized around dozens to hundreds of repeating
*compound records* (a compound name or procedure label, followed by its
characterization data and NMR spectra). Forcing that into the main text's
prose section/subsection tree would lose the thing that actually matters
here — "text blob and spectra per compound" — so `si_parse.py` reuses
`section_parse.py`'s `parse_page`/`assemble_sections` unmodified to build the
same kind of heading tree as the main text, then does one SI-specific thing:
any *leaf* heading (nothing nested under it) is recast as a flat `record`
(name + text blob + linked figures) instead of a subsection with a single
paragraph. No SI-specific parsing logic exists anymore — it's a shape
transform on top of the exact same document-relative heading detection used
everywhere else.

That heading detection had to generalize further to get here. The original
version required a heading to be *strictly larger* than body text — true for
every main-text heading tier seen so far, but SI documents don't always
follow it:

- **Nature's SI** uses genuine font-size-tiered headings, same convention as
  its main text (`Supplementary Methods` at 16pt containing `General
  Considerations` at 14pt, both clearly larger than 11pt body) — real
  nesting, up to 4 levels deep in practice.
- **ACS's SI** uses flat body-size-bold headings distinguished only by
  numbering (`1. General experimental`, `12. Experimental procedures...`,
  all at the exact same size as body text) — no size signal at all.

`compute_heading_levels`/`classify_line` now rank heading tiers the way a
human reader would: font size first, then secondary cues (a numbered prefix,
ALL CAPS) as tie-breakers within equal sizes — so a same-size tier
distinguished only by numbering still ranks as its own level instead of
collapsing into body text. The relaxed (non-strictly-larger) tier is gated
more conservatively than the strict one, since it's a weaker signal:
eligible only at the very start of a block (so a coincidentally-short,
coincidentally-bold last line of a wrapped paragraph — e.g. a chemistry
compound ID like `3a` sitting alone at a line break — can't qualify; a real
heading always starts a new block), and requires more repetitions across the
document before being trusted as a real level (a bold "label lead-in" phrase
in boilerplate prose, like "Correspondence and requests... :", can
coincidentally repeat 2-3 times without being a heading).

Two more general fixes came out of validating this against both SI
documents:

- **Trailing colon ≠ end of a sentence.** The shared sentence-suppression
  check (used to reject a bold-but-prose-like fragment as "not a heading")
  treated a trailing colon as terminal punctuation, which silently swallowed
  every ACS compound header (`Synthesis of ... (14):`) into body text. A
  colon introduces something, it doesn't end a declarative sentence — never
  a valid "reject as heading" signal in the first place.
- **A Table of Contents page isn't a real container.** Its entries are an
  index of headings that appear again later, not genuine children of "Table
  of Contents" — no typographic signal can tell a TOC's structure apart from
  real nesting, so without an explicit check, a TOC page could validate a
  heading level that then never legitimately closes for the rest of the
  document (everything after it nests one level deeper, forever). Detecting
  a dedicated TOC page and excluding it from heading-level validation is no
  more publisher-specific than recognizing numbered lists or figure-caption
  prefixes as conventions — it's a standard document convention independent
  of either paper here.

Figures are attached to whichever section/record is "open" at a given page
(the most recent heading at or before it) — coarse, but matches how these
documents are laid out: a compound's structure and spectra sit on or right
after its own heading.

```
python3 ingest/si_parse.py data/papers/suzuki_iron_2024
python3 ingest/si_parse.py data/papers/copper_iron_2025
```

Outputs `SI_sections.json`. Validated on all three SI documents so far: 250
records (Suzuki), 152 records (copper-iron), 159 records (`suzuki_nickel_2026`,
see below) — fewer than an earlier hand-rolled, SI-specific version for the
first two, but a quality improvement, not a regression: verified zero
orphaned figures (144/144 images in Suzuki's NMR section attached to a
record) and confirmed the drop is entirely from removing noise a
publisher-specific heuristic had let through (raw DFT-coordinate-dump
fragments misread as headings, and NMR sub-spectra like `11B NMR` that used
to spuriously split off from their parent compound now correctly merge back
in). Known limitation, out of scope for this stage: dense comparison tables
and reaction-scheme labels still fragment into spurious low-content records
— real table extraction is Stage 4, not something this layout-level parsing
is expected to solve.

## Validated against a third publisher

A third paper (Lu et al., *ACS Catal.* 16, 2417-2426 (2026),
`suzuki_nickel_2026` — main text + 159-page SI) surfaced new conventions none
of the first two papers had, each fixed as a further generalization of the
existing document-relative signals rather than a new special case:

- **Leading marker glyph as a heading signal.** ACS section headings are
  preceded by a bullet-like glyph (`■`) in a completely different font/size
  than the heading text. This is now a positive signal in
  `heading_style_key`, ranked alongside a numbered prefix and ALL CAPS — the
  specific character varies by publisher and can't be hardcoded, but "a
  short, non-alphanumeric, differently-styled span immediately before the
  heading text" is a generic convention. It's found relative to wherever the
  dominant-style run of the line actually starts, not just at index 0, so it
  survives the same fused-line pollution (`paper.■AUTHOR INFORMATION`, an
  unrelated preceding sentence's tail landing on the same PDF line with zero
  separating whitespace) that the marker itself needs to be robust to.
  Restricted to non-alphanumeric text specifically so a superscript
  affiliation number at a coincidental line-wrap point isn't mistaken for a
  decorative bullet.
- **A controlled vocabulary for chemistry-nomenclature italics.** Terms like
  `tert`, `sec`, `cis`, `trans`, `R`/`S`/`E`/`Z` are conventionally
  italicized inline (`lithium tert-butyl aryl boronates`) and are now kept
  verbatim when reconstructing a heading's clean text, rather than relying on
  their weight/size happening to match the surrounding heading style, which
  isn't something to count on in general.
- **Figure-region exclusion made length-aware.** Stage 1's figure detection
  is purely geometric — it has no idea what text is nearby, so an oversized
  or badly-merged figure region can coincidentally overlap real headings and
  even whole paragraphs of body prose. `block_in_figure` now only ever
  excludes short fragments (≤8 words for body text, ≤2 characters for
  anything already classified as a heading): genuine figure debris (a
  compound ID, a yield, a single-letter panel label) is always short;
  a real heading or a real sentence never is, regardless of geometric
  overlap with a bad figure box. This fixed two headings (`CONCLUSIONS`,
  `MATERIALS AND METHODS`) that a badly-merged figure region had been
  silently swallowing whole.
- **Asymmetric figure-clustering padding.** `cluster_drawing_rects` (Stage 1)
  now pads 25pt vertically but only 8pt horizontally. A multi-panel scheme
  (5 sub-panels stacked down a page with generous whitespace between them)
  needs a wide vertical gap tolerance to merge into one region; the same
  tolerance horizontally risks bridging across a two-column page's gutter
  and merging a figure with unrelated text in the other column. This also
  surfaced and fixed a crash: a drawing rect extending into the page's
  negative-coordinate margin/bleed area (previously too small on its own to
  matter) became part of a larger, otherwise-valid cluster after the wider
  vertical merge, and PyMuPDF's renderer rejects an out-of-bounds clip
  region — fixed by clipping cluster bounding boxes to the page's actual
  visible bounds before rendering.
- **Body size computed across all text, not just non-bold.** This SI
  document's characterization data — normally the clearest non-bold prose in
  a document — renders entirely bold, including the data itself, not just
  compound-name headers. Excluding bold text left the body-size estimate
  reflecting only a small, unrepresentative non-bold sliver (running
  headers, stray captions), well below the real compound-name headings —
  they never qualified as headings at all. Bold text is a small enough share
  of total characters in a normal prose document that including it doesn't
  move the mode; it only matters when, as here, it's the majority of the
  page.

The two earlier papers were re-validated against every change in this batch
and show zero regressions from their existing baselines. Known remaining
limitation: two figures vertically close together on the same page (e.g. a
scheme and a table six lines apart) can still merge into one region and
share a caption match — lower priority, since it degrades gracefully (both
captions still resolve to *a* relevant image) rather than losing content.

## Validated against a fourth paper (same publisher family)

A fourth paper (Daley-Dee et al., *Org. Lett.* 27, 197-201 (2025),
`miyaura_iron_2025` — main text + 83-page SI) is the same publisher as
`copper_iron_2025`, used as a check on whether the generalization actually
holds within a family rather than being tuned to one example of it. It did,
with no code changes needed: main text (3/3 real sections, 4 figures, 46
references via GROBID) and SI (10 sections, 120 records — including a
decimal-numbered convention, `3.2.3 Precatalyst screen`, not seen before and
not matched by `NUMBERED_PREFIX_RE`, correctly nested anyway via the size/
marker signals) both parsed correctly on the first run.

One thing this surfaced worth noting as a genuine document-type difference,
not a detection gap: *Organic Letters* is a short-communication format with
a strict length limit, so a paper like this one has no `RESULTS AND
DISCUSSION` heading at all — the whole body is continuous discussion prose,
with only the standardized back-matter sections formally headed. Two
`General Procedure for...` subheadings inside that prose (9.5pt, just under
10pt body size, occurring only twice) don't get elevated to heading status —
the procedure text itself is preserved in the body, just not organized under
them. Investigated and left as-is rather than risk a regression: loosening
either the size threshold or the recurrence requirement to catch a
twice-occurring, unmarked, non-numbered, non-caps heading would also let
back in the earlier false positive (a bold "label lead-in" phrase in
boilerplate prose, e.g. "Correspondence and requests..."), which looks
identical by every signal currently available.
