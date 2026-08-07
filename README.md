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
  (the size with the most non-bold characters), and flags a line as a
  heading candidate if it's bold and larger than that — no hardcoded font
  names or size thresholds. A size tier only becomes a real heading *level*
  if it recurs across ≥2 distinct blocks (a one-off large bold block, like a
  title or byline, isn't a heading — real section headings are a family of
  different short blocks sharing one style). A candidate is also rejected if
  it reads like a sentence (ends in `.`/`?`/`!`, not all-caps, >3 words) or
  runs over ~20 words, since real headings are short phrases, not emphasized
  prose.
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
here — "text blob and spectra per compound" — so the output is a section
tree where each section holds a flat `records` list instead of only
paragraphs.

Two SI documents, two different heading conventions, discovered by testing
against both rather than assumed up front:

- **Nature's SI** uses genuine font-size-tiered headings, same as its main
  text (`Supplementary Methods` at 16pt containing `General Considerations`
  at 14pt, both clearly larger than 11pt body) — real nesting, up to 4 levels
  deep in practice (`Supplementary Methods` → `Preparation of Boronic
  Esters...` → `General Procedure 1` → per-compound record).
- **ACS's SI** uses flat body-size-bold headings distinguished only by
  numbering (`1. General experimental`, `12. Experimental procedures...`,
  all at the same size as body text) — no size signal available at all.

`classify_si_line` tries the size-tier signal first (reusing the same
document-relative heading-level machinery as the main text); if a bold line
doesn't belong to any size tier, it falls back to a numbered-list-prefix
check. Anything bold-and-short that's neither is a compound/procedure record
rather than a top-level section. Figures are attached to whichever
section/record is "open" at a given page (the most recent heading at or
before it) — coarse, but matches how these documents are laid out: a
compound's structure and spectra sit on or right after its own heading.

One bug this surfaced in the shared `is_sentence_like` heading-suppression
check: it treated a trailing colon as "looks like end of a sentence, reject
as heading," which silently swallowed every ACS compound header (`Synthesis
of ... (14):`) into body text — colons introduce something, they don't end a
declarative sentence, so they were never a valid signal for "not a heading"
in the first place. Fixed at the source since it's shared with main-text
parsing.

```
python3 ingest/si_parse.py data/papers/suzuki_iron_2024
python3 ingest/si_parse.py data/papers/copper_iron_2025
```

Outputs `SI_sections.json`. Validated on both SI documents: 458 records
(Suzuki, including 75 clean NMR-data records each linked to their spectrum
images) and 197 records (copper-iron, 73 experimental-procedure records + 74
NMR records each linked to exactly its 1H/13C spectra). Known limitation,
out of scope for this stage: dense comparison tables and reaction-scheme
labels fragment into spurious low-content "records" — real table extraction
is Stage 4, not something this layout-level parsing is expected to solve.
stays in the section tree instead of being silently dropped.
