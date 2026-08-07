# AI_Paper_Extraction

Multimodal pipeline for extracting structured, provenance-linked scientific
knowledge (synthesis conditions, characterization data, reaction outcomes)
from chemistry/materials literature — text, figures, and tables.

## Status

Early build, in progress. Working through the pipeline one stage at a time:

1. **PDF ingestion** (done) — parse text blocks and figures out of raw PDFs.
2. **Section/layout parsing** (in progress) — group raw text blocks into
   headings/paragraphs/figure captions and assemble a document structure.
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
