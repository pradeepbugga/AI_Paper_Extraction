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

The nontrivial parts:
- PyMuPDF's `blocks` text mode (used in Stage 1) strips font metadata, so a
  heading and the paragraph immediately following it often land in the same
  block with no visual gap between them. Stage 2 re-reads each page with
  `dict` mode to get per-line font name/size, and splits blocks into
  heading/body units by font (`HardingText-Bold` ≥10pt = top-level heading,
  `HardingText-Semibold` = subsection heading).
- Small text blocks that are actually chemical-structure labels sitting
  inside a figure (not body text) are dropped by checking overlap against
  the figure bounding boxes already found in Stage 1.
- Running headers/footers (journal name, DOI banner, page number) are
  filtered by page-position band plus a small text-match rule for the
  masthead logo.
- Figure captions (`Fig. N | ...`) are matched back to the nearest figure
  image above them on the page.

```
python3 ingest/section_parse.py data/papers/suzuki_iron_2024
```

Outputs `sections.json`: front-matter paragraphs, a list of top-level
sections (each with paragraphs and nested subsections), and a list of
figures with captions linked to their `image_id`/`image_path` from Stage 1.
