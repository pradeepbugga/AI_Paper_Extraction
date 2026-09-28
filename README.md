# AI Paper Extraction

A multimodal ML pipeline that reads chemistry/materials-science papers — main
text, figures, and Supporting Information — and turns them into structured,
provenance-linked data: chemical structures as machine-readable SMILES,
reaction conditions, yields, and characterization data, each one traceable
back to the exact page, figure, or table it came from.

## Highlights

- **2,219 chemical structures** extracted from paper figures across a
  7-paper corpus (up to ~400-page Supporting Information PDFs, 4 different
  publisher formats). Optical structure recognition (OCSR) error rate cut
  from a **9.8% baseline down to 1.6%** by integrating a stronger OCSR model
  (MolScribe) and building a correction pipeline around it; ambiguous/
  unresolved structures cut from 24.0% to **11.3%** over two full-corpus
  triage passes.
- **Two full-stack review tools built from scratch** (FastAPI + custom
  canvas/JS frontends) to route model uncertainty to a human reviewer in
  under a minute per structure: a freeform lasso-and-eraser tool for
  figures where the model merged multiple compounds into one crop (154
  reviewed), and a chemistry-aware sketch tool — an embedded Ketcher
  molecular editor wired to correctly preserve dative bonds and metal
  hapticity, which Ketcher's own SMILES export silently drops — for
  structures the model missed outright (339 reviewed, 78 hand-corrected).
- **Vision-LLM-grounded data linking**: each extracted structure's bounding
  box is used as a Set-of-Mark prompt so a VLM can read the surrounding
  figure and attach the right compound ID, yield, and reaction conditions —
  no hand-written per-publisher parsing rules, and a large accuracy jump
  over an earlier deterministic (PDF-text-proximity) approach that only
  reached a 5.8% hit rate.
- **37 data tables** (SI characterization data, reaction-optimization
  tables) extracted corpus-wide with structured cell/row parsing.
- ~8,800 lines of Python across PDF ingestion, layout parsing, computer
  vision, OCSR, table extraction, and LLM-grounded linking, plus the two
  standalone review applications.

## Pipeline

| Stage | What it does | Status |
|---|---|---|
| 1. PDF ingestion | Extracts text blocks and figures from paper + SI PDFs, including vector-drawn (not embedded-raster) figures | Done |
| 2. Layout parsing | Reconstructs headings/paragraphs/figure captions/references into a document structure, generalized across publishers via document-relative statistics rather than hardcoded per-journal rules | Done |
| 3. Figure understanding | CLIP-based figure tagging → segmentation → OCSR (MolScribe) → RDKit validation, with two human review tools closing the loop on model errors | Corpus-wide, iterating |
| 4. Table extraction | Structured cell/row extraction from genuine data tables (SI characterization data, screening tables) | First working version, done |
| 5. Structure-to-data linking | VLM (Claude), grounded on each structure's bounding box, links it to its compound ID/yield/conditions | First slice, in progress |
| 6. Entity normalization | Canonical compound IDs across papers | Not started |
| 7. Knowledge graph | Provenance-linked graph over the above | Not started |

## Engineering deep-dives

A few of the harder problems this pipeline had to solve:

- **Vector-graphic figure detection.** Chemistry figures in these PDFs are
  almost always hundreds of individual vector path/line/fill operations, not
  embedded raster images — naively pulling embedded images misses them
  entirely. Stage 1 clusters nearby vector-drawing bounding boxes into
  figure-sized regions instead, with asymmetric padding (wide vertically,
  narrow horizontally) so multi-panel schemes merge correctly without
  bridging across a two-column page's gutter.
- **Publisher-agnostic layout parsing.** An early version hardcoded font
  names and page-position bands tuned to one journal and silently produced
  an empty document structure on a different publisher. Rewritten to rank
  heading tiers by document-relative statistics (font size, then
  numbered-list/ALL-CAPS/marker-glyph tie-breakers), validated cleanly
  across 4 publisher formats with zero regressions as each new one was
  added.
- **OCSR error correction.** Beyond swapping in a stronger base OCSR model,
  built a correction layer: multi-mode Tesseract OCR voting to catch
  compound-label abbreviations a single PSM mode misreads (e.g. a tosyl
  group silently dropping its trailing "s"), a dictionary patch for
  domain-specific fragments, and a fragment-count heuristic to flag crops
  where the model under-merged multiple real compounds into one image.
- **Chemically-correct molecule encoding for organometallic structures.**
  This corpus is rich in NHC/organometallic complexes with dative
  metal-ligand bonds and π-hapticity (η⁵-Cp, η⁶-arene, etc.). The sketch
  tool exports Ketcher's Molfile (never its SMILES, which silently drops
  dative bond orders) and lets RDKit re-serialize it — the only reliable
  path found for round-tripping these structures correctly.
- **Set-of-Mark VLM grounding.** Rather than write per-publisher rules to
  match a structure to its caption text, each structure's own bounding box
  is drawn onto the figure and handed to a vision-LLM as a grounding
  anchor — a general technique that replaced a bespoke, much lower-accuracy
  deterministic matcher.

## Tech stack

Python · PyMuPDF · RDKit · DECIMER / MolScribe (OCSR) · OpenCLIP ·
Tesseract / EasyOCR · GROBID (reference parsing) · FastAPI · Ketcher
(React/Vite, built from source) · Claude (Anthropic API) for VLM grounding.

## Setup

```
conda create -n paper_extraction python=3.13 -y
conda activate paper_extraction
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install open_clip_torch pymupdf requests decimer "tensorflow[and-cuda]"
pip install -r requirements.txt
```

`torch`/`torchvision` are pinned to CPU builds deliberately; only
`tensorflow` (DECIMER) uses a GPU if one's available, falling back to CPU
otherwise with no code changes required.

## Running the pipeline

```
python3 ingest/pdf_ingest.py data/papers/<paper>       # Stage 1: PDF -> text + figures
python3 ingest/section_parse.py data/papers/<paper>    # Stage 2: document structure
python3 ingest/si_parse.py data/papers/<paper>         # Stage 2b: SI compound records
python3 ingest/batch_tag_figures.py data/papers/<paper> # Stage 3a: figure classification
python3 ingest/batch_segment.py data/papers/<paper>     # Stage 3b: crop individual structures
python3 ingest/batch_molscribe_extract.py data/papers/<paper> # Stage 3c: OCSR
python3 ingest/table_extract.py data/papers/<paper>     # Stage 4: table extraction
python3 ingest/reaction_link.py data/papers/<paper>     # Stage 5: structure-to-data linking
```

References are parsed via a local [GROBID](https://github.com/kermitt2/grobid)
container (`docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.1`);
Stage 2 falls back to a prose reference section if it isn't running.

The two review tools (`ingest/split_ui/`, `ingest/draw_ui/`) run as
standalone FastAPI apps (`uvicorn server:app --port 8420` / `8430`) and pull
their queues from whatever the pipeline currently has flagged for review.

## Corpus

Seven papers spanning four publisher formats, chosen to stress-test the
pipeline against real-world layout and chemistry diversity: `suzuki_iron_2024`,
`copper_iron_2025`, `suzuki_nickel_2026`, `miyaura_iron_2025`,
`redox_neutral_2024`, `nickelocene_2025`, `suzuki_nhc_2026`.
