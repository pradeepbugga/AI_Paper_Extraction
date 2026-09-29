"""Extracts genuine data tables (SI characterization data -- GC-MS/NMR peak
assignments, crystallography stats; reaction-optimization/screening tables
-- catalyst/ligand/solvent/yield) from each paper's PDFs.

Two real, separate detection paths are needed, not one, because this
corpus's "tables" split into three distinct cases, confirmed by direct
inspection before writing any extraction logic:

1. Most "Table N" captions in main-text results sections are actually
   scope-table *scheme* graphics (compound structures + conditions +
   yields), not text-cell data tables at all -- confirmed directly on
   redox_neutral_2024's "Table 1", which is a 3-panel ChemDraw grid.
   Stage 1 (pdf_ingest.py) already captures these as vector_region figures
   regardless of caption label, and Stage 3 already segments/extracts
   their structures, so this module must actively EXCLUDE them (see
   bordered.py's `_is_real_data_table`) rather than treat every "Table N"
   as its job.

2. Real bordered/gridded data tables (a full ruled grid, e.g. a GC-MS peak
   table) are reliably found by PyMuPDF's own `page.find_tables()` --
   Path A, `bordered.py`.

3. Real "borderless" data tables -- common specifically for reaction-
   optimization/screening tables (catalyst, ligand, T, yield, ee columns)
   in both main text and SI, in this corpus's journals' house style: only
   a header row and a closing rule, no grid lines between data rows -- are
   MISSED by `find_tables()` entirely (confirmed: it detects at most the
   header row, sometimes not even that). These need to be reconstructed
   directly from text position -- Path B, `borderless.py`. A raster-image
   variant of the same house style (no extractable text/vector content at
   all) is handled by Path C, `raster_ocr.py`.

False positives in `find_tables()` itself, checked directly across all 7
papers before writing any filter: journal masthead/byline boxes (tiny
1-3 row grids near the top of page 1), a two-column body-text layout
occasionally misdetected as a spurious "1 row" table (spanning the full
column height -- no real table row is 150-330pt tall), and a degenerate
zero-height rule-line misread as a table. See bordered.py's
`_is_real_data_table`.

Row/header reconstruction logic shared across Paths B and C lives in
`common.py`; this module owns caption matching and page/paper-level
orchestration (multi-page table continuation, in particular)."""

import json
from pathlib import Path

import fitz

from bordered import _extract_bordered_tables, _vector_regions_by_page
from borderless import _extract_borderless_tables, _reconstruct_rows
from common import CAPTION_RE, CAPTION_SEARCH_ABOVE_PT
from raster_ocr import _extract_ocr_tables

PAPERS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "papers"


def _find_caption(page, bbox):
    """Table captions sit above the table in this corpus's house style
    (confirmed on every real table found), the opposite of figure
    captions -- checked directly rather than assumed, since
    section_parse.py's figure-caption logic searches below."""
    x0, y0, x1, y1 = bbox
    best = None
    for block in page.get_text("blocks"):
        bx0, by0, bx1, by1, text, _, block_type = block
        if block_type != 0:
            continue
        if by1 > y0 or by1 < y0 - CAPTION_SEARCH_ABOVE_PT:
            continue
        stripped = text.strip()
        if CAPTION_RE.match(stripped):
            if best is None or by1 > best[0]:
                best = (by1, stripped)
    return best[1] if best else None


# A table's last row must end within this margin of the page's bottom
# edge to be treated as possibly truncated by the page break rather than
# a table that simply, genuinely ended -- confirmed on suzuki_iron_2024/
# SI pages 106-107 ("Supplementary Table 19", a DFT energy table): its
# last row on page 106 sits well within this margin of the page bottom,
# and continues with no header, caption, or border at all on page 107.
PAGE_BOTTOM_MARGIN_PT = 100
# Skip likely running-header/page-number content at the very top of a
# continuation page before looking for column-aligned data rows.
PAGE_TOP_SKIP_PT = 40
# How far down a continuation page to check for a *new* table's own
# caption before trusting that unheaded content there is a continuation
# of the previous page's table, not the start of an unrelated one.
CONTINUATION_CAPTION_CHECK_PT = 150


def extract_tables_for_page(page, page_number, page_vector_regions):
    tables = _extract_bordered_tables(page, page_vector_regions) + _extract_borderless_tables(page)
    results = []
    found_captions = set()
    for t in tables:
        caption = _find_caption(page, t["bbox"])
        if caption:
            found_captions.add(caption)
        results.append({
            "page_number": page_number,
            "caption": caption,
            "header": t["header"],
            "rows": t["rows"],
            "bbox": [round(v, 1) for v in t["bbox"]],
            "col_anchors": t.get("col_anchors"),
            "closed": t.get("closed"),
        })

    # Path C (OCR) is only a fallback for a "Table N" caption that Paths
    # A/B found no real text/vector table for at all -- confirmed
    # necessary to avoid a duplicate when a caption's table IS real text/
    # vector content that Paths A/B already extracted correctly.
    for t in _extract_ocr_tables(page):
        if t["caption"] in found_captions:
            continue
        results.append({
            "page_number": page_number,
            "caption": t["caption"],
            "header": t["header"],
            "rows": t["rows"],
            "bbox": [round(v, 1) for v in t["bbox"]],
            "col_anchors": t.get("col_anchors"),
            "closed": t.get("closed"),
        })
    return results


def _has_own_caption_near_top(page):
    for block in page.get_text("blocks"):
        bx0, by0, bx1, by1, text, _, block_type = block
        if block_type != 0 or by0 > CONTINUATION_CAPTION_CHECK_PT:
            continue
        if CAPTION_RE.match(text.strip()):
            return True
    return False


def _try_continue_table(last_table, page):
    """A table whose last row landed close to the previous page's bottom
    margin, with no closing rule found (see `closed` on rule-line-seeded
    tables) and no table of its own detected on this page, is checked for
    continuation: reconstruct rows from this page's top using the
    previous table's own column positions, the same way a borderless
    table's body is built from its header -- a continuation page has
    exactly the same shape (a header-less block of column-aligned rows)
    as a headerless closing fragment does. Refuses to merge if this page
    starts with its own "Table N" caption -- a real new table, not a
    continuation, even if it coincidentally has the same column count.

    Deliberately does NOT try to splice these rows into the previous
    page's row list as one seamless table -- confirmed that's a much
    harder problem than it looks (miyaura_iron_2025/SI's GC-MS table,
    continuing from page 27 to 28, has wrapped multi-line cells of
    inconsistent height, one row with an embedded structure image
    disrupting the normal row-spacing pattern _reconstruct_rows leans on
    to know when to stop -- real column-misalignment and early-stopping
    problems were confirmed on this exact case). Instead returns a
    self-contained continuation fragment (its own header, copied from the
    table it continues, so every row is still independently interpretable
    even if it never gets stitched into one unified table) -- the
    per-fragment row reconstruction can still be locally imperfect on an
    irregular row without corrupting or truncating the rest of a much
    larger table the way a single shared row list would."""
    if last_table.get("closed"):
        return None
    if not last_table.get("col_anchors"):
        return None
    if _has_own_caption_near_top(page):
        return None
    x0, y0, x1, y1 = last_table["bbox"]
    rows, last_y1 = _reconstruct_rows(page, last_table["col_anchors"], x0, x1, PAGE_TOP_SKIP_PT, None)
    if not rows:
        return None
    return {
        "header": list(last_table["header"]),
        "rows": rows,
        "bbox": [x0, PAGE_TOP_SKIP_PT, x1, last_y1],
        "col_anchors": last_table["col_anchors"],
        "closed": last_table.get("closed", False),
    }


def extract_tables(pdf_path, paper_dir, source):
    """A table can span more than one page break, not just one -- confirmed
    on suzuki_iron_2024/SI's "Supplementary Table 19" (a DFT energy table),
    which continues across three consecutive pages with no header, rule,
    or caption of its own on the second or third. `open_table` tracks a
    table that might still be continuing across an arbitrary run of
    header-less pages, not just a single next page: it's carried forward
    as long as each successive page keeps extending it, and only dropped
    once a page fails to continue it (a real end) or a page produces its
    own new tables (a different context). Each continuation page becomes
    its own table entry with a copied header (see _try_continue_table for
    why), tagged with `continues_page` pointing at the table it follows,
    rather than one table whose row list silently spans several pages."""
    doc = fitz.open(pdf_path)
    vector_regions = _vector_regions_by_page(paper_dir, source)
    all_tables = []
    open_table = None
    for page_number, page in enumerate(doc, start=1):
        page_tables = extract_tables_for_page(page, page_number, vector_regions.get(page_number, []))
        if not page_tables and open_table is not None:
            continuation = _try_continue_table(open_table, page)
            if continuation is None:
                open_table = None
            else:
                continuation["page_number"] = page_number
                continuation["caption"] = None
                continuation["continues_page"] = open_table["page_number"]
                all_tables.append(continuation)
                open_table = continuation
        elif page_tables:
            last = page_tables[-1]
            near_bottom = (page.rect.height - last["bbox"][3]) <= PAGE_BOTTOM_MARGIN_PT
            open_table = last if near_bottom else None
            all_tables.extend(page_tables)
        else:
            all_tables.extend(page_tables)
    doc.close()
    for t in all_tables:
        t.pop("col_anchors", None)
        t.pop("closed", None)
        t["bbox"] = [round(v, 1) for v in t["bbox"]]
        t.setdefault("continues_page", None)
    return all_tables


def main():
    for paper_dir in sorted(p for p in PAPERS_DIR.iterdir() if p.is_dir()):
        main_pdf = paper_dir / "paper.pdf"
        if main_pdf.exists():
            tables = extract_tables(main_pdf, paper_dir, "main")
            with open(paper_dir / "tables.json", "w") as f:
                json.dump(tables, f, indent=2)
            print(f"{paper_dir.name} main: {len(tables)} tables")

        si_pdf = paper_dir / "SI.pdf"
        if si_pdf.exists():
            tables = extract_tables(si_pdf, paper_dir, "SI")
            with open(paper_dir / "SI_tables.json", "w") as f:
                json.dump(tables, f, indent=2)
            print(f"{paper_dir.name} SI: {len(tables)} tables")


if __name__ == "__main__":
    main()
