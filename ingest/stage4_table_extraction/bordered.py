"""Path A: real bordered/gridded data tables (a full ruled grid, e.g. a
GC-MS peak table) are reliably found by PyMuPDF's own `page.find_tables()`
-- this module's job is filtering its false positives, not finding tables
itself. See table_extract.py's module docstring for the corpus-wide
survey behind this split."""

import json

import fitz

from common import MAX_HEADER_CELL_CHARS

# A real single-row table is rare in this corpus and every row_count==1
# candidate found in a corpus-wide survey was either a two-column-text
# misdetection or a header fragment of a table whose body find_tables()
# failed to see at all (handled separately by Path B) -- never a complete,
# useful table on its own. Simpler and safer to require >=2 rows outright
# than to guess a height/width threshold for the false-positive shape.
MIN_ROWS_BORDERED = 2
# A genuine data table's cells are real text throughout; a scope-table
# scheme masquerading as "Table N" has mostly-empty cells (its content is
# vector-drawn chemistry, not text) -- confirmed as a clean, reliable
# discriminator across every case checked.
MIN_NONEMPTY_CELL_FRAC = 1.0
# Stage 1 can partially swallow a real table's rows into a neighboring
# vector_region figure crop (table rule lines are vector-drawn too, so
# cluster_drawing_rects' pad_y can bridge across them) -- confirmed on
# miyaura_iron_2025/SI pages 24 and 28, where the real table only
# partially, not fully, overlaps the swallowing figure. A high overlap
# threshold (vs. a naive "any overlap") is what separates that from an
# actual scheme-graphic false positive, where the detected "table" IS
# essentially the whole figure.
FIGURE_OVERLAP_REJECT_FRAC = 0.9


def _vector_regions_by_page(paper_dir, source):
    fname = "raw_extraction.json" if source == "main" else "SI_raw_extraction.json"
    path = paper_dir / fname
    if not path.exists():
        return {}
    data = json.load(open(path))
    return {
        p["page_number"]: [fitz.Rect(img["bbox"]) for img in p["images"] if img["source"] == "vector_region"]
        for p in data["pages"]
    }


def _overlaps_figure(bbox, regions, min_frac=FIGURE_OVERLAP_REJECT_FRAC):
    r = fitz.Rect(bbox)
    for vr in regions:
        inter = r & vr
        if inter.is_empty:
            continue
        if (inter.width * inter.height) / (r.width * r.height) >= min_frac:
            return True
    return False


def _is_real_data_table(cells, table, page_vector_regions):
    """Takes cells = table.extract() from the caller rather than
    re-extracting -- and deliberately never reads table.header.names,
    which turned out to be unreliable in PyMuPDF: confirmed directly on a
    real table (miyaura_iron_2025/SI page 2, a genuine HPLC gradient
    table) where .header.names returned ['gs:', '', '', ''] -- a
    fragment matching nothing on the page -- while table.extract()[0]
    correctly returned the real header every time. extract() is the only
    header source used anywhere in this module now."""
    if table.row_count < MIN_ROWS_BORDERED:
        return False
    total = sum(len(row) for row in cells)
    nonempty = sum(1 for row in cells for cell in row if (cell or "").strip())
    if total == 0 or nonempty / total < MIN_NONEMPTY_CELL_FRAC:
        return False
    if any(len(h or "") > MAX_HEADER_CELL_CHARS for h in cells[0]):
        return False
    if _overlaps_figure(table.bbox, page_vector_regions):
        return False
    return True


def _extract_bordered_tables(page, page_vector_regions):
    tables = []
    for t in page.find_tables().tables:
        cells = t.extract()
        if _is_real_data_table(cells, t, page_vector_regions):
            col_anchors = [(c[0] + c[2]) / 2 for c in t.rows[0].cells]
            tables.append({"header": cells[0], "rows": cells[1:], "bbox": list(t.bbox), "col_anchors": col_anchors})
    return tables
