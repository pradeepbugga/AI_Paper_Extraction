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
   `_is_real_data_table`) rather than treat every "Table N" as its job.

2. Real bordered/gridded data tables (a full ruled grid, e.g. a GC-MS peak
   table) are reliably found by PyMuPDF's own `page.find_tables()`.

3. Real "borderless" data tables -- common specifically for reaction-
   optimization/screening tables (catalyst, ligand, T, yield, ee columns)
   in both main text and SI, in this corpus's journals' house style: only
   a header row and a closing rule, no grid lines between data rows -- are
   MISSED by `find_tables()` entirely (confirmed: it detects at most the
   header row, sometimes not even that). These need to be reconstructed
   directly from text position, seeded from a header row detected one of
   two ways (see `_extract_borderless_tables`): a header with some visual
   marker `find_tables()` itself can key off (a shaded background band,
   confirmed on suzuki_nickel_2026's Table 1), or -- when even that's
   missing, confirmed on redox_neutral_2024/SI's Table S6 -- the ruling
   lines above/below the header, which can be drawn as several short
   per-column segments rather than one continuous line (also why
   `find_tables()` misses the whole table: its line-detector expects
   continuous rules). The segment gaps directly give column boundaries.

False positives in `find_tables()` itself, checked directly across all 7
papers before writing any filter: journal masthead/byline boxes (tiny
1-3 row grids near the top of page 1), a two-column body-text layout
occasionally misdetected as a spurious "1 row" table (spanning the full
column height -- no real table row is 150-330pt tall), and a degenerate
zero-height rule-line misread as a table. See `_is_real_data_table`.
"""

import json
import re
from pathlib import Path

import fitz

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"

CAPTION_RE = re.compile(r"^Table\s+S?\d+[\.\:]", re.IGNORECASE)
CAPTION_SEARCH_ABOVE_PT = 400  # captions sit above the table in this
                               # corpus's house style (confirmed on every
                               # real table found), unlike figure captions
                               # (below) -- and can sit well above it, with
                               # a reaction-scheme graphic (no real text)
                               # in between the two: confirmed directly on
                               # suzuki_nickel_2026's Table 1, where the
                               # caption-to-header gap is 311.8pt. Takes the
                               # CLOSEST matching caption-shaped line found
                               # in this whole window, so a wide window
                               # doesn't risk preferring a distant false
                               # match over a near one.

# --- Path A: bordered/gridded tables -----------------------------------

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
# A chart/graph's axis-tick labels and equation annotations can pack
# densely enough into a small grid to also pass the two filters above --
# confirmed on suzuki_iron_2024/SI pages 84-85 (kinetics plots), where
# find_tables() misread the tick-label grid as a legitimate small table.
# Real header cells in this corpus are short labels ("T (°C)", "Fe Source
# (mol%)", 11-17 chars); the chart-garbage cells are 100+ chars of
# concatenated, newline-joined tick values -- a clean, evidence-based
# length cutoff, not a guess.
MAX_HEADER_CELL_CHARS = 60


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


# --- Path B: borderless tables -------------------------------------------

ROW_Y_TOLERANCE_PT = 5.0  # absorbs a subscript/superscript span's vertical
                            # offset from its cell's baseline (confirmed:
                            # ~3.4pt on real cases) while staying well under
                            # real row-to-row spacing (confirmed: ~11pt)
SPACING_DEVIATION_FRAC = 0.4  # once a stable row cadence is established, a
                                # candidate row breaks the table if its gap
                                # from the previous row deviates this much
                                # from the running median spacing -- only
                                # used as a fallback when no closing rule
                                # was found (see RULE_* below)
HEADER_ROW_GAP_MAX_PT = 30  # a header's own top/bottom bounding rules (or
                              # a header table + the next rule below it)
                              # are one text line apart, not a whole table
RULE_MAX_HEIGHT_PT = 3.0
RULE_MIN_SEGMENTS = 2
RULE_Y_TOLERANCE_PT = 0.5
RULE_MIN_TOTAL_WIDTH_FRAC = 0.4  # a real table-width rule spans a large
                                    # fraction of the page's content width;
                                    # excludes short decorative underlines


def _find_rule_lines(page):
    """Groups thin, dark filled rects sharing a y-position into candidate
    table rule lines. Some journals' PDF renderers draw a table's ruling
    as one short segment per column (with a real gap at each column
    boundary) instead of one continuous line -- confirmed on
    redox_neutral_2024/SI page 40, where find_tables() fails to recognize
    the table at all because of this. The segments' own x-ranges double as
    column boundaries once found."""
    page_width = page.rect.width
    candidates = {}
    for d in page.get_drawings():
        rect = d["rect"]
        if rect.height > RULE_MAX_HEIGHT_PT or rect.width < 5:
            continue
        fill = d.get("fill")
        if fill is None or sum(fill[:3]) / 3 >= 0.3:  # dark fill only
            continue
        y_center = round((rect.y0 + rect.y1) / 2, 1)
        key = next((k for k in candidates if abs(k - y_center) <= RULE_Y_TOLERANCE_PT), y_center)
        candidates.setdefault(key, []).append((rect.x0, rect.x1))

    rule_lines = []
    for y_center, segments in candidates.items():
        segments.sort()
        total_width = sum(x1 - x0 for x0, x1 in segments)
        if len(segments) >= RULE_MIN_SEGMENTS and total_width >= page_width * RULE_MIN_TOTAL_WIDTH_FRAC:
            rule_lines.append((y_center, segments))
    rule_lines.sort(key=lambda r: r[0])
    return rule_lines


def _text_in_band(page, y0, y1, x0, x1):
    entries = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for s in line["spans"]:
                text = s["text"].strip()
                if not text:
                    continue
                sx0, sy0 = s["bbox"][0], s["bbox"][1]
                if y0 - 1 <= sy0 <= y1 + 1 and x0 - 5 <= sx0 <= x1 + 5:
                    entries.append((sy0, sx0, text))
    return entries


def _finalize_cell_buckets(buckets):
    """Each bucket holds (y0, x0, text) entries possibly spanning several
    physical lines (a wrapped cell). Groups by y0 into physical lines,
    sorts each line left-to-right, and joins lines with "\\n" -- the same
    convention PyMuPDF's own table.extract() uses for a real multi-line
    cell, confirmed on suzuki_nickel/miyaura's bordered tables (e.g.
    "m/z (% relative\\nintensity, ion)")."""
    cells = []
    for bucket in buckets:
        by_line = {}
        for y0, x0, text in bucket:
            by_line.setdefault(round(y0, 1), []).append((x0, text))
        lines = [" ".join(t for _, t in sorted(spans)) for _, spans in sorted(by_line.items())]
        cells.append("\n".join(lines).strip())
    return cells


def _reconstruct_rows(page, col_anchors, table_x0, table_x1, start_y, close_y):
    """Groups text below a known header into rows, buckets each row's text
    into columns by nearest anchor, and stops either at a known closing
    rule (`close_y`, authoritative when available) or, when none exists,
    once the row spacing or column-fill count breaks from what's been
    established -- distinguishes real, evenly-spaced, mostly-full rows
    from prose/footnote text that can coincidentally hit a couple of
    column anchors.

    A row can itself span more than one physical text line -- a wrapped
    cell, confirmed on miyaura_iron_2025/SI's GC-MS table (an m/z reading
    wrapping across two lines) -- so grouping can't just be "one physical
    line = one row" the way the single-line reaction-optimization tables
    this was first validated against allow. The real signal: a genuine
    new row has its first (leftmost/anchor) column populated -- a wrapped
    continuation line doesn't, since it's continuing a *later* column's
    cell, never restarting the row's own index/key value. So a physical
    line with column 0 empty is appended onto the still-open previous row
    instead of starting a new one. Validated end-to-end against three real
    tables: suzuki_nickel_2026 Table 1 (20/20 rows, single-line);
    redox_neutral_2024/SI Table S6 (10/10 rows, single-line);
    miyaura_iron_2025/SI's GC-MS table continuing from page 27 onto page
    28 (correctly reconstructs the wrapped m/z cells there, previously
    produced a garbled partial row under the old one-line-per-row model).

    Returns (rows, last_row_y1) -- the latter approximates the bottom edge
    of the actual last reconstructed row (not just the header/start_y),
    needed so a caller checking "does this table end near the page
    bottom" (for continuation-merging across a page break) isn't comparing
    against the wrong, far-too-early position when there's no closing
    rule to give an exact bound."""
    entries = _text_in_band(page, start_y, close_y if close_y else start_y + 2000, table_x0, table_x1)
    entries.sort(key=lambda e: (e[0], e[1]))

    lines = []
    current, last_y, group_y0 = [], None, None
    for y0, x0, text in entries:
        if last_y is None or abs(y0 - last_y) <= ROW_Y_TOLERANCE_PT:
            current.append((x0, text))
            group_y0 = group_y0 if group_y0 is not None else y0
        else:
            lines.append((group_y0, current))
            current, group_y0 = [(x0, text)], y0
        last_y = y0
    if current:
        lines.append((group_y0, current))

    rows, row_y0s = [], []
    min_matched = max(2, (len(col_anchors) + 1) // 2)
    open_row = None
    for group_y0, line in lines:
        buckets, matched = [[] for _ in col_anchors], set()
        for x0, text in line:
            nearest = min(range(len(col_anchors)), key=lambda k: abs(col_anchors[k] - x0))
            buckets[nearest].append((group_y0, x0, text))
            matched.add(nearest)

        if 0 not in matched:
            # a wrapped continuation line of the still-open row's cell(s)
            if open_row is not None:
                for i, b in enumerate(buckets):
                    open_row[i].extend(b)
            continue

        if len(matched) < min_matched:
            if close_y is None:
                break
            continue  # geometric bound is authoritative -- skip stray lines within it
        if close_y is None and len(row_y0s) >= 2:
            spacings = [row_y0s[k] - row_y0s[k - 1] for k in range(1, len(row_y0s))]
            median = sorted(spacings)[len(spacings) // 2]
            if median > 0 and abs(group_y0 - row_y0s[-1] - median) > SPACING_DEVIATION_FRAC * median:
                break

        if open_row is not None:
            rows.append(_finalize_cell_buckets(open_row))
        open_row = buckets
        row_y0s.append(group_y0)

    if open_row is not None:
        rows.append(_finalize_cell_buckets(open_row))

    last_row_y1 = (row_y0s[-1] + ROW_Y_TOLERANCE_PT * 2) if row_y0s else start_y
    return rows, last_row_y1


HEADER_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def _header_looks_valid(header):
    """Shared sanity gate for a candidate header row, for both Path B
    sources: needs at least 2 populated cells to be a real multi-column
    header, no cell so long it's actually chart/graph text (axis tick
    labels, equation annotations) rather than a short column label (see
    MAX_HEADER_CELL_CHARS), and most populated cells must contain a real
    alphabetic word -- confirmed necessary on redox_neutral_2024/SI page
    130, a pasted block of raw NMR chemical-shift values
    ("143. 133. 129. ...") that's short enough to dodge the length check
    but, unlike every real header checked, is purely numeric. Majority
    rather than all, since a genuinely short column label can legitimately
    have no letter pair at all (confirmed: suzuki_nickel_2026's real
    header has "T (°C)", where neither "T" nor "C" clears a 2-letter run
    alone) as long as most of the row is real words."""
    nonempty = [h for h in header if h.strip()]
    if len(nonempty) < 2:
        return False
    if any(len(h) > MAX_HEADER_CELL_CHARS for h in header):
        return False
    word_count = sum(1 for h in nonempty if HEADER_WORD_RE.search(h))
    if word_count < len(nonempty) / 2:
        return False
    return True


def _borderless_from_visual_header(page):
    """Path B, header source 1: find_tables() can itself find a header row
    when it has some visual marker to key off (e.g. a shaded background
    band, confirmed on suzuki_nickel_2026's Table 1) even though it can't
    see the borderless body below it. Reads the header via t.extract()[0],
    never t.header.names -- see _is_real_data_table's docstring for why
    that property is unreliable."""
    results = []
    for t in page.find_tables().tables:
        if t.row_count != 1:
            continue
        header = t.extract()[0]
        if not _header_looks_valid(header):
            continue
        col_anchors = [(c[0] + c[2]) / 2 for c in t.rows[0].cells]
        table_x0, table_x1 = t.bbox[0], t.bbox[2]
        rows, last_y1 = _reconstruct_rows(page, col_anchors, table_x0, table_x1, t.bbox[3], None)
        if rows:
            bbox = [t.bbox[0], t.bbox[1], t.bbox[2], max(t.bbox[3], last_y1)]
            results.append({"header": header, "rows": rows, "bbox": bbox, "col_anchors": col_anchors, "closed": False})
    return results


def _borderless_from_rule_lines(page):
    """Path B, header source 2: a header with no visual marker at all,
    seeded instead from its own bounding rule lines -- see
    `_find_rule_lines` for why those can be fragmented into per-column
    segments rather than one line. Also recovers the table's closing rule
    when a matching third rule line exists further down, giving an exact
    geometric stop instead of the content-based fallback."""
    rules = _find_rule_lines(page)
    results = []
    for i in range(len(rules) - 1):
        y_a, segs_a = rules[i]
        y_b, segs_b = rules[i + 1]
        if len(segs_a) != len(segs_b) or not (3 <= y_b - y_a <= HEADER_ROW_GAP_MAX_PT):
            continue
        col_anchors = [(x0 + x1) / 2 for x0, x1 in segs_b]
        table_x0, table_x1 = segs_b[0][0], segs_b[-1][1]

        header_entries = _text_in_band(page, y_a, y_b, table_x0, table_x1)
        if not header_entries:
            continue
        header_buckets = [[] for _ in col_anchors]
        for sy0, sx0, text in header_entries:
            nearest = min(range(len(col_anchors)), key=lambda k: abs(col_anchors[k] - sx0))
            header_buckets[nearest].append(text)
        header = [" ".join(b).strip() for b in header_buckets]
        if not _header_looks_valid(header):
            continue

        close_y = next((y_c for y_c, segs_c in rules[i + 2:] if len(segs_c) == len(segs_b)), None)
        rows, last_y1 = _reconstruct_rows(page, col_anchors, table_x0, table_x1, y_b, close_y)
        if rows:
            results.append({
                "header": header, "rows": rows,
                "bbox": [table_x0, y_a, table_x1, close_y if close_y is not None else last_y1],
                "col_anchors": col_anchors,
                "closed": close_y is not None,
            })
    return results


def _extract_borderless_tables(page):
    """Tries both header sources and combines their results, but a row
    elsewhere in the page that happens to carry its own visual marker (a
    highlighted/colored row, not the true header) can also independently
    pass `_borderless_from_visual_header`'s bar -- confirmed on
    redox_neutral_2024/SI page 40, where a green-highlighted data row
    (entry 8) got misread as a second, spurious one-row "table" alongside
    the real 10-row reconstruction from `_borderless_from_rule_lines`.
    Any visual-header result whose own y-range falls inside a rule-line
    result's row-span on the same page is that same table's data, already
    captured -- drop it rather than emit a duplicate fragment."""
    from_lines = _borderless_from_rule_lines(page)
    from_visual = _borderless_from_visual_header(page)
    kept_visual = []
    for v in from_visual:
        v_y0, v_y1 = v["bbox"][1], v["bbox"][3]
        if any(t["bbox"][1] - 5 <= v_y0 and v_y1 <= t["bbox"][3] + 5 for t in from_lines):
            continue
        kept_visual.append(v)
    return from_lines + kept_visual


# --- Captions and page/paper orchestration --------------------------------


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
    for t in tables:
        caption = _find_caption(page, t["bbox"])
        results.append({
            "page_number": page_number,
            "caption": caption,
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
