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
import numpy as np
import pytesseract
from PIL import Image

PAPERS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "papers"

CAPTION_RE = re.compile(r"^(?:Supplementary\s+)?Table\s+S?\d+\s*[\.\:–—-]", re.IGNORECASE)
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

ROW_Y_TOLERANCE_PT = 6.0  # absorbs a subscript/superscript span's vertical
                            # offset from its cell's baseline (confirmed:
                            # ~3.4pt on real cases, up to 5.04pt on
                            # suzuki_iron_2024/SI's own multi-part species
                            # labels like "HS I_U *") while staying well
                            # under real row-to-row spacing (confirmed as
                            # low as ~8.56pt on suzuki_nhc_2026's Table 2)
SPACING_DEVIATION_FRAC = 0.4  # once a stable row cadence is established, a
                                # candidate row breaks the table if its gap
                                # from the previous row deviates this much
                                # from the running median spacing -- only
                                # used as a fallback when no closing rule
                                # was found (see RULE_* below)
NEW_ROW_VS_WRAP_FRAC = 0.6  # a line with column 0 empty is normally a
                              # wrapped continuation of the previous cell,
                              # but a real new row can also legitimately
                              # have nothing recognized in column 0 -- OCR
                              # failing to read a repeated/ditto-marked
                              # value, confirmed on nickelocene_2025's
                              # Table 2 (entry 2's catalyst name never
                              # got OCR'd at all). Distinguished from a
                              # genuine wrap by gap size against the
                              # established row spacing: a real wrapped
                              # line sits close to its parent row
                              # (confirmed ~3.4pt on the DFT-table
                              # subscript case, ~31% of that table's
                              # ~10.9pt row spacing), while a real row
                              # missing its column-0 value still sits a
                              # full row apart (confirmed ~9.8pt on
                              # nickelocene Table 2, ~100% of its ~9.8pt
                              # spacing) -- 0.6 sits with margin between
                              # both confirmed cases.
HEADER_ROW_GAP_MAX_PT = 30  # a header's own top/bottom bounding rules (or
                              # a header table + the next rule below it)
                              # are one text line apart, not a whole table
RULE_MAX_HEIGHT_PT = 3.0
RULE_Y_TOLERANCE_PT = 0.5
# A real table-width rule spans a large fraction of the page's content
# width, excluding short decorative underlines -- but "page" here means
# whatever column the table sits in, not necessarily the full page.
# Originally required >=2 segments AND >=0.4 of the full page width, both
# calibrated only against single-column SI pages (where a rule spans the
# whole page). Confirmed wrong on nickelocene_2025's main-text Table 1: a
# real, single continuous rule (not fragmented into per-column segments
# at all) spanning its own two-column layout's column width -- 240pt,
# only 0.395 of the 607pt full page width, and just one segment, so it
# failed both the old segment-count and width checks. A single segment is
# now accepted (real rules aren't always fragmented, see
# redox_neutral_2024/SI page 40 vs. this case), and the width fraction is
# lowered with margin below both confirmed-real cases (single-column
# ~0.83, two-column ~0.395) -- re-verified corpus-wide after lowering that
# this doesn't newly accept a false positive.
RULE_MIN_TOTAL_WIDTH_FRAC = 0.3
# Two unrelated tables sitting side by side in a two-column page layout
# can share close-to-identical y-positions for their own rules purely by
# coincidence of column layout (sometimes even deliberately, for visual
# alignment) -- confirmed on suzuki_nhc_2026, where Table 1 (left column)
# and Table 3 (right column) each have their own header-bounding rules at
# the *same* y, and grouping by y alone merged them into one rule
# spanning both, which then made the header-row pairing fail entirely (a
# real header-bounding pair on one side no longer matched the other
# side's unrelated segment count). Within one real table, adjacent
# per-column segments touch or overlap slightly (confirmed ~1.4-1.5pt
# apart on redox_neutral_2024/SI's fragmented rule); the gap between two
# actually-different tables/columns is much larger (confirmed 17.9pt on
# suzuki_nhc_2026) -- comfortably separated by this threshold.
RULE_REGION_GAP_PT = 10.0
# Threshold for collapsing same-y segments into one when they overlap --
# see the comment at the merge site in _find_rule_lines. Genuine per-column
# segments can overlap by rendering-rounding noise (confirmed ~0.24-0.49pt
# on suzuki_nickel_2026/SI page 65's closing rule); the PDF-export artifact
# this merge targets overlaps by tens of points (confirmed 53.6pt on
# suzuki_nhc_2026's Table 1 header-bottom rule). This sits comfortably
# between the two.
RULE_SEGMENT_OVERLAP_MERGE_PT = 5.0


def _cluster_x_regions(intervals, gap_pt=RULE_REGION_GAP_PT):
    """Merges (x0, x1, ...) intervals into disjoint regions wherever a
    real gap (not just adjacency) separates them -- see RULE_REGION_GAP_PT
    for why. Returns a list of region (x0, x1) bounds; the caller matches
    its own items back to a region by containment."""
    ordered = sorted(intervals, key=lambda iv: iv[0])
    regions = []
    for iv in ordered:
        x0, x1 = iv[0], iv[1]
        if regions and x0 - regions[-1][1] <= gap_pt:
            regions[-1] = (regions[-1][0], max(regions[-1][1], x1))
        else:
            regions.append((x0, x1))
    return regions


def _find_rule_lines(page):
    """Groups thin, dark filled rects sharing a y-position into candidate
    table rule lines. Some journals' PDF renderers draw a table's ruling
    as one short segment per column (with a real gap at each column
    boundary) instead of one continuous line -- confirmed on
    redox_neutral_2024/SI page 40, where find_tables() fails to recognize
    the table at all because of this. The segments' own x-ranges double as
    column boundaries once found.

    Segments are first split into x-regions (_cluster_x_regions) and
    grouped by y independently *within* each region -- see
    RULE_REGION_GAP_PT for why two side-by-side tables in a two-column
    layout must never be allowed to merge into one rule just because they
    share a y-position. Returns one sorted rule-line list *per region*
    (not a single flattened list) -- a flat list interleaves two columns'
    rules whenever their y-values happen to be close, and the caller's
    pairing logic (adjacent rules = one table's header bounds) would then
    pair a rule from one column with an unrelated rule from the other."""
    page_width = page.rect.width
    raw = []
    for d in page.get_drawings():
        rect = d["rect"]
        if rect.height > RULE_MAX_HEIGHT_PT or rect.width < 5:
            continue
        fill = d.get("fill")
        if fill is None or sum(fill[:3]) / 3 >= 0.3:  # dark fill only
            continue
        raw.append((rect.x0, rect.x1, round((rect.y0 + rect.y1) / 2, 1)))

    # A lone decorative rule (e.g. a heading underline) that happens to
    # straddle the gutter between two columns must not be allowed to bridge
    # the region split -- confirmed on suzuki_nhc_2026 page 4, where a 94pt
    # segment at y=76.5 (not part of any table, no sibling segment anywhere
    # near that y) sits across the column gap and re-merges Table 1's and
    # Table 3's regions into one, reproducing the exact bug this region
    # split exists to fix. Only segments that look like real rule-line
    # pieces -- part of a multi-segment y-band (a fragmented per-column
    # rule) or individually wide enough to be a genuine full rule on their
    # own -- are used to determine region boundaries.
    y_sibling_counts = {}
    for x0, x1, y_center in raw:
        key = next((k for k in y_sibling_counts if abs(k - y_center) <= RULE_Y_TOLERANCE_PT), y_center)
        y_sibling_counts[key] = y_sibling_counts.get(key, 0) + 1
    region_seed_raw = [
        (x0, x1, y_center)
        for x0, x1, y_center in raw
        if y_sibling_counts[next(k for k in y_sibling_counts if abs(k - y_center) <= RULE_Y_TOLERANCE_PT)] >= 2
        or (x1 - x0) >= page_width * RULE_MIN_TOTAL_WIDTH_FRAC
    ]

    regions = _cluster_x_regions(region_seed_raw)

    rule_lines_by_region = []
    for rx0, rx1 in regions:
        region_raw = [(x0, x1, y) for x0, x1, y in raw if rx0 <= x0 and x1 <= rx1]
        candidates = {}
        for x0, x1, y_center in region_raw:
            key = next((k for k in candidates if abs(k - y_center) <= RULE_Y_TOLERANCE_PT), y_center)
            candidates.setdefault(key, []).append((x0, x1))
        region_lines = []
        for y_center, segments in candidates.items():
            segments.sort()
            # Segments that overlap deeply are a PDF-export artifact, not
            # real per-column fragments -- confirmed on suzuki_nhc_2026's
            # own Table 1/3 header-bottom rule, where 3 segments overlap
            # each other by tens of points (e.g. (37.6,112.6) and
            # (59.0,253.5) overlap by 53.6pt) yet render as one visually
            # continuous line. But a genuine fragmented per-column rule can
            # *also* overlap its neighbor slightly, just by a rendering
            # rounding amount rather than a real one -- confirmed on
            # suzuki_nickel_2026/SI page 65's own closing rule, whose 4
            # legitimate per-column segments overlap by only ~0.24-0.49pt
            # each; merging on any overlap at all collapsed them to one
            # segment and broke this table's close_y match against its
            # 4-segment header. RULE_SEGMENT_OVERLAP_MERGE_PT sits well
            # above that rounding noise and well below the confirmed
            # artifact's overlap, so only the latter gets merged.
            merged = []
            for x0, x1 in segments:
                if merged and merged[-1][1] - x0 > RULE_SEGMENT_OVERLAP_MERGE_PT:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], x1))
                else:
                    merged.append((x0, x1))
            segments = merged
            total_width = sum(x1 - x0 for x0, x1 in segments)
            if total_width >= page_width * RULE_MIN_TOTAL_WIDTH_FRAC:
                region_lines.append((y_center, segments))
        region_lines.sort(key=lambda r: r[0])
        if region_lines:
            rule_lines_by_region.append(region_lines)
    return rule_lines_by_region


def _text_spans_in_band(page, y0, y1, x0, x1, y1_tol=1):
    """Like _text_in_band but keeps each span's own x1 too -- needed to
    cluster a header's words into columns by real gap (_cluster_header_
    columns) when the only column-boundary signal is the header text
    itself, not fragmented rule segments (see _borderless_from_rule_lines'
    single-continuous-rule case).

    y1_tol defaults to a small fuzz margin so a row starting right at
    start_y isn't excluded when this is used for row reconstruction (see
    _reconstruct_rows). But a header extraction call bounded by its own
    two rules must pass y1_tol=0: confirmed on suzuki_iron_2024/SI's
    Supplementary Table 19, where the first data row ("PhCl (1a)") starts
    only 0.25pt below the header-bottom rule -- comfortably inside the
    default +1pt margin -- and got vacuumed into the header text itself,
    producing a header with the row's own values concatenated into it
    and a duplicated first row."""
    entries = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for s in line["spans"]:
                text = s["text"].strip()
                if not text:
                    continue
                sx0, sy0, sx1 = s["bbox"][0], s["bbox"][1], s["bbox"][2]
                if y0 - 1 <= sy0 <= y1 + y1_tol and x0 - 5 <= sx0 <= x1 + 5:
                    entries.append((sy0, sx0, sx1, text))
    return entries


def _text_in_band(page, y0, y1, x0, x1, y1_tol=1):
    entries = []
    for sy0, sx0, sx1, text in _text_spans_in_band(page, y0, y1, x0, x1, y1_tol):
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


def _looks_like_new_row_gap(group_y0, row_y0s):
    """See NEW_ROW_VS_WRAP_FRAC. Only usable once >=2 rows have already
    been confirmed (need a real median spacing to compare against) --
    with fewer, falls back to treating a column-0-empty line as a wrap,
    the safer default when there's no established cadence yet to check
    against."""
    if len(row_y0s) < 2:
        return False
    spacings = [row_y0s[k] - row_y0s[k - 1] for k in range(1, len(row_y0s))]
    median = sorted(spacings)[len(spacings) // 2]
    if median <= 0:
        return False
    return (group_y0 - row_y0s[-1]) >= median * NEW_ROW_VS_WRAP_FRAC


def _reconstruct_rows(page, col_anchors, table_x0, table_x1, start_y, close_y):
    """PDF-text-backed wrapper around _reconstruct_rows_from_entries -- see
    that function for the actual reconstruction logic. Kept separate so
    the OCR-backed path (_ocr_reconstruct_rows, for tables with no real
    text/vector content at all) can feed the same core algorithm from a
    different entries source without duplicating it."""
    entries = _text_in_band(page, start_y, close_y if close_y else start_y + 2000, table_x0, table_x1)
    return _reconstruct_rows_from_entries(entries, col_anchors, start_y, close_y)


def _reconstruct_rows_from_entries(entries, col_anchors, start_y, close_y):
    """Groups (y0, x0, text) entries into rows, buckets each row's text
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
    this was first validated against allow. The general signal: a genuine
    new row has its first (leftmost/anchor) column populated -- a wrapped
    continuation line doesn't, since it's continuing a *later* column's
    cell, never restarting the row's own index/key value. So a physical
    line with column 0 empty is normally appended onto the still-open
    previous row instead of starting a new one -- UNLESS its gap from the
    last row looks like a full row apart rather than a tight wrap (see
    _looks_like_new_row_gap/NEW_ROW_VS_WRAP_FRAC), which means it's a
    real new row that's simply missing its own column-0 value (confirmed
    necessary on nickelocene_2025's Table 2/OCR path: entry 2's catalyst
    name never got OCR'd at all, and without this check that row silently
    vanished into entry 1's instead of becoming its own row). Validated
    end-to-end against four real tables: suzuki_nickel_2026 Table 1
    (20/20 rows, single-line); redox_neutral_2024/SI Table S6 (10/10
    rows, single-line); miyaura_iron_2025/SI's GC-MS table continuing
    from page 27 onto page 28 (correctly reconstructs the wrapped m/z
    cells there, previously produced a garbled partial row under the old
    one-line-per-row model); nickelocene_2025's Table 2, OCR-sourced (see
    above -- entry 2 now correctly recovered as its own row).

    Returns (rows, last_row_y1) -- the latter approximates the bottom edge
    of the actual last reconstructed row (not just the header/start_y),
    needed so a caller checking "does this table end near the page
    bottom" (for continuation-merging across a page break) isn't comparing
    against the wrong, far-too-early position when there's no closing
    rule to give an exact bound."""
    entries = sorted(entries, key=lambda e: (e[0], e[1]))

    # Compared against the group's own first (topmost) y0, not the previous
    # entry's y0 -- comparing to the previous entry lets tolerance chain
    # transitively across genuinely different rows: confirmed on
    # suzuki_nhc_2026's Table 2, where row 20's subscript ("4" in "K3PO4",
    # y=523.55) sat only 4.65pt from row 21's superscript footnote marker
    # ("d", y=528.20) -- each within ROW_Y_TOLERANCE_PT of its neighbor,
    # chaining the two rows' entries into one line even though the rows'
    # own baselines are 8.56pt apart, well over tolerance. Comparing to the
    # group's first y0 instead still correctly merges a single physical
    # line's own super/subscript spread (confirmed within ~4pt here) while
    # refusing to bridge into the next row.
    lines = []
    current, group_y0 = [], None
    for y0, x0, text in entries:
        if group_y0 is None or abs(y0 - group_y0) <= ROW_Y_TOLERANCE_PT:
            current.append((x0, text))
            group_y0 = group_y0 if group_y0 is not None else y0
        else:
            lines.append((group_y0, current))
            current, group_y0 = [(x0, text)], y0
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

        if 0 not in matched and not _looks_like_new_row_gap(group_y0, row_y0s):
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
    # A real column label is never a bare number by itself (even a short
    # one -- "T" always has its letter) -- confirmed necessary on
    # suzuki_nhc_2026, where a rule-pair mismatch produced a garbled
    # 3-cell "header" (['7', '[PtCl2(DMS)]', 'NR']) that otherwise passed
    # the word-majority check below (2 of 3 cells have real letters).
    if any(re.fullmatch(r"\d+", h.strip()) for h in nonempty):
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
    rule_lines_by_region = _find_rule_lines(page)
    results = []
    for rules in rule_lines_by_region:
        for i in range(len(rules) - 1):
            y_a, segs_a = rules[i]
            y_b, segs_b = rules[i + 1]
            if len(segs_a) != len(segs_b) or not (3 <= y_b - y_a <= HEADER_ROW_GAP_MAX_PT):
                continue
            table_x0, table_x1 = segs_b[0][0], segs_b[-1][1]

            if len(segs_b) > 1:
                # fragmented per-column segments double as column boundaries
                col_anchors = [(x0 + x1) / 2 for x0, x1 in segs_b]
                header_entries = _text_in_band(page, y_a, y_b, table_x0, table_x1, y1_tol=0)
                if not header_entries:
                    continue
                header_buckets = [[] for _ in col_anchors]
                for sy0, sx0, text in header_entries:
                    nearest = min(range(len(col_anchors)), key=lambda k: abs(col_anchors[k] - sx0))
                    header_buckets[nearest].append(text)
                header = [" ".join(b).strip() for b in header_buckets]
            else:
                # a single continuous rule carries no per-column information
                # at all -- confirmed on nickelocene_2025's main Table 1 and
                # SI Tables S1/S2, both ruled with one solid line rather than
                # fragmented segments. Derive columns from the header text's
                # own word spacing instead (the same approach Path C's OCR
                # detector uses for the same underlying problem).
                header_spans = _text_spans_in_band(page, y_a, y_b, table_x0, table_x1, y1_tol=0)
                if len(header_spans) < 2:
                    continue
                header_words = [(sx0, sx1, text) for sy0, sx0, sx1, text in header_spans]
                col_anchors, header = _cluster_header_columns(sorted(header_words))
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


# --- Path C: raster-image tables (OCR) ------------------------------------

# A raster-image table has real visual structure to key off even though
# it has zero extractable text or vector paths -- the same journal house
# style seen everywhere else in this corpus (a header row bounded by bold
# rule lines, sometimes a shaded background too) is still *rendered* into
# the picture, just as pixels instead of PDF drawing/text objects. Tried
# inferring the header from OCR content instead (a fixed column-name
# vocabulary, then a structural best-fit search over every line) and
# confirmed both too unreliable: nickelocene_2025's own two tables use
# different column names entirely ("[Ni]/base/additive/solvent" vs.
# "metallocene/equivalents"), and the best-fit search picked a data row
# over the real header on the second table (a coincidental column-anchor
# fit). Detecting the same bold rule lines this session's vector-based
# Path B already looks for, but directly in the rendered pixels, is both
# more reliable and simpler -- confirmed identical rule positions (to
# 0.3pt) on both of nickelocene_2025's tables despite their completely
# different column layouts, since both share the same journal template.
OCR_RULE_DARK_PIXEL_THRESHOLD = 100  # 0-255 grayscale; a genuine printed
                                       # rule is near-black
OCR_RULE_DARK_FRAC = 0.7  # fraction of a pixel-row that must be this dark
                            # to count as a rule spanning the image width
                            # -- confirmed real rules measure ~0.99,
                            # ordinary text rows measure far lower (text
                            # only darkens the small fraction of a row's
                            # width the glyphs themselves occupy)
OCR_HEADER_RULE_GAP_MAX_PT = 15.0  # the two rules bounding a header sit
                                     # one text line apart -- confirmed
                                     # ~9.3pt on both nickelocene tables
OCR_TABLE_ZOOM = 5.0  # checked directly: 3.0 (this module's original
                        # default) missed real words at a meaningfully
                        # higher rate on nickelocene_2025's Table 1 (94
                        # words recognized vs. 136 at 5.0) -- higher DPI
                        # matters more here than in decimer_extract.py's
                        # OCR checks, which upscale an already-cropped
                        # single-structure image rather than a whole
                        # multi-row table region
OCR_TABLE_MIN_CONFIDENCE = 40
OCR_HEADER_COLUMN_GAP_PT = 15.0
# A caption's own table must start close below it -- generous enough to
# clear a reaction-scheme graphic sitting between the caption and the
# table's own header (confirmed gap on nickelocene_2025 Table 1: caption
# sits directly above the combined scheme+table raster image).
OCR_CAPTION_TO_IMAGE_MAX_GAP_PT = 40


def _render_table_region(page, rect, zoom=OCR_TABLE_ZOOM):
    """Renders a raster-image region once for both pixel-based rule
    detection and OCR, so callers needing both don't pay for two
    pixmaps. Returns (PIL RGB image, grayscale numpy array)."""
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    gray = np.array(img.convert("L"))
    return img, gray


def _find_pixel_rule_bands(gray, rect, zoom=OCR_TABLE_ZOOM):
    """Finds horizontal rule lines directly in the rendered pixels -- see
    the Path C module comment for why this replaced inferring the header
    from OCR content. A rule is a pixel-row where a large fraction of
    pixels are near-black (a solid printed line spans nearly the full
    image width; ordinary text only darkens the narrow fraction of a
    row's width its glyphs occupy). Adjacent matching rows (a rule is
    several pixels thick at this zoom) are merged into one band. Returns
    a list of (y0, y1) rule bands in PDF-point coordinates, top to
    bottom."""
    dark_frac = (gray < OCR_RULE_DARK_PIXEL_THRESHOLD).mean(axis=1)
    rule_rows = [i for i, f in enumerate(dark_frac) if f > OCR_RULE_DARK_FRAC]
    bands = []
    for i in rule_rows:
        if bands and i - bands[-1][-1] <= 2:
            bands[-1].append(i)
        else:
            bands.append([i])
    return [(rect.y0 + b[0] / zoom, rect.y0 + b[-1] / zoom) for b in bands]


def _find_ocr_header_bounds(rule_bands):
    """The header sits between the first two rule bands close enough
    together to be one text line apart (see OCR_HEADER_RULE_GAP_MAX_PT) --
    confirmed identical on both of nickelocene_2025's tables (~9.3pt
    apart) despite their different column layouts. Deliberately returns
    on the FIRST qualifying pair, scanning top to bottom, rather than
    picking whichever pair looks "best" some other way: the real header
    is always the table's first row (continuation pages, with no header
    of their own, are handled separately -- see _try_continue_table), and
    a highlighted/colored data row further down can carry its own
    visual marker too (confirmed on redox_neutral_2024/SI page 40's
    green-highlighted entry 8, which is exactly the kind of false
    candidate a "pick whichever looks most header-like" rule would risk
    picking over the real one). Returns (header_y0, header_y1) -- the
    OCR'd text band -- or None if no such pair exists (not every raster
    image under a "Table N" caption is actually a table with this rule
    style; safer to find nothing than to guess)."""
    for i in range(len(rule_bands) - 1):
        y0 = rule_bands[i][1]
        y1 = rule_bands[i + 1][0]
        if 0 < y1 - y0 <= OCR_HEADER_RULE_GAP_MAX_PT:
            return y0, y1
    return None


def _ocr_words_in_image(img, rect, zoom=OCR_TABLE_ZOOM):
    """OCRs an already-rendered region image and returns (y0, x0, x1,
    text) entries in PDF-point coordinates -- x1 is kept alongside x0
    here (unlike the PDF-text paths) because OCR gives no separate
    per-word column structure to lean on; finding column boundaries has
    to start from real word widths.

    `--psm 6` (assume one uniform text block) matters here, the same way
    it already does for decimer_extract.py's own OCR checks: tesseract's
    default full-page auto-segmentation badly misreads a narrow, mostly-
    numeric column in isolation (confirmed on nickelocene_2025's Table 2
    -- clearly legible entry numbers "1".."6" came back as "=", "o",
    "na", "&", garbage under the default mode, and correctly as
    "1".."6" at 95%+ confidence under psm 6). Checked this doesn't cost
    anything on a wider/busier table -- word count was flat on Table 1
    (129 vs. 131) while recovering 18 more real words on Table 2."""
    data = pytesseract.image_to_data(img, config="--psm 6", output_type=pytesseract.Output.DICT)
    entries = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < OCR_TABLE_MIN_CONFIDENCE:
            continue
        x0 = rect.x0 + data["left"][i] / zoom
        y0 = rect.y0 + data["top"][i] / zoom
        x1 = x0 + data["width"][i] / zoom
        entries.append((y0, x0, x1, text))
    return entries


def _cluster_header_columns(words, gap_pt=OCR_HEADER_COLUMN_GAP_PT):
    """Groups a header line's individual OCR words into columns by real
    x-gap (e.g. "yield" and "(%)" are two separate OCR tokens that belong
    to one "yield (%)" column) -- the same gap-based clustering principle
    pdf_ingest.py's cluster_drawing_rects uses for vector rects, applied
    to 1D word spans instead. Returns (col_anchors, header_cell_texts)."""
    clusters = [[words[0]]]
    for x0, x1, text in words[1:]:
        prev_x0, prev_x1, prev_text = clusters[-1][-1]
        if x0 - prev_x1 > gap_pt:
            clusters.append([])
        clusters[-1].append((x0, x1, text))
    anchors = [sum((x0 + x1) / 2 for x0, x1, _ in c) / len(c) for c in clusters]
    header = [" ".join(t for _, _, t in c) for c in clusters]
    return anchors, header


def _extract_ocr_table(page, image_rect):
    """Path C entry point: renders a raster-image region once, finds its
    header from bold rule lines in the pixels themselves (see the Path C
    module comment for why -- OCR-content-based header detection was
    tried twice and confirmed too unreliable), OCRs the header band for
    column labels and the rest of the region for data, then reconstructs
    rows with the same shared core the text-based paths use
    (_reconstruct_rows_from_entries): column-bucketing and row-stopping
    logic doesn't care whether an entry came from real PDF text or an OCR
    word, both are just (y0, x0, text)."""
    img, gray = _render_table_region(page, image_rect)
    rule_bands = _find_pixel_rule_bands(gray, image_rect)
    header_bounds = _find_ocr_header_bounds(rule_bands)
    if header_bounds is None:
        return None
    header_y0, header_y1 = header_bounds

    entries = _ocr_words_in_image(img, image_rect)
    if not entries:
        return None
    header_words = [(x0, x1, text) for y0, x0, x1, text in entries if header_y0 - 1 <= y0 <= header_y1 + 1]
    if len(header_words) < 2:
        return None
    col_anchors, header = _cluster_header_columns(sorted(header_words))
    if not _header_looks_valid(header):
        return None

    # header_y1 is the rule's own bottom edge, already a precise bound --
    # only a small buffer is needed to skip the header text itself (not
    # a full ROW_Y_TOLERANCE_PT, confirmed that margin was wide enough to
    # exclude row 1 by 0.01pt on nickelocene_2025's Table 1).
    row_entries = [(y0, x0, text) for y0, x0, x1, text in entries if y0 > header_y1 + 1]
    rows, last_y1 = _reconstruct_rows_from_entries(row_entries, col_anchors, header_y1, None)
    if not rows:
        return None

    # A raster-image table has no rule/caption below it to bound
    # reconstruction with (unlike Paths A/B), and this corpus's
    # optimization tables are routinely followed immediately by ligand/
    # catalyst structure drawings with no real gap -- confirmed on
    # nickelocene_2025's Table 1, where OCR noise from those drawings
    # ("aad ipr...", "iPr Cl UNi dipp...") cleared the spacing-deviation
    # check (the drawings start close enough below the last real row that
    # the y-gap alone doesn't look anomalous) and got appended as two
    # garbage trailing rows. Every real row in every table extracted by
    # any path this session uses a small integer entry number, optionally
    # with a lowercase letter suffix, in column 0 -- truncate at the
    # first row that doesn't, rather than trust the spacing/column-count
    # heuristics alone for content with no real structural bound at all.
    entry_number_re = re.compile(r"^\d{1,3}[a-z]?$")
    for i, row in enumerate(rows):
        if not entry_number_re.match(row[0].strip()):
            rows = rows[:i]
            break
    if not rows:
        return None

    return {
        "header": header,
        "rows": rows,
        "bbox": [image_rect.x0, image_rect.y0, image_rect.x1, last_y1],
        "col_anchors": col_anchors,
        "closed": False,
    }


def _extract_ocr_tables(page):
    """Only attempted for a raster image sitting close below this page's
    own "Table N" caption -- OCR is expensive and noisy, not run
    speculatively on every raster image on a page (most are real figures,
    already Stage 1/3's job)."""
    results = []
    for block in page.get_text("blocks"):
        bx0, by0, bx1, by1, text, _, block_type = block
        if block_type != 0 or not CAPTION_RE.match(text.strip()):
            continue
        caption_text = text.strip()
        for img in page.get_images(full=True):
            xref = img[0]
            for rect in page.get_image_rects(xref):
                if not (by1 - 5 <= rect.y0 <= by1 + OCR_CAPTION_TO_IMAGE_MAX_GAP_PT):
                    continue
                table = _extract_ocr_table(page, rect)
                if table:
                    table["caption"] = caption_text
                    results.append(table)
    return results


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
