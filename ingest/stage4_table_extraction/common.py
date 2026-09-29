"""Shared helpers used by more than one of table_extract.py's detection
paths (bordered.py / borderless.py / raster_ocr.py) -- row/header
reconstruction doesn't care whether an entry came from real PDF text, a
fragmented rule line, or an OCR word, so that shared core lives here
instead of being duplicated per path."""

import re

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

# A chart/graph's axis-tick labels and equation annotations can pack
# densely enough into a small grid to also pass the two filters above --
# confirmed on suzuki_iron_2024/SI pages 84-85 (kinetics plots), where
# find_tables() misread the tick-label grid as a legitimate small table.
# Real header cells in this corpus are short labels ("T (°C)", "Fe Source
# (mol%)", 11-17 chars); the chart-garbage cells are 100+ chars of
# concatenated, newline-joined tick values -- a clean, evidence-based
# length cutoff, not a guess.
MAX_HEADER_CELL_CHARS = 60

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
                                # was found (see borderless.py's RULE_*)
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


OCR_HEADER_COLUMN_GAP_PT = 15.0


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
