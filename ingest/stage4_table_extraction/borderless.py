"""Path B: "borderless" data tables -- common specifically for reaction-
optimization/screening tables (catalyst, ligand, T, yield, ee columns) in
both main text and SI, in this corpus's journals' house style: only a
header row and a closing rule, no grid lines between data rows -- are
MISSED by PyMuPDF's `find_tables()` entirely (confirmed: it detects at
most the header row, sometimes not even that). These need to be
reconstructed directly from text position, seeded from a header row
detected one of two ways (see `_extract_borderless_tables`): a header
with some visual marker `find_tables()` itself can key off (a shaded
background band, confirmed on suzuki_nickel_2026's Table 1), or -- when
even that's missing, confirmed on redox_neutral_2024/SI's Table S6 -- the
ruling lines above/below the header, which can be drawn as several short
per-column segments rather than one continuous line (also why
`find_tables()` misses the whole table: its line-detector expects
continuous rules). The segment gaps directly give column boundaries. See
table_extract.py's module docstring for the corpus-wide survey behind
this split."""

from common import _cluster_header_columns, _header_looks_valid, _reconstruct_rows_from_entries

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


def _reconstruct_rows(page, col_anchors, table_x0, table_x1, start_y, close_y):
    """PDF-text-backed wrapper around _reconstruct_rows_from_entries -- see
    that function (common.py) for the actual reconstruction logic. Kept
    separate so the OCR-backed path (raster_ocr.py, for tables with no
    real text/vector content at all) can feed the same core algorithm
    from a different entries source without duplicating it."""
    entries = _text_in_band(page, start_y, close_y if close_y else start_y + 2000, table_x0, table_x1)
    return _reconstruct_rows_from_entries(entries, col_anchors, start_y, close_y)


def _borderless_from_visual_header(page):
    """Path B, header source 1: find_tables() can itself find a header row
    when it has some visual marker to key off (e.g. a shaded background
    band, confirmed on suzuki_nickel_2026's Table 1) even though it can't
    see the borderless body below it. Reads the header via t.extract()[0],
    never t.header.names -- see bordered.py's _is_real_data_table
    docstring for why that property is unreliable."""
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
