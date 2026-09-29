"""Path C: raster-image tables. A raster-image table has real visual
structure to key off even though it has zero extractable text or vector
paths -- the same journal house style seen everywhere else in this
corpus (a header row bounded by bold rule lines, sometimes a shaded
background too) is still *rendered* into the picture, just as pixels
instead of PDF drawing/text objects. Tried inferring the header from OCR
content instead (a fixed column-name vocabulary, then a structural
best-fit search over every line) and confirmed both too unreliable:
nickelocene_2025's own two tables use different column names entirely
("[Ni]/base/additive/solvent" vs. "metallocene/equivalents"), and the
best-fit search picked a data row over the real header on the second
table (a coincidental column-anchor fit). Detecting the same bold rule
lines Path B (borderless.py) already looks for, but directly in the
rendered pixels, is both more reliable and simpler -- confirmed identical
rule positions (to 0.3pt) on both of nickelocene_2025's tables despite
their completely different column layouts, since both share the same
journal template."""

import re

import fitz
import numpy as np
import pytesseract
from PIL import Image

from common import CAPTION_RE, _cluster_header_columns, _header_looks_valid, _reconstruct_rows_from_entries

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
    the module docstring for why this replaced inferring the header from
    OCR content. A rule is a pixel-row where a large fraction of pixels
    are near-black (a solid printed line spans nearly the full image
    width; ordinary text only darkens the narrow fraction of a row's
    width its glyphs occupy). Adjacent matching rows (a rule is several
    pixels thick at this zoom) are merged into one band. Returns a list
    of (y0, y1) rule bands in PDF-point coordinates, top to bottom."""
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
    of their own, are handled separately -- see table_extract.py's
    _try_continue_table), and a highlighted/colored data row further
    down can carry its own visual marker too (confirmed on
    redox_neutral_2024/SI page 40's green-highlighted entry 8, which is
    exactly the kind of false candidate a "pick whichever looks most
    header-like" rule would risk picking over the real one). Returns
    (header_y0, header_y1) -- the OCR'd text band -- or None if no such
    pair exists (not every raster image under a "Table N" caption is
    actually a table with this rule style; safer to find nothing than to
    guess)."""
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


def _extract_ocr_table(page, image_rect):
    """Path C entry point: renders a raster-image region once, finds its
    header from bold rule lines in the pixels themselves (see the module
    docstring for why -- OCR-content-based header detection was tried
    twice and confirmed too unreliable), OCRs the header band for column
    labels and the rest of the region for data, then reconstructs rows
    with the same shared core the text-based paths use
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
