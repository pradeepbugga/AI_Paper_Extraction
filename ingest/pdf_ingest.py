import io
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


def extract_text_blocks(page):
    blocks = []
    for x0, y0, x1, y1, text, block_no, block_type in page.get_text("blocks"):
        if block_type != 0 or not text.strip():
            continue
        blocks.append({
            "block_id": block_no,
            "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
            "text": text.strip(),
        })
    return blocks


MIN_RASTER_DIMENSION_PT = 72  # 1 inch


def is_decorative_raster(bbox):
    """Publisher templates embed small raster images that are real PDF
    content but not figures -- journal cover-art thumbnails, "Editors'
    Choice" badges, publisher logos, corner marks. These cluster almost
    exclusively on page 1 and, across a sample of six papers, none of them
    reach even 1 inch in either dimension while every genuine figure does
    -- a clean, document-relative size gap rather than a guessed constant."""
    return bbox.width < MIN_RASTER_DIMENSION_PT and bbox.height < MIN_RASTER_DIMENSION_PT


STRIP_ASPECT_RATIO = 2.0  # a fragment must be at least this much wider than tall
MIN_STRIP_CLUSTER_SIZE = 3
STRIP_TOLERANCE_PT = 2.0


def _find_strip_clusters(entries):
    """Detects raster fragments that are one taller image sliced into
    horizontal strips on export -- same x-extent, y-ranges tiling
    contiguously with no gap, each strip much wider than tall. Seen in SI
    spectra pages (a landscape chart rotated + sliced into bands on
    embedding); never in main text, since publishers don't ship sideways
    figures there. Purely geometric, no format/rotation metadata needed --
    see pdf_ingest.py's investigation of this exact case, which found no
    such metadata exists to detect it from."""
    used = set()
    clusters = []
    ordered = sorted(range(len(entries)), key=lambda i: entries[i]["bbox"].y0)

    for idx in ordered:
        if idx in used:
            continue
        bbox = entries[idx]["bbox"]
        if bbox.width < bbox.height * STRIP_ASPECT_RATIO:
            continue

        group = [idx]
        group_used = {idx}
        cur_y1 = bbox.y1
        for j in ordered:
            if j in group_used or j in used:
                continue
            ob = entries[j]["bbox"]
            same_x = abs(ob.x0 - bbox.x0) < STRIP_TOLERANCE_PT and abs(ob.x1 - bbox.x1) < STRIP_TOLERANCE_PT
            contiguous_y = abs(ob.y0 - cur_y1) < STRIP_TOLERANCE_PT
            if same_x and contiguous_y:
                group.append(j)
                group_used.add(j)
                cur_y1 = ob.y1

        if len(group) >= MIN_STRIP_CLUSTER_SIZE:
            clusters.append(group)
            used |= group_used

    return clusters, used


def extract_raster_images(doc, page, page_number, images_dir, doc_source, zoom=3.0):
    entries = []
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        entries.append({"img_index": img_index, "xref": xref, "bbox": rects[0]})

    strip_clusters, used_in_strips = ([], set())
    if doc_source == "SI":
        strip_clusters, used_in_strips = _find_strip_clusters(entries)

    images = []
    matrix = fitz.Matrix(zoom, zoom)

    for cluster_index, group in enumerate(strip_clusters):
        union_bbox = fitz.Rect(entries[group[0]]["bbox"])
        for i in group[1:]:
            union_bbox |= entries[i]["bbox"]
        pix = page.get_pixmap(matrix=matrix, clip=union_bbox)
        pil_image = Image.open(io.BytesIO(pix.tobytes("png"))).rotate(-90, expand=True)
        image_filename = f"page{page_number}_rasterstrip{cluster_index}.png"
        pil_image.save(images_dir / image_filename)
        images.append({
            "image_id": f"{doc_source}_p{page_number}_rasterstrip{cluster_index}",
            "source": "embedded_raster_strip_merged",
            "bbox": [round(union_bbox.x0, 1), round(union_bbox.y0, 1), round(union_bbox.x1, 1), round(union_bbox.y1, 1)],
            "path": f"{images_dir.name}/{image_filename}",
            "width": pil_image.width,
            "height": pil_image.height,
        })
        pix = None

    for i, entry in enumerate(entries):
        if i in used_in_strips:
            continue
        bbox = entry["bbox"]
        if is_decorative_raster(bbox):
            continue
        pix = fitz.Pixmap(doc, entry["xref"])
        if pix.n - pix.alpha >= 4:  # CMYK -> RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        image_filename = f"page{page_number}_raster{entry['img_index']}.png"
        pix.save(images_dir / image_filename)
        images.append({
            "image_id": f"{doc_source}_p{page_number}_raster{entry['img_index']}",
            "source": "embedded_raster",
            "bbox": [round(bbox.x0, 1), round(bbox.y0, 1), round(bbox.x1, 1), round(bbox.y1, 1)],
            "path": f"{images_dir.name}/{image_filename}",
            "width": pix.width,
            "height": pix.height,
        })
        pix = None
    return images


def cluster_drawing_rects(rects, pad_x=8.0, pad_y=25.0, min_area=2000.0):
    """Merge nearby/overlapping vector-drawing bboxes into figure-sized regions.

    Journal figures (chemical structures, plots) are almost always built from
    hundreds of individual vector path ops, not one embedded image. Padding
    each rect before checking overlap lets disjoint strokes that belong to
    the same drawing (e.g. a bond line and an atom label gap) merge together.

    Padding is asymmetric on purpose. A multi-panel scheme (e.g. panels a-e
    stacked down a page, each with its own caption and generous whitespace
    before the next) needs a generous *vertical* gap tolerance -- observed
    gaps up to ~23pt between panels that visually belong to one figure.
    Horizontally, that same generosity is actively harmful: on a two-column
    page, the gap between columns is often close to that same distance, so a
    wide pad_x can bridge across it and merge a left-column figure with
    unrelated right-column body text. Real within-figure horizontal gaps
    (e.g. a bond and an atom label) are much smaller than a column gutter,
    so a tight pad_x doesn't cost us anything there.
    """
    clusters = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]:
                continue
            base = fitz.Rect(clusters[i])
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                padded = fitz.Rect(base.x0 - pad_x, base.y0 - pad_y, base.x1 + pad_x, base.y1 + pad_y)
                if padded.intersects(clusters[j]):
                    base |= clusters[j]
                    used[j] = True
                    changed = True
            used[i] = True
            merged.append(base)
        clusters = merged

    return [c for c in clusters if c.width * c.height >= min_area]


def is_invisible_drawing(d, white_tol=0.02):
    """A filled-white rect with no stroke renders no visible ink — some
    publisher templates draw one as a full-page/column background or layout
    guide. Left in, a single such rect can dwarf every real figure on the
    page and drag unrelated content into its cluster."""
    fill = d.get("fill")
    stroke = d.get("color")
    if stroke is not None:
        return False
    if fill is None:
        return False
    return all(abs(c - 1.0) < white_tol for c in fill)


DEFAULT_PAD_X = 8.0
SINGLE_COLUMN_PAD_X = 12.0  # no column gutter to guard against at all -- see measure_column_gutter's None case
PAD_X_SAFETY_BUFFER_PT = 6.0  # stay this far clear of a document's own measured gutter, never just barely under it
MAX_PAD_X = 20.0


def measure_column_gutter(doc, sample_pages=8):
    """Finds the narrowest real gap between two-column body text across a
    sample of pages, so extract_vector_figures can widen its horizontal
    merge tolerance (pad_x) up to -- but staying safely clear of -- what
    this specific document's own layout actually uses as a column gutter,
    instead of guessing one fixed constant for every document.

    Returns None if no multi-column page is found in the sample. This
    matters beyond "nothing to stay clear of": SI documents in this corpus
    are commonly single-column, one-compound-per-page (name, structure,
    two spectra stacked vertically) -- confirmed directly on a real case
    where a structure's own colored sub-groups (a two-tone ChemDraw
    highlight) sat 9.1pt apart, just past the then-fixed 8.0 default, and
    fragmented into separate images that DECIMER Segmentation then
    mis-cropped further downstream. There the two spectra are raster
    images (a separate extraction path entirely) and the two repeated
    structure instances sit ~280pt apart vertically -- far past pad_y's
    reach -- so a wider pad_x here has nothing false to merge with. That
    won't hold for every single-column layout (a main-text scope table
    packs many distinct compounds' vector structures onto one page, where
    over-merging is a real risk), so this is a modest, evidence-sized
    bump (SINGLE_COLUMN_PAD_X), not the same generous ceiling a confirmed
    real gutter earns.

    Splitting blocks at a fixed x (e.g. page-width midpoint) breaks the
    moment one block straddles that line -- one early attempt at this
    picked up a block that started just left of the true column edge but
    ran on across most of the right column, which corrupted the whole
    page's measurement into a large negative "gap." Finding the actual
    empty band between merged block intervals sidesteps that -- it never
    assumes where the split is.

    Two more filters, both earned by checking a real false positive rather
    than guessed upfront: narrow blocks only (< 0.55 * page width) drops a
    full-width block (a caption, a table spanning both columns) that would
    otherwise report a fake zero/negative gap where it overlaps both
    halves; a minimum height (> 24pt, roughly two text lines) drops both
    individual chemical-shift/atom-label fragments (each its own tiny
    single-line text block in this corpus) and, less obviously, multi-line
    reaction-scheme annotation labels ("S / O", "Cl / S / O" stacked
    vertically) -- tall enough to clear a naive line-count check despite
    not being body prose at all. Neither filter alone was enough; both
    were added only after a specific real page produced a wrong gutter
    without it.
    """
    gutters = []
    for page in list(doc)[:sample_pages]:
        page_width = page.rect.width
        blocks = [
            b for b in page.get_text("blocks")
            if b[6] == 0 and (b[2] - b[0]) < page_width * 0.55 and (b[3] - b[1]) > 24
        ]
        intervals = sorted((b[0], b[2]) for b in blocks)
        if not intervals:
            continue
        merged = [list(intervals[0])]
        for x0, x1 in intervals[1:]:
            if x0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], x1)
            else:
                merged.append([x0, x1])
        if len(merged) < 2:
            continue
        page_gap = max(merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1))
        gutters.append(page_gap)
    return min(gutters) if gutters else None


CROP_MARGIN_PT = 5.0  # see extract_vector_figures docstring for why this exists

# Atom labels (S, O, N-H, CO2Me, ...) are frequently real embedded PDF text,
# not vector-drawn glyphs -- page.get_drawings() never sees them, so a crop
# built purely from vector paths can slice a label in half at the crop edge.
# Confirmed directly: N-phenylmethanesulfonamide's second "O" (a 3-line text
# block "N\nH\nO") got cut mid-glyph because the vector-only cluster ended
# right where that text block did, sized only for its own bond geometry.
# Checked across a 30-page sample before picking these thresholds: height is
# a reliable signal (short label-style text blocks top out at 33.4pt in that
# sample; real prose runs to many more lines), width has a long tail even
# for short text (a handful of characters can still span 200+pt depending on
# spacing), so it's capped tighter and backed by an explicit char/line-count
# gate rather than relied on alone.
LABEL_TEXT_MAX_WIDTH_PT = 60.0
LABEL_TEXT_MAX_HEIGHT_PT = 35.0
LABEL_TEXT_MAX_CHARS = 20

# Block-level pre-filter, applied before the per-line checks below. A real
# label cluster (atom labels scattered around one structure) stays within
# a modest width and a handful of lines even when PyMuPDF's block
# segmentation bundles several of them together (confirmed: the H2N case
# below is 91.1pt / 4 lines). Real prose blocks are much wider (a title or
# SI-procedure paragraph measured 451-454pt in this corpus) -- checked
# first, because per-line dimensions alone aren't enough to tell the two
# apart (see _label_text_rects docstring).
LABEL_BLOCK_MAX_WIDTH_PT = 150.0
LABEL_BLOCK_MAX_LINES = 6


def _label_text_rects(page):
    """Operates on individual text *lines* (page.get_text("dict")), not
    whole blocks -- PyMuPDF's block segmentation can bundle several
    unrelated, spatially-separated labels into one wide block. Confirmed
    directly on copper_iron_2025/page89 (SI): "N", "H", "O" (the amide
    N-H, next to the S=O) and "H2N" (a separate amino group ~66pt further
    left, on the opposite side of the structure) landed in one PyMuPDF
    block spanning 91.1pt -- comfortably over LABEL_TEXT_MAX_WIDTH_PT, so
    a block-level-only check discarded all four labels at once, including
    "H2N", which was the only text anchoring the crop's left edge.

    Per-line dimensions alone are NOT a safe replacement for the block
    check, though -- confirmed by a real regression on copper_iron_2025/
    page32 (SI): narrow-column justified paragraph text (an SI procedure
    write-up wrapping around a structure) breaks into individual lines
    like "procedure ", "for ", "15 h," that are each well under the
    line-level width/char thresholds on their own, even though the parent
    block is obviously prose (15 lines, 454pt wide). So LABEL_BLOCK_MAX_*
    gates the enclosing block first (comfortably wider/more-lines than any
    real label cluster, comfortably narrower/shorter than the observed
    prose blocks), and only lines within a block that passes get checked
    individually and kept."""
    rects = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = block["bbox"]
        if bx1 - bx0 > LABEL_BLOCK_MAX_WIDTH_PT or len(block["lines"]) > LABEL_BLOCK_MAX_LINES:
            continue
        for line in block["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            if x1 - x0 > LABEL_TEXT_MAX_WIDTH_PT or y1 - y0 > LABEL_TEXT_MAX_HEIGHT_PT:
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text or len(text) > LABEL_TEXT_MAX_CHARS:
                continue
            rects.append(fitz.Rect(x0, y0, x1, y1))
    return rects


def extract_vector_figures(page, page_number, images_dir, doc_source, zoom=3.0, pad_x=DEFAULT_PAD_X):
    """pad_x/pad_y (via cluster_drawing_rects) only ever decide whether two
    nearby vector-drawing rects get merged into the same cluster -- they are
    a clustering *tolerance*, never applied to the resulting bbox. A cluster
    that never merges with anything (a small, isolated, standalone
    structure -- no neighbor within reach to test against) keeps the exact
    tight bounding box of its own ink, zero margin, no matter how generous
    pad_x/pad_y are. Confirmed as the root cause of two separate downstream
    symptoms discovered independently: DECIMER OCSR misreading a bond that
    sits exactly on the crop edge (page4_fig0_seg1.png, a chlorobenzene
    misread as containing a nonexistent iodine), and DECIMER Segmentation
    detecting zero structures at all on some isolated compounds (15 of 18
    zero-segment figures corpus-wide traced to this same zero-margin
    pattern). Fixed at the source here -- a small fixed margin added to
    every rendered crop -- rather than patched separately in each
    downstream consumer.

    Small label-style text blocks (see _label_text_rects) are folded into
    the same clustering pass as the vector rects, for the separate reason
    above -- atom labels can be real text, not vector paths, and a crop
    sized only from vector geometry can truncate one. A cluster is kept
    only if at least one real vector rect intersects it, so a stray small
    text fragment elsewhere on the page (a footnote marker, a page number)
    can never form a "figure" on text alone -- it only ever extends an
    already-real structure cluster.
    """
    drawings = page.get_drawings()
    if not drawings:
        return []

    vector_rects = [d["rect"] for d in drawings
                     if d["rect"].width > 0 and d["rect"].height > 0 and not is_invisible_drawing(d)]
    label_rects = _label_text_rects(page)
    clusters = cluster_drawing_rects(vector_rects + label_rects, pad_x=pad_x)
    clusters = [c for c in clusters if any(c.intersects(v) for v in vector_rects)]
    clusters = [
        fitz.Rect(c.x0 - CROP_MARGIN_PT, c.y0 - CROP_MARGIN_PT, c.x1 + CROP_MARGIN_PT, c.y1 + CROP_MARGIN_PT)
        for c in clusters
    ]
    # A drawing can sit partly or entirely in the page's margin/bleed area
    # (off the visible canvas) -- rare on its own, but merging (or the
    # margin above) can push a cluster there. Clip to the page's actual
    # bounds before rendering; drop anything that becomes degenerate as a
    # result.
    clusters = [bbox & page.rect for bbox in clusters]
    clusters = [bbox for bbox in clusters if bbox.width > 0 and bbox.height > 0]
    clusters.sort(key=lambda r: (round(r.y0), r.x0))

    figures = []
    matrix = fitz.Matrix(zoom, zoom)
    for fig_index, bbox in enumerate(clusters):
        pix = page.get_pixmap(matrix=matrix, clip=bbox)
        image_filename = f"page{page_number}_fig{fig_index}.png"
        pix.save(images_dir / image_filename)
        figures.append({
            "image_id": f"{doc_source}_p{page_number}_fig{fig_index}",
            "source": "vector_region",
            "bbox": [round(bbox.x0, 1), round(bbox.y0, 1), round(bbox.x1, 1), round(bbox.y1, 1)],
            "path": f"{images_dir.name}/{image_filename}",
            "width": pix.width,
            "height": pix.height,
        })
        pix = None
    return figures


def ingest_pdf(pdf_path: Path, output_dir: Path, doc_source: str = "main", images_dirname: str = "images"):
    """doc_source tags every page 'main' or 'SI' — main text and Supporting
    Information are ingested as one job (see __main__) but stay two documents
    with independent page numbering, so paths/image_ids/output files are kept
    in source-specific namespaces to avoid collisions (both start at page 1)."""
    images_dir = output_dir / images_dirname
    if images_dir.exists():
        # Re-running extraction can change how many figures a page produces
        # (e.g. a clustering fix that merges what used to be split into
        # several images down to one). Without clearing first, the extra
        # files from a previous run's higher count linger on disk and look
        # like current output -- found by hitting exactly this after a
        # clustering change actually worked.
        for stale in images_dir.glob("*.png"):
            stale.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    gutter = measure_column_gutter(doc)
    if gutter is None:
        pad_x = SINGLE_COLUMN_PAD_X
    else:
        pad_x = min(MAX_PAD_X, max(DEFAULT_PAD_X, gutter - PAD_X_SAFETY_BUFFER_PT))

    pages = []
    for page_number, page in enumerate(doc, start=1):
        raster_images = extract_raster_images(doc, page, page_number, images_dir, doc_source)
        vector_figures = extract_vector_figures(page, page_number, images_dir, doc_source, pad_x=pad_x)
        pages.append({
            "page_number": page_number,
            "source": doc_source,
            "text_blocks": extract_text_blocks(page),
            "images": raster_images + vector_figures,
        })

    result = {
        "source_pdf": pdf_path.name,
        "source": doc_source,
        "page_count": len(pages),
        "pages": pages,
    }

    output_filename = "raw_extraction.json" if doc_source == "main" else f"{doc_source}_raw_extraction.json"
    output_path = output_dir / output_filename
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    doc.close()
    return result


def _print_summary(label, result):
    total_blocks = sum(len(p["text_blocks"]) for p in result["pages"])
    total_images = sum(len(p["images"]) for p in result["pages"])
    print(f"[{label}] Pages: {result['page_count']}  Text blocks: {total_blocks}  Images: {total_images}")


if __name__ == "__main__":
    paper_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/papers/suzuki_iron_2024")

    main_result = ingest_pdf(paper_dir / "paper.pdf", paper_dir, doc_source="main")
    _print_summary("main", main_result)

    si_path = paper_dir / "SI.pdf"
    if si_path.exists():
        si_result = ingest_pdf(si_path, paper_dir, doc_source="SI", images_dirname="images_SI")
        _print_summary("SI", si_result)
