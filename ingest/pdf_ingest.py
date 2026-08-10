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


def extract_vector_figures(page, page_number, images_dir, doc_source, zoom=3.0):
    drawings = page.get_drawings()
    if not drawings:
        return []

    rects = [d["rect"] for d in drawings
             if d["rect"].width > 0 and d["rect"].height > 0 and not is_invisible_drawing(d)]
    clusters = cluster_drawing_rects(rects)
    # A drawing can sit partly or entirely in the page's margin/bleed area
    # (off the visible canvas) -- rare on its own, but merging can absorb
    # one into an otherwise-valid cluster. Clip to the page's actual bounds
    # before rendering; drop anything that becomes degenerate as a result.
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
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages = []
    for page_number, page in enumerate(doc, start=1):
        raster_images = extract_raster_images(doc, page, page_number, images_dir, doc_source)
        vector_figures = extract_vector_figures(page, page_number, images_dir, doc_source)
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
