import json
import sys
from pathlib import Path

import fitz  # PyMuPDF


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


def extract_raster_images(doc, page, page_number, images_dir):
    images = []
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        bbox = rects[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha >= 4:  # CMYK -> RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        image_filename = f"page{page_number}_raster{img_index}.png"
        pix.save(images_dir / image_filename)
        images.append({
            "image_id": f"p{page_number}_raster{img_index}",
            "source": "embedded_raster",
            "bbox": [round(bbox.x0, 1), round(bbox.y0, 1), round(bbox.x1, 1), round(bbox.y1, 1)],
            "path": f"images/{image_filename}",
            "width": pix.width,
            "height": pix.height,
        })
        pix = None
    return images


def cluster_drawing_rects(rects, pad=8.0, min_area=2000.0):
    """Merge nearby/overlapping vector-drawing bboxes into figure-sized regions.

    Journal figures (chemical structures, plots) are almost always built from
    hundreds of individual vector path ops, not one embedded image. Padding
    each rect before checking overlap lets disjoint strokes that belong to
    the same drawing (e.g. a bond line and an atom label gap) merge together.
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
                padded = fitz.Rect(base.x0 - pad, base.y0 - pad, base.x1 + pad, base.y1 + pad)
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


def extract_vector_figures(page, page_number, images_dir, zoom=3.0):
    drawings = page.get_drawings()
    if not drawings:
        return []

    rects = [d["rect"] for d in drawings
             if d["rect"].width > 0 and d["rect"].height > 0 and not is_invisible_drawing(d)]
    clusters = cluster_drawing_rects(rects)
    clusters.sort(key=lambda r: (round(r.y0), r.x0))

    figures = []
    matrix = fitz.Matrix(zoom, zoom)
    for fig_index, bbox in enumerate(clusters):
        pix = page.get_pixmap(matrix=matrix, clip=bbox)
        image_filename = f"page{page_number}_fig{fig_index}.png"
        pix.save(images_dir / image_filename)
        figures.append({
            "image_id": f"p{page_number}_fig{fig_index}",
            "source": "vector_region",
            "bbox": [round(bbox.x0, 1), round(bbox.y0, 1), round(bbox.x1, 1), round(bbox.y1, 1)],
            "path": f"images/{image_filename}",
            "width": pix.width,
            "height": pix.height,
        })
        pix = None
    return figures


def ingest_pdf(pdf_path: Path, output_dir: Path):
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages = []
    for page_number, page in enumerate(doc, start=1):
        raster_images = extract_raster_images(doc, page, page_number, images_dir)
        vector_figures = extract_vector_figures(page, page_number, images_dir)
        pages.append({
            "page_number": page_number,
            "text_blocks": extract_text_blocks(page),
            "images": raster_images + vector_figures,
        })

    result = {
        "source_pdf": pdf_path.name,
        "page_count": len(pages),
        "pages": pages,
    }

    output_path = output_dir / "raw_extraction.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    doc.close()
    return result


if __name__ == "__main__":
    paper_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/papers/suzuki_iron_2024")
    pdf_path = paper_dir / "paper.pdf"

    result = ingest_pdf(pdf_path, paper_dir)

    total_blocks = sum(len(p["text_blocks"]) for p in result["pages"])
    total_images = sum(len(p["images"]) for p in result["pages"])
    print(f"Pages: {result['page_count']}")
    print(f"Text blocks: {total_blocks}")
    print(f"Images extracted: {total_images}")
