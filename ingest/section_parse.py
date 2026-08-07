import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

FIGURE_CAPTION_RE = re.compile(r"^Fig\.\s*\d+\s*\|")
FURNITURE_TEXT_RE = re.compile(r"^(nature catalysis|article)$", re.IGNORECASE)
HYPHEN_WRAP_RE = re.compile(r"(?<=[a-z])-$")
HEADER_BAND = 40.0
FOOTER_BAND = 755.0


def in_furniture_band(bbox, page_height):
    y0, y1 = bbox[1], bbox[3]
    return y0 < HEADER_BAND or y1 > page_height - (790.0 - FOOTER_BAND)


def join_lines(texts):
    out = ""
    for text in texts:
        if out and HYPHEN_WRAP_RE.search(out):
            out = out[:-1] + text
        elif out:
            out = out + " " + text
        else:
            out = text
    return out


def classify_line(spans):
    if not spans:
        return "body"
    font, size = spans[0]["font"], spans[0]["size"]
    if font == "HardingText-Bold" and size >= 10.0:
        return "heading1"
    if font == "HardingText-Semibold":
        return "heading2"
    return "body"


def line_text(spans):
    return "".join(s["text"] for s in spans).strip()


def overlap_ratio(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = (ax1 - ax0) * (ay1 - ay0)
    return inter / area if area > 0 else 0.0


def block_in_figure(bbox, figure_bboxes, threshold=0.4):
    return any(overlap_ratio(bbox, fb) >= threshold for fb in figure_bboxes)


def parse_block_lines(block):
    """Split a raw block's lines into ordered (kind, text) units, merging
    consecutive lines of the same kind (heading immediately followed by its
    own paragraph text lands in one PyMuPDF block with no blank-line gap)."""
    units = []
    for line in block["lines"]:
        text = line_text(line["spans"])
        if not text:
            continue
        kind = classify_line(line["spans"])
        if units and units[-1][0] == kind:
            units[-1][1] = join_lines([units[-1][1], text])
        else:
            units.append([kind, text])
    return units


def parse_page(page, page_number, figure_bboxes):
    page_height = page.rect.height
    mid_x = page.rect.width / 2
    d = page.get_text("dict")

    elements = []
    for block in d["blocks"]:
        if block["type"] != 0:
            continue
        bbox = [round(v, 1) for v in block["bbox"]]
        if in_furniture_band(bbox, page_height):
            continue
        if block_in_figure(bbox, figure_bboxes):
            continue

        units = parse_block_lines(block)
        if not units:
            continue
        if len(units) == 1 and FURNITURE_TEXT_RE.match(units[0][1].strip()):
            continue

        column = 0 if (bbox[0] + bbox[2]) / 2 < mid_x else 1
        full_text = " ".join(t for _, t in units)
        if FIGURE_CAPTION_RE.match(full_text):
            elements.append({
                "type": "figure_caption",
                "text": full_text,
                "page": page_number,
                "column": column,
                "bbox": bbox,
            })
            continue

        for kind, text in units:
            elements.append({
                "type": kind,
                "text": text,
                "page": page_number,
                "column": column,
                "bbox": bbox,
            })

    elements.sort(key=lambda e: (e["column"], e["bbox"][1]))
    return elements


def nearest_figure(caption, page_images):
    above = [img for img in page_images if img["bbox"][3] <= caption["bbox"][1]]
    if above:
        return min(above, key=lambda img: caption["bbox"][1] - img["bbox"][3])
    if page_images:
        return max(page_images, key=lambda img: img["width"] * img["height"])
    return None


def assemble_sections(elements):
    front_matter = []
    sections = []
    current_section = None
    current_subsection = None

    def flush_subsection():
        nonlocal current_subsection
        if current_subsection is not None and current_section is not None:
            current_section["subsections"].append(current_subsection)
        current_subsection = None

    for el in elements:
        if el["type"] == "heading1":
            flush_subsection()
            current_section = {"heading": el["text"], "page_start": el["page"],
                                "paragraphs": [], "subsections": []}
            sections.append(current_section)
        elif el["type"] == "heading2":
            flush_subsection()
            if current_section is None:
                current_section = {"heading": None, "page_start": el["page"],
                                    "paragraphs": [], "subsections": []}
                sections.append(current_section)
            current_subsection = {"heading": el["text"], "page_start": el["page"], "paragraphs": []}
        elif el["type"] == "body":
            target = current_subsection["paragraphs"] if current_subsection else (
                current_section["paragraphs"] if current_section else front_matter
            )
            target.append(el["text"])

    flush_subsection()
    return front_matter, sections


def parse_captions(elements, images_by_page):
    captions = []
    for el in elements:
        if el["type"] != "figure_caption":
            continue
        match = re.match(r"^Fig\.\s*(\d+)", el["text"])
        fig_num = match.group(1) if match else None
        page_images = images_by_page.get(el["page"], [])
        image = nearest_figure(el, page_images)
        captions.append({
            "figure_number": fig_num,
            "caption": el["text"],
            "page": el["page"],
            "image_id": image["image_id"] if image else None,
            "image_path": image["path"] if image else None,
        })
    return captions


def parse_sections(pdf_path: Path, raw_extraction: dict, output_dir: Path):
    doc = fitz.open(pdf_path)

    images_by_page = {}
    for page_data in raw_extraction["pages"]:
        images_by_page[page_data["page_number"]] = page_data["images"]

    all_elements = []
    for page_number, page in enumerate(doc, start=1):
        figure_bboxes = [img["bbox"] for img in images_by_page.get(page_number, [])]
        all_elements.extend(parse_page(page, page_number, figure_bboxes))

    doc.close()

    front_matter, sections = assemble_sections(
        [e for e in all_elements if e["type"] != "figure_caption"]
    )
    figures = parse_captions(all_elements, images_by_page)

    result = {
        "source_pdf": pdf_path.name,
        "front_matter": front_matter,
        "sections": sections,
        "figures": figures,
    }

    output_path = output_dir / "sections.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    paper_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/papers/suzuki_iron_2024")
    pdf_path = paper_dir / "paper.pdf"
    raw_extraction = json.load(open(paper_dir / "raw_extraction.json"))

    result = parse_sections(pdf_path, raw_extraction, paper_dir)

    print(f"Front-matter paragraphs: {len(result['front_matter'])}")
    print(f"Sections: {len(result['sections'])}")
    for s in result["sections"]:
        print(f"  - {s['heading']!r}: {len(s['paragraphs'])} paragraphs, {len(s['subsections'])} subsections")
        for sub in s["subsections"]:
            print(f"      - {sub['heading']!r}: {len(sub['paragraphs'])} paragraphs")
    print(f"Figures with captions: {len(result['figures'])}")
    for fig in result["figures"]:
        print(f"  - Fig {fig['figure_number']} (page {fig['page']}) -> {fig['image_id']}")
