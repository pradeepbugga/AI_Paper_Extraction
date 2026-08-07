import json
import re
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from grobid_client import fetch_references

CAPTION_RE = re.compile(r"^(Fig(?:ure)?|Scheme|Table|Chart)\.?\s*\d+\s*[\.\|:]", re.IGNORECASE)
HYPHEN_WRAP_RE = re.compile(r"(?<=[a-z])-$")
TERMINAL_PUNCT_RE = re.compile(r"[.?!:;]$")
TRAILING_PAGE_NUM_RE = re.compile(r"\s*\d{1,4}$")
BOLD_FLAG = 1 << 4

HEADER_FRACTION = 0.05    # top of page treated as running-header territory
FOOTER_FRACTION = 0.05    # bottom of page treated as running-footer territory
REPEAT_FRACTION = 0.3     # line text repeated on this share of pages is furniture
HEADING_SIZE_RATIO = 1.02  # heading candidate must be strictly larger than body size
MAX_HEADING_LEN = 120
MAX_HEADING_WORDS = 20     # real headings read as short phrases, not sentences
MAX_HEADING_LEVELS = 6
MIN_HEADING_BLOCKS = 2  # a size tier must recur across >=N distinct blocks to count as a real heading level
COLUMN_GAP_MIN_FRACTION = 0.08  # min x-gap (as fraction of page width) to call it two columns


def line_text(spans):
    return "".join(s["text"] for s in spans).strip()


def dominant_span(spans):
    """The span covering the most characters in a line — robust to a leading
    decorative glyph (bullet/icon) sharing the line with the real heading text."""
    non_blank = [s for s in spans if s["text"].strip()]
    return max(non_blank, key=lambda s: len(s["text"].strip()))


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


def collect_raw_blocks(doc):
    """One pass over every page's dict-mode text, keeping enough per-line
    metadata (own bbox, dominant font size/boldness) to make document-relative
    calls later: what counts as body size, what counts as furniture, etc.
    Line-level (not block-level) because a running-header line can end up
    fused into the same PyMuPDF block as real content when there's no gap
    between them (e.g. a title sitting right under the running head)."""
    pages = []
    for page_number, page in enumerate(doc, start=1):
        page_height = page.rect.height
        page_width = page.rect.width
        blocks = []
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            bbox = [round(v, 1) for v in block["bbox"]]
            lines = []
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"]]
                text = line_text(spans)
                if not text:
                    continue
                dom = dominant_span(spans)
                lines.append({
                    "text": text,
                    "bbox": [round(v, 1) for v in line["bbox"]],
                    "size": dom["size"],
                    "bold": bool(dom["flags"] & BOLD_FLAG),
                })
            if lines:
                blocks.append({"bbox": bbox, "lines": lines})
        pages.append({"page_number": page_number, "height": page_height,
                       "width": page_width, "blocks": blocks})
    return pages


def compute_body_size(pages):
    """Body text size = the (size, rounded) bucket with the most total
    characters, restricted to non-bold lines so headings can't skew it."""
    char_count = Counter()
    for page in pages:
        for block in page["blocks"]:
            for line in block["lines"]:
                if not line["bold"]:
                    char_count[round(line["size"])] += len(line["text"])
    if not char_count:
        return 10.0
    return char_count.most_common(1)[0][0]


def furniture_key(text):
    """Normalize a page-footer-style line (e.g. 'Nature Catalysis | Vol 7 | 1189')
    by stripping a trailing page number, so 'same footer, different page number'
    still counts as one recurring line instead of many one-off lines."""
    return TRAILING_PAGE_NUM_RE.sub("", text).strip()


def compute_furniture_lines(pages):
    """Line text that recurs across a large share of pages (running headers/
    footers, DOI banners, journal masthead) — publisher-agnostic, no literal
    strings. Checked per-line (not per-block) so it still catches a running
    header even on the one page where it happens to be fused into a bigger
    content block. Matched both verbatim and with a trailing page number
    stripped, since footers often differ only by that number."""
    total_pages = len(pages)
    text_pages = Counter()
    key_pages = Counter()
    for page in pages:
        seen_text, seen_key = set(), set()
        for block in page["blocks"]:
            for line in block["lines"]:
                text = line["text"].strip()
                if not text:
                    continue
                if text not in seen_text:
                    text_pages[text] += 1
                    seen_text.add(text)
                key = furniture_key(text)
                if key and key not in seen_key:
                    key_pages[key] += 1
                    seen_key.add(key)
    threshold = max(2, int(total_pages * REPEAT_FRACTION))
    texts = {text for text, count in text_pages.items() if count >= threshold}
    keys = {key for key, count in key_pages.items() if count >= threshold}
    return texts, keys


def is_furniture_line(line, page_height, furniture_lines):
    texts, keys = furniture_lines
    y0, y1 = line["bbox"][1], line["bbox"][3]
    if y1 < page_height * HEADER_FRACTION or y0 > page_height * (1 - FOOTER_FRACTION):
        return True
    text = line["text"].strip()
    return text in texts or furniture_key(text) in keys


def compute_heading_levels(pages, body_size, furniture_lines):
    """Rank the distinct sizes seen among bold, larger-than-body lines
    (document-relative, so it adapts to whatever the source PDF's heading
    typography actually is) into up to MAX_HEADING_LEVELS tiers.

    A size tier only counts as a heading level if it recurs across multiple
    distinct blocks — a one-off large bold block (title, byline) reads the
    same as a heading by font alone, but real section headings are a family
    of *different* short blocks sharing one style, not a single wrapped one."""
    blocks_by_size = {}
    for page in pages:
        for block in page["blocks"]:
            block_key = (page["page_number"], tuple(block["bbox"]))
            for line in block["lines"]:
                if is_furniture_line(line, page["height"], furniture_lines):
                    continue
                text = line["text"]
                if (line["bold"] and line["size"] >= body_size * HEADING_SIZE_RATIO
                        and len(text) < MAX_HEADING_LEN and len(text.split()) <= MAX_HEADING_WORDS
                        and not is_sentence_like(text)):
                    bucket = round(line["size"] * 2) / 2
                    blocks_by_size.setdefault(bucket, set()).add(block_key)

    sizes = [size for size, blocks in blocks_by_size.items() if len(blocks) >= MIN_HEADING_BLOCKS]
    ranked = sorted(sizes, reverse=True)[:MAX_HEADING_LEVELS]
    return {size: level for level, size in enumerate(ranked, start=1)}


def is_sentence_like(text):
    """A short bold fragment that reads as a sentence (terminal punctuation,
    not all-caps, more than a few words) is more likely emphasis inside a
    paragraph than an actual heading."""
    words = text.split()
    return (TERMINAL_PUNCT_RE.search(text.strip()) is not None
            and not text.isupper()
            and len(words) > 3)


def classify_line(line, body_size, heading_levels):
    text = line["text"]
    words = text.split()
    if (not line["bold"] or len(text) >= MAX_HEADING_LEN or len(words) > MAX_HEADING_WORDS
            or is_sentence_like(text)):
        return "body"
    bucket = round(line["size"] * 2) / 2
    level = heading_levels.get(bucket)
    if level is None or line["size"] < body_size * HEADING_SIZE_RATIO:
        return "body"
    return f"heading{level}"


def find_column_split(blocks, page_width):
    """Detect a two-column layout from the actual x0 distribution instead of
    assuming an exact page-center split; falls back to single-column."""
    xs = sorted(b["bbox"][0] for b in blocks)
    if len(xs) < 4:
        return None
    best_gap, best_pos = 0.0, None
    for i in range(1, len(xs)):
        gap = xs[i] - xs[i - 1]
        left_share = i / len(xs)
        if gap > best_gap and 0.2 <= left_share <= 0.8:
            best_gap, best_pos = gap, (xs[i - 1] + xs[i]) / 2
    if best_gap >= page_width * COLUMN_GAP_MIN_FRACTION:
        return best_pos
    return None


def parse_page(page_data, figure_bboxes, body_size, heading_levels, furniture_lines):
    candidate_blocks = [b for b in page_data["blocks"] if not block_in_figure(b["bbox"], figure_bboxes)]
    split_x = find_column_split(candidate_blocks, page_data["width"])

    elements = []
    for block in candidate_blocks:
        units = []
        for line in block["lines"]:
            if is_furniture_line(line, page_data["height"], furniture_lines):
                continue
            kind = classify_line(line, body_size, heading_levels)
            if units and units[-1][0] == kind:
                units[-1][1] = join_lines([units[-1][1], line["text"]])
            else:
                units.append([kind, line["text"]])
        if not units:
            continue

        column = 0 if split_x is None or block["bbox"][0] < split_x else 1
        full_text = " ".join(t for _, t in units)
        if CAPTION_RE.match(full_text):
            elements.append({"type": "figure_caption", "text": full_text,
                              "page": page_data["page_number"], "column": column,
                              "bbox": block["bbox"]})
            continue

        for kind, text in units:
            elements.append({"type": kind, "text": text, "page": page_data["page_number"],
                              "column": column, "bbox": block["bbox"]})

    elements.sort(key=lambda e: (e["column"], e["bbox"][1]))
    return elements


def merge_boundary_paragraphs(elements):
    """A paragraph cut off at a page or column break has no terminal
    punctuation at the break point; if the next element is also body text,
    it's a continuation, not a new paragraph — stitch them back together."""
    merged = []
    for el in elements:
        if (merged and el["type"] == "body" and merged[-1]["type"] == "body"
                and (el["page"] != merged[-1]["page"] or el["column"] != merged[-1]["column"])
                and not TERMINAL_PUNCT_RE.search(merged[-1]["text"].strip())):
            merged[-1] = dict(merged[-1])
            merged[-1]["text"] = join_lines([merged[-1]["text"], el["text"]])
            continue
        merged.append(el)
    return merged


def vertical_gap(caption_bbox, image_bbox):
    """Distance between a caption and an image, whichever side it's on —
    some publishers put the caption below the figure, others (e.g. ACS,
    where the caption sits inside the same boxed scheme) put it above."""
    if image_bbox[3] <= caption_bbox[1]:
        return caption_bbox[1] - image_bbox[3]
    if image_bbox[1] >= caption_bbox[3]:
        return image_bbox[1] - caption_bbox[3]
    return 0.0  # overlapping/adjacent


def horizontal_overlap(caption_bbox, image_bbox):
    x0 = max(caption_bbox[0], image_bbox[0])
    x1 = min(caption_bbox[2], image_bbox[2])
    return max(0.0, x1 - x0)


def nearest_figure(caption, page_images):
    """Nearest by vertical gap, but only among images that share a column
    with the caption (multi-column pages can have two side-by-side figures
    at nearly the same height — pure vertical distance would pick either)."""
    if not page_images:
        return None
    same_column = [img for img in page_images if horizontal_overlap(caption["bbox"], img["bbox"]) > 0]
    candidates = same_column or page_images
    return min(candidates, key=lambda img: vertical_gap(caption["bbox"], img["bbox"]))


def assemble_sections(elements):
    front_matter = []
    sections = []
    stack = []  # (level, section_dict) currently open, outermost first

    def container_for(level):
        while stack and stack[-1][0] >= level:
            stack.pop()
        return stack[-1][1] if stack else None

    for el in elements:
        m = re.fullmatch(r"heading(\d+)", el["type"])
        if m:
            level = int(m.group(1))
            parent = container_for(level)
            node = {"heading": el["text"], "page_start": el["page"],
                    "paragraphs": [], "subsections": []}
            if parent is None:
                sections.append(node)
            else:
                parent["subsections"].append(node)
            stack.append((level, node))
        elif el["type"] == "body":
            target = stack[-1][1]["paragraphs"] if stack else front_matter
            target.append(el["text"])

    return front_matter, sections


def parse_captions(elements, images_by_page):
    captions = []
    for el in elements:
        if el["type"] != "figure_caption":
            continue
        match = re.match(r"^(Fig(?:ure)?|Scheme|Table|Chart)\.?\s*(\d+)", el["text"], re.IGNORECASE)
        label = f"{match.group(1)} {match.group(2)}" if match else None
        page_images = images_by_page.get(el["page"], [])
        image = nearest_figure(el, page_images)
        captions.append({
            "label": label,
            "caption": el["text"],
            "page": el["page"],
            "image_id": image["image_id"] if image else None,
            "image_path": image["path"] if image else None,
        })
    return captions


def pop_references_section(sections):
    """Pull the prose 'References' section out of the tree by heading text
    (stripped of ACS-style '■' markers), so it can be replaced with GROBID's
    structured bibliography or, on failure, reinserted at the same spot."""
    for i, s in enumerate(sections):
        heading = (s.get("heading") or "").strip("■ ").strip()
        if heading.lower() == "references":
            return sections.pop(i), i
    return None, None


def parse_sections(pdf_path: Path, raw_extraction: dict, output_dir: Path):
    doc = fitz.open(pdf_path)
    pages = collect_raw_blocks(doc)
    doc.close()

    body_size = compute_body_size(pages)
    furniture_lines = compute_furniture_lines(pages)
    heading_levels = compute_heading_levels(pages, body_size, furniture_lines)

    images_by_page = {p["page_number"]: p["images"] for p in raw_extraction["pages"]}

    all_elements = []
    for page_data in pages:
        figure_bboxes = [img["bbox"] for img in images_by_page.get(page_data["page_number"], [])]
        all_elements.extend(parse_page(page_data, figure_bboxes, body_size, heading_levels, furniture_lines))

    all_elements = merge_boundary_paragraphs(all_elements)

    front_matter, sections = assemble_sections(
        [e for e in all_elements if e["type"] != "figure_caption"]
    )
    figures = parse_captions(all_elements, images_by_page)

    ref_node, ref_idx = pop_references_section(sections)
    structured_refs = fetch_references(pdf_path)
    if structured_refs is not None:
        references = structured_refs
    else:
        references = []
        if ref_node is not None:
            sections.insert(ref_idx, ref_node)

    result = {
        "source_pdf": pdf_path.name,
        "front_matter": front_matter,
        "sections": sections,
        "figures": figures,
        "references": references,
    }

    output_path = output_dir / "sections.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_section(s, indent=2):
    print(f"{' ' * indent}- {s['heading']!r}: {len(s['paragraphs'])} paragraphs, {len(s['subsections'])} subsections")
    for sub in s["subsections"]:
        print_section(sub, indent + 4)


if __name__ == "__main__":
    paper_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/papers/suzuki_iron_2024")
    pdf_path = paper_dir / "paper.pdf"
    raw_extraction = json.load(open(paper_dir / "raw_extraction.json"))

    result = parse_sections(pdf_path, raw_extraction, paper_dir)

    print(f"Front-matter paragraphs: {len(result['front_matter'])}")
    print(f"Sections: {len(result['sections'])}")
    for s in result["sections"]:
        print_section(s)
    print(f"Figures with captions: {len(result['figures'])}")
    for fig in result["figures"]:
        print(f"  - {fig['label']} (page {fig['page']}) -> {fig['image_id']}")
    print(f"References: {len(result['references'])}")
