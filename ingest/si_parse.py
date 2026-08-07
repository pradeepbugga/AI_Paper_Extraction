import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

from section_parse import (
    HEADING_SIZE_RATIO, MAX_HEADING_LEN,
    collect_raw_blocks, compute_body_size, compute_furniture_lines,
    compute_heading_levels, is_furniture_line, find_column_split, join_lines,
    is_sentence_like, block_in_figure, merge_boundary_paragraphs,
)

SECTION_NUM_RE = re.compile(r"^\d{1,3}\.\s+\S")
MAX_BOUNDARY_WORDS = 20


def classify_si_line(line, body_size, heading_levels):
    """SI documents split into two conventions we've seen so far: genuine
    font-size-tiered headings just like the main text (Nature's SI has
    'Supplementary Methods' at 16pt containing 'General Considerations' at
    14pt, both clearly larger than 11pt body), and flat body-size-bold
    headings distinguished only by numbering (ACS's '1. General
    experimental', '12. Experimental procedures...', all at the same 12pt
    as body text). Try the size-tier signal first (reusing the same
    document-relative logic as main-text parsing); if a bold line doesn't
    belong to any size tier, fall back to the numbering pattern. Anything
    bold-and-short that's neither is a record (compound name, procedure
    label) rather than a top-level section."""
    text = line["text"]
    if (not line["bold"] or len(text.split()) > MAX_BOUNDARY_WORDS
            or is_sentence_like(text) or len(text) >= MAX_HEADING_LEN):
        return "body"

    bucket = round(line["size"] * 2) / 2
    level = heading_levels.get(bucket)
    if level is not None and line["size"] >= body_size * HEADING_SIZE_RATIO:
        return f"section{level}"

    if SECTION_NUM_RE.match(text):
        return "section1"

    return "record"


def parse_si_page(page_data, figure_bboxes, body_size, heading_levels, furniture_lines):
    candidate_blocks = [b for b in page_data["blocks"] if not block_in_figure(b["bbox"], figure_bboxes)]
    split_x = find_column_split(candidate_blocks, page_data["width"])

    elements = []
    for block in candidate_blocks:
        units = []
        for line in block["lines"]:
            if is_furniture_line(line, page_data["height"], furniture_lines):
                continue
            kind = classify_si_line(line, body_size, heading_levels)
            if units and units[-1][0] == kind:
                units[-1][1] = join_lines([units[-1][1], line["text"]])
            else:
                units.append([kind, line["text"]])
        if not units:
            continue
        column = 0 if split_x is None or block["bbox"][0] < split_x else 1
        for kind, text in units:
            elements.append({"type": kind, "text": text, "page": page_data["page_number"],
                              "column": column, "bbox": block["bbox"]})

    elements.sort(key=lambda e: (e["column"], e["bbox"][1]))
    return elements


def assemble_records(elements):
    """Compound-record-oriented structure: sections can nest (some SI docs
    have real sub-headings), and each holds a flat list of records (compound
    names, 'Synthesis of ...' procedure headers, general-procedure blocks).
    Flat records are a more useful shape for downstream schema extraction
    than forcing SI into the same nested prose tree used for the main text
    -- the goal here is "text blob per compound," not narrative structure."""
    front_matter = []
    sections = []
    stack = []  # (level, section_dict), outermost first
    current_record = None

    def container_for(level):
        while stack and stack[-1][0] >= level:
            stack.pop()
        return stack[-1][1] if stack else None

    for el in elements:
        m = re.fullmatch(r"section(\d+)", el["type"])
        if m:
            level = int(m.group(1))
            parent = container_for(level)
            node = {"heading": el["text"], "page_start": el["page"], "preamble": [],
                    "records": [], "subsections": [], "figures": []}
            if parent is None:
                sections.append(node)
            else:
                parent["subsections"].append(node)
            stack.append((level, node))
            current_record = None
        elif el["type"] == "record":
            if stack:
                current_record = {"heading": el["text"], "page_start": el["page"],
                                   "text": [], "figures": []}
                stack[-1][1]["records"].append(current_record)
            else:
                current_record = None  # a record needs a section to live in
        elif el["type"] == "body":
            if current_record is not None:
                current_record["text"].append(el["text"])
            elif stack:
                stack[-1][1]["preamble"].append(el["text"])
            else:
                front_matter.append(el["text"])

    return front_matter, sections


def collect_timeline(sections):
    timeline = []

    def walk(nodes):
        for node in nodes:
            timeline.append((node["page_start"], node))
            walk(node["subsections"])
            for rec in node["records"]:
                timeline.append((rec["page_start"], rec))

    walk(sections)
    timeline.sort(key=lambda t: t[0])
    return timeline


def attach_figures(sections, images_by_page):
    """Assign each page's figures to whichever section/record is 'open' at
    that page number -- i.e. the most recent heading at or before it. Coarse
    (page-level, not per-compound-within-page) but matches how these SI
    documents are laid out: a compound's structure + spectra sit on/near
    its own page."""
    timeline = collect_timeline(sections)

    for page_num in sorted(images_by_page):
        images = images_by_page[page_num]
        if not images:
            continue
        target = None
        for page_start, node in timeline:
            if page_start <= page_num:
                target = node
            else:
                break
        if target is not None:
            target["figures"].extend(img["image_id"] for img in images)


def parse_si(si_path: Path, si_raw_extraction: dict, output_dir: Path):
    doc = fitz.open(si_path)
    pages = collect_raw_blocks(doc)
    doc.close()

    body_size = compute_body_size(pages)
    furniture_lines = compute_furniture_lines(pages)
    heading_levels = compute_heading_levels(pages, body_size, furniture_lines)

    images_by_page = {p["page_number"]: p["images"] for p in si_raw_extraction["pages"]}

    all_elements = []
    for page_data in pages:
        figure_bboxes = [img["bbox"] for img in images_by_page.get(page_data["page_number"], [])]
        all_elements.extend(parse_si_page(page_data, figure_bboxes, body_size, heading_levels, furniture_lines))

    all_elements = merge_boundary_paragraphs(all_elements)

    front_matter, sections = assemble_records(all_elements)
    attach_figures(sections, images_by_page)

    def flatten_text(node):
        node["preamble"] = "\n".join(node["preamble"])
        for rec in node["records"]:
            rec["text"] = "\n".join(rec["text"])
        for sub in node["subsections"]:
            flatten_text(sub)

    for sec in sections:
        flatten_text(sec)

    result = {
        "source_pdf": si_path.name,
        "front_matter": front_matter,
        "sections": sections,
    }

    output_path = output_dir / "SI_sections.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_section(s, indent=2):
    print(f"{' ' * indent}- {s['heading']!r} (p{s['page_start']}): {len(s['records'])} records, "
          f"{len(s['subsections'])} subsections, {len(s['figures'])} figures")
    for r in s["records"][:2]:
        print(f"{' ' * (indent + 4)}- {r['heading']!r} (p{r['page_start']}): "
              f"{len(r['text'])} chars, {len(r['figures'])} figures")
    if len(s["records"]) > 2:
        print(f"{' ' * (indent + 4)}... and {len(s['records']) - 2} more records")
    for sub in s["subsections"]:
        print_section(sub, indent + 4)


if __name__ == "__main__":
    paper_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/papers/copper_iron_2025")
    si_path = paper_dir / "SI.pdf"
    si_raw_extraction = json.load(open(paper_dir / "SI_raw_extraction.json"))

    result = parse_si(si_path, si_raw_extraction, paper_dir)

    print(f"Front-matter paragraphs: {len(result['front_matter'])}")
    print(f"Sections: {len(result['sections'])}")

    def count_records(nodes):
        return sum(len(s["records"]) + count_records(s["subsections"]) for s in nodes)

    print(f"Total records: {count_records(result['sections'])}")
    for s in result["sections"]:
        print_section(s)
