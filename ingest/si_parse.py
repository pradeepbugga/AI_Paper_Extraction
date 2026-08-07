import json
import sys
from pathlib import Path

import fitz  # PyMuPDF

from section_parse import (
    collect_raw_blocks, compute_body_size, compute_furniture_lines,
    compute_heading_levels, parse_page, merge_boundary_paragraphs,
    assemble_sections,
)


def restructure_as_records(node):
    """SI documents are organized around dozens to hundreds of repeating
    compound/procedure records rather than narrative prose subsections, so
    a leaf heading (one with no headings nested under it) is recast as a
    'record' — a flat name + text blob — instead of a subsection with a
    single paragraph list. This uses no SI-specific detection at all: it's
    a shape transform applied after the exact same heading/paragraph tree
    the main text produces, via the same document-relative heading-level
    logic (see section_parse.compute_heading_levels)."""
    real_subsections = []
    records = []
    for sub in node["subsections"]:
        restructure_as_records(sub)
        if not sub["subsections"] and not sub["records"]:
            records.append({
                "heading": sub["heading"],
                "page_start": sub["page_start"],
                "text": sub["preamble"],
                "figures": [],
            })
        else:
            real_subsections.append(sub)

    node["subsections"] = real_subsections
    node["records"] = records
    node["preamble"] = "\n".join(node.pop("paragraphs"))
    node["figures"] = []


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
        all_elements.extend(parse_page(page_data, figure_bboxes, body_size, heading_levels, furniture_lines))

    all_elements = merge_boundary_paragraphs(all_elements)

    front_matter, sections = assemble_sections(
        [e for e in all_elements if e["type"] != "figure_caption"]
    )

    for sec in sections:
        restructure_as_records(sec)
    attach_figures(sections, images_by_page)

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
