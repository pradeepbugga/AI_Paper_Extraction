"""Extracts a caption (or best-available text context) for every Stage 1
figure crop, main text and SI, across all papers.

Main-text figures already have a matched caption from section_parse.py
(sections.json). SI figures usually have no formal caption -- si_parse.py
attaches each figure to whichever compound/section record is "open" at that
page, and that record's own text (characterization data: NMR/HRMS/IR/X-ray
descriptions) is the best substitute context available for a figure that
was never given its own caption.
"""

import json
import re
from pathlib import Path

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"
SI_FIGURE_ID_RE = re.compile(r"^SI_p(\d+)_(fig|raster|rasterstrip)(\d+)$")


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main_text_captions(sections):
    """Returns {image_path: {"caption": str, "source": "main_caption"}}."""
    result = {}
    for fig in sections.get("figures", []):
        result[fig["image_path"]] = {
            "caption": fig["caption"],
            "source": "main_caption",
        }
    return result


def si_figure_id_to_path(figure_id):
    m = SI_FIGURE_ID_RE.match(figure_id)
    if not m:
        return None
    page, kind, idx = m.groups()
    return f"images_SI/page{page}_{kind}{idx}.png"


def walk_records(node, results):
    """Recurses through the SI section/subsection/record tree, attaching
    each record's own heading+text to every figure it lists."""
    if isinstance(node, dict):
        if "figures" in node and node["figures"]:
            heading = node.get("heading") or ""
            text = node.get("text") or ""
            if text.strip():
                caption = f"{heading} | {text}".strip(" |")
                source = "SI_record_text"
            elif heading.strip():
                caption = heading
                source = "SI_heading_only"
            else:
                caption = ""
                source = "SI_none"
            for figure_id in node["figures"]:
                image_path = si_figure_id_to_path(figure_id)
                if image_path:
                    results[image_path] = {"caption": caption, "source": source}
        for value in node.values():
            walk_records(value, results)
    elif isinstance(node, list):
        for item in node:
            walk_records(item, results)


def si_captions(si_sections):
    result = {}
    walk_records(si_sections.get("sections", []), result)
    return result


def extract_for_paper(paper_dir):
    result = {}

    sections = load_json(paper_dir / "sections.json")
    if sections:
        result.update(main_text_captions(sections))

    si_sections = load_json(paper_dir / "SI_sections.json")
    if si_sections:
        result.update(si_captions(si_sections))

    return result


def extract_all(papers_dir=PAPERS_DIR):
    all_results = {}
    for paper_dir in sorted(papers_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        captions = extract_for_paper(paper_dir)
        all_results[paper_dir.name] = captions
        out_path = paper_dir / "figure_captions.json"
        with open(out_path, "w") as f:
            json.dump(captions, f, indent=2)
        print(f"{paper_dir.name}: {len(captions)} figures -> {out_path}")
    return all_results


if __name__ == "__main__":
    extract_all()
