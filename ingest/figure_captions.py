"""Extracts a caption (or best-available text context) for every Stage 1
figure crop, main text and SI, across all papers.

Main-text figures already have a matched caption from section_parse.py
(sections.json). SI figures usually have no formal caption -- si_parse.py
attaches each figure to whichever compound/section record is "open" at that
page, and that record's own text (characterization data: NMR/HRMS/IR/X-ray
descriptions) is the best substitute context available for a figure that
was never given its own caption.

Some SI documents have a dedicated "spectra appendix" section where each
figure has no real heading at all -- just a short plain-text label on the
same page (e.g. "1H NMR (400 MHz, CDCl3) of compound 1e."), too short/
unstyled to ever be promoted to a heading by section_parse.py (correctly --
it isn't one). Left alone, attach_figures' "most recent heading at or
before this page" logic falls back to whatever real heading came last
(often "References"), so every figure in that whole section gets the same
useless caption. local_captions_by_page reads the raw per-page text
directly and prefers it whenever a page's real content is short enough to
read as a standalone label rather than a chunk of a larger procedure.
"""

import json
import re
from collections import Counter
from pathlib import Path

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"
SI_FIGURE_ID_RE = re.compile(r"^SI_p(\d+)_(fig|raster|rasterstrip)(\d+)$")
TRAILING_PAGE_NUM_RE = re.compile(r"\s*\d+\s*$")
MAX_LOCAL_CAPTION_WORDS = 20  # a genuine per-figure label reads as a short
                               # phrase; real procedure/record text runs to
                               # 50-100+ words -- the same gap section_parse.py
                               # uses to separate headings from body prose


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


def local_captions_by_page(si_raw_extraction):
    """Returns {image_path: {"caption": str, "source": "SI_local_caption"}}
    for every image on a page whose real (non-furniture) text is short
    enough to read as a standalone label rather than record prose.

    Furniture (running headers/footers) is detected the same document-
    relative way as elsewhere in this pipeline: text repeating near-
    verbatim across most pages isn't page content, computed here
    independently since SI_raw_extraction.json's simplified blocks don't
    carry the per-line style info section_parse.py's own furniture
    detection needs.
    """
    pages = si_raw_extraction.get("pages", [])
    if not pages:
        return {}

    text_counts = Counter()
    page_texts = {}
    page_images = {}
    for page in pages:
        texts = [b["text"].strip() for b in page.get("text_blocks", []) if b["text"].strip()]
        page_texts[page["page_number"]] = texts
        page_images[page["page_number"]] = page.get("images", [])
        for t in texts:
            text_counts[TRAILING_PAGE_NUM_RE.sub("", t)] += 1

    total_pages = len(pages)
    furniture = {t for t, count in text_counts.items() if count >= max(3, total_pages * 0.3)}

    result = {}
    for page_num, texts in page_texts.items():
        content = [t for t in texts if TRAILING_PAGE_NUM_RE.sub("", t) not in furniture]
        if not content:
            continue
        joined = " ".join(content)
        if len(joined.split()) > MAX_LOCAL_CAPTION_WORDS:
            continue
        for img in page_images[page_num]:
            result[img["path"]] = {"caption": joined, "source": "SI_local_caption"}
    return result


def extract_for_paper(paper_dir):
    result = {}

    sections = load_json(paper_dir / "sections.json")
    if sections:
        result.update(main_text_captions(sections))

    si_sections = load_json(paper_dir / "SI_sections.json")
    if si_sections:
        result.update(si_captions(si_sections))

    si_raw = load_json(paper_dir / "SI_raw_extraction.json")
    if si_raw:
        result.update(local_captions_by_page(si_raw))

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
