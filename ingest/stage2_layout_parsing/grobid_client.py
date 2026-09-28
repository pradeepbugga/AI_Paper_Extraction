import xml.etree.ElementTree as ET
from pathlib import Path

import requests

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
DEFAULT_GROBID_URL = "http://localhost:8070"


def is_available(grobid_url: str = DEFAULT_GROBID_URL, timeout: float = 2.0) -> bool:
    try:
        resp = requests.get(f"{grobid_url}/api/isalive", timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _text(el):
    return "".join(el.itertext()).strip() if el is not None else None


def _parse_bibl_struct(bibl):
    authors = []
    for pers in bibl.findall(".//tei:author/tei:persName", TEI_NS):
        forename = " ".join(f.text.strip() for f in pers.findall("tei:forename", TEI_NS) if f.text)
        surname = _text(pers.find("tei:surname", TEI_NS))
        name = " ".join(p for p in (forename, surname) if p)
        if name:
            authors.append(name)

    title = _text(bibl.find(".//tei:analytic/tei:title", TEI_NS)) or _text(bibl.find(".//tei:monogr/tei:title", TEI_NS))
    journal = _text(bibl.find('.//tei:monogr/tei:title[@level="j"]', TEI_NS))
    doi = _text(bibl.find('.//tei:idno[@type="DOI"]', TEI_NS))

    imprint = bibl.find(".//tei:monogr/tei:imprint", TEI_NS)
    volume = issue = year = page_from = page_to = publisher = None
    if imprint is not None:
        volume = _text(imprint.find('tei:biblScope[@unit="volume"]', TEI_NS))
        issue = _text(imprint.find('tei:biblScope[@unit="issue"]', TEI_NS))
        publisher = _text(imprint.find("tei:publisher", TEI_NS))
        page_el = imprint.find('tei:biblScope[@unit="page"]', TEI_NS)
        if page_el is not None:
            page_from = page_el.get("from")
            page_to = page_el.get("to")
        date_el = imprint.find('tei:date[@type="published"]', TEI_NS)
        if date_el is not None:
            year = date_el.get("when") or _text(date_el)

    pages = None
    if page_from and page_to:
        pages = f"{page_from}-{page_to}"
    elif page_from:
        pages = page_from

    return {
        "id": bibl.get("{http://www.w3.org/XML/1998/namespace}id"),
        "title": title,
        "authors": authors,
        "journal": journal,
        "volume": volume,
        "issue": issue,
        "pages": pages,
        "year": year,
        "doi": doi,
        "publisher": publisher,
    }


def parse_references_tei(tei_xml: str):
    root = ET.fromstring(tei_xml)
    return [_parse_bibl_struct(b) for b in root.findall(".//tei:biblStruct", TEI_NS)]


def fetch_references(pdf_path: Path, grobid_url: str = DEFAULT_GROBID_URL, timeout: float = 60.0):
    """Structured bibliography via GROBID's dedicated references endpoint
    (lighter than full-document processing, since that's all we need here).
    Returns None (not []) if GROBID is unreachable or errors, so callers can
    fall back to whatever they had rather than silently reporting zero refs."""
    if not is_available(grobid_url):
        return None
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                f"{grobid_url}/api/processReferences",
                files={"input": f},
                data={"consolidateCitations": "1"},
                timeout=timeout,
            )
        resp.raise_for_status()
        return parse_references_tei(resp.text)
    except (requests.RequestException, ET.ParseError):
        return None


if __name__ == "__main__":
    import json
    import sys

    paper_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/papers/suzuki_iron_2024")
    refs = fetch_references(paper_dir / "paper.pdf")
    if refs is None:
        print("GROBID unavailable or request failed")
    else:
        print(f"{len(refs)} references")
        for r in refs[:3]:
            print(json.dumps(r, indent=2))
