"""Generates audit_pages_v{N}/audit_<paper>.html from each paper's current
reaction_links.json -- a static, self-contained (base64-embedded images)
review page: one card per linked segment, filterable by has-id/no-id/
flagged status plus free-text search, click-to-zoom. Matches audit_pages_v3's
layout exactly; regenerate into a new numbered dir after any corpus re-run
that changes reaction_links.json, rather than overwriting the old one, so
the previous state stays available for comparison.

Usage: python3 ingest/stage3_figure_extraction/generate_audit_pages.py <output_dir_name>
e.g.:  python3 ingest/stage3_figure_extraction/generate_audit_pages.py audit_pages_v4
"""
import base64
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PAPERS_DIR = ROOT / "data" / "papers"

STYLE = """
<style>
:root {
  --bg: #eef0ec;
  --surface: #ffffff;
  --surface-border: #d8dcd3;
  --ink: #1c2321;
  --ink-muted: #5b655f;
  --accent: #3e7c74;
  --accent-ink: #ffffff;
  --ok: #4b7a51;
  --ok-bg: #e4efe3;
  --warn: #b8791c;
  --warn-bg: #f6e9d4;
  --null: #8a8f98;
  --null-bg: #e9eae7;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", "IBM Plex Mono", monospace;
  --sans: ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141917;
    --surface: #1d2422;
    --surface-border: #2e3733;
    --ink: #e8ebe6;
    --ink-muted: #98a199;
    --accent: #6fbfb3;
    --accent-ink: #10201d;
    --ok: #7fbb84;
    --ok-bg: #223226;
    --warn: #e0a63d;
    --warn-bg: #3a2f18;
    --null: #7c8680;
    --null-bg: #232925;
  }
}
:root[data-theme="dark"] {
  --bg: #141917;
  --surface: #1d2422;
  --surface-border: #2e3733;
  --ink: #e8ebe6;
  --ink-muted: #98a199;
  --accent: #6fbfb3;
  --accent-ink: #10201d;
  --ok: #7fbb84;
  --ok-bg: #223226;
  --warn: #e0a63d;
  --warn-bg: #3a2f18;
  --null: #7c8680;
  --null-bg: #232925;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  -webkit-font-smoothing: antialiased;
}
header.top {
  position: sticky;
  top: 0;
  z-index: 5;
  background: var(--bg);
  border-bottom: 1px solid var(--surface-border);
  padding: 20px 28px 16px;
}
h1 {
  font-family: var(--mono);
  font-size: 1.15rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  margin: 0 0 2px;
  text-wrap: balance;
}
.subtitle {
  color: var(--ink-muted);
  font-size: 0.85rem;
  margin: 0 0 14px;
}
.stats {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}
.stat {
  background: var(--surface);
  border: 1px solid var(--surface-border);
  border-radius: 6px;
  padding: 8px 14px;
  min-width: 92px;
}
.stat .n {
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
  font-size: 1.3rem;
  font-weight: 600;
  display: block;
}
.stat .label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-muted);
}
.controls {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
}
.filter-btn {
  font-family: var(--mono);
  font-size: 0.78rem;
  background: var(--surface);
  border: 1px solid var(--surface-border);
  color: var(--ink);
  padding: 6px 12px;
  border-radius: 5px;
  cursor: pointer;
}
.filter-btn[aria-pressed="true"] {
  background: var(--accent);
  border-color: var(--accent);
  color: var(--accent-ink);
}
.filter-btn:focus-visible, #search:focus-visible, .card:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}
#search {
  font-family: var(--mono);
  font-size: 0.8rem;
  background: var(--surface);
  border: 1px solid var(--surface-border);
  color: var(--ink);
  padding: 6px 10px;
  border-radius: 5px;
  min-width: 220px;
}
#count {
  font-family: var(--mono);
  font-size: 0.78rem;
  color: var(--ink-muted);
  margin-left: auto;
}
main {
  padding: 20px 28px 60px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 14px;
}
.card {
  background: var(--surface);
  border: 1px solid var(--surface-border);
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  cursor: zoom-in;
}
.swatch {
  background: #ffffff;
  padding: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 140px;
  border-bottom: 1px solid var(--surface-border);
  position: relative;
}
.swatch img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
.pill {
  position: absolute;
  top: 6px;
  right: 6px;
  font-family: var(--mono);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 2px 7px;
  border-radius: 999px;
}
.pill.ok { background: var(--ok-bg); color: var(--ok); }
.pill.warn { background: var(--warn-bg); color: var(--warn); }
.pill.null { background: var(--null-bg); color: var(--null); }
.body { padding: 10px 12px 12px; display: flex; flex-direction: column; gap: 6px; flex: 1; }
.idrow { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
.cid { font-family: var(--mono); font-weight: 600; font-size: 1rem; }
.cid.empty { color: var(--null); font-weight: 400; }
.yield { font-family: var(--mono); font-size: 0.78rem; color: var(--ink-muted); }
.path { font-family: var(--mono); font-size: 0.65rem; color: var(--ink-muted); word-break: break-all; }
.note { font-size: 0.75rem; color: var(--ink-muted); line-height: 1.35; }
.flag { font-size: 0.72rem; color: var(--warn); line-height: 1.35; }
.source { font-family: var(--mono); font-size: 0.62rem; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.03em; }
.card.hidden { display: none; }
#lightbox {
  position: fixed;
  inset: 0;
  background: rgba(10, 12, 11, 0.82);
  display: none;
  align-items: center;
  justify-content: center;
  padding: 40px;
  z-index: 10;
  cursor: zoom-out;
}
#lightbox.open { display: flex; }
#lightbox img {
  max-width: 90vw;
  max-height: 85vh;
  background: #fff;
  border-radius: 6px;
  padding: 20px;
}
</style>
"""

SCRIPT = """
<script>
const cards = Array.from(document.querySelectorAll('.card'));
const buttons = Array.from(document.querySelectorAll('.filter-btn'));
const search = document.getElementById('search');
const count = document.getElementById('count');
const lightbox = document.getElementById('lightbox');
const lightboxImg = lightbox.querySelector('img');
let activeFilter = 'all';

function apply() {
  const q = search.value.trim().toLowerCase();
  let shown = 0;
  for (const c of cards) {
    const matchesFilter = activeFilter === 'all' || c.dataset.status === activeFilter;
    const matchesSearch = !q || c.dataset.search.includes(q);
    const visible = matchesFilter && matchesSearch;
    c.classList.toggle('hidden', !visible);
    if (visible) shown++;
  }
  count.textContent = shown + ' / ' + cards.length + ' shown';
}

for (const b of buttons) {
  b.addEventListener('click', () => {
    for (const other of buttons) other.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-pressed', 'true');
    activeFilter = b.dataset.filter;
    apply();
  });
}
search.addEventListener('input', apply);

for (const c of cards) {
  c.addEventListener('click', () => {
    lightboxImg.src = c.querySelector('img').src;
    lightbox.classList.add('open');
  });
}
lightbox.addEventListener('click', () => lightbox.classList.remove('open'));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') lightbox.classList.remove('open');
});

apply();
</script>
"""


def _img_b64(paper_dir, segment_path):
    p = paper_dir / segment_path
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _status(rec):
    if rec.get("flag"):
        return "warn"
    if rec.get("compound_id"):
        return "ok"
    return "null"


def _card(paper_dir, image_path, rec):
    status = _status(rec)
    compound_id = rec.get("compound_id")
    yield_percent = rec.get("yield_percent")
    panel_label = rec.get("panel_label")
    segment_path = rec["segment_path"]
    note = rec.get("note")
    flag = rec.get("flag")
    source = rec.get("source", "")

    b64 = _img_b64(paper_dir, segment_path)
    if b64 is None:
        return ""

    search_parts = []
    if compound_id:
        search_parts.append(compound_id)
    search_parts.append(segment_path)
    search_parts.append(image_path)
    if note:
        search_parts.append(note)
    if flag:
        search_parts.append(flag)
    data_search = html.escape(" ".join(search_parts).lower(), quote=True)

    cid_html = (
        f'<span class="cid">{html.escape(compound_id)}</span>'
        if compound_id else '<span class="cid empty">null</span>'
    )
    yield_bits = []
    if yield_percent:
        yield_bits.append(f"yield {html.escape(str(yield_percent))}")
    if panel_label:
        yield_bits.append(f"({html.escape(str(panel_label))})")
    yield_html = f'<div class="yield">{" ".join(yield_bits)}</div>' if yield_bits else ""

    note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
    flag_html = f'<div class="flag">⚑ {html.escape(flag)}</div>' if flag else ""

    return f"""<article class="card" data-status="{status}" data-search="{data_search}" tabindex="0">
  <div class="swatch">
    <span class="pill {status}">{status}</span>
    <img src="data:image/png;base64,{b64}" alt="">
  </div>
  <div class="body">
    <div class="idrow">{cid_html}{yield_html}</div>
    <div class="path">{html.escape(segment_path)}</div>
    {note_html}
    {flag_html}
    <div class="source">{html.escape(source)}</div>
  </div>
</article>
"""


def generate_paper(paper_dir, out_dir):
    links_path = paper_dir / "reaction_links.json"
    if not links_path.exists():
        return None
    links = json.loads(links_path.read_text())

    cards_html = []
    total = has_id = no_id = flagged = 0
    for image_path, records in links.items():
        for rec in records:
            card = _card(paper_dir, image_path, rec)
            if not card:
                continue
            cards_html.append(card)
            total += 1
            status = _status(rec)
            if status == "warn":
                flagged += 1
            elif status == "ok":
                has_id += 1
            else:
                no_id += 1

    name = paper_dir.name
    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Audit — {html.escape(name)}</title>
{STYLE}
</head>
<body>
<header class="top">
  <h1>{html.escape(name)}</h1>
  <p class="subtitle">Stage 5 structure ↔ identifier links — click a card to zoom, use filters to spot-check</p>
  <div class="stats">
<div class="stat"><span class="n">{total}</span><span class="label">total</span></div>
<div class="stat"><span class="n">{has_id}</span><span class="label">has id</span></div>
<div class="stat"><span class="n">{no_id}</span><span class="label">no id</span></div>
<div class="stat"><span class="n">{flagged}</span><span class="label">flagged</span></div>
</div>
  <div class="controls">
    <button class="filter-btn" data-filter="all" aria-pressed="true">All</button>
    <button class="filter-btn" data-filter="ok" aria-pressed="false">Has ID</button>
    <button class="filter-btn" data-filter="null" aria-pressed="false">No ID</button>
    <button class="filter-btn" data-filter="warn" aria-pressed="false">Flagged</button>
    <input id="search" type="text" placeholder="search id / path / note…">
    <span id="count"></span>
  </div>
</header>
<main>
{"".join(cards_html)}</main>
<div id="lightbox"><img src="" alt=""></div>
{SCRIPT}
</body>
</html>
"""
    out_path = out_dir / f"audit_{name}.html"
    out_path.write_text(page)
    return {"total": total, "has_id": has_id, "no_id": no_id, "flagged": flagged, "path": out_path}


def main():
    out_dirname = sys.argv[1] if len(sys.argv) > 1 else "audit_pages_v4"
    out_dir = ROOT / out_dirname
    out_dir.mkdir(exist_ok=True)

    for paper_dir in sorted(p for p in PAPERS_DIR.iterdir() if p.is_dir()):
        result = generate_paper(paper_dir, out_dir)
        if result is None:
            print(f"{paper_dir.name}: no reaction_links.json, skipped")
            continue
        print(
            f"{paper_dir.name}: {result['total']} total, {result['has_id']} has id, "
            f"{result['no_id']} no id, {result['flagged']} flagged -> {result['path']}"
        )


if __name__ == "__main__":
    main()
