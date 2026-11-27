"""Build docs/brief.pdf (2 A4 pages) from docs/BRIEF.md.

Usage:  python docs/build_brief_pdf.py
Needs (dev only, never imported by serve.py): markdown, playwright (+ `playwright install chromium`), pypdf.
"Page 2" in BRIEF.md starts on a new sheet. The script fails if the PDF is not exactly 2 pages.
"""
from __future__ import annotations

import sys
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "BRIEF.md"
OUT = ROOT / "docs" / "brief.pdf"

CSS = """
@page { size: A4; margin: 12mm 14mm 11mm 14mm; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 9pt; line-height: 1.3;
       color: #1B2A41; }
h1 { font-size: 15pt; margin: 0 0 2pt 0; letter-spacing: -0.01em; }
h2 { font-size: 11.5pt; margin: 8pt 0 3pt 0; color: #2F6F9F; border-bottom: 0.6pt solid #c9d6df; padding-bottom: 1pt; }
h2.page2 { break-before: page; margin-top: 0; }
p { margin: 0 0 4.5pt 0; }
ul { margin: 0 0 4pt 0; padding-left: 12pt; }
li { margin: 0 0 1pt 0; }
em { color: #44566b; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 3pt 0; font-size: 8.9pt; }
th, td { border: 0.5pt solid #c9d6df; padding: 2pt 5pt; text-align: left; }
th { background: #EEF3F6; }
td:nth-child(2), td:nth-child(3), th:nth-child(2), th:nth-child(3) { text-align: right; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #EEF3F6; padding: 0 2pt; border-radius: 2pt; }
"""


def _blank_line_before_lists(text: str) -> str:
    """Python-Markdown needs a blank line before a list (GitHub does not); add it where missing."""
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith("- ") and out and out[-1].strip() and not out[-1].startswith("- "):
            out.append("")
        out.append(line)
    return "\n".join(out) + "\n"


def build() -> int:
    src = _blank_line_before_lists(SRC.read_text(encoding="utf-8"))
    html_body = markdown.markdown(src, extensions=["tables"])
    html_body = html_body.replace("<h2>Page 2", '<h2 class="page2">Page 2', 1)
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        page.pdf(path=str(OUT), format="A4", print_background=True, prefer_css_page_size=True)
        browser.close()
    n = len(PdfReader(str(OUT)).pages)
    print(f"wrote {OUT.relative_to(ROOT)}: {n} pages")
    if n != 2:
        print("FAIL: the brief must be exactly 2 pages; shorten docs/BRIEF.md", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(build())
