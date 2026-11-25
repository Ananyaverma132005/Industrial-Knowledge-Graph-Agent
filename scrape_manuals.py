"""Scrape public-domain maintenance and safety manuals and save them for ingestion.

Sources are US OSHA pages (works of the US Government, public domain). For each page the
scraper keeps the paragraphs that talk about maintenance, inspection, lockout or repair,
and writes them to data/manuals/public/<MACHINES>_<name>.txt. The file name prefix tells
ingest_pdf.py which machines the text applies to (ALL, M3, M2_M6 ...).

Usage:  python scrape_manuals.py           (then: python ingest_pdf.py)
If the network is down, the sample manuals in data/manuals are still used.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from html.parser import HTMLParser

import httpx

import config

PUBLIC_DIR = config.MANUALS_DIR / "public"
MAX_CHARS = 4800  # keep the most relevant text only, so each source becomes a few chunks

SOURCES = [
    {"file": "ALL_osha_lockout_tagout", "title": "OSHA 1910.147 Control of hazardous energy (lockout/tagout)",
     "url": "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147"},
    {"file": "ALL_osha_machine_guarding", "title": "OSHA 1910.212 General requirements for all machines",
     "url": "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.212"},
    {"file": "M3_osha_power_presses", "title": "OSHA 1910.217 Mechanical power presses",
     "url": "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.217"},
    {"file": "M4_osha_power_transmission", "title": "OSHA 1910.219 Mechanical power-transmission apparatus (belts, pulleys)",
     "url": "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.219"},
    {"file": "M2_M6_osha_robotics", "title": "OSHA Robotics safety",
     "url": "https://www.osha.gov/robotics"},
]

KEYWORDS = re.compile(
    r"maint|inspect|repair|servic|lockout|tagout|energy|adjust|lubricat|belt|pulley|brake|clutch|guard|"
    r"press|robot|hydraulic|pneumatic|worn|wear|replace|test|verify|isolat|release|restart",
    re.I,
)


class _TextBlocks(HTMLParser):
    """Collects text of <p>, <li> and <h*> elements, skipping scripts, nav and footer."""

    BLOCK = {"p", "li", "h1", "h2", "h3", "h4"}
    SKIP = {"script", "style", "nav", "footer", "header", "noscript"}

    def __init__(self):
        super().__init__()
        self.blocks, self._buf, self._depth_skip, self._in_block = [], [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._depth_skip += 1
        elif tag in self.BLOCK:
            self._in_block += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._depth_skip:
            self._depth_skip -= 1
        elif tag in self.BLOCK and self._in_block:
            self._in_block -= 1
            if not self._in_block:
                text = " ".join("".join(self._buf).split())
                if text:
                    self.blocks.append(text)
                self._buf = []

    def handle_data(self, data):
        if self._in_block and not self._depth_skip:
            self._buf.append(data)


def extract(html: str) -> str:
    parser = _TextBlocks()
    parser.feed(html)
    blocks = [b for b in parser.blocks if len(b) > 60]
    # Rank paragraphs by keyword hits, keep the best ones but in their original order.
    ranked = sorted(range(len(blocks)), key=lambda i: -len(KEYWORDS.findall(blocks[i])))
    keep, size = set(), 0
    for i in ranked:
        if size > MAX_CHARS or not KEYWORDS.search(blocks[i]):
            break
        keep.add(i)
        size += len(blocks[i])
    return "\n".join(blocks[i] for i in sorted(keep))


def scrape() -> list[dict]:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    headers = {"User-Agent": "Mozilla/5.0 (maintenance knowledge graph - student project)"}
    with httpx.Client(headers=headers, timeout=30, follow_redirects=True) as client:
        for src in SOURCES:
            try:
                resp = client.get(src["url"])
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"[scrape] skipped {src['url']} ({exc.__class__.__name__})")
                continue
            text = extract(resp.text)
            if not text:
                print(f"[scrape] no usable text in {src['url']}")
                continue
            out = PUBLIC_DIR / f"{src['file']}.txt"
            out.write_text(f"{src['title']}\nSource: {src['url']}\n\n{text}\n", encoding="utf-8")
            manifest.append({**src, "chars": len(text), "scraped_on": date.today().isoformat(),
                             "license": "US Government work (public domain)"})
            print(f"[scrape] {out.name}: {len(text)} chars")
            time.sleep(1)  # be polite to the server
    if manifest:
        (PUBLIC_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    got = scrape()
    print(f"[scrape] saved {len(got)} of {len(SOURCES)} public manuals to {PUBLIC_DIR.relative_to(config.BASE_DIR)}")
