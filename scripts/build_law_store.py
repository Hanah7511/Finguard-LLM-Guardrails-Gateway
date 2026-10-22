"""Build FinGuard's trusted law store from official US government sources.

Downloads the exact text of each section, splits it into top-level subsections
((a), (b), ...) and writes data/law_store/regulations.json. Every entry records
where it came from and when, so an auditor can trace any grounding decision.

Only public-domain US law is fetched here. FATF and UAE texts are copyrighted or
not script-downloadable, so they are added separately and kept out of git.

Usage:  python scripts/build_law_store.py [--out PATH]
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "law_store" / "regulations.json"
ECFR_TITLES = "https://www.ecfr.gov/api/versioner/v1/titles.json"
ECFR_SECTION = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-31.xml?section={section}"
ECFR_PUBLIC = "https://www.ecfr.gov/current/title-31/section-{section}"
USC_EDITION = 2024
USC_URL = (
    "https://www.govinfo.gov/content/pkg/USCODE-{ed}-title31/html/"
    "USCODE-{ed}-title31-subtitleIV-chap53-subchapII-sec{section}.htm"
)

# (section, only these top-level paragraphs or None for all)
CFR_SECTIONS: list[tuple[str, list[str] | None]] = [
    ("1010.100", ["xx"]),  # definition of "structure"; the rest of 1010.100 is other definitions
    ("1010.311", None),    # CTR: currency transactions over $10,000
    ("1010.313", None),    # aggregation of multiple currency transactions
    ("1010.314", None),    # structured transactions prohibited
    ("1010.230", None),    # beneficial ownership of legal entity customers
    ("1010.410", None),    # funds transfer records and the travel rule
    ("1020.320", None),    # SARs by banks
]
USC_SECTIONS = ["5324"]   # structuring (statute)

_LABEL = re.compile(r"^\(([a-z]+|\d+)\)")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "FinGuard-law-store/1.0", "Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
        return gzip.decompress(body) if resp.headers.get("Content-Encoding") == "gzip" else body


def split_top_level(paragraphs: list[str]) -> list[tuple[str, str]]:
    """Group paragraphs under top-level letters (a), (b), ...

    "(i)" is ambiguous: the letter after (h), or roman numeral one. It is roman
    when the next labelled paragraph is "(ii)".
    """
    labels = [m.group(1) if (m := _LABEL.match(p)) else None for p in paragraphs]
    groups: list[tuple[str, list[str]]] = []
    expected = "a"
    for i, (para, label) in enumerate(zip(paragraphs, labels)):
        starts_group = label == expected
        if starts_group and label == "i":
            following = next((l for l in labels[i + 1:] if l), None)
            starts_group = following != "ii"
        if starts_group:
            groups.append((label, [para]))
            expected = chr(ord(expected) + 1)
        elif groups:
            groups[-1][1].append(para)
        else:
            groups.append(("", [para]))  # intro text before (a)
    # Intro text (e.g. "No person shall ...:") belongs with every subsection it introduces,
    # so it is kept as a prefix of each one.
    intro = " ".join(groups[0][1]) if groups and groups[0][0] == "" else ""
    out = []
    for label, paras in groups:
        if label == "":
            continue
        text = " ".join(paras)
        out.append((label, f"{intro} {text}".strip() if intro else text))
    return out or [("", intro)]


def cfr_entries(issue_date: str, retrieved_on: str) -> list[dict]:
    entries = []
    for section, only in CFR_SECTIONS:
        root = ET.fromstring(fetch(ECFR_SECTION.format(date=issue_date, section=section)))
        head = "".join(root.find("HEAD").itertext()).strip()
        title = re.sub(r"^§\s*[\d.]+\s*", "", head).rstrip(".")
        paragraphs = [re.sub(r"\s+", " ", "".join(p.itertext())).strip() for p in root.iter("P")]
        if only:
            # Definitions like (xx) are flat: take the paragraph itself, not a letter sequence.
            parts = [(lbl, p) for p in paragraphs if (m := _LABEL.match(p)) and (lbl := m.group(1)) in only]
        else:
            parts = split_top_level(paragraphs)
        sec_citation = f"31 CFR {section}"
        sec_id = "CFR_" + section.replace(".", "_")
        for label, text in parts:
            entries.append({
                "doc_id": f"{sec_id}_{label}" if label else sec_id,
                "citation": f"{sec_citation}({label})" if label else sec_citation,
                "section": sec_citation,
                "section_id": sec_id,
                "title": title,
                "jurisdiction": "US",
                "source": "eCFR (Electronic Code of Federal Regulations)",
                "source_url": ECFR_PUBLIC.format(section=section),
                "version_date": issue_date,
                "retrieved_on": retrieved_on,
                "text": text,
            })
    return entries


def usc_entries(retrieved_on: str) -> list[dict]:
    entries = []
    for section in USC_SECTIONS:
        url = USC_URL.format(ed=USC_EDITION, section=section)
        page = fetch(url).decode("utf-8", errors="replace")
        heading = re.search(r'section-head">(.*?)</h3>', page, re.S)
        title = html.unescape(re.sub(r"<[^>]+>", "", heading.group(1))) if heading else ""
        title = re.sub(r"^§\s*\d+\.\s*", "", title).strip().rstrip(".")
        body = page.split("field-start:statute", 1)[1].split("field-end:statute", 1)[0]
        paragraphs = [
            re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", p))).strip()
            for p in re.findall(r"<p[^>]*>(.*?)</p>", body, re.S)
        ]
        sec_citation = f"31 U.S.C. § {section}"
        sec_id = f"USC_31_{section}"
        for label, text in split_top_level([p for p in paragraphs if p]):
            entries.append({
                "doc_id": f"{sec_id}_{label}",
                "citation": f"{sec_citation}({label})",
                "section": sec_citation,
                "section_id": sec_id,
                "title": title,
                "jurisdiction": "US",
                "source": f"United States Code, {USC_EDITION} edition (govinfo.gov)",
                "source_url": url,
                "version_date": str(USC_EDITION),
                "retrieved_on": retrieved_on,
                "text": text,
            })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    titles = json.loads(fetch(ECFR_TITLES))["titles"]
    issue_date = next(t["latest_issue_date"] for t in titles if t["number"] == 31)
    today = date.today().isoformat()

    documents = usc_entries(today) + cfr_entries(issue_date, today)
    store = {
        "_notice": (
            "Official text of US federal law (public domain), downloaded by "
            "scripts/build_law_store.py. Re-run the script to refresh. Each entry "
            "records its source_url, version_date and retrieved_on date."
        ),
        "documents": documents,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sections = sorted({d["section"] for d in documents})
    print(f"Wrote {len(documents)} entries from {len(sections)} sections to {args.out}")
    for s in sections:
        print(f"  {s}: {sum(d['section'] == s for d in documents)} part(s)")


if __name__ == "__main__":
    main()


