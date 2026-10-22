
"""Build FinGuard's LOCAL law store from official PDFs you downloaded yourself.

These texts are copyrighted (FATF: "all rights reserved") or their reuse terms
are unclear, so the PDFs and the output stay on your machine and are gitignored:

    data/legal_sources/*.pdf  ->  data/law_store/regulations_local.json

Expected PDFs in data/legal_sources/ (any file name matching the pattern):
    fatf*.pdf     FATF Recommendations          fatf-gafi.org
    uae*.pdf      UAE Federal Decree-Law No. 10 of 2025   uaelegislation.gov.ae
    cbuae*.pdf    CBUAE Guidance on Transaction Monitoring and Sanctions Screening
                  rulebook.centralbank.ae

Usage:  python scripts/build_local_law_store.py   (needs: pip install pypdf)
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES_DIR = PROJECT_ROOT / "data" / "legal_sources"
OUT = PROJECT_ROOT / "data" / "law_store" / "regulations_local.json"

FATF_RECOMMENDATIONS = [1, 6, 10, 11, 12, 16, 19, 20, 24]

UAE_LAW = "Federal Decree-Law No. 10 of 2025"
CBUAE_GUIDANCE = "CBUAE Guidance on Transaction Monitoring and Sanctions Screening"


def find_pdf(pattern: str) -> Path:
    matches = sorted(SOURCES_DIR.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one '{pattern}' in {SOURCES_DIR}, found {len(matches)}")
    return matches[0]


def read_lines(pdf: Path) -> list[str]:
    lines = []
    for page in PdfReader(pdf).pages:
        lines.extend(l.strip() for l in (page.extract_text() or "").splitlines())
    return [l for l in lines if l]


def file_info(pdf: Path) -> dict:
    return {
        "source_file": pdf.name,
        "source_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }


def join(lines: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def split_by_headings(lines: list[str], is_heading) -> list[tuple[str, list[str]]]:
    """[(heading_key, body_lines)]; is_heading(line) returns a key or None."""
    parts: list[tuple[str, list[str]]] = []
    for line in lines:
        key = is_heading(line)
        if key is not None:
            parts.append((key, [line]))
        elif parts:
            parts[-1][1].append(line)
    return parts


# --- FATF ---------------------------------------------------------------------

def fatf_entries(pdf: Path, retrieved_on: str) -> list[dict]:
    lines = read_lines(pdf)
    updated = next((m.group(1) for l in lines[:40] if (m := re.search(r"Updated (\w+ \d{4})", l))), "")
    # The 40 Recommendations come first; the interpretive notes repeat the numbers.
    start = next(i for i, l in enumerate(lines) if re.match(r"1\.\s+Assessing risks", l))
    end = next(i for i, l in enumerate(lines) if i > start and "INTERPRETIVE NOTES TO THE FATF" in l.upper())
    noise = re.compile(r"^(THE FATF RECOMMENDATIONS|INTERNATIONAL STANDARDS ON COMBATING.*|\d*\s*2012-\d{4}\s*\d*)$")
    body = [l for l in lines[start:end] if not noise.match(l)]
    # Drop all-caps group titles such as "D. PREVENTIVE MEASURES".
    body = [l for l in body if not (l.upper() == l and re.search(r"[A-Z]{4}", l))]

    expected = [1]

    def heading(line: str):
        m = re.match(r"(\d{1,2})\.\s+[A-Z]", line)
        if m and int(m.group(1)) == expected[0] and len(line) < 110:
            expected[0] += 1
            return int(m.group(1))
        return None

    entries = []
    for number, part in split_by_headings(body, heading):
        if number not in FATF_RECOMMENDATIONS:
            continue
        title = re.sub(r"^\d+\.\s*|\s*\*$", "", part[0]).strip()
        entries.append({
            "doc_id": f"FATF_R{number}",
            "citation": f"FATF Recommendation {number}",
            "aliases": [f"FATF Rec {number}", f"FATF R.{number}", f"FATF R{number}"],
            "section": "FATF Recommendations",
            "section_id": "FATF",
            "title": title,
            "jurisdiction": "International",
            "source": f"The FATF Recommendations (updated {updated})",
            "source_url": "https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html",
            "version_date": updated,
            "retrieved_on": retrieved_on,
            "notes": "FATF standard, not law. (c) FATF/OECD, all rights reserved: local use only.",
            "text": join(part[1:]),
            **file_info(pdf),
        })
    return entries


# --- UAE Decree-Law No. 10 of 2025 ---------------------------------------------

def uae_entries(pdf: Path, retrieved_on: str) -> list[dict]:
    lines = read_lines(pdf)
    noise = re.compile(r"^(Federal Decree by Law of 2025 Regarding.*|and Proliferation Financing|\d{1,3})$")
    lines = [l for l in lines if not noise.match(l)]
    # Chapter titles sit between "Chapter Two" and the next "Article (n)": drop them.
    kept, in_chapter_title = [], False
    for l in lines:
        if re.match(r"^Chapter [A-Z][a-z]+$", l):
            in_chapter_title = True
            continue
        if in_chapter_title and not re.match(r"^Article \(\d+\)$", l):
            continue
        in_chapter_title = False
        kept.append(l)

    expected = [1]

    def heading(line: str):
        m = re.match(r"^Article \((\d+)\)$", line)
        if m and int(m.group(1)) == expected[0]:
            expected[0] += 1
            return int(m.group(1))
        return None

    entries = []
    for number, part in split_by_headings(kept, heading):
        entries.append({
            "doc_id": f"UAE_FDL_10_2025_A{number}",
            "citation": f"{UAE_LAW}, Article {number}",
            "aliases": [
                f"Federal Decree by Law No. 10 of 2025, Article {number}",
                f"Decree-Law No. 10 of 2025, Article {number}",
                f"Article {number} of Federal Decree-Law No. 10 of 2025",
                f"Article {number} of Federal Decree by Law No. 10 of 2025",
            ],
            "section": UAE_LAW,
            "section_id": "UAE_FDL_10_2025",
            "title": f"Article ({number})",
            "jurisdiction": "UAE",
            "source": "UAE Legislation portal, official English translation",
            "source_url": "https://uaelegislation.gov.ae/en/legislations/3314",
            "version_date": "2025-10-14",
            "retrieved_on": retrieved_on,
            "defined_terms": {"the Unit": "the Financial Intelligence Unit"},
            "notes": "In force from 14 October 2025; replaces Decree-Law No. 20 of 2018. "
                     "English translation: the Arabic text is authoritative.",
            "text": join(part[1:]),
            **file_info(pdf),
        })
    return entries


# --- CBUAE guidance ------------------------------------------------------------

def cbuae_entries(pdf: Path, retrieved_on: str) -> list[dict]:
    lines = read_lines(pdf)
    noise = re.compile(r"^(Page \d+ of \d+|CBUAE Classification: Public|\d+ Available at:.*)$")
    lines = [l for l in lines if not noise.match(l)]
    start = next(i for i, l in enumerate(lines) if re.match(r"^1\.\s+Introduction$", l))
    seen: set[str] = set()

    def heading(line: str):
        m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+[A-Z][A-Za-z ,&/()\-]{2,80}$", line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            return m.group(1)
        return None

    entries = []
    for number, part in split_by_headings(lines[start:], heading):
        text = join(part[1:])
        if not text:
            continue  # a heading directly followed by its first subsection
        title = re.sub(r"^[\d.]+\s*", "", part[0]).strip()
        entries.append({
            "doc_id": "CBUAE_TMSS_" + number.replace(".", "_"),
            "citation": f"{CBUAE_GUIDANCE}, Section {number}",
            "aliases": [f"CBUAE TM and Sanctions Screening Guidance, Section {number}"],
            "section": CBUAE_GUIDANCE,
            "section_id": "CBUAE_TMSS",
            "title": title,
            "jurisdiction": "UAE",
            "source": "CBUAE Rulebook",
            "source_url": "https://rulebook.centralbank.ae/en/rulebook/"
                          "guidance-licensed-financial-institutions-transaction-monitoring-and-sanctions-screening",
            "version_date": "2021-09-08",
            "retrieved_on": retrieved_on,
            "defined_terms": {
                "LFIs": "licensed financial institutions",
                "LFI": "licensed financial institution",
            },
            "notes": "Issued under Decree-Law No. 20 of 2018 (since replaced by Decree-Law No. 10 of 2025); "
                     "check the CBUAE Rulebook for updated guidance.",
            "text": text,
            **file_info(pdf),
        })
    return entries


def main() -> None:
    today = date.today().isoformat()
    documents = (
        fatf_entries(find_pdf("fatf*.pdf"), today)
        + uae_entries(find_pdf("uae*.pdf"), today)
        + cbuae_entries(find_pdf("cbuae*.pdf"), today)
    )
    store = {
        "_notice": (
            "LOCAL ONLY - do not commit. Extracted by scripts/build_local_law_store.py from "
            "PDFs in data/legal_sources/. Contains copyrighted text (FATF)."
        ),
        "documents": documents,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(documents)} entries to {OUT}")
    for section in dict.fromkeys(d["section"] for d in documents):
        print(f"  {section}: {sum(d['section'] == section for d in documents)} part(s)")


if __name__ == "__main__":
    main()



