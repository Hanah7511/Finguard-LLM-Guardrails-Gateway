
"""Download the official sanctions lists and convert them to one FinGuard format.

    UN Security Council Consolidated List   scsanctions.un.org            (XML)
    UAE Local Terrorist List                 uaeiec.gov.ae (EOCN)          (Excel)
    OFAC Specially Designated Nationals      sanctionslistservice.ofac.treas.gov (XML)

Writes data/sanctions/sanctions_list.json (gitignored: large and changes often).
Re-run it to refresh; banks refresh at least daily and on every list update.

Each entry keeps the identifiers the list provides (dates of birth, nationalities,
ID documents), so a name match can be confirmed or ruled out, not just guessed.

Usage:  python scripts/build_sanctions_lists.py   (needs: pip install xlrd)
"""
from __future__ import annotations

import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "sanctions" / "sanctions_list.json"

UN_URL = "https://scsanctions.un.org/resources/xml/en/consolidated.xml"
OFAC_URL = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.XML"
UAE_PAGE = "https://www.uaeiec.gov.ae/en-us/un-page?p=1"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

# OFAC ID types worth keeping (the list also stores gender, websites, legal notes...).
_ID_TYPES = re.compile(
    r"passport|national|identification|id no|cedula|registration|tax|licen|ssn|dni|curp|rfc|nit|ruc|imo|mmsi|serial",
    re.I,
)

# The UAE list writes nationality in Arabic, sometimes as an adjective ("Lebanese").
# Arabic text is written as \u escapes so copy-paste cannot reverse it (right-to-left).
_UAE_NATIONALITY = {
    "\u0645\u0635\u0631": "Egypt", "\u0627\u0644\u064a\u0645\u0646": "Yemen", "\u0627\u0644\u0635\u0648\u0645\u0627\u0644": "Somalia", "\u0644\u0628\u0646\u0627\u0646": "Lebanon", "\u0644\u0628\u0646\u0627\u0646\u064a": "Lebanon",
    "\u0642\u0637\u0631": "Qatar", "\u0644\u064a\u0628\u064a\u0627": "Libya", "\u0623\u0641\u063a\u0627\u0646\u0633\u062a\u0627\u0646": "Afghanistan", "\u0627\u0644\u0625\u0645\u0627\u0631\u0627\u062a": "United Arab Emirates",
    "\u0627\u0644\u0627\u0645\u0627\u0631\u0627\u062a": "United Arab Emirates", "\u0625\u064a\u0631\u0627\u0646": "Iran", "\u0627\u064a\u0631\u0627\u0646": "Iran", "\u0625\u064a\u0631\u0627\u0646\u064a": "Iran",
    "\u0646\u064a\u062c\u064a\u0631\u064a": "Nigeria", "\u0646\u064a\u062c\u064a\u0631\u064a\u0627": "Nigeria", "\u0627\u0644\u0623\u0631\u062f\u0646": "Jordan", "\u0627\u0644\u0643\u0648\u064a\u062a": "Kuwait", "\u0627\u0644\u0628\u062d\u0631\u064a\u0646": "Bahrain",
    "\u0633\u0648\u0631\u064a": "Syria", "\u0633\u0648\u0631\u064a\u0627": "Syria", "\u0628\u0627\u0643\u0633\u062a\u0627\u0646": "Pakistan", "\u0633\u0627\u0646\u062a \u0643\u064a\u062a\u0633": "Saint Kitts and Nevis",
    "\u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0629": "Saudi Arabia", "\u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0647": "Saudi Arabia", "\u0623\u0645\u0631\u064a\u0643\u0627": "United States",
    "\u0627\u0644\u0646\u0645\u0633\u0627": "Austria", "\u0628\u0631\u064a\u0637\u0627\u0646\u064a\u0627": "United Kingdom", "\u0631\u0648\u0633\u064a\u0627": "Russia", "\u062a\u0631\u0643\u064a\u0627": "Turkey",
    "\u0627\u0644\u0633\u0648\u064a\u062f": "Sweden", "\u0644\u064a\u0628\u064a\u0631\u064a\u0627": "Liberia", "\u0627\u0644\u0639\u0631\u0627\u0642": "Iraq", "\u062a\u0648\u0646\u0633": "Tunisia", "\u0627\u0644\u0645\u063a\u0631\u0628": "Morocco",
    "\u0627\u0644\u062c\u0632\u0627\u0626\u0631": "Algeria", "\u0627\u0644\u0633\u0648\u062f\u0627\u0646": "Sudan", "\u0639\u0645\u0627\u0646": "Oman", "\u0641\u0644\u0633\u0637\u064a\u0646": "Palestine",
    "\u062f\u0628\u064a": "United Arab Emirates",
}
# Keyed by the first 5 characters, so every passport spelling ("passport",
# "the passport", "Yemeni passport") maps to Passport.
_UAE_DOC_TYPES = {"\u062c\u0648\u0627\u0632 ": "Passport", "\u0627\u0644\u0631\u0642\u0645": "UAE Unified Number", "\u062a\u0623\u0634\u064a\u0631": "Visa"}
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_ARABIC_MONTHS = {
    "\u064a\u0646\u0627\u064a\u0631": 1, "\u0641\u0628\u0631\u0627\u064a\u0631": 2, "\u0645\u0627\u0631\u0633": 3, "\u0623\u0628\u0631\u064a\u0644": 4, "\u0625\u0628\u0631\u064a\u0644": 4, "\u0627\u0628\u0631\u064a\u0644": 4, "\u0645\u0627\u064a\u0648": 5, "\u064a\u0648\u0646\u064a\u0648": 6,
    "\u064a\u0648\u0644\u064a\u0648": 7, "\u0623\u063a\u0633\u0637\u0633": 8, "\u0627\u063a\u0633\u0637\u0633": 8, "\u0633\u0628\u062a\u0645\u0628\u0631": 9, "\u0623\u0643\u062a\u0648\u0628\u0631": 10, "\u0627\u0643\u062a\u0648\u0628\u0631": 10, "\u0646\u0648\u0641\u0645\u0628\u0631": 11, "\u062f\u064a\u0633\u0645\u0628\u0631": 12,
}


def uae_country(arabic: str) -> list[str]:
    return [en for ar, en in _UAE_NATIONALITY.items() if ar in arabic] or ([arabic] if arabic else [])


def fetch(url: str, user_agent: str = "FinGuard-sanctions/1.0") -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip(" -")


def entry(**fields) -> dict:
    """One sanctions entry in FinGuard's common format (empty fields dropped)."""
    base = {
        "name": "", "aliases": [], "weak_aliases": [], "names_original_script": [],
        "list": "", "list_id": "", "program": "", "entity_type": "individual",
        "dates_of_birth": [], "nationalities": [], "id_numbers": [], "listed_on": "", "remarks": "",
    }
    base.update(fields)
    for key in ("aliases", "weak_aliases", "names_original_script", "dates_of_birth", "nationalities"):
        base[key] = list(dict.fromkeys(v for v in base[key] if v and v != base["name"]))
    return base


# --- UN ------------------------------------------------------------------------

def un_dob(node: ET.Element) -> str:
    """'1965-05-04', '1971', or a range '1960..1965'."""
    if node.findtext("DATE"):
        return node.findtext("DATE")[:10]
    if node.findtext("YEAR"):
        return node.findtext("YEAR")
    if node.findtext("FROM_YEAR") and node.findtext("TO_YEAR"):
        return f"{node.findtext('FROM_YEAR')}..{node.findtext('TO_YEAR')}"
    return ""


def un_entries() -> tuple[list[dict], dict]:
    root = ET.fromstring(fetch(UN_URL))
    out = []
    for kind, tag, alias_tag in (("individual", "INDIVIDUALS/INDIVIDUAL", "INDIVIDUAL_ALIAS"),
                                 ("entity", "ENTITIES/ENTITY", "ENTITY_ALIAS")):
        for node in root.findall(tag):
            name = clean(" ".join(node.findtext(t) or "" for t in
                                  ("FIRST_NAME", "SECOND_NAME", "THIRD_NAME", "FOURTH_NAME")))
            aliases, weak = [], []
            for a in node.findall(alias_tag):
                alias = clean(a.findtext("ALIAS_NAME"))
                (weak if (a.findtext("QUALITY") or "").lower() == "low" else aliases).append(alias)
            out.append(entry(
                name=name, aliases=aliases, weak_aliases=weak,
                names_original_script=[clean(node.findtext("NAME_ORIGINAL_SCRIPT"))],
                list="UN", list_id=clean(node.findtext("REFERENCE_NUMBER")),
                program=clean(node.findtext("UN_LIST_TYPE")), entity_type=kind,
                dates_of_birth=[un_dob(d) for d in node.findall("INDIVIDUAL_DATE_OF_BIRTH")],
                nationalities=[clean(v.text) for v in node.findall("NATIONALITY/VALUE")],
                id_numbers=[
                    {"type": clean(d.findtext("TYPE_OF_DOCUMENT")), "number": clean(d.findtext("NUMBER")),
                     "country": clean(d.findtext("ISSUING_COUNTRY"))}
                    for d in node.findall("INDIVIDUAL_DOCUMENT") if clean(d.findtext("NUMBER"))
                ],
                listed_on=clean(node.findtext("LISTED_ON")),
            ))
    meta = {"source_url": UN_URL, "published": root.attrib.get("dateGenerated", "")[:10], "count": len(out)}
    return out, meta


# --- OFAC ----------------------------------------------------------------------

def ofac_dob(text: str) -> str:
    """'10 Dec 1948' -> '1948-12-10', 'Dec 1948' -> '1948-12', '1960 to 1962' -> '1960..1962'."""
    parts = text.lower().replace("circa", "").split()
    years = [p for p in parts if re.fullmatch(r"\d{4}", p)]
    if len(years) == 2:
        return f"{years[0]}..{years[1]}"
    if len(parts) == 3 and parts[1][:3] in _MONTHS and parts[0].isdigit():
        return f"{parts[2]}-{_MONTHS[parts[1][:3]]:02d}-{int(parts[0]):02d}"
    if len(parts) == 2 and parts[0][:3] in _MONTHS:
        return f"{parts[1]}-{_MONTHS[parts[0][:3]]:02d}"
    return years[0] if years else ""


def ofac_entries() -> tuple[list[dict], dict]:
    root = ET.fromstring(fetch(OFAC_URL))
    ns = {"o": root.tag.split("}")[0][1:]}
    full = lambda n: clean(f"{n.findtext('o:firstName', '', ns)} {n.findtext('o:lastName', '', ns)}")
    out = []
    for node in root.findall("o:sdnEntry", ns):
        aliases, weak = [], []
        for a in node.findall("o:akaList/o:aka", ns):
            (weak if a.findtext("o:category", "", ns) == "weak" else aliases).append(full(a))
        out.append(entry(
            name=full(node), aliases=aliases, weak_aliases=weak,
            list="OFAC", list_id=node.findtext("o:uid", "", ns),
            program=", ".join(p.text for p in node.findall("o:programList/o:program", ns)),
            entity_type=node.findtext("o:sdnType", "", ns).lower(),
            dates_of_birth=[ofac_dob(d.text or "") for d in node.findall(
                "o:dateOfBirthList/o:dateOfBirthItem/o:dateOfBirth", ns)],
            nationalities=[c.text for c in node.findall("o:nationalityList/o:nationality/o:country", ns)]
            + [c.text for c in node.findall("o:citizenshipList/o:citizenship/o:country", ns)],
            id_numbers=[
                {"type": i.findtext("o:idType", "", ns), "number": i.findtext("o:idNumber", "", ns),
                 "country": i.findtext("o:idCountry", "", ns)}
                for i in node.findall("o:idList/o:id", ns) if _ID_TYPES.search(i.findtext("o:idType", "", ns))
            ],
        ))
    info = root.find("o:publshInformation", ns)
    published = info.findtext("o:Publish_Date", "", ns) if info is not None else ""
    if published:
        published = datetime.strptime(published, "%m/%d/%Y").date().isoformat()
    return out, {"source_url": OFAC_URL, "published": published, "count": len(out)}


# --- UAE Local Terrorist List ----------------------------------------------------

def uae_entries() -> tuple[list[dict], dict]:
    import xlrd  # the EOCN publishes the list as an old-style .xls file

    page = fetch(UAE_PAGE, BROWSER_UA).decode("utf-8", "replace")
    # The page shows a "Download Excel File" label followed by its download link.
    link = re.search(r'Download Excel File.{0,600}?href="([^"]*DownloadFile\?FileID=[^"]+)"', page, re.S)
    if not link:
        raise SystemExit(f"Could not find the Local Terrorist List Excel link on {UAE_PAGE}")
    url = urljoin(UAE_PAGE, unescape(link.group(1)))
    book = xlrd.open_workbook(file_contents=fetch(url, BROWSER_UA))

    def dob(value) -> str:
        if isinstance(value, float):
            if value < 3000:  # a bare year
                return str(int(value))
            y, m, d, *_ = xlrd.xldate_as_tuple(value, book.datemode)
            return date(y, m, d).isoformat()
        value = clean(str(value))
        if m := re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", value):  # 17/02/1967
            return f"{m[3]}-{int(m[2]):02d}-{int(m[1]):02d}"
        if m := re.fullmatch(r"(\d{1,2}) (\S+) (\d{4})\u0645?", value):  # "21 February 1987" written in Arabic
            if month := _ARABIC_MONTHS.get(m[2]):
                return f"{m[3]}-{month:02d}-{int(m[1]):02d}"
        year = re.search(r"\d{4}", value)
        return year.group(0) if year else ""

    def text(value) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return clean(str(value))

    out = []
    for sheet in book.sheets():
        if "\u0631\u0641\u0639" in sheet.name:  # sheets named "delisting": no longer sanctioned
            continue
        header = [str(h).strip() for h in sheet.row_values(2)]
        col = lambda *keys: next((i for i, h in enumerate(header) if all(k in h for k in keys)), None)
        # "Full name" columns; the individuals sheet also has family-name-only columns.
        latin, arabic = col("\u0627\u0644\u0627\u0633\u0645 \u0627\u0644\u0643\u0627\u0645\u0644", "\u0627\u0644\u0644\u0627\u062a\u064a\u0646\u064a\u0629"), col("\u0627\u0644\u0627\u0633\u0645 \u0627\u0644\u0643\u0627\u0645\u0644", "\u0627\u0644\u0639\u0631\u0628\u064a\u0629")
        alias = header.index("\u0627\u0644\u0627\u0633\u0645") if "\u0627\u0644\u0627\u0633\u0645" in header else None
        nat, birth, doc_no, doc_type, issuer, info = (
            col("\u0627\u0644\u062c\u0646\u0633\u064a\u0629"), col("\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0645\u064a\u0644\u0627\u062f"), col("\u0631\u0642\u0645 \u0627\u0644\u0648\u062b\u064a\u0642\u0629"), col("\u0627\u0644\u0646\u0648\u0639"), col("\u062c\u0647\u0629 \u0627\u0644\u0625\u0635\u062f\u0627\u0631"), col("\u0645\u0639\u0644\u0648\u0645\u0627\u062a \u0623\u062e\u0631\u0649"))
        kind = "individual" if nat is not None else "entity"
        for r in range(3, sheet.nrows):
            row = sheet.row_values(r)
            name = text(row[latin]) if latin is not None else ""
            if not name:
                continue
            nationality = text(row[nat]) if nat is not None else ""
            out.append(entry(
                name=name,
                names_original_script=[text(row[i]) for i in (arabic, alias) if i is not None],
                list="UAE_LOCAL", list_id=f"UAE-{kind[0].upper()}-{text(row[0])}",
                program="UAE Local Terrorist List", entity_type=kind,
                dates_of_birth=[dob(row[birth])] if birth is not None else [],
                nationalities=uae_country(nationality),
                id_numbers=[{"type": _UAE_DOC_TYPES.get(text(row[doc_type])[:5], text(row[doc_type])),
                             "number": text(row[doc_no]),
                             "country": ", ".join(uae_country(text(row[issuer]))) if issuer is not None else ""}]
                if doc_no is not None and text(row[doc_no]) else [],
                remarks=text(row[info]) if info is not None else "",
            ))
    return out, {"source_url": url, "published": "", "count": len(out)}


def main() -> None:
    lists, meta = [], {}
    for name, build in (("UN", un_entries), ("UAE_LOCAL", uae_entries), ("OFAC", ofac_entries)):
        entries, info = build()
        lists += entries
        meta[name] = {**info, "retrieved_on": date.today().isoformat()}
        print(f"  {name}: {len(entries)} entries")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"_meta": meta, "entries": lists}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(lists)} entries to {OUT}")


if __name__ == "__main__":
    main()