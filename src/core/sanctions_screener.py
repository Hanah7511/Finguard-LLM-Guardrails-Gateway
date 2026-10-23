"""Sanctions screening: exact + fuzzy name matching against consolidated lists.

The real lists (UN, UAE Local Terrorist List, OFAC) are built by
scripts/build_sanctions_lists.py; data/sanctions/test_sanctions_list.json is
FICTIONAL test data in the same format. Refresh the real lists at least daily.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

# A candidate must share this share of the query's letter-triples before the (slow)
# full comparison runs. Benchmarked on the full OFAC list: same matches as comparing
# every name, about 100x faster.
_MIN_TRIGRAM_SHARE = 0.3

_STOPWORDS = {"llc", "ltd", "limited", "inc", "co", "company", "corp", "corporation", "sa", "the", "of", "and"}


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    tokens = [t for t in re.findall(r"[a-z0-9]+", text) if t not in _STOPWORDS]
    return " ".join(tokens)


def name_similarity(a: str, b: str) -> float:
    """Max of plain and token-sorted similarity, so word order does not matter."""
    if not a or not b:
        return 0.0
    plain = SequenceMatcher(None, a, b).ratio()
    sorted_a, sorted_b = " ".join(sorted(a.split())), " ".join(sorted(b.split()))
    return max(plain, SequenceMatcher(None, sorted_a, sorted_b).ratio())


def _trigrams(name: str) -> set[str]:
    padded = f" {name} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def _norm_id(number: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", number.upper())


def _dob_consistent(listed: str, dob: str) -> bool:
    """Listed DOBs are "YYYY", "YYYY-MM", "YYYY-MM-DD" or a year range "YYYY..YYYY"."""
    if ".." in listed:
        low, high = listed.split("..", 1)
        return low <= dob[:4] <= high
    return dob.startswith(listed) or listed.startswith(dob)


@dataclass(frozen=True)
class SanctionsEntry:
    name: str
    list_name: str
    program: str = ""
    entity_type: str = "individual"
    country: str = ""
    aliases: tuple[str, ...] = ()
    list_id: str = ""
    # Identifiers that confirm or rule out a name match
    dates_of_birth: tuple[str, ...] = ()
    nationalities: tuple[str, ...] = ()
    id_numbers: tuple[tuple[str, str, str], ...] = ()  # (type, number, issuing country)


@dataclass
class SanctionsMatch:
    name: str
    matched_on: str
    list_name: str
    program: str
    score: float
    status: str = ""  # with identifiers on: "confirmed", "possible" or "ruled_out"
    reason: str = ""


@dataclass
class SanctionsResult:
    match_found: bool
    exact_match: bool
    checked_lists: list[str]
    matches: list[SanctionsMatch] = field(default_factory=list)
    confidence: float = 1.0
    country_risk: str = "UNKNOWN"
    recommendation: str = "clear"


class SanctionsScreener:
    def __init__(
        self,
        entries: list[SanctionsEntry],
        lists: list[str],
        fuzzy_threshold: float,
        high_risk_jurisdictions: list[str],
        source: str = "",
        use_identifiers: bool = False,
        monitored_jurisdictions: tuple[str, ...] = (),
    ) -> None:
        self.source = source  # which list file was loaded, e.g. "official" or "test"
        self._use_identifiers = use_identifiers
        self._lists = list(lists)
        self._threshold = fuzzy_threshold
        self._jurisdictions = {j.strip().lower() for j in high_risk_jurisdictions}
        self._monitored = {j.strip().lower() for j in monitored_jurisdictions}
        self._index = [
            (entry, normalize_name(n))
            for entry in entries
            if entry.list_name in self._lists
            for n in (entry.name, *entry.aliases)
            if normalize_name(n)
        ]
        self._by_trigram: dict[str, list[int]] = defaultdict(list)
        for i, (_, candidate) in enumerate(self._index):
            for gram in _trigrams(candidate):
                self._by_trigram[gram].append(i)

    @classmethod
    def from_json(
        cls,
        path: Path,
        lists: list[str],
        fuzzy_threshold: float,
        high_risk: list[str],
        source: str = "",
        use_identifiers: bool = False,
        monitored: tuple[str, ...] = (),
    ) -> "SanctionsScreener":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # Weak aliases and original-script (e.g. Arabic) names are kept in the file
        # but not screened: weak aliases cause false hits, and matching other
        # scripts needs transliteration.
        entries = [
            SanctionsEntry(
                name=e["name"],
                list_name=e["list"],
                program=e.get("program", ""),
                entity_type=e.get("entity_type", "individual"),
                country=e.get("country", ""),
                aliases=tuple(e.get("aliases", [])),
                list_id=e.get("list_id", ""),
                dates_of_birth=tuple(e.get("dates_of_birth", [])),
                nationalities=tuple(e.get("nationalities", [])),
                id_numbers=tuple(
                    (i.get("type", ""), i.get("number", ""), i.get("country", "")) for i in e.get("id_numbers", [])
                ),
            )
            for e in raw["entries"]
        ]
        return cls(entries, lists, fuzzy_threshold, high_risk, source, use_identifiers, tuple(monitored))

    def _candidates(self, query: str) -> list[tuple[SanctionsEntry, str]]:
        """Index entries sharing enough letter-triples with the query to be worth comparing."""
        grams = _trigrams(query)
        if not grams:
            return []
        shared = Counter(i for gram in grams for i in self._by_trigram.get(gram, ()))
        return [self._index[i] for i, n in shared.items() if n / len(grams) >= _MIN_TRIGRAM_SHARE]

    @staticmethod
    def _check_identifiers(
        entry: SanctionsEntry,
        score: float,
        dob: Optional[str],
        nationality: Optional[str],
        id_number: Optional[str],
    ) -> tuple[str, str]:
        """Is the screened person the listed person? Returns (status, reason)."""
        wanted = _norm_id(id_number or "")
        if wanted and wanted in {_norm_id(n) for _, n, _ in entry.id_numbers}:
            return "confirmed", "ID number matches"
        if dob and entry.dates_of_birth:
            if not any(_dob_consistent(d, dob) for d in entry.dates_of_birth):
                return "ruled_out", "date of birth differs"
            if len(dob) == 10 and dob in entry.dates_of_birth:
                return "confirmed", "name and full date of birth match"
            status, reason = "possible", "birth year fits, full date not confirmed"
        elif entry.entity_type != "individual" and score == 1.0:
            # Companies have no date of birth; an exact company name is strong enough.
            return "confirmed", "exact company name match"
        else:
            status, reason = "possible", "only the name matches; no date of birth to compare"
        # Nationality is shown to the reviewer but never decides alone (dual nationals).
        if nationality and entry.nationalities:
            same = nationality.strip().lower() in {n.lower() for n in entry.nationalities}
            reason += "; nationality matches" if same else "; nationality differs"
        return status, reason

    def screen(
        self,
        entity_name: str,
        country: Optional[str] = None,
        dob: Optional[str] = None,
        nationality: Optional[str] = None,
        id_number: Optional[str] = None,
    ) -> SanctionsResult:
        """dob is "YYYY-MM-DD" (or "YYYY" if only the year is known)."""
        query = normalize_name(entity_name)
        best: dict[tuple[str, str], tuple[SanctionsEntry, SanctionsMatch]] = {}
        best_score = 0.0
        for entry, candidate in self._candidates(query):
            score = 1.0 if query == candidate else name_similarity(query, candidate)
            best_score = max(best_score, score)
            if score < self._threshold:
                continue
            key = (entry.name, entry.list_name)
            if key not in best or score > best[key][1].score:
                match = SanctionsMatch(entry.name, candidate, entry.list_name, entry.program, round(score, 4))
                best[key] = (entry, match)

        ranked = sorted(best.values(), key=lambda pair: -pair[1].score)
        if self._use_identifiers:
            for entry, m in ranked:
                m.status, m.reason = self._check_identifiers(entry, m.score, dob, nationality, id_number)
        matches = [m for _, m in ranked]
        active = [m for m in matches if m.status != "ruled_out"]
        exact = any(m.score == 1.0 for m in matches)
        country_risk = "UNKNOWN"
        if country:
            place = country.strip().lower()
            if place in self._jurisdictions:
                country_risk = "HIGH"
            elif place in self._monitored:
                country_risk = "ELEVATED"
            else:
                country_risk = "STANDARD"
            

        # Without identifiers an exact name blocks; with them, a name alone goes to review.
        confirmed = any(m.status == "confirmed" for m in matches)
        if confirmed or (exact and not self._use_identifiers):
            recommendation = "block_and_escalate"
        elif active or country_risk == "HIGH":
            recommendation = "escalate_for_review"
        else:
            recommendation = "clear"

        # Confidence in the stated outcome: for a match, the match score; for "no
        # match", how far the closest list entry was from being a match. Clearing a name
        # match on date of birth uses a fixed 0.9 until evals (Step 8) calibrate it.
        if active:
            confidence = active[0].score
        elif matches:
            confidence = 0.9
        else:
            confidence = 1.0 - best_score
        return SanctionsResult(
            match_found=bool(matches),
            exact_match=exact,
            checked_lists=list(self._lists),
            matches=matches,
            confidence=round(confidence, 2),
            country_risk=country_risk,
            recommendation=recommendation,
        )
