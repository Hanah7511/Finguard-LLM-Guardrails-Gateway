
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


@dataclass(frozen=True)
class SanctionsEntry:
    name: str
    list_name: str
    program: str = ""
    entity_type: str = "individual"
    country: str = ""
    aliases: tuple[str, ...] = ()
    list_id: str = ""
    # Identifiers that confirm or rule out a name match (used by the screener in 2C)
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
    ) -> None:
        self.source = source  # which list file was loaded, e.g. "official" or "test"
        self._lists = list(lists)
        self._threshold = fuzzy_threshold
        self._jurisdictions = {j.strip().lower() for j in high_risk_jurisdictions}
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
        cls, path: Path, lists: list[str], fuzzy_threshold: float, high_risk: list[str], source: str = ""
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
        return cls(entries, lists, fuzzy_threshold, high_risk, source)

    def _candidates(self, query: str) -> list[tuple[SanctionsEntry, str]]:
        """Index entries sharing enough letter-triples with the query to be worth comparing."""
        grams = _trigrams(query)
        if not grams:
            return []
        shared = Counter(i for gram in grams for i in self._by_trigram.get(gram, ()))
        return [self._index[i] for i, n in shared.items() if n / len(grams) >= _MIN_TRIGRAM_SHARE]

    def screen(self, entity_name: str, country: Optional[str] = None) -> SanctionsResult:
        query = normalize_name(entity_name)
        best: dict[tuple[str, str], SanctionsMatch] = {}
        best_score = 0.0
        for entry, candidate in self._candidates(query):
            score = 1.0 if query == candidate else name_similarity(query, candidate)
            best_score = max(best_score, score)
            if score < self._threshold:
                continue
            key = (entry.name, entry.list_name)
            if key not in best or score > best[key].score:
                best[key] = SanctionsMatch(entry.name, candidate, entry.list_name, entry.program, round(score, 4))

        matches = sorted(best.values(), key=lambda m: -m.score)
        exact = any(m.score == 1.0 for m in matches)
        country_risk = "UNKNOWN"
        if country:
            country_risk = "HIGH" if country.strip().lower() in self._jurisdictions else "STANDARD"

        if exact:
            recommendation = "block_and_escalate"
        elif matches or country_risk == "HIGH":
            recommendation = "escalate_for_review"
        else:
            recommendation = "clear"

        # Confidence in the stated outcome: for a match, the match score; for "no
        # match", how far the closest list entry was from being a match.
        confidence = matches[0].score if matches else 1.0 - best_score
        return SanctionsResult(
            match_found=bool(matches),
            exact_match=exact,
            checked_lists=list(self._lists),
            matches=matches,
            confidence=round(confidence, 2),
            country_risk=country_risk,
            recommendation=recommendation,
        )
