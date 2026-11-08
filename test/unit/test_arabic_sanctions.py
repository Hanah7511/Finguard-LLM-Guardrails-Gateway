"""Screening names written in Arabic script (policy v1.11.0). Every name is made up.
Arabic is written as \\u escapes so copy-paste cannot reverse it.

Run: python -m pytest test/unit/test_arabic_sanctions.py -v
"""
from __future__ import annotations

import pytest

from src.config.policy_loader import load_policy
from src.config.settings import get_settings
from src.core.arabic_names import consonant_key, has_arabic, normalize_arabic
from src.core.sanctions_screener import SanctionsScreener, normalize_name

KHALID_TESTANI = "خالد تستاني"  # the list's own Arabic name
AHMED_FAKERI = "أحمد فكري"  # the list has this one in English letters only
KHALID_TESFAYE = "خالد تسفاي"  # innocent, close in letters
MOHAMMED = "محمد"


def screener(version: str) -> SanctionsScreener:
    policy = load_policy(version).sanctions_screening
    return SanctionsScreener.from_policy(get_settings().sanctions_test_list_path, policy, "test")


NEW, OLD = screener("1.11.0"), screener("1.10.0")


# Normalising and keys

def test_letter_forms_are_unified_and_vowel_marks_removed():
    with_marks = "أَحْمَد"  # Ahmad with vowel marks
    assert normalize_arabic(with_marks) == "احمد"
    assert normalize_arabic("تستانى") == normalize_arabic("تستاني")


@pytest.mark.parametrize("spelling", ["Mohammed", "Muhammad", "Mohamad", "Mouhamed"])
def test_english_spellings_and_arabic_share_one_key(spelling):
    assert consonant_key(normalize_name(spelling)) == consonant_key(normalize_arabic(MOHAMMED)) == "mhmd"


def test_al_is_dropped_on_both_sides():
    assert consonant_key("mohammed al exampleh") == consonant_key(normalize_arabic(
        "محمد الاكسامبله"))


def test_has_arabic():
    assert has_arabic(AHMED_FAKERI) and not has_arabic("Ahmed Fakeri")


# Screening

def test_the_lists_own_arabic_name_is_an_exact_match():
    r = NEW.screen(KHALID_TESTANI)
    assert r.exact_match and r.matches[0].matched_by == "arabic_script"
    assert r.recommendation != "clear"


def test_an_arabic_name_matches_a_name_listed_only_in_english_letters():
    r = NEW.screen(AHMED_FAKERI)
    assert [m.name for m in r.matches] == ["Ahmed Fakeri"]
    assert r.matches[0].matched_by == "consonant_key"


def test_a_key_match_goes_to_a_person_and_never_blocks_on_the_name_alone():
    r = NEW.screen(AHMED_FAKERI)
    assert not r.exact_match and r.matches[0].score < 1.0
    assert r.recommendation == "escalate_for_review"


def test_a_key_match_plus_a_matching_id_number_blocks():
    viktor = "فيكتور سامبليوف"
    assert NEW.screen(viktor, id_number="TP0000001").recommendation == "block_and_escalate"


def test_a_key_match_is_cleared_by_a_different_date_of_birth():
    r = NEW.screen(AHMED_FAKERI, dob="1990-01-01")
    assert r.matches[0].status == "ruled_out" and r.recommendation == "clear"


def test_an_innocent_arabic_name_one_letter_apart_stays_clear():
    assert NEW.screen(KHALID_TESFAYE).recommendation == "clear"


def test_names_in_english_letters_are_screened_as_before():
    for name in ("Khalid Testani", "Ahmed Fakeri", "Clara Openfield"):
        assert NEW.screen(name).recommendation == OLD.screen(name).recommendation


def test_the_older_policy_still_ignores_arabic_script():
    assert OLD.screen(KHALID_TESTANI).recommendation == "clear"
    assert OLD.screen(AHMED_FAKERI).matches == []
