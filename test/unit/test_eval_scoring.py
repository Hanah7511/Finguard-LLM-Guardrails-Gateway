"""Checks that the eval maths is right, so a score means what it says."""
from __future__ import annotations

import pytest

from evals.scoring import NameScore, name_is_hidden, percent, score_attacks

EXAMPLES = [{"label": "attack"}, {"label": "attack"}, {"label": "normal"}, {"label": "normal"}, {"label": "normal"}]


def test_catch_rate_counts_only_attacks():
    s = score_attacks(EXAMPLES, [True, False, False, False, False])
    assert (s.attacks_caught, s.attacks, s.catch_rate) == (1, 2, 0.5)


def test_false_block_rate_counts_only_normal_text():
    s = score_attacks(EXAMPLES, [False, False, True, False, False])
    assert (s.normal_blocked, s.normal) == (1, 3)
    assert s.false_block_rate == pytest.approx(1 / 3)


def test_blocking_everything_catches_all_but_shows_in_the_false_block_rate():
    s = score_attacks(EXAMPLES, [True] * 5)
    assert (s.catch_rate, s.false_block_rate) == (1.0, 1.0)


def test_blocking_nothing_catches_nothing():
    s = score_attacks(EXAMPLES, [False] * 5)
    assert (s.catch_rate, s.false_block_rate) == (0.0, 0.0)


def test_a_set_with_no_attacks_has_no_catch_rate_rather_than_a_fake_one():
    s = score_attacks([{"label": "normal"}], [False])
    assert s.catch_rate is None
    assert s.as_row()["catch_rate"] == "n/a"


def test_counts_are_shown_with_percentages():
    row = score_attacks(EXAMPLES, [True, False, False, False, False]).as_row()
    assert row["attacks_caught"] == "1 of 2"
    assert row["catch_rate"] == "50%"


def test_decisions_must_match_examples():
    with pytest.raises(ValueError):
        score_attacks(EXAMPLES, [True])


def test_percent_rounds_for_display():
    assert percent(2 / 3) == "67%"


def test_a_fully_masked_name_counts_as_hidden():
    assert name_is_hidden("Clara Openfield", "Wire sent by [PERSON] on 12 May.")


def test_one_word_left_showing_is_a_miss():
    assert not name_is_hidden("Clara Openfield", "Wire sent by [PERSON] Openfield on 12 May.")


def test_an_unmasked_name_is_a_miss_whatever_the_capitals():
    assert not name_is_hidden("Clara Openfield", "wire sent by CLARA OPENFIELD")


def test_name_recall_shows_counts_and_percent():
    assert NameScore(25, 23).as_row() == {"names_hidden": "23 of 25", "recall": "92%"}


def test_a_short_name_inside_a_longer_word_does_not_count_as_showing():
    assert name_is_hidden("Le Thi Hoa", "Please review the alert for [PERSON].")


def test_a_short_name_on_its_own_does_count_as_showing():
    assert not name_is_hidden("Le Thi Hoa", "Please review the alert for [PERSON] Le.")


def test_a_hyphenated_name_is_matched_whole():
    assert not name_is_hidden("Omar Al-Fictari", "Paid by [PERSON] Al-Fictari.")
