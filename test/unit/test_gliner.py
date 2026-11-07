"""Tests for the hybrid detector (policy v1.10.0). All names and values are made up.

Run:                 python -m pytest test/unit/test_gliner.py -v
With the real model: set RUN_REAL_GLINER=1 first.
"""
from __future__ import annotations

import pytest

from src.config.policy_loader import load_policy
from src.input_guardrails.pii.base import PIIEntity
from src.input_guardrails.pii.gliner_detector import GlinerPIIDetector, HybridPIIDetector, windows
from src.input_guardrails.pii.presidio_detector import build_pii_detector

POLICY = load_policy("1.10.0")
NAME = "Thandiwe Nkosi"


class FindsTheName:
    """Stands in for the model: finds one known name wherever it is, and notes what it was given."""
    def __init__(self):
        self.pieces = []

    def predict_entities(self, text, labels, threshold=0.5):
        self.pieces.append(text)
        spans, start = [], text.find(NAME)
        while start != -1:
            spans.append({"label": "person", "start": start, "end": start + len(NAME), "score": 0.99})
            start = text.find(NAME, start + 1)
        return spans


def fake_detector() -> tuple[GlinerPIIDetector, FindsTheName]:
    detector = GlinerPIIDetector("not-used", 0.9, {"person": "PERSON"})
    detector._model = FindsTheName()
    return detector, detector._model


# Long text must not leak

def test_a_name_far_into_a_long_report_is_still_found():
    detector, model = fake_detector()
    text = ("Ordinary text about a deposit. " * 200) + f"Wire sent by {NAME}."
    found = detector.detect(text)
    assert len(model.pieces) > 1
    assert [text[e.start:e.end] for e in found] == [NAME]


def test_a_name_across_a_piece_boundary_is_still_found():
    for padding in range(960, 1010):  # puts the name at every position around the first boundary
        detector, _ = fake_detector()
        text = "x" * padding + f" {NAME} " + "y" * 1200
        assert [text[e.start:e.end] for e in detector.detect(text)] == [NAME], padding


def test_a_name_found_in_two_pieces_is_counted_once():
    detector, _ = fake_detector()
    text = "word " * 190 + NAME + " word" * 300
    assert len(detector.detect(text)) == 1


def test_masking_a_long_report_puts_the_token_in_the_right_place():
    detector, _ = fake_detector()
    text = ("Ordinary text. " * 150) + f"Paid by {NAME} today."
    assert detector.mask(text).masked_text.endswith("Paid by [PERSON] today.")


def test_windows_cover_the_whole_text():
    text = "word " * 1000
    pieces = list(windows(text))
    covered = set()
    for start, piece in pieces:
        assert text[start:start + len(piece)] == piece
        covered.update(range(start, start + len(piece)))
    assert covered == set(range(len(text)))


@pytest.mark.parametrize("text", ["", "short", "x" * 5000])
def test_windows_finish_on_any_text(text):
    assert sum(len(p) for _, p in windows(text)) >= len(text)


# Hybrid

def test_hybrid_returns_what_every_detector_found():
    class Finds:
        def __init__(self, entity):
            self.entity = entity

        def detect(self, text):
            return [PIIEntity(self.entity, 0, 1, 1.0)]

    found = HybridPIIDetector([Finds("IBAN_CODE"), Finds("PERSON")]).detect("x")
    assert {e.entity_type for e in found} == {"IBAN_CODE", "PERSON"}


def test_the_policy_builds_a_hybrid_with_rules_for_numbers():
    detector = build_pii_detector(POLICY.input_guardrails, "presidio")
    assert isinstance(detector, HybridPIIDetector)
    assert detector.mask("IBAN GB82 WEST 1234 5698 7654 32 was used.").masked_text == "IBAN [IBAN_CODE] was used."


def test_a_policy_with_gliner_refuses_to_run_on_regex_alone():
    with pytest.raises(RuntimeError, match="Presidio"):
        build_pii_detector(POLICY.input_guardrails, "regex")


def test_the_older_policy_stays_presidio_only():
    assert not isinstance(build_pii_detector(load_policy("1.9.0").input_guardrails, "presidio"), HybridPIIDetector)


# The real model

@pytest.mark.real_gliner
@pytest.mark.parametrize("text, expected", [
    ("The account holder is Thandiwe Nkosi.", "The account holder is [PERSON]."),
    ("The customer lives at 45 Marina Walk, Dubai according to the file.", "The customer lives at [ADDRESS] according to the file."),
    ("IBAN GB82 WEST 1234 5698 7654 32 belongs to Thandiwe Nkosi.", "IBAN [IBAN_CODE] belongs to [PERSON]."),
    ("Paris is the destination of the payment.", "Paris is the destination of the payment."),
])
def test_real_model(text, expected):
    assert build_pii_detector(POLICY.input_guardrails, "presidio").mask(text).masked_text == expected
