"""Most tests must not load the 1.1 GB GLiNER model: they get a stand-in that finds nothing.
Tests marked real_gliner use the real model, and only run with RUN_REAL_GLINER=1."""
from __future__ import annotations

import os

import pytest


class NoNamesModel:
    def predict_entities(self, text, labels, threshold=0.5):
        return []


def pytest_configure(config):
    config.addinivalue_line("markers", "real_gliner: needs the real GLiNER model (set RUN_REAL_GLINER=1)")


@pytest.fixture(autouse=True)
def no_real_gliner(request, monkeypatch):
    if request.node.get_closest_marker("real_gliner"):
        if not os.environ.get("RUN_REAL_GLINER"):
            pytest.skip("set RUN_REAL_GLINER=1 to use the real GLiNER model")
        return
    monkeypatch.setattr("src.input_guardrails.pii.gliner_detector._load_model", lambda model_id: NoNamesModel())
