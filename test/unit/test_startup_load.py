"""The model loads when a door starts, once, and never twice at the same moment. No real model needed.

Run: python -m pytest test/unit/test_startup_load.py -v
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import main as api
from src.input_guardrails.pii import gliner_detector as g
from src.input_guardrails.pii.base import PIIDetector
from src.mcp_server import server as mcp_door


class NoModel(PIIDetector):
    def detect(self, text):
        return []


class CountsLoads:
    def __init__(self):
        self.loads = 0

    def load(self):
        self.loads += 1


def test_many_requests_at_once_load_the_model_once(monkeypatch):
    loaded = []

    def slow_load(model_id):
        loaded.append(model_id)
        time.sleep(0.2)
        return SimpleNamespace(predict_entities=lambda *a, **k: [])

    monkeypatch.setattr(g, "_load_model", slow_load)
    detector = g.GlinerPIIDetector("model", 0.5, {"person": "PERSON"})
    threads = [threading.Thread(target=detector.detect, args=("some text",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(loaded) == 1


def test_hybrid_loads_every_detector_and_a_detector_with_no_model_does_nothing():
    inner = CountsLoads()
    hybrid = g.HybridPIIDetector([NoModel(), inner])
    hybrid.load()
    assert inner.loads == 1


def test_the_rest_door_loads_the_model_as_it_starts(monkeypatch):
    detector = CountsLoads()
    orch = SimpleNamespace(require_app_key_is_set=lambda: None, require_analyst_key_is_set=lambda: None, load_models=detector.load)
    monkeypatch.setattr(api, "get_orchestrator", lambda: orch)
    with TestClient(api.app):
        assert detector.loads == 1


def run_mcp_main(monkeypatch, transport):
    detector = CountsLoads()
    runs = []
    monkeypatch.setattr(mcp_door, "get_settings", lambda: SimpleNamespace(mcp_transport=transport, mcp_host="h", mcp_port=1))
    monkeypatch.setattr(mcp_door, "get_orchestrator", lambda: SimpleNamespace(load_models=detector.load))
    monkeypatch.setattr(mcp_door, "create_server", lambda orch: SimpleNamespace(run=lambda *a, **k: runs.append(detector.loads)))
    mcp_door.main()
    return runs


def test_the_mcp_door_in_docker_has_the_model_before_it_opens(monkeypatch):
    assert run_mcp_main(monkeypatch, "streamable-http") == [1]


def test_the_mcp_door_started_by_a_client_stays_quick(monkeypatch):
    assert run_mcp_main(monkeypatch, "stdio") == [0]
