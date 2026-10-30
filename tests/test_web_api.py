"""
tests/test_web_api.py
Unit tests for FastAPI endpoints of the SAFEPASS Multi-Camera Dashboard.
"""

import pytest
from fastapi.testclient import TestClient
from web.app import app, tracker_instance, playback_controller


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "SAFEPASS" in res.text
    assert "Multi-Camera Strategic Topology Tracker" in res.text


def test_status_endpoint(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "current_time" in data
    assert "camera_states" in data
    assert len(data["camera_states"]) == 8


def test_playback_control(client):
    res = client.post("/api/playback", json={"playing": True, "speed": 2.0})
    assert res.status_code == 200
    assert res.json()["is_playing"] is True
    assert res.json()["speed"] == 2.0

    # Pause again
    res = client.post("/api/playback", json={"playing": False, "speed": 1.0})
    assert res.status_code == 200
    assert res.json()["is_playing"] is False


def test_seek_control(client):
    res = client.post("/api/seek", json={"time": 42.5})
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_search_endpoint(client):
    body = {
        "upper_color": "black",
        "lower_color": "blue",
        "has_backpack": "Yes",
        "has_bag": "Any",
        "has_hat": "No",
        "has_jacket": "Any",
        "has_longpants": "Yes",
        "starting_cameras": ["cam2", "cam3"],
    }
    res = client.post("/api/search", json=body)
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_offsets_endpoint(client):
    body = {
        "offsets": {
            "cam1": 1.5,
            "cam2": -2.0,
            "cam3": 0.0,
        }
    }
    res = client.post("/api/offsets", json=body)
    assert res.status_code == 200
    assert res.json()["offsets"]["cam1"] == 1.5


def test_reset_endpoint(client):
    res = client.post("/api/reset")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_snapshot_endpoint(client):
    res = client.get("/api/snapshot/cam2")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert len(res.content) > 1000
