"""
tests/test_demo_profiles.py
===========================
Unit tests verifying the 3 sequential person re-ID demo profiles,
camera handoff sequence, destination resolution, and dossier report generation.
"""

import pytest
from fastapi.testclient import TestClient
from web.app import app, playback_controller, tracker_instance, DEMO_PROFILES


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_demo_profile_1_blue_black(client):
    """Profile 1: Blue upper, Black bottom -> CAM 1 -> CAM 2 -> CAM 3 -> CAM 4 -> CAM 5 -> CAM 2 (SE Gate)"""
    client.post("/api/reset")

    res = client.post("/api/search", json={
        "upper_color": "blue",
        "lower_color": "black",
        "starting_cameras": ["cam1"],
    })
    assert res.status_code == 200
    assert res.json().get("demo_mode") is True
    assert playback_controller.demo_active is True
    assert playback_controller.demo_key == "profile_1"
    assert tracker_instance.active_camera_id == "cam1"
    assert playback_controller.current_time < 2.0

    # Advance through all segments
    for i in range(len(DEMO_PROFILES["profile_1"]["segments"]) - 1):
        adv_res = client.post("/api/demo/next_segment")
        assert adv_res.status_code == 200
        assert adv_res.json()["advanced"] is True

    # Advance last segment -> should finalize
    final_res = client.post("/api/demo/next_segment")
    assert final_res.status_code == 200
    assert tracker_instance.global_state.value == "RESOLVED"
    assert tracker_instance.final_verdict is not None
    assert tracker_instance.final_verdict.last_camera == "cam2"
    assert "Main Gate Exit" in tracker_instance.final_verdict.inferred_location

    # Verify status payload report
    st = client.get("/api/status").json()
    assert st["report"] is not None
    assert st["report"]["upper_color"] == "Blue"
    assert st["report"]["lower_color"] == "Black"
    assert len(st["report"]["sighting_rows"]) == 6
    assert st["report"]["final_camera"] == "CAM2"


def test_demo_profile_2_white_blue(client):
    """Profile 2: White upper, Blue bottom -> CAM 2 -> CAM 3 -> CAM 6 -> CAM 7 -> CAM 8 (Terrace)"""
    client.post("/api/reset")

    res = client.post("/api/search", json={
        "upper_color": "white",
        "lower_color": "blue",
    })
    assert res.status_code == 200
    assert res.json().get("demo_mode") is True
    assert playback_controller.demo_key == "profile_2"
    assert tracker_instance.active_camera_id == "cam2"
    assert playback_controller.current_time == pytest.approx(1.0, abs=2.0)

    # Advance to end
    while playback_controller.demo_idx < len(DEMO_PROFILES["profile_2"]["segments"]) - 1:
        client.post("/api/demo/next_segment")
    client.post("/api/demo/next_segment")

    assert tracker_instance.global_state.value == "RESOLVED"
    assert tracker_instance.final_verdict.last_camera == "cam8"
    assert "Terrace" in tracker_instance.final_verdict.inferred_location


def test_demo_profile_3_black_black(client):
    """Profile 3: Black upper, Black bottom -> CAM 3 -> CAM 6 -> CAM 7 -> CAM 8 (Terrace)"""
    client.post("/api/reset")

    res = client.post("/api/search", json={
        "upper_color": "black",
        "lower_color": "black",
    })
    assert res.status_code == 200
    assert res.json().get("demo_mode") is True
    assert playback_controller.demo_key == "profile_3"
    assert tracker_instance.active_camera_id == "cam3"
    assert playback_controller.current_time == pytest.approx(570.0, abs=2.0)

    # Advance to end
    while playback_controller.demo_idx < len(DEMO_PROFILES["profile_3"]["segments"]) - 1:
        client.post("/api/demo/next_segment")
    client.post("/api/demo/next_segment")

    assert tracker_instance.global_state.value == "RESOLVED"
    assert tracker_instance.final_verdict.last_camera == "cam8"
    assert "Terrace" in tracker_instance.final_verdict.inferred_location
