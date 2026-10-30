"""
tests/test_topology_tracker.py
Unit tests for the SAFEPASS Multi-Camera Strategic Topology Tracker state machine.
Verified per Owner Ground Truth:
- CAM 1-5: Ground Floor (CAM 3 & 4: Entrance, CAM 1, 2, 5: Sides)
- CAM 6: 1st Floor Staircase (Flat 1A & 1B)
- CAM 7: 2nd Floor Staircase (Flat 2A & 2B)
- CAM 8: Terrace (Terrace Door & Terrace Area)
"""

from collections import deque
from pathlib import Path
import numpy as np
import pytest

from src.topology_tracker import (
    TopologyTracker,
    TargetCriteria,
    ComputeState,
    GlobalTrackerState,
    ExitZone,
)
from src.record import ColorInfo, AttributeInfo, QualityInfo


def test_topology_initialization():
    tracker = TopologyTracker(
        topo_cfg_path="configs/topology.yaml",
        default_cfg_path="configs/default.yaml",
    )
    assert len(tracker.cameras) == 8
    assert "cam1" in tracker.cameras
    assert "cam8" in tracker.cameras

    # Check Ground Floor cameras (CAM 1-5)
    for cid in ["cam1", "cam2", "cam3", "cam4", "cam5"]:
        assert tracker.cameras[cid].floor == 0

    # Check cam6 (1st floor staircase)
    cam6 = tracker.cameras["cam6"]
    assert cam6.floor == 1
    assert "cam3" in cam6.neighbors
    assert "cam4" in cam6.neighbors
    assert "cam7" in cam6.neighbors
    assert any(z.name == "Flat 1A" for z in cam6.exit_zones)
    assert any(z.name == "Flat 1B" for z in cam6.exit_zones)

    # Check cam7 (2nd floor staircase)
    cam7 = tracker.cameras["cam7"]
    assert cam7.floor == 2
    assert "cam6" in cam7.neighbors
    assert "cam8" in cam7.neighbors
    assert any(z.name == "Flat 2A (Left)" for z in cam7.exit_zones)
    assert any(z.name == "Flat 2B (Right)" for z in cam7.exit_zones)

    # Check cam8 (Terrace)
    cam8 = tracker.cameras["cam8"]
    assert cam8.floor == 3
    assert cam8.fov_angle == 180.0
    assert "cam7" in cam8.neighbors
    assert "cam5" not in cam8.neighbors  # No phantom leap to 1F
    assert any(z.name == "Terrace Door" for z in cam8.exit_zones)
    assert any(z.name == "Terrace Area" for z in cam8.exit_zones)


def test_edges_symmetric():
    """Verify that every topological edge is 100% symmetric with identical transit bounds."""
    tracker = TopologyTracker(
        topo_cfg_path="configs/topology.yaml",
        default_cfg_path="configs/default.yaml",
    )
    for cid, node in tracker.cameras.items():
        for neighbor in node.neighbors:
            rev_node = tracker.cameras[neighbor]
            assert cid in rev_node.neighbors, f"Asymmetry: {cid} -> {neighbor} exists, but {neighbor} does not have {cid}"
            fwd_b = node.transit_bounds.get(neighbor)
            rev_b = rev_node.transit_bounds.get(cid)
            assert fwd_b == rev_b, f"Bound mismatch: {cid}->{neighbor} {fwd_b} != {neighbor}->{cid} {rev_b}"


def test_each_edge_has_handoff_zone():
    """Verify that every directed edge has an explicit exit zone on its source camera."""
    tracker = TopologyTracker(
        topo_cfg_path="configs/topology.yaml",
        default_cfg_path="configs/default.yaml",
    )
    for cid, node in tracker.cameras.items():
        handoff_targets = {z.handoff_cam for z in node.exit_zones if not z.is_terminal}
        for neighbor in node.neighbors:
            assert neighbor in handoff_targets, f"Camera {cid} missing exit zone handoff to neighbor {neighbor}"


def test_video_file_exists_per_cam():
    """Verify that the designated video file exists for every camera."""
    tracker = TopologyTracker(
        topo_cfg_path="configs/topology.yaml",
        default_cfg_path="configs/default.yaml",
    )
    for cid, node in tracker.cameras.items():
        p = Path(node.video_path)
        assert p.exists(), f"Video file {node.video_path} for {cid} does not exist"


def test_floor_per_cam_matches_floorplan():
    """Verify floor assignments per camera match architectural floorplans."""
    tracker = TopologyTracker(
        topo_cfg_path="configs/topology.yaml",
        default_cfg_path="configs/default.yaml",
    )
    expected_floors = {
        "cam1": 0,
        "cam2": 0,
        "cam3": 0,
        "cam4": 0,
        "cam5": 0,
        "cam6": 1,
        "cam7": 2,
        "cam8": 3,
    }
    for cid, exp_floor in expected_floors.items():
        assert tracker.cameras[cid].floor == exp_floor, f"{cid} floor is {tracker.cameras[cid].floor}, expected {exp_floor}"


def test_each_terminal_reachable_from_entrance():
    """Verify BFS traversal from entrance cameras (cam3, cam4) reaches all terminal destinations."""
    tracker = TopologyTracker(
        topo_cfg_path="configs/topology.yaml",
        default_cfg_path="configs/default.yaml",
    )
    # Collect all cameras that have terminal exit zones
    cameras_with_terminals = {
        cid for cid, node in tracker.cameras.items() if any(z.is_terminal for z in node.exit_zones)
    }

    # BFS from entrance cameras
    visited = set()
    queue = deque(["cam3", "cam4"])
    while queue:
        curr = queue.popleft()
        if curr in visited:
            continue
        visited.add(curr)
        for n in tracker.cameras[curr].neighbors:
            if n not in visited:
                queue.append(n)

    for term_cam in cameras_with_terminals:
        assert term_cam in visited, f"Terminal camera {term_cam} is not reachable from entrance cameras"


def test_non_topological_jumps_rejected():
    """Verify that invalid non-topological jumps are rejected by is_valid_transition."""
    tracker = TopologyTracker(
        topo_cfg_path="configs/topology.yaml",
        default_cfg_path="configs/default.yaml",
    )
    invalid_transitions = [
        ("cam5", "cam2"),  # North Walkway directly to SE Driveway
        ("cam8", "cam5"),  # Terrace directly to Ground / 1F
        ("cam1", "cam7"),  # East Walkway directly to 2F
        ("cam6", "cam1"),  # 1F Staircase directly to East Walkway
        ("cam3", "cam8"),  # Ground Entrance directly to Terrace
    ]
    for src, dst in invalid_transitions:
        assert not tracker.is_valid_transition(src, dst), f"Transition {src} -> {dst} should be rejected"


def test_camera_eof_marks_offline():
    """Verify that when a camera exceeds its duration, it transitions to OFFLINE without crashing."""
    tracker = TopologyTracker(
        topo_cfg_path="configs/topology.yaml",
        default_cfg_path="configs/default.yaml",
    )
    cam8 = tracker.cameras["cam8"]
    # cam8 duration is 631.88s; query at global_t = 700s
    ret, frame = cam8.read_frame_at_time(global_t=700.0)
    assert not ret
    assert frame is not None
    assert cam8.compute_state == ComputeState.OFFLINE


def test_exit_zone_intersection():
    zone = ExitZone(
        name="Flat 1A",
        box=(0.1, 0.2, 0.4, 0.6),
        is_terminal=True,
    )
    assert zone.contains_point(0.2, 0.3)
    assert not zone.contains_point(0.05, 0.3)
    assert zone.intersects((0.15, 0.25, 0.35, 0.55))


def test_state_machine_flow_simulated():
    tracker = TopologyTracker(
        topo_cfg_path="configs/topology.yaml",
        default_cfg_path="configs/default.yaml",
    )
    assert tracker.global_state == GlobalTrackerState.UNINITIALIZED

    # 1. Start Acquisition on entrance cameras
    criteria = TargetCriteria(
        upper_color="black",
        lower_color="blue",
        has_backpack=True,
        starting_cameras=["cam2", "cam3"],
    )
    tracker.start_acquisition(criteria)
    assert tracker.global_state == GlobalTrackerState.ACQUIRING
    assert tracker.cameras["cam2"].compute_state == ComputeState.SEARCHING_NEIGHBOR
    assert tracker.cameras["cam3"].compute_state == ComputeState.SEARCHING_NEIGHBOR
    assert tracker.cameras["cam1"].compute_state == ComputeState.IDLE

    # 2. Lock Target on CAM 2
    dummy_crop = np.zeros((128, 64, 3), dtype=np.uint8)
    tracker._embedder = type("MockEmbedder", (), {"extract": lambda self, c: np.ones(512, dtype=np.float32) / np.sqrt(512)})()

    cols = ColorInfo(upper="black", lower="blue", upper_conf=0.9, lower_conf=0.85)
    attrs = AttributeInfo(has_backpack=0.92, has_longpants=0.95)
    qual = QualityInfo(overall=0.88)

    tracker.lock_target(
        camera_id="cam2",
        track_id=1,
        bbox=(100, 100, 200, 300),
        crop_bgr=dummy_crop,
        colors=cols,
        attrs=attrs,
        quality=qual,
    )
    assert tracker.global_state == GlobalTrackerState.TRACKING
    assert tracker.active_camera_id == "cam2"
    assert tracker.cameras["cam2"].compute_state == ComputeState.ACTIVE
    assert tracker.cameras["cam1"].compute_state == ComputeState.IDLE
    assert tracker.cameras["cam3"].compute_state == ComputeState.IDLE

    # 3. Simulate target loss -> Transition to Neighbor Search
    tracker._initiate_transition("cam2", global_t=10.0)
    assert tracker.global_state == GlobalTrackerState.TRANSITIONING
    # cam2 neighbors: cam1, cam3
    assert tracker.cameras["cam1"].compute_state == ComputeState.SEARCHING_NEIGHBOR
    assert tracker.cameras["cam3"].compute_state == ComputeState.SEARCHING_NEIGHBOR
    assert tracker.cameras["cam2"].compute_state == ComputeState.LOW_RATE_CHECK
    assert tracker.cameras["cam4"].compute_state == ComputeState.IDLE

    # 4. Re-acquire on CAM 3
    tracker._reacquire_target(
        from_cam="cam2",
        to_cam="cam3",
        track_id=2,
        bbox=(150, 150, 250, 350),
        crop_bgr=dummy_crop,
        colors=cols,
        attrs=attrs,
        quality=qual,
        fusion_score=0.82,
        global_t=15.0,
    )
    assert tracker.global_state == GlobalTrackerState.TRACKING
    assert tracker.active_camera_id == "cam3"
    assert tracker.cameras["cam3"].compute_state == ComputeState.ACTIVE
    assert tracker.cameras["cam1"].compute_state == ComputeState.IDLE
    assert tracker.cameras["cam2"].compute_state == ComputeState.IDLE
    assert tracker.camera_path == ["cam2", "cam3"]

    # 5. Resolve Final Destination
    tracker._resolve_final_destination("cam3", global_t=40.0)
    assert tracker.global_state == GlobalTrackerState.RESOLVED
    assert tracker.final_verdict is not None
    assert tracker.final_verdict.last_camera == "cam3"
    assert "cam2" in tracker.final_verdict.camera_path
    assert "cam3" in tracker.final_verdict.camera_path
