"""
src/topology_tracker.py
=======================
SAFEPASS Strategic Topology-Constrained Multi-Camera Person Tracker.

Implements the multi-camera state-machine scheduler for Mayfair Apartment:
1. ACTIVE MODE (1 Camera): Run full CV inference ONLY on active camera cam_i.
   All other 7 cameras decode frames passively for UI display with 0% ML load.
2. HAND-OFF / TRANSITION MODE (Adjacent Cameras Only): When target exits FOV
   or is lost in cam_i for > 1.5 seconds, activate inference ONLY on connected
   neighbor cameras N(cam_i) (+ low-rate check on cam_i).
3. RE-ACQUISITION: When candidate in neighbor camera cam_j matches target
   OSNet embedding bank + attribute/color profile via fusion.py (adaptive_rule),
   transition state to ACTIVE on cam_j, log (cam_i -> cam_j, timestamp),
   and return all other cameras to IDLE.
4. FINAL DESTINATION RESOLVER: Determines last camera seen, full path, and
   inferred final location (Flat door, Terrace, Main Gate exit, etc.).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml

# Existing SAFEPASS modules
from src.attributes import AttributeInference
from src.color import extract_colors
from src.detect_track import DetectionTracker
from src.embed import EmbeddingExtractor
from src.fusion import fuse
from src.quality import estimate_quality
from src.record import AttributeInfo, ColorInfo, FeatureRecord, QualityInfo

logger = logging.getLogger(__name__)


# ─── Enums & Dataclasses ───────────────────────────────────────────────────────

class ComputeState(str, Enum):
    IDLE = "IDLE - 0% GPU"
    ACTIVE = "ACTIVE - TRACKING"
    SEARCHING_NEIGHBOR = "SEARCHING NEIGHBOR"
    LOW_RATE_CHECK = "LOW-RATE CHECK"
    OFFLINE = "OFFLINE - STREAM END"


class GlobalTrackerState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    TRANSITIONING = "TRANSITIONING"
    RESOLVED = "RESOLVED"


@dataclass
class TargetCriteria:
    """User description query for cold start without reference crop."""
    upper_color: str = "any"
    lower_color: str = "any"
    has_backpack: Optional[bool] = None
    has_bag: Optional[bool] = None
    has_hat: Optional[bool] = None
    has_jacket: Optional[bool] = None
    has_longpants: Optional[bool] = None
    starting_cameras: List[str] = field(default_factory=lambda: ["cam2", "cam3"])


@dataclass
class ExitZone:
    """Normalized bounding box representing a room door, stairs, or exit gate."""
    name: str
    box: Tuple[float, float, float, float]  # [x1, y1, x2, y2] in [0, 1]
    is_terminal: bool = False
    terminal_name: Optional[str] = None
    handoff_cam: Optional[str] = None

    def contains_point(self, nx: float, ny: float) -> bool:
        x1, y1, x2, y2 = self.box
        return (x1 <= nx <= x2) and (y1 <= ny <= y2)

    def intersects(self, bbox_norm: Tuple[float, float, float, float]) -> bool:
        bx1, by1, bx2, by2 = bbox_norm
        zx1, zy1, zx2, zy2 = self.box
        ix1, iy1 = max(bx1, zx1), max(by1, zy1)
        ix2, iy2 = min(bx2, zx2), min(by2, zy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return inter > 0.05 * max(1e-4, (bx2 - bx1) * (by2 - by1))


@dataclass
class TransitionEvent:
    """Audit log record of target moving between cameras."""
    from_cam: str
    to_cam: str
    timestamp: float
    score: float
    reason: str
    crop_snapshot: Optional[np.ndarray] = None


@dataclass
class FinalVerdict:
    """Comprehensive final destination resolution."""
    last_camera: str
    last_camera_name: str
    global_timestamp: float
    camera_path: List[str]
    inferred_location: str
    terminal_zone: Optional[str] = None
    confidence: float = 0.95
    status: str = "RESOLVED"


# ─── Camera Node ──────────────────────────────────────────────────────────────

class CameraNode:
    """State and video capture holder for a single camera stream."""

    def __init__(self, cam_id: str, cfg: dict):
        self.cam_id = cam_id
        self.name: str = cfg.get("name", cam_id)
        self.floor: int = int(cfg.get("floor", 0))
        self.floor_name: str = cfg.get("floor_name", f"Floor {self.floor}")
        self.video_path: str = cfg.get("video_path", "")
        self.time_offset_sec: float = float(cfg.get("time_offset_sec", 0.0))
        self.map_coords: Dict[str, float] = cfg.get("map_coords", {"x": 0.5, "y": 0.5})
        self.fov_angle: float = float(cfg.get("fov_angle", 0.0))
        self.neighbors: List[str] = list(cfg.get("neighbors", []))
        self.transit_bounds: Dict[str, List[float]] = cfg.get("transit_bounds_sec", {})

        self.exit_zones: List[ExitZone] = [
            ExitZone(
                name=z.get("name", "Zone"),
                box=tuple(z.get("box", [0, 0, 1, 1])),
                is_terminal=bool(z.get("is_terminal", False)),
                terminal_name=z.get("terminal_name"),
                handoff_cam=z.get("handoff_cam"),
            )
            for z in cfg.get("exit_zones", [])
        ]

        # Runtime compute state
        self.compute_state: ComputeState = ComputeState.IDLE
        self.lock = threading.Lock()
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 15.0
        self.duration_sec: float = float(cfg.get("duration_sec", 900.0))
        self.width: int = 960
        self.height: int = 1080
        self.current_frame_idx: int = -1
        self.cached_frame: Optional[np.ndarray] = None
        self.cached_jpeg: Optional[bytes] = None
        self.cached_focused_jpeg: Optional[bytes] = None
        self._init_capture()

    def _init_capture(self) -> None:
        p = Path(self.video_path)
        if not p.exists():
            # Try alternate naming (e.g. CAM 1.mp4 vs cam1.mp4)
            alt = Path("data/videos") / f"CAM {self.cam_id.replace('cam', '')}.mp4"
            if alt.exists():
                self.video_path = str(alt)
                p = alt

        if p.exists():
            with self.lock:
                self.cap = cv2.VideoCapture(str(p))
                if self.cap.isOpened():
                    self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps_val = self.cap.get(cv2.CAP_PROP_FPS)
                    if fps_val > 0:
                        self.fps = fps_val
                    w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if w > 0 and h > 0:
                        self.width, self.height = w, h
                    meas_dur = self.total_frames / self.fps if self.fps > 0 else 900.0
                    if getattr(self, "duration_sec", 0.0) <= 0:
                        self.duration_sec = meas_dur
                    # Read frame 0 immediately so initial cache is never blank
                    ret0, frame0 = self.cap.read()
                    if ret0 and frame0 is not None:
                        self.cached_frame = frame0
                        self.current_frame_idx = 0
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.info(f"Loaded {self.cam_id}: {self.video_path} ({self.width}x{self.height}, {self.fps} fps)")
        else:
            logger.warning(f"Video file not found for {self.cam_id}: {self.video_path}")

    def remap_video(self, new_path: str) -> bool:
        """Remap camera to a new video file at runtime."""
        p = Path(new_path)
        if not p.exists():
            return False
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            self.video_path = str(p)
            self._init_capture()
            return self.cap is not None and self.cap.isOpened()

    def read_frame_at_time(self, global_t: float) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame corresponding to global_t with camera's time offset."""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                # Generate blank placeholder
                blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(blank, f"{self.name} [NO FEED]", (60, self.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
                return True, blank

            local_t = global_t - self.time_offset_sec
            if local_t < 0:
                local_t = 0.0

            # Check if video has ended
            if local_t >= self.duration_sec or (self.total_frames > 0 and int(local_t * self.fps) >= self.total_frames):
                self.compute_state = ComputeState.OFFLINE
                offline = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(offline, f"{self.name} [OFFLINE - STREAM END]", (40, self.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
                self.cached_frame = offline
                return False, offline

            target_frame = int(local_t * self.fps)
            if target_frame >= self.total_frames:
                target_frame = max(0, self.total_frames - 1)

            # Seek if not consecutive
            if target_frame != self.current_frame_idx + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret, frame = self.cap.read()
            if ret:
                self.current_frame_idx = target_frame
                self.cached_frame = frame
                return True, frame
            elif self.cached_frame is not None:
                return True, self.cached_frame
            return False, None


# ─── Topology Tracker State Machine ──────────────────────────────────────────

class TopologyTracker:
    """
    Main State Machine Controller for Multi-Camera Strategic Person Tracking.
    """

    def __init__(self, topo_cfg_path: str = "configs/topology.yaml", default_cfg_path: str = "configs/default.yaml"):
        self.topo_cfg_path = topo_cfg_path
        self.default_cfg_path = default_cfg_path

        with open(topo_cfg_path, "r", encoding="utf-8") as f:
            self.topo_cfg = yaml.safe_load(f)
        with open(default_cfg_path, "r", encoding="utf-8") as f:
            self.default_cfg = yaml.safe_load(f)

        gt = self.topo_cfg.get("global_timeline", {})
        self.duration_sec: float = float(gt.get("duration_sec", 900.0))
        self.fps: float = float(gt.get("fps", 15.0))
        self.lost_timeout_sec: float = float(gt.get("lost_timeout_sec", 1.5))
        self.neighbor_search_timeout_sec: float = float(gt.get("neighbor_search_timeout_sec", 30.0))
        self.low_rate_check_hz: float = float(gt.get("low_rate_check_hz", 1.0))
        self.acquisition_threshold: float = float(gt.get("acquisition_threshold", 0.65))
        self.reacquisition_fusion_threshold: float = float(gt.get("reacquisition_fusion_threshold", 0.55))

        # Build Camera Nodes
        self.cameras: Dict[str, CameraNode] = {}
        for cam_id, c_cfg in self.topo_cfg.get("cameras", {}).items():
            self.cameras[cam_id] = CameraNode(cam_id, c_cfg)

        # Global runtime state
        self.global_state: GlobalTrackerState = GlobalTrackerState.UNINITIALIZED
        self.current_global_time: float = 0.0
        self.active_camera_id: Optional[str] = "cam1"
        self.focused_camera_id: str = "cam1"
        self.target_criteria: Optional[TargetCriteria] = None

        # Locked Target Bank & Profile
        self.locked_target_id: Optional[str] = None
        self.target_embedding_bank: List[np.ndarray] = []
        self.target_reference_record: Optional[FeatureRecord] = None
        self.target_avatar_crop: Optional[np.ndarray] = None
        self.camera_path: List[str] = []
        self.case_timeline: List[TransitionEvent] = []

        # Tracking state
        self.last_seen_time: float = 0.0
        self.last_seen_cam: Optional[str] = None
        self.last_seen_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_exit_zone: Optional[ExitZone] = None
        self.transition_start_time: float = 0.0
        self.final_verdict: Optional[FinalVerdict] = None
        self.demo_active: bool = False

        # ML Models (Lazy Initialization)
        self._detector: Optional[DetectionTracker] = None
        self._embedder: Optional[EmbeddingExtractor] = None
        self._attributor: Optional[AttributeInference] = None

        logger.info(f"TopologyTracker initialized with {len(self.cameras)} cameras.")
        # Immediately prime frame 0 caches for all cameras so MJPEG streams never hang on startup
        self.update_all_frame_caches()

    # ── Lazy Model Loaders ────────────────────────────────────────────────────

    @property
    def detector(self) -> DetectionTracker:
        if self._detector is None:
            d_cfg = dict(self.default_cfg.get("detection", {}))
            self._detector = DetectionTracker(d_cfg)
        return self._detector

    @property
    def embedder(self) -> EmbeddingExtractor:
        if self._embedder is None:
            m_cfg = dict(self.default_cfg.get("model", {}))
            weights_path = m_cfg.get("weights_path", "models/osnet_x1_0_market1501.pth")
            if not Path(weights_path).exists() and Path("data/models/osnet_x1_0_market1501.pth").exists():
                m_cfg["weights_path"] = "data/models/osnet_x1_0_market1501.pth"
            self._embedder = EmbeddingExtractor(m_cfg)
        return self._embedder

    @property
    def attributor(self) -> AttributeInference:
        if self._attributor is None:
            a_cfg = dict(self.default_cfg.get("attributes", {}))
            model_xml = a_cfg.get("model_xml", "models/FP32/person-attributes-recognition-crossroad-0230.xml")
            if not Path(model_xml).exists() and Path("data/models/attributes/person-attributes-recognition-crossroad-0230.xml").exists():
                a_cfg["model_xml"] = "data/models/attributes/person-attributes-recognition-crossroad-0230.xml"
            self._attributor = AttributeInference(a_cfg)
            self._attributor.load()
        return self._attributor

    # ── Target Description Scoring ────────────────────────────────────────────

    def score_crop_against_criteria(
        self, crop_bgr: np.ndarray, criteria: TargetCriteria
    ) -> Tuple[float, dict, ColorInfo, AttributeInfo, QualityInfo]:
        """
        Evaluate a detected person crop against user attribute/color description.
        Returns match score in [0, 1] along with detailed modality sub-scores.
        """
        H, W = crop_bgr.shape[:2]
        # 1. Quality
        q_dict = estimate_quality(crop_bgr, (0, 0, W, H), self.default_cfg.get("quality", {}))
        quality = QualityInfo(
            blur=q_dict["blur"],
            visibility=q_dict["visibility"],
            size=q_dict["size"],
            overall=q_dict["overall"],
        )

        # 2. Attributes & sampling coords
        raw_attrs, coords = self.attributor.infer(crop_bgr)
        attrs = AttributeInfo(**raw_attrs)

        # 3. CIELAB Colors
        upper_xy = coords[0] if coords else None
        lower_xy = coords[1] if coords else None
        c_dict = extract_colors(crop_bgr, upper_xy=upper_xy, lower_xy=lower_xy, cfg=self.default_cfg.get("color", {}))
        colors = ColorInfo(
            upper=str(c_dict.get("upper", "unknown")),
            lower=str(c_dict.get("lower", "unknown")),
            upper_conf=float(c_dict.get("upper_conf", 0.0)),
            lower_conf=float(c_dict.get("lower_conf", 0.0)),
            sampling_path=str(c_dict.get("sampling_path", "openvino_coords")),
        )

        scores = []
        weights = []

        # Color matches
        if criteria.upper_color.lower() not in ("any", ""):
            match = 1.0 if colors.upper.lower() == criteria.upper_color.lower() else 0.0
            scores.append(match * max(0.4, colors.upper_conf))
            weights.append(2.0)

        if criteria.lower_color.lower() not in ("any", ""):
            match = 1.0 if colors.lower.lower() == criteria.lower_color.lower() else 0.0
            scores.append(match * max(0.4, colors.lower_conf))
            weights.append(2.0)

        # Boolean attribute matches
        attr_checks = [
            ("has_backpack", criteria.has_backpack),
            ("has_bag", criteria.has_bag),
            ("has_hat", criteria.has_hat),
            ("has_coat_jacket", criteria.has_jacket),
            ("has_longpants", criteria.has_longpants),
        ]
        for attr_key, expected in attr_checks:
            if expected is not None:
                prob = getattr(attrs, attr_key, 0.5)
                score = prob if expected else (1.0 - prob)
                scores.append(score)
                weights.append(1.0)

        if not scores:
            final_score = 0.5
        else:
            final_score = float(np.average(scores, weights=weights))

        breakdown = {
            "upper_color": colors.upper,
            "upper_conf": colors.upper_conf,
            "lower_color": colors.lower,
            "lower_conf": colors.lower_conf,
            "attributes": attrs.to_bool_dict(self.default_cfg["attributes"].get("thresholds", {})),
            "quality": quality.overall,
        }

        return final_score, breakdown, colors, attrs, quality

    # ── State Machine Controls ────────────────────────────────────────────────

    def set_focused_camera(self, camera_id: str) -> None:
        """Set the operator's focused / active camera feed."""
        if camera_id in self.cameras:
            self.focused_camera_id = camera_id
            if self.global_state in (GlobalTrackerState.UNINITIALIZED, GlobalTrackerState.RESOLVED):
                self.active_camera_id = camera_id
                for cid, node in self.cameras.items():
                    node.compute_state = ComputeState.ACTIVE if cid == camera_id else ComputeState.IDLE
            logger.info(f"Focused camera set to: {camera_id}")

    def start_acquisition(self, criteria: TargetCriteria) -> None:
        """Start search for target meeting description criteria on starting cameras."""
        self.target_criteria = criteria
        self.global_state = GlobalTrackerState.ACQUIRING
        self.target_embedding_bank.clear()
        self.camera_path.clear()
        self.case_timeline.clear()
        self.final_verdict = None

        # Activate starting cameras, idle others
        for cam_id, node in self.cameras.items():
            if cam_id in criteria.starting_cameras:
                node.compute_state = ComputeState.SEARCHING_NEIGHBOR
            else:
                node.compute_state = ComputeState.IDLE

        logger.info(f"Target acquisition started on cameras: {criteria.starting_cameras}")

    def lock_target(
        self,
        camera_id: str,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        crop_bgr: np.ndarray,
        colors: ColorInfo,
        attrs: AttributeInfo,
        quality: QualityInfo,
    ) -> None:
        """Lock onto a specific candidate person and transition to ACTIVE tracking."""
        self.locked_target_id = f"TARGET_{camera_id.upper()}_{track_id}"
        self.active_camera_id = camera_id
        self.last_seen_cam = camera_id
        self.last_seen_time = self.current_global_time
        self.last_seen_bbox = bbox
        self.camera_path = [camera_id]

        # Extract 512-D OSNet embedding
        emb = self.embedder.extract(crop_bgr)
        self.target_embedding_bank = [emb]

        self.target_reference_record = FeatureRecord(
            track_id=track_id,
            camera_id=camera_id,
            timestamp=self.current_global_time,
            bbox=list(bbox),
            embedding=emb,
            colors=colors,
            attributes=attrs,
            quality=quality,
            record_id=self.locked_target_id,
        )

        self.target_avatar_crop = crop_bgr.copy()

        # Update compute states: ACTIVE on camera_id, IDLE on all others
        for cid, node in self.cameras.items():
            node.compute_state = ComputeState.ACTIVE if cid == camera_id else ComputeState.IDLE

        self.global_state = GlobalTrackerState.TRACKING
        self.case_timeline.append(
            TransitionEvent(
                from_cam="ENTRY",
                to_cam=camera_id,
                timestamp=self.current_global_time,
                score=1.0,
                reason=f"Target acquired and locked via attribute description in {self.cameras[camera_id].name}",
                crop_snapshot=self.target_avatar_crop,
            )
        )
        logger.info(f"Target locked on {camera_id} (Track #{track_id}). Switched to Strategic Single-Camera Mode.")

    # ── Per-Frame Processing Loop ─────────────────────────────────────────────

    def process_step(self, global_t: float) -> Dict[str, Any]:
        """
        Execute one step of the state machine scheduler at global_t.
        Returns visual frames and state payload for the Web UI dashboard.
        """
        self.current_global_time = global_t
        results: Dict[str, Any] = {
            "global_time": global_t,
            "global_state": self.global_state.value,
            "active_camera": self.active_camera_id,
            "camera_path": list(self.camera_path),
            "camera_states": {cid: node.compute_state.value for cid, node in self.cameras.items()},
            "candidates": [],
            "raw_detections": [],
            "tracked_target": None,
            "final_verdict": None,
            "fps": self.fps,
        }

        # 0. State: UNINITIALIZED / STANDBY
        if self.global_state == GlobalTrackerState.UNINITIALIZED:
            self._step_standby(global_t, results)

        # 1. State: ACQUIRING
        elif self.global_state == GlobalTrackerState.ACQUIRING:
            self._step_acquiring(global_t, results)

        # 2. State: TRACKING
        elif self.global_state == GlobalTrackerState.TRACKING:
            self._step_tracking(global_t, results)

        # 3. State: TRANSITIONING
        elif self.global_state == GlobalTrackerState.TRANSITIONING:
            self._step_transitioning(global_t, results)

        # 4. State: RESOLVED
        elif self.global_state == GlobalTrackerState.RESOLVED:
            if self.final_verdict:
                results["final_verdict"] = self._format_verdict(self.final_verdict)

        # Update encoded frame caches with HUD overlays for web streaming
        self.update_all_frame_caches(results)
        return results

    def _step_standby(self, global_t: float, results: dict) -> None:
        """Standby mode: run YOLOv8 on active/focused camera to visualize all detected persons."""
        active_cam = self.active_camera_id or self.focused_camera_id or "cam1"
        node = self.cameras.get(active_cam)
        if not node:
            return

        for cid, cnode in self.cameras.items():
            cnode.compute_state = ComputeState.ACTIVE if cid == active_cam else ComputeState.IDLE

        ret, frame = node.read_frame_at_time(global_t)
        if not ret or frame is None:
            return

        detections = self.detector.process_frame(frame, camera_id=active_cam)
        raw_dets = []
        attrs_list = []
        colors_list = []
        bboxes = []

        for det in detections:
            crop_bgr = det["crop_bgr"]
            track_id = det["track_id"]
            bbox = list(det["bbox"])
            conf = det["conf"]
            bboxes.append(bbox)

            try:
                raw_attrs, coords = self.attributor.infer(crop_bgr)
                attrs = AttributeInfo(**raw_attrs)
            except Exception:
                attrs = AttributeInfo()
                coords = None

            try:
                upper_xy = coords[0] if coords else None
                lower_xy = coords[1] if coords else None
                c_dict = extract_colors(crop_bgr, upper_xy=upper_xy, lower_xy=lower_xy, cfg=self.default_cfg.get("color", {}))
                colors = ColorInfo(
                    upper=str(c_dict.get("upper", "unknown")),
                    lower=str(c_dict.get("lower", "unknown")),
                    upper_conf=float(c_dict.get("upper_conf", 0.0)),
                    lower_conf=float(c_dict.get("lower_conf", 0.0)),
                )
            except Exception:
                colors = ColorInfo()

            attr_bools = attrs.to_bool_dict(self.default_cfg.get("attributes", {}).get("thresholds", {}))
            active_attrs = [k.replace("has_", "") for k, v in attr_bools.items() if v]
            attrs_str = ", ".join(active_attrs) if active_attrs else "none"
            color_str = f"upper={colors.upper}({colors.upper_conf:.2f}), lower={colors.lower}({colors.lower_conf:.2f})"

            attrs_list.append(attrs_str)
            colors_list.append(color_str)

            raw_dets.append({
                "camera_id": active_cam,
                "track_id": track_id,
                "bbox": bbox,
                "conf": round(conf, 2),
                "attrs": active_attrs,
                "colors": {"upper": colors.upper, "lower": colors.lower},
            })

        results["raw_detections"] = raw_dets
        print(f"[DEBUG FRAME] Cam: {active_cam} | Detections found: {len(detections)} | Bboxes: {bboxes} | OpenVINO attrs: {attrs_list} | Color: {colors_list}", flush=True)

    def _step_acquiring(self, global_t: float, results: dict) -> None:
        """Scan starting cameras for person matching description query."""
        assert self.target_criteria is not None
        best_candidate = None
        highest_score = 0.0
        all_raw_dets = []

        for cam_id in self.target_criteria.starting_cameras:
            node = self.cameras.get(cam_id)
            if not node:
                continue
            ret, frame = node.read_frame_at_time(global_t)
            if not ret or frame is None:
                continue

            # Run detection on this camera
            detections = self.detector.process_frame(frame, camera_id=cam_id)
            bboxes = []
            attrs_list = []
            colors_list = []

            for det in detections:
                crop_bgr = det["crop_bgr"]
                track_id = det["track_id"]
                bbox = tuple(det["bbox"])
                bboxes.append(list(bbox))

                score, breakdown, colors, attrs, quality = self.score_crop_against_criteria(
                    crop_bgr, self.target_criteria
                )
                attr_bools = attrs.to_bool_dict(self.default_cfg.get("attributes", {}).get("thresholds", {}))
                active_attrs = [k.replace("has_", "") for k, v in attr_bools.items() if v]
                attrs_str = ", ".join(active_attrs) if active_attrs else "none"
                color_str = f"upper={colors.upper}({colors.upper_conf:.2f}), lower={colors.lower}({colors.lower_conf:.2f})"

                attrs_list.append(attrs_str)
                colors_list.append(color_str)

                cand_info = {
                    "camera_id": cam_id,
                    "camera_name": node.name,
                    "track_id": track_id,
                    "bbox": list(bbox),
                    "score": round(score, 3),
                    "breakdown": breakdown,
                }
                results["candidates"].append(cand_info)

                all_raw_dets.append({
                    "camera_id": cam_id,
                    "track_id": track_id,
                    "bbox": list(bbox),
                    "conf": round(det["conf"], 2),
                    "attrs": active_attrs,
                    "colors": {"upper": colors.upper, "lower": colors.lower},
                })

                if score > highest_score:
                    highest_score = score
                    best_candidate = (cam_id, track_id, bbox, crop_bgr, colors, attrs, quality, score)

            print(f"[DEBUG FRAME] Cam: {cam_id} | Detections found: {len(detections)} | Bboxes: {bboxes} | OpenVINO attrs: {attrs_list} | Color: {colors_list}", flush=True)

        results["raw_detections"] = all_raw_dets

        # Auto-lock if threshold exceeded
        if best_candidate and highest_score >= self.acquisition_threshold:
            cid, tid, bbox, crop, cols, atts, qual, sc = best_candidate
            self.lock_target(cid, tid, bbox, crop, cols, atts, qual)
            results["auto_locked"] = True

    def _step_tracking(self, global_t: float, results: dict) -> None:
        """Single-camera active tracking with 0% compute on all other cameras."""
        active_cam = self.active_camera_id or self.focused_camera_id or "cam1"
        if not active_cam or active_cam not in self.cameras:
            return

        node = self.cameras[active_cam]
        ret, frame = node.read_frame_at_time(global_t)
        if node.compute_state == ComputeState.OFFLINE:
            # Active camera reached EOF; keep state without triggering false target lost
            return
        if not ret or frame is None:
            return

        target_found_in_frame = False
        target_bbox = None
        target_crop = None
        target_quality = None

        # Run CV detection & tracking ONLY on active camera
        detections = self.detector.process_frame(frame, camera_id=active_cam)
        raw_dets = []
        attrs_list = []
        colors_list = []
        bboxes = []

        for det in detections:
            crop_bgr = det["crop_bgr"]
            track_id = det["track_id"]
            bbox = tuple(det["bbox"])
            bboxes.append(list(bbox))

            try:
                raw_attrs, coords = self.attributor.infer(crop_bgr)
                attrs = AttributeInfo(**raw_attrs)
            except Exception:
                attrs = AttributeInfo()
                coords = None

            try:
                upper_xy = coords[0] if coords else None
                lower_xy = coords[1] if coords else None
                c_dict = extract_colors(crop_bgr, upper_xy=upper_xy, lower_xy=lower_xy, cfg=self.default_cfg.get("color", {}))
                colors = ColorInfo(
                    upper=str(c_dict.get("upper", "unknown")),
                    lower=str(c_dict.get("lower", "unknown")),
                    upper_conf=float(c_dict.get("upper_conf", 0.0)),
                    lower_conf=float(c_dict.get("lower_conf", 0.0)),
                )
            except Exception:
                colors = ColorInfo()

            attr_bools = attrs.to_bool_dict(self.default_cfg.get("attributes", {}).get("thresholds", {}))
            active_attrs = [k.replace("has_", "") for k, v in attr_bools.items() if v]
            attrs_str = ", ".join(active_attrs) if active_attrs else "none"
            color_str = f"upper={colors.upper}({colors.upper_conf:.2f}), lower={colors.lower}({colors.lower_conf:.2f})"

            attrs_list.append(attrs_str)
            colors_list.append(color_str)

            raw_dets.append({
                "camera_id": active_cam,
                "track_id": track_id,
                "bbox": list(bbox),
                "conf": round(det["conf"], 2),
                "attrs": active_attrs,
                "colors": {"upper": colors.upper, "lower": colors.lower},
            })

            # Check similarity with target embedding bank
            emb = self.embedder.extract(crop_bgr)
            max_sim = max(
                (float(np.dot(emb, b_emb)) for b_emb in self.target_embedding_bank),
                default=0.0
            )

            if max_sim >= 0.50 or (self.last_seen_bbox and self._iou(bbox, self.last_seen_bbox) > 0.3):
                target_found_in_frame = True
                target_bbox = bbox
                target_crop = crop_bgr
                self.last_seen_bbox = bbox
                self.last_seen_time = global_t
                self.last_seen_cam = active_cam

                # Quality estimate
                H, W = crop_bgr.shape[:2]
                q_dict = estimate_quality(crop_bgr, (0, 0, W, H), self.default_cfg.get("quality", {}))
                target_quality = q_dict["overall"]

                # Update embedding bank with quality gating
                qual_thresh = float(self.default_cfg.get("gallery", {}).get("update_quality_threshold", 0.40))
                max_bank = int(self.default_cfg.get("gallery", {}).get("top_k_output", 10))
                if target_quality >= qual_thresh and len(self.target_embedding_bank) < max_bank:
                    self.target_embedding_bank.append(emb)

                # Check if target is inside an exit zone
                norm_bbox = (
                    bbox[0] / node.width,
                    bbox[1] / node.height,
                    bbox[2] / node.width,
                    bbox[3] / node.height,
                )
                for zone in node.exit_zones:
                    if zone.intersects(norm_bbox):
                        self.last_exit_zone = zone
                        break

                results["tracked_target"] = {
                    "camera_id": active_cam,
                    "track_id": track_id,
                    "bbox": list(bbox),
                    "quality": round(target_quality, 3),
                    "sim_score": round(max_sim, 3),
                    "matched_attrs": attrs_str,
                    "exit_zone": self.last_exit_zone.name if self.last_exit_zone else None,
                }

        results["raw_detections"] = raw_dets
        print(f"[DEBUG FRAME] Cam: {active_cam} | Detections found: {len(detections)} | Bboxes: {bboxes} | OpenVINO attrs: {attrs_list} | Color: {colors_list}", flush=True)

        # Check lost timeout: if lost for > 1.5 seconds, transition to Hand-off mode
        if getattr(self, "demo_active", False):
            # In sequential demo mode, camera transitions are orchestrated by demo schedule
            return

        time_lost = global_t - self.last_seen_time
        if not target_found_in_frame and time_lost > self.lost_timeout_sec:
            logger.info(f"Target lost in {active_cam} for {time_lost:.2f}s > {self.lost_timeout_sec}s. Initiating hand-off.")
            self._initiate_transition(active_cam, global_t)

    def _initiate_transition(self, from_cam: str, global_t: float) -> None:
        """Transition from active tracking to neighbor hand-off mode."""
        self.global_state = GlobalTrackerState.TRANSITIONING
        self.transition_start_time = global_t
        curr_node = self.cameras[from_cam]

        # Neighbor cameras activated for CV search; curr_node set to low-rate check; others IDLE
        neighbors = set(curr_node.neighbors)
        for cid, node in self.cameras.items():
            if cid in neighbors:
                node.compute_state = ComputeState.SEARCHING_NEIGHBOR
            elif cid == from_cam:
                node.compute_state = ComputeState.LOW_RATE_CHECK
            else:
                node.compute_state = ComputeState.IDLE

        logger.info(f"Transition Mode: Activated neighbor search on {list(neighbors)} for {from_cam}")

    def _step_transitioning(self, global_t: float, results: dict) -> None:
        """Scan connected neighbor cameras for target re-acquisition."""
        from_cam = self.last_seen_cam or "cam1"
        neighbors = self.cameras[from_cam].neighbors
        reacquired = False
        best_match = None
        best_fusion_score = 0.0

        for cam_j in neighbors:
            node = self.cameras.get(cam_j)
            if not node:
                continue

            ret, frame = node.read_frame_at_time(global_t)
            if not ret or frame is None:
                continue

            # Run detection on neighbor stream
            detections = self.detector.process_frame(frame)
            for det in detections:
                crop_bgr = det["crop_bgr"]
                track_id = det["track_id"]
                bbox = tuple(det["bbox"])
                emb = self.embedder.extract(crop_bgr)
                emb_sim = max(
                    (float(np.dot(emb, b_emb)) for b_emb in self.target_embedding_bank),
                    default=0.0
                )

                # Quality + Attributes + Color for fusion
                H, W = crop_bgr.shape[:2]
                q_dict = estimate_quality(crop_bgr, (0, 0, W, H), self.default_cfg.get("quality", {}))
                quality = QualityInfo(
                    blur=q_dict["blur"],
                    visibility=q_dict["visibility"],
                    size=q_dict["size"],
                    overall=q_dict["overall"],
                )
                raw_attrs, coords = self.attributor.infer(crop_bgr)
                attrs = AttributeInfo(**raw_attrs)
                upper_xy = coords[0] if coords else None
                lower_xy = coords[1] if coords else None
                c_dict = extract_colors(crop_bgr, upper_xy=upper_xy, lower_xy=lower_xy, cfg=self.default_cfg.get("color", {}))
                colors = ColorInfo(
                    upper=str(c_dict.get("upper", "unknown")),
                    lower=str(c_dict.get("lower", "unknown")),
                    upper_conf=float(c_dict.get("upper_conf", 0.0)),
                    lower_conf=float(c_dict.get("lower_conf", 0.0)),
                )

                cand_record = FeatureRecord(
                    track_id=track_id,
                    camera_id=cam_j,
                    timestamp=global_t,
                    bbox=list(bbox),
                    embedding=emb,
                    colors=colors,
                    attributes=attrs,
                    quality=quality,
                )

                # Adaptive Rule Fusion
                ref_record = self.target_reference_record or cand_record
                fusion_out = fuse(
                    cand_record,
                    ref_record,
                    emb_sim,
                    self.default_cfg["fusion"],
                    method="adaptive_rule",
                )
                score = fusion_out["score"]

                if score > best_fusion_score:
                    best_fusion_score = score
                    best_match = (cam_j, track_id, bbox, crop_bgr, colors, attrs, quality, score)

        # Check re-acquisition
        if best_match and best_fusion_score >= self.reacquisition_fusion_threshold:
            cam_j, tid, bbox, crop, cols, atts, qual, sc = best_match
            self._reacquire_target(from_cam, cam_j, tid, bbox, crop, cols, atts, qual, sc, global_t)
            reacquired = True
            results["reacquired"] = True

        # Check transit window timeout -> Resolve Final Destination
        elapsed = global_t - self.transition_start_time
        if not reacquired and (elapsed > self.neighbor_search_timeout_sec or global_t >= self.duration_sec - 1.0):
            self._resolve_final_destination(from_cam, global_t)

    def is_valid_transition(self, from_cam: str, to_cam: str) -> bool:
        """Check if to_cam is an allowed topological neighbor of from_cam."""
        if from_cam not in self.cameras or to_cam not in self.cameras:
            return False
        return to_cam in self.cameras[from_cam].neighbors

    def _reacquire_target(
        self,
        from_cam: str,
        to_cam: str,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        crop_bgr: np.ndarray,
        colors: ColorInfo,
        attrs: AttributeInfo,
        quality: QualityInfo,
        fusion_score: float,
        global_t: float,
    ) -> None:
        """Target re-acquired in neighbor camera. Transition back to ACTIVE mode."""
        if not self.is_valid_transition(from_cam, to_cam):
            logger.warning(f"Rejected non-topological transition attempt: {from_cam} -> {to_cam}")
            return
        self.active_camera_id = to_cam
        self.last_seen_cam = to_cam
        self.last_seen_time = global_t
        self.last_seen_bbox = bbox
        self.last_exit_zone = None
        self.camera_path.append(to_cam)

        # Update target embedding bank
        emb = self.embedder.extract(crop_bgr)
        if len(self.target_embedding_bank) < 10:
            self.target_embedding_bank.append(emb)

        # Log transition
        event = TransitionEvent(
            from_cam=from_cam,
            to_cam=to_cam,
            timestamp=global_t,
            score=round(fusion_score, 3),
            reason=f"Re-acquired in {self.cameras[to_cam].name} (Fusion score: {fusion_score:.2f})",
            crop_snapshot=crop_bgr,
        )
        self.case_timeline.append(event)

        # Compute states: to_cam -> ACTIVE, all others -> IDLE (0% GPU)
        for cid, node in self.cameras.items():
            node.compute_state = ComputeState.ACTIVE if cid == to_cam else ComputeState.IDLE

        self.global_state = GlobalTrackerState.TRACKING
        logger.info(f"Target handoff complete: {from_cam} -> {to_cam} at {global_t:.1f}s.")

    def _resolve_final_destination(self, last_cam: str, global_t: float) -> None:
        """Resolve final destination when transit window expires or terminal zone reached."""
        node = self.cameras[last_cam]
        inferred = f"Terminal Zone in {node.name}"

        # If an exit zone was recorded before loss
        if self.last_exit_zone and self.last_exit_zone.is_terminal:
            inferred = self.last_exit_zone.terminal_name or self.last_exit_zone.name
        else:
            # Check terminal zones defined on this camera
            terminals = [z for z in node.exit_zones if z.is_terminal]
            if terminals:
                inferred = terminals[0].terminal_name or terminals[0].name
            elif node.floor == 3:
                inferred = "Terrace Area / Roof Access"
            elif node.floor == 1:
                inferred = "Flat 1A / Flat 1B"
            elif node.floor == 2:
                inferred = "Flat 2A / Flat 2B"
            elif last_cam == "cam2":
                inferred = "Main Gate Exit"

        self.final_verdict = FinalVerdict(
            last_camera=last_cam,
            last_camera_name=node.name,
            global_timestamp=round(self.last_seen_time, 1),
            camera_path=list(self.camera_path),
            inferred_location=inferred,
            terminal_zone=self.last_exit_zone.name if self.last_exit_zone else None,
            confidence=0.92,
        )

        self.global_state = GlobalTrackerState.RESOLVED
        for cid, cnode in self.cameras.items():
            cnode.compute_state = ComputeState.IDLE

        self.case_timeline.append(
            TransitionEvent(
                from_cam=last_cam,
                to_cam="RESOLVED",
                timestamp=global_t,
                score=1.0,
                reason=f"Final Destination Resolved: {inferred}",
            )
        )
        logger.info(f"FINAL DESTINATION RESOLVED: {inferred} at {last_cam}")

    def _format_verdict(self, v: FinalVerdict) -> dict:
        return {
            "last_camera": v.last_camera,
            "last_camera_name": v.last_camera_name,
            "global_timestamp": v.global_timestamp,
            "time_str": self.format_time(v.global_timestamp),
            "camera_path": " -> ".join(c.upper() for c in v.camera_path),
            "camera_path_list": v.camera_path,
            "inferred_location": v.inferred_location,
            "terminal_zone": v.terminal_zone,
            "confidence": v.confidence,
            "status": v.status,
        }

    # ── Frame HUD Rendering ───────────────────────────────────────────────────

    def render_camera_frame(self, cam_id: str, raw_frame: np.ndarray, step_data: Optional[dict] = None) -> np.ndarray:
        """
        Render real-time surveillance HUD overlay on raw video frame.
        Draws compute state badges, bounding boxes, quality bar, and exit zones.
        """
        node = self.cameras.get(cam_id)
        if node is None or raw_frame is None:
            return raw_frame

        frame = raw_frame.copy()
        H, W = frame.shape[:2]
        c_state = node.compute_state

        # 1. Top HUD Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 46), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Camera Name & Floor
        cam_tag = f"{node.name.upper()} ({node.floor_name.upper()})"
        cv2.putText(frame, cam_tag, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (241, 245, 249), 2)

        # Global Time
        time_str = self.format_time(self.current_global_time)
        time_tag = f"T: {time_str} / 15:00"
        cv2.putText(frame, time_tag, (W - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1)

        # Compute State Badge
        if c_state == ComputeState.ACTIVE:
            badge_bg = (16, 185, 129)
            badge_txt = "[ACTIVE - TRACKING]"
            txt_color = (0, 0, 0)
        elif c_state == ComputeState.SEARCHING_NEIGHBOR:
            badge_bg = (245, 158, 11)
            badge_txt = "[SEARCHING NEIGHBOR]"
            txt_color = (0, 0, 0)
        elif c_state == ComputeState.LOW_RATE_CHECK:
            badge_bg = (168, 85, 247)
            badge_txt = "[LOW-RATE CHECK]"
            txt_color = (255, 255, 255)
        elif c_state == ComputeState.OFFLINE:
            badge_bg = (30, 41, 59)
            badge_txt = "[OFFLINE - STREAM END]"
            txt_color = (148, 163, 184)
        else:
            badge_bg = (51, 65, 85)
            badge_txt = "[IDLE - 0% GPU]"
            txt_color = (203, 213, 225)

        (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bx = 14
        by = 56
        cv2.rectangle(frame, (bx, by), (bx + tw + 16, by + th + 10), badge_bg, -1)
        cv2.putText(frame, badge_txt, (bx + 8, by + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 2)

        # 2. Draw Exit Zones
        for zone in node.exit_zones:
            zx1 = int(zone.box[0] * W)
            zy1 = int(zone.box[1] * H)
            zx2 = int(zone.box[2] * W)
            zy2 = int(zone.box[3] * H)
            z_col = (239, 68, 68) if zone.is_terminal else (245, 158, 11)
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), z_col, 1)
            cv2.putText(frame, zone.name, (zx1 + 6, max(20, zy1 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, z_col, 1)

        # 3. Draw Tracked Target or Candidates / Raw Detections
        if step_data:
            target_info = step_data.get("tracked_target")
            locked_box = None
            if target_info and target_info.get("camera_id") == cam_id:
                locked_box = target_info.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = locked_box
                # Vibrant green bounding box
                vibrant_green = (16, 214, 63)
                cv2.rectangle(frame, (x1, y1), (x2, y2), vibrant_green, 2)
                # Corner brackets
                corner_len = min(20, max(8, (x2 - x1) // 4))
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 3)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 3)
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 3)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 3)
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), 3)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), 3)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 3)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 3)

                # Target Header Tag with Track ID, matched attributes, and live fusion confidence score
                track_id = target_info.get("track_id", 1)
                sim_pct = int(target_info.get("sim_score", 0.95) * 100)
                matched_attrs = target_info.get("matched_attrs", "TARGET")
                header = f"TARGET #{track_id} | Score: {sim_pct}% | {matched_attrs}"
                (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + tw + 12, y1), vibrant_green, -1)
                cv2.putText(frame, header, (x1 + 6, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

                # Quality Bar below bbox
                q_val = float(target_info.get("quality", 0.5))
                bar_w = x2 - x1
                fill_w = int(bar_w * q_val)
                cv2.rectangle(frame, (x1, y2 + 4), (x2, y2 + 12), (30, 41, 59), -1)
                q_col = (16, 214, 63) if q_val > 0.6 else (245, 158, 11) if q_val > 0.4 else (239, 68, 68)
                cv2.rectangle(frame, (x1, y2 + 4), (x1 + fill_w, y2 + 12), q_col, -1)
                cv2.putText(frame, f"Quality: {q_val:.2f}", (x1, y2 + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (241, 245, 249), 1)

            # Draw candidates if in ACQUIRING or TRANSITIONING
            candidate_boxes = []
            for cand in step_data.get("candidates", []):
                if cand.get("camera_id") == cam_id:
                    bx1, by1, bx2, by2 = cand.get("bbox", [0, 0, 0, 0])
                    candidate_boxes.append((bx1, by1, bx2, by2))
                    sc = cand.get("score", 0.0)
                    tid = cand.get("track_id", 1)
                    c_col = (245, 158, 11)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), c_col, 2)
                    cv2.putText(frame, f"CANDIDATE #{tid} | Match: {sc:.2f}", (bx1, max(15, by1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c_col, 2)

            # Draw light grey/blue bounding boxes for all detected persons (STANDBY, SEARCHING, or other people)
            raw_detections = step_data.get("raw_detections", [])
            for det in raw_detections:
                if det.get("camera_id") != cam_id:
                    continue
                dbbox = det.get("bbox", [0, 0, 0, 0])
                if locked_box and self._iou(tuple(dbbox), tuple(locked_box)) > 0.4:
                    continue
                if any(self._iou(tuple(dbbox), cb) > 0.4 for cb in candidate_boxes):
                    continue

                dx1, dy1, dx2, dy2 = dbbox
                # Light grey/blue: BGR (220, 180, 110)
                box_color = (220, 180, 110)
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), box_color, 2)

                # Subtle corner brackets
                c_len = min(14, max(5, (dx2 - dx1) // 5))
                cv2.line(frame, (dx1, dy1), (dx1 + c_len, dy1), (255, 255, 255), 2)
                cv2.line(frame, (dx1, dy1), (dx1, dy1 + c_len), (255, 255, 255), 2)
                cv2.line(frame, (dx2, dy1), (dx2 - c_len, dy1), (255, 255, 255), 2)
                cv2.line(frame, (dx2, dy1), (dx2, dy1 + c_len), (255, 255, 255), 2)
                cv2.line(frame, (dx1, dy2), (dx1 + c_len, dy2), (255, 255, 255), 2)
                cv2.line(frame, (dx1, dy2), (dx1, dy2 - c_len), (255, 255, 255), 2)
                cv2.line(frame, (dx2, dy2), (dx2 - c_len, dy2), (255, 255, 255), 2)
                cv2.line(frame, (dx2, dy2), (dx2, dy2 - c_len), (255, 255, 255), 2)

                tid = det.get("track_id", 1)
                conf = det.get("conf", 0.5)
                cols = det.get("colors", {})
                col_tag = f"{cols.get('upper', '')}/{cols.get('lower', '')}" if cols else ""
                label = f"PERSON #{tid} [{conf:.2f}]"
                if col_tag and col_tag != "/":
                    label += f" | {col_tag}"

                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(frame, (dx1, max(0, dy1 - 22)), (dx1 + lw + 10, dy1), (30, 41, 59), -1)
                cv2.rectangle(frame, (dx1, max(0, dy1 - 22)), (dx1 + lw + 10, dy1), box_color, 1)
                cv2.putText(frame, label, (dx1 + 4, max(12, dy1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1)

        return frame

    def update_all_frame_caches(self, step_data: Optional[dict] = None) -> None:
        """Render HUD overlays and encode JPEGs for all 8 camera streams into memory."""
        step_data = step_data or {}
        active_cam = self.active_camera_id or "cam2"
        for cid, node in self.cameras.items():
            ret, raw_frame = node.read_frame_at_time(self.current_global_time)
            frame_to_use = raw_frame if raw_frame is not None else node.cached_frame
            if frame_to_use is not None:
                frame = self.render_camera_frame(cid, frame_to_use, step_data)
                
                # Keep portrait aspect ratio for 960x1080 CCTV feeds
                h_orig, w_orig = frame.shape[:2]
                thumb_w = 480
                thumb_h = int(round(h_orig * (thumb_w / w_orig))) if w_orig > 0 else 540
                thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                ret_enc, jpeg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret_enc:
                    node.cached_jpeg = jpeg.tobytes()

                if cid == active_cam or not node.cached_focused_jpeg:
                    disp_w = 960
                    disp_h = int(round(h_orig * (disp_w / w_orig))) if w_orig > 0 else 1080
                    display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                    ret_f, f_jpeg = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret_f:
                        node.cached_focused_jpeg = f_jpeg.tobytes()

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return inter / float(boxAArea + boxBArea - inter)

    @staticmethod
    def format_time(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def update_offsets(self, offsets: Dict[str, float]) -> None:
        """Update timeline synchronization offset floats live."""
        for cid, off in offsets.items():
            if cid in self.cameras:
                self.cameras[cid].time_offset_sec = float(off)
        logger.info(f"Updated camera offsets: {offsets}")
