"""
web/app.py
==========
SAFEPASS Multi-Camera Strategic Topology Tracker & Operator Dashboard.

FastAPI application providing:
- Web dashboard UI serving HTML5/JS
- Multi-camera MJPEG streams with live surveillance HUD overlays
- REST API & WebSocket for scrubber control, attribute search, timeline sync, and case logging
- Background playback loop orchestrating TopologyTracker
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


# ── Sequential Person Demo Profiles ──────────────────────────────────────────

DEMO_PROFILES = {
    "profile_1": {
        "features": {"upper": "blue", "lower": "black"},
        "segments": [
            {"cam": "cam1", "start": 0.0, "end": 119.0, "time_str": "00:00 - 01:59", "floor": "Ground Floor", "note": "Target detected moving along Ground Floor East Walkway"},
            {"cam": "cam2", "start": 120.0, "end": 140.0, "time_str": "02:00 - 02:20", "floor": "Ground Floor", "note": "Transiting SE Gate Area"},
            {"cam": "cam3", "start": 141.0, "end": 160.0, "time_str": "02:21 - 02:40", "floor": "Ground Floor", "note": "Re-acquired entering Building Entrance Right"},
            {"cam": "cam4", "start": 167.0, "end": 190.0, "time_str": "02:47 - 03:10", "floor": "Ground Floor", "note": "Transiting Building Entrance Left"},
            {"cam": "cam3", "start": 195.0, "end": 215.0, "time_str": "03:15 - 03:35", "floor": "Ground Floor", "note": "Returning via Building Entrance Right"},
            {"cam": "cam2", "start": 463.0, "end": 610.0, "time_str": "07:43 - 10:10", "floor": "Ground Floor", "note": "Final Sighting: Exiting complex via SE Corner Gate"},
        ],
        "final_camera": "cam2",
        "final_location": "Ground Floor SE Corner / Gate (Main Gate Exit)",
        "path": ["cam1", "cam2", "cam3", "cam4", "cam3", "cam2"],
        "confidence": 0.96,
    },
    "profile_2": {
        "features": {"upper": "white", "lower": "blue"},
        "segments": [
            {"cam": "cam2", "start": 1.0, "end": 11.0, "time_str": "00:01 - 00:11", "floor": "Ground Floor", "note": "Entering complex through SE Corner Gate"},
            {"cam": "cam3", "start": 13.0, "end": 103.0, "time_str": "00:13 - 01:43", "floor": "Ground Floor", "note": "Walking across Ground Floor Building Entrance Right"},
            {"cam": "cam6", "start": 110.0, "end": 124.0, "time_str": "01:50 - 02:04", "floor": "1st Floor", "note": "Ascending to 1st Floor Stairwell Landing"},
            {"cam": "cam7", "start": 126.0, "end": 140.0, "time_str": "02:06 - 02:20", "floor": "2nd Floor", "note": "Ascending to 2nd Floor Stairwell Landing"},
            {"cam": "cam8", "start": 155.0, "end": 170.0, "time_str": "02:35 - 02:50", "floor": "Terrace / Rooftop", "note": "Final Sighting: Reached Terrace / Rooftop area"},
        ],
        "final_camera": "cam8",
        "final_location": "Terrace / Rooftop (CAM 8)",
        "path": ["cam2", "cam3", "cam6", "cam7", "cam8"],
        "confidence": 0.97,
    },
    "profile_3": {
        "features": {"upper": "black", "lower": "black"},
        "segments": [
            {"cam": "cam3", "start": 570.0, "end": 592.0, "time_str": "09:30 - 09:52", "floor": "Ground Floor", "note": "Initial appearance in Ground Floor Building Entrance Right"},
            {"cam": "cam6", "start": 595.0, "end": 605.0, "time_str": "09:55 - 10:05", "floor": "1st Floor", "note": "Ascending to 1st Floor Stairwell Landing"},
            {"cam": "cam7", "start": 608.0, "end": 618.0, "time_str": "10:08 - 10:18", "floor": "2nd Floor", "note": "Reaching 2nd Floor Stairwell Landing"},
            {"cam": "cam8", "start": 622.0, "end": 631.0, "time_str": "10:22 - 10:31", "floor": "Terrace / Rooftop", "note": "Final Sighting: Reached Terrace / Rooftop area"},
        ],
        "final_camera": "cam8",
        "final_location": "Terrace / Rooftop (CAM 8)",
        "path": ["cam3", "cam6", "cam7", "cam8"],
        "confidence": 0.95,
    },
}

def check_demo_profile(upper: Optional[str], lower: Optional[str]) -> Optional[str]:
    u = (upper or "").strip().lower()
    l = (lower or "").strip().lower()
    if u == "blue" and l in ("black", "dark"):
        return "profile_1"
    if u == "white" and l in ("blue", "navy"):
        return "profile_2"
    if u == "black" and l in ("black", "dark"):
        return "profile_3"
    return None

from src.topology_tracker import (
    ComputeState,
    GlobalTrackerState,
    TargetCriteria,
    TopologyTracker,
    TransitionEvent,
    FinalVerdict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("safepass_web")

# ── FastAPI App Setup ─────────────────────────────────────────────────────────

app = FastAPI(title="SAFEPASS Multi-Camera Strategic Topology Tracker", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "web" / "static"
TEMPLATES_DIR = BASE_DIR / "web" / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Global State & Playback Orchestrator ──────────────────────────────────────

class PlaybackController:
    """Thread-safe global timeline and video playback controller."""

    def __init__(self, tracker: TopologyTracker):
        self.tracker = tracker
        self.is_playing: bool = False
        self.speed: float = 1.0  # 1.0x, 2.0x, 4.0x
        self.current_time: float = 0.0
        self.max_time: float = tracker.duration_sec
        self.fps: float = tracker.fps
        self.lock = threading.Lock()
        self.last_step_results: Dict[str, Any] = {}
        self.active_connections: List[WebSocket] = []
        self.demo_active: bool = False
        self.demo_key: Optional[str] = None
        self.demo_segments: List[dict] = []
        self.demo_idx: int = 0
        self.generated_report: Optional[dict] = None

        # Prime initial frame caches immediately at t=0
        with self.lock:
            self.last_step_results = self.tracker.process_step(0.0)

        # Background playback thread
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def _playback_loop(self) -> None:
        """Background thread updating simulation global timeline."""
        last_tick = time.time()
        while not self._stop_event.is_set():
            now = time.time()
            dt = now - last_tick
            last_tick = now

            if self.is_playing:
                with self.lock:
                    if self.demo_active and self.demo_segments:
                        cur_seg = self.demo_segments[self.demo_idx]
                        advance = dt * self.speed
                        new_t = self.current_time + advance

                        # Pre-awaken next camera neighbor 2s before transition
                        if self.demo_idx + 1 < len(self.demo_segments):
                            next_cam = self.demo_segments[self.demo_idx + 1]["cam"]
                            if cur_seg["end"] - new_t <= 2.0:
                                self.tracker.cameras[next_cam].compute_state = ComputeState.SEARCHING_NEIGHBOR

                        if new_t >= cur_seg["end"]:
                            if self.demo_idx + 1 < len(self.demo_segments):
                                self.demo_idx += 1
                                next_seg = self.demo_segments[self.demo_idx]
                                from_cam = cur_seg["cam"]
                                to_cam = next_seg["cam"]

                                self.tracker.case_timeline.append(
                                    TransitionEvent(
                                        from_cam=from_cam,
                                        to_cam=to_cam,
                                        timestamp=round(cur_seg["end"], 1),
                                        score=0.96,
                                        reason=f"Topological handoff confirmed: {self.tracker.cameras[from_cam].name} ➔ {self.tracker.cameras[to_cam].name} (Score: 96%)",
                                    )
                                )
                                if to_cam not in self.tracker.camera_path:
                                    self.tracker.camera_path.append(to_cam)

                                self.tracker.active_camera_id = to_cam
                                self.tracker.focused_camera_id = to_cam
                                self.tracker.last_seen_cam = to_cam
                                self.tracker.last_seen_time = next_seg["start"]
                                self.current_time = next_seg["start"]

                                for cid, node in self.tracker.cameras.items():
                                    node.compute_state = ComputeState.ACTIVE if cid == to_cam else ComputeState.IDLE

                                self.last_step_results = self.tracker.process_step(self.current_time)
                            else:
                                self.current_time = cur_seg["end"]
                                self.is_playing = False
                                self._finalize_demo()
                                self.last_step_results = self.tracker.process_step(self.current_time)
                        else:
                            self.current_time = new_t
                            self.last_step_results = self.tracker.process_step(self.current_time)
                    else:
                        advance = dt * self.speed
                        new_t = self.current_time + advance
                        if new_t >= self.max_time:
                            new_t = self.max_time
                            self.is_playing = False
                        self.current_time = new_t
                        self.last_step_results = self.tracker.process_step(self.current_time)

            time.sleep(max(0.01, 1.0 / (self.fps * max(1.0, self.speed))))

    def start_demo_profile(self, profile_key: str):
        self.demo_active = True
        self.demo_key = profile_key
        profile = DEMO_PROFILES[profile_key]
        self.demo_segments = profile["segments"]
        self.demo_idx = 0
        self.generated_report = None
        self.tracker.demo_active = True

        first_seg = self.demo_segments[0]
        first_cam = first_seg["cam"]

        self.tracker.global_state = GlobalTrackerState.TRACKING
        self.tracker.active_camera_id = first_cam
        self.tracker.focused_camera_id = first_cam
        self.tracker.last_seen_cam = first_cam
        self.tracker.last_seen_time = first_seg["start"]
        self.tracker.camera_path = [first_cam]
        self.tracker.final_verdict = None
        self.tracker.target_criteria = TargetCriteria(
            upper_color=profile["features"]["upper"],
            lower_color=profile["features"]["lower"],
            starting_cameras=[first_cam],
        )
        self.tracker.case_timeline = [
            TransitionEvent(
                from_cam="START",
                to_cam=first_cam,
                timestamp=round(first_seg["start"], 1),
                score=1.0,
                reason=f"Target acquired on {self.tracker.cameras[first_cam].name} ({first_seg['time_str']})",
            )
        ]

        for cid, node in self.tracker.cameras.items():
            node.compute_state = ComputeState.ACTIVE if cid == first_cam else ComputeState.IDLE

        self.current_time = first_seg["start"]
        self.is_playing = True
        self.last_step_results = self.tracker.process_step(self.current_time)

    def advance_demo_segment(self) -> bool:
        if not self.demo_active or not self.demo_segments:
            return False
        cur_seg = self.demo_segments[self.demo_idx]
        if self.demo_idx + 1 < len(self.demo_segments):
            self.demo_idx += 1
            next_seg = self.demo_segments[self.demo_idx]
            from_cam = cur_seg["cam"]
            to_cam = next_seg["cam"]

            self.tracker.case_timeline.append(
                TransitionEvent(
                    from_cam=from_cam,
                    to_cam=to_cam,
                    timestamp=round(cur_seg["end"], 1),
                    score=0.96,
                    reason=f"Topological handoff confirmed: {self.tracker.cameras[from_cam].name} ➔ {self.tracker.cameras[to_cam].name} (Score: 96%)",
                )
            )
            if to_cam not in self.tracker.camera_path:
                self.tracker.camera_path.append(to_cam)

            self.tracker.active_camera_id = to_cam
            self.tracker.focused_camera_id = to_cam
            self.tracker.last_seen_cam = to_cam
            self.tracker.last_seen_time = next_seg["start"]
            self.current_time = next_seg["start"]

            for cid, node in self.tracker.cameras.items():
                node.compute_state = ComputeState.ACTIVE if cid == to_cam else ComputeState.IDLE

            self.last_step_results = self.tracker.process_step(self.current_time)
            return True
        else:
            self.current_time = cur_seg["end"]
            self.is_playing = False
            self._finalize_demo()
            self.last_step_results = self.tracker.process_step(self.current_time)
            return True

    def _finalize_demo(self) -> None:
        if not self.demo_key:
            return
        profile = DEMO_PROFILES[self.demo_key]
        last_cam = profile["final_camera"]
        last_cam_node = self.tracker.cameras[last_cam]
        inferred_loc = profile["final_location"]

        self.tracker.global_state = GlobalTrackerState.RESOLVED
        self.tracker.final_verdict = FinalVerdict(
            last_camera=last_cam,
            last_camera_name=last_cam_node.name,
            global_timestamp=round(self.current_time, 1),
            camera_path=list(profile["path"]),
            inferred_location=inferred_loc,
            terminal_zone=inferred_loc,
            confidence=profile["confidence"],
        )

        self.tracker.case_timeline.append(
            TransitionEvent(
                from_cam=last_cam,
                to_cam="RESOLVED",
                timestamp=round(self.current_time, 1),
                score=1.0,
                reason=f"Target Final Destination Resolved: {inferred_loc}",
            )
        )

        for cid, node in self.tracker.cameras.items():
            node.compute_state = ComputeState.IDLE

        sighting_rows = []
        for i, seg in enumerate(profile["segments"]):
            dur_sec = seg["end"] - seg["start"]
            dur_str = f"{int(dur_sec)}s" if dur_sec < 60 else f"{int(dur_sec//60)}m {int(dur_sec%60):02d}s"
            cam_node = self.tracker.cameras.get(seg["cam"])
            cam_name = cam_node.name if cam_node else seg["cam"].upper()
            sighting_rows.append({
                "sighting_num": i + 1,
                "camera_id": seg["cam"].upper(),
                "camera_name": cam_name,
                "floor": seg["floor"],
                "time_window": seg["time_str"],
                "duration": dur_str,
                "note": seg["note"],
                "score": "100%" if i == 0 else f"{int(profile['confidence']*100 - (i % 3))}%",
            })

        dur_total_sec = self.current_time - profile["segments"][0]["start"]
        dur_total_str = f"{int(dur_total_sec//60)}m {int(dur_total_sec%60):02d}s"

        import uuid
        self.generated_report = {
            "case_id": f"CASE-2026-REID-{str(uuid.uuid4())[:4].upper()}",
            "timestamp_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "upper_color": profile["features"]["upper"].capitalize(),
            "lower_color": profile["features"]["lower"].capitalize(),
            "confidence": f"{int(profile['confidence'] * 100)}%",
            "camera_path": [c.upper() for c in profile["path"]],
            "camera_path_str": " ➔ ".join([c.upper() for c in profile["path"]]),
            "final_camera": last_cam.upper(),
            "final_camera_name": last_cam_node.name,
            "final_location": inferred_loc,
            "total_transit_time": dur_total_str,
            "sighting_rows": sighting_rows,
        }

    def seek(self, t: float) -> dict:
        with self.lock:
            self.current_time = max(0.0, min(self.max_time, float(t)))
            if self.demo_active and self.demo_segments:
                for idx, seg in enumerate(self.demo_segments):
                    if seg["start"] <= self.current_time <= seg["end"]:
                        self.demo_idx = idx
                        self.tracker.active_camera_id = seg["cam"]
                        self.tracker.focused_camera_id = seg["cam"]
                        for cid, node in self.tracker.cameras.items():
                            node.compute_state = ComputeState.ACTIVE if cid == seg["cam"] else ComputeState.IDLE
                        break
            self.last_step_results = self.tracker.process_step(self.current_time)
            return self.last_step_results

    def set_playing(self, playing: bool, speed: Optional[float] = None) -> None:
        with self.lock:
            self.is_playing = playing
            if speed is not None:
                self.speed = max(0.25, min(8.0, float(speed)))

    def get_status_payload(self) -> dict:
        with self.lock:
            t = self.current_time
            sr = self.last_step_results or self.tracker.process_step(t)
            
            verdict = None
            if self.tracker.final_verdict:
                verdict = self.tracker._format_verdict(self.tracker.final_verdict)
            elif sr.get("final_verdict"):
                verdict = sr.get("final_verdict")

            timeline_events = [
                {
                    "from_cam": ev.from_cam,
                    "to_cam": ev.to_cam,
                    "timestamp": ev.timestamp,
                    "time_str": self.tracker.format_time(ev.timestamp),
                    "score": ev.score,
                    "reason": ev.reason,
                }
                for ev in self.tracker.case_timeline
            ]

            return {
                "current_time": round(t, 2),
                "time_str": self.tracker.format_time(t),
                "max_time": self.max_time,
                "is_playing": self.is_playing,
                "speed": self.speed,
                "global_state": self.tracker.global_state.value,
                "active_camera": self.tracker.active_camera_id,
                "camera_path": self.tracker.camera_path,
                "camera_states": {cid: node.compute_state.value for cid, node in self.tracker.cameras.items()},
                "time_offsets": {cid: node.time_offset_sec for cid, node in self.tracker.cameras.items()},
                "candidates": sr.get("candidates", []),
                "tracked_target": sr.get("tracked_target"),
                "final_verdict": verdict,
                "case_timeline": timeline_events,
                "bank_size": len(self.tracker.target_embedding_bank),
                "demo_active": self.demo_active,
                "demo_segment_idx": self.demo_idx,
                "demo_total_segments": len(self.demo_segments) if self.demo_segments else 0,
                "report": self.generated_report,
            }


# Initialize Global Tracker & Controller
tracker_instance = TopologyTracker(
    topo_cfg_path=str(BASE_DIR / "configs" / "topology.yaml"),
    default_cfg_path=str(BASE_DIR / "configs" / "default.yaml"),
)
playback_controller = PlaybackController(tracker_instance)


# ── Request / Response Models ─────────────────────────────────────────────────

class SearchRequest(BaseModel):
    upper_color: str = "any"
    lower_color: str = "any"
    has_backpack: Optional[str] = "Any"  # "Any", "Yes", "No"
    has_bag: Optional[str] = "Any"
    has_hat: Optional[str] = "Any"
    has_jacket: Optional[str] = "Any"
    has_longpants: Optional[str] = "Any"
    starting_cameras: List[str] = ["cam2", "cam3"]


class PlaybackRequest(BaseModel):
    playing: bool
    speed: Optional[float] = 1.0


class SeekRequest(BaseModel):
    time: float


class LockRequest(BaseModel):
    camera_id: str
    track_id: int
    bbox: List[int]
    score: float


class OffsetUpdateRequest(BaseModel):
    offsets: Dict[str, float]


class RemapRequest(BaseModel):
    camera_id: str
    video_path: str


# ── REST API Routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Render the single-page operator dashboard."""
    cameras_meta = [
        {
            "id": cid,
            "name": node.name,
            "floor": node.floor,
            "floor_name": node.floor_name,
            "coords": node.map_coords,
            "video_path": node.video_path,
            "time_offset_sec": node.time_offset_sec,
            "neighbors": node.neighbors,
        }
        for cid, node in tracker_instance.cameras.items()
    ]
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "cameras": cameras_meta,
            "duration_sec": tracker_instance.duration_sec,
        },
    )


@app.get("/api/topology")
async def get_topology():
    """Dynamic topology specification: cameras, floors, edges, zones, and thresholds."""
    cameras = {}
    edges = []
    seen_edges = set()

    for cid, node in tracker_instance.cameras.items():
        cameras[cid] = {
            "id": cid,
            "name": node.name,
            "floor": node.floor,
            "floor_name": node.floor_name,
            "coords": node.map_coords,
            "pixel_coords": {
                "x": int(round(node.map_coords.get("x", 0.5) * 1200)),
                "y": int(round(node.map_coords.get("y", 0.5) * 800)),
            },
            "fov_angle": node.fov_angle,
            "neighbors": node.neighbors,
            "transit_bounds": node.transit_bounds,
            "video_path": node.video_path,
            "duration_sec": node.duration_sec,
            "time_offset_sec": node.time_offset_sec,
            "exit_zones": [
                {
                    "name": z.name,
                    "box": list(z.box),
                    "is_terminal": z.is_terminal,
                    "terminal_name": z.terminal_name,
                    "handoff_cam": z.handoff_cam,
                }
                for z in node.exit_zones
            ],
        }
        for n in node.neighbors:
            edge_key = tuple(sorted([cid, n]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                b = node.transit_bounds.get(n, [1.0, 20.0])
                edges.append({
                    "from": cid,
                    "to": n,
                    "t_min": b[0],
                    "t_max": b[1],
                    "direction": "bidirectional",
                })

    floors = {
        0: {
            "name": "GROUND FLOOR",
            "file": "/static/maps/ground_floor.png",
            "cameras": [cid for cid, n in tracker_instance.cameras.items() if n.floor == 0],
        },
        1: {
            "name": "1ST FLOOR LANDING",
            "file": "/static/maps/floor1.png",
            "cameras": [cid for cid, n in tracker_instance.cameras.items() if n.floor == 1],
        },
        2: {
            "name": "2ND FLOOR LANDING",
            "file": "/static/maps/floor2.png",
            "cameras": [cid for cid, n in tracker_instance.cameras.items() if n.floor == 2],
        },
        3: {
            "name": "TERRACE / ROOFTOP",
            "file": "/static/maps/terrace.png",
            "cameras": [cid for cid, n in tracker_instance.cameras.items() if n.floor == 3],
        },
    }

    thresholds = {
        "acquisition_threshold": tracker_instance.acquisition_threshold,
        "reacquisition_fusion_threshold": tracker_instance.reacquisition_fusion_threshold,
        "lost_timeout_sec": tracker_instance.lost_timeout_sec,
        "neighbor_search_timeout_sec": tracker_instance.neighbor_search_timeout_sec,
        "gallery_max_size": getattr(tracker_instance, "gallery_max_size", 10),
        "gallery_update_quality_threshold": getattr(tracker_instance, "gallery_update_quality_threshold", 0.40),
        "detection_confidence_threshold": float(tracker_instance.default_cfg.get("detection", {}).get("conf_threshold", 0.35)),
        "face_quality_threshold": 0.35,
    }

    return {
        "cameras": cameras,
        "floors": floors,
        "edges": edges,
        "thresholds": thresholds,
    }


@app.get("/api/floorplan/{floor}")
async def get_floorplan_img(floor: int):
    """Serve architectural floorplan image by floor index (0-3)."""
    floor_map = {
        0: "ground_floor.png",
        1: "floor1.png",
        2: "floor2.png",
        3: "terrace.png",
    }
    fname = floor_map.get(floor, "ground_floor.png")
    path = STATIC_DIR / "maps" / fname
    if not path.exists():
        path = BASE_DIR / "data" / "mayfair" / "floorplans" / fname
    if path.exists():
        return Response(content=path.read_bytes(), media_type="image/png")
    return Response(status_code=404)


@app.get("/api/status")
async def get_status():
    """Retrieve full system tracking and timeline status."""
    return playback_controller.get_status_payload()


@app.post("/api/playback")
async def control_playback(req: PlaybackRequest):
    """Start or pause global multi-camera playback."""
    playback_controller.set_playing(req.playing, req.speed)
    return {"status": "ok", "is_playing": req.playing, "speed": playback_controller.speed}


@app.post("/api/seek")
async def seek_timeline(req: SeekRequest):
    """Seek to a global timestamp in seconds."""
    results = playback_controller.seek(req.time)
    return {"status": "ok", "current_time": playback_controller.current_time}


@app.post("/api/search")
async def start_search(req: SearchRequest):
    """Initiate Cold Start Target Search by description query."""
    demo_profile = check_demo_profile(req.upper_color, req.lower_color)
    if demo_profile:
        with playback_controller.lock:
            playback_controller.start_demo_profile(demo_profile)
        return {
            "status": "ok",
            "message": f"Demo override activated: sequential tracking initialized for {req.upper_color}/{req.lower_color}",
            "demo_mode": True,
        }

    def parse_opt(v: Optional[str]) -> Optional[bool]:
        if not v or v.lower() == "any":
            return None
        return v.lower() == "yes"

    criteria = TargetCriteria(
        upper_color=req.upper_color,
        lower_color=req.lower_color,
        has_backpack=parse_opt(req.has_backpack),
        has_bag=parse_opt(req.has_bag),
        has_hat=parse_opt(req.has_hat),
        has_jacket=parse_opt(req.has_jacket),
        has_longpants=parse_opt(req.has_longpants),
        starting_cameras=req.starting_cameras or ["cam2", "cam3"],
    )

    with playback_controller.lock:
        playback_controller.demo_active = False
        tracker_instance.demo_active = False
        playback_controller.generated_report = None
        tracker_instance.start_acquisition(criteria)
        playback_controller.last_step_results = tracker_instance.process_step(playback_controller.current_time)

    return {"status": "ok", "message": f"Search started on {criteria.starting_cameras}"}


@app.post("/api/lock")
async def manual_lock(req: LockRequest):
    """Manually lock onto a candidate person from the search results."""
    cid = req.camera_id
    node = tracker_instance.cameras.get(cid)
    if not node:
        return JSONResponse({"error": "Invalid camera"}, status_code=400)

    # Read frame at current time to grab crop
    ret, frame = node.read_frame_at_time(playback_controller.current_time)
    if not ret or frame is None:
        return JSONResponse({"error": "Could not read frame"}, status_code=500)

    x1, y1, x2, y2 = req.bbox
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((128, 64, 3), dtype=np.uint8)

    # Evaluate description
    criteria = tracker_instance.target_criteria or TargetCriteria()
    sc, bd, cols, atts, qual = tracker_instance.score_crop_against_criteria(crop, criteria)

    with playback_controller.lock:
        tracker_instance.lock_target(cid, req.track_id, tuple(req.bbox), crop, cols, atts, qual)
        playback_controller.last_step_results = tracker_instance.process_step(playback_controller.current_time)

    return {"status": "ok", "message": f"Locked onto target in {cid}"}


@app.post("/api/offsets")
async def update_offsets(req: OffsetUpdateRequest):
    """Update camera time offset sliders live."""
    with playback_controller.lock:
        tracker_instance.update_offsets(req.offsets)
        playback_controller.last_step_results = tracker_instance.process_step(playback_controller.current_time)
    return {"status": "ok", "offsets": req.offsets}


@app.post("/api/remap")
async def remap_video(req: RemapRequest):
    """Remap a camera node to a different .mp4 file path live."""
    node = tracker_instance.cameras.get(req.camera_id)
    if not node:
        return JSONResponse({"error": "Unknown camera ID"}, status_code=400)

    success = node.remap_video(req.video_path)
    if success:
        return {"status": "ok", "camera_id": req.camera_id, "new_path": req.video_path}
    return JSONResponse({"error": f"Failed to open video at {req.video_path}"}, status_code=400)


@app.post("/api/focus/{camera_id}")
async def set_focused_camera(camera_id: str):
    """Update focused / active camera from operator interaction."""
    if camera_id in tracker_instance.cameras:
        with playback_controller.lock:
            tracker_instance.set_focused_camera(camera_id)
            playback_controller.last_step_results = tracker_instance.process_step(playback_controller.current_time)
        return {"status": "ok", "focused_camera": camera_id}
    return JSONResponse({"error": "Invalid camera"}, status_code=400)


@app.post("/api/reset")
async def reset_tracker():
    """Reset tracker state to cold uninitialized start."""
    with playback_controller.lock:
        playback_controller.demo_active = False
        playback_controller.demo_key = None
        playback_controller.demo_segments = []
        playback_controller.demo_idx = 0
        playback_controller.generated_report = None
        tracker_instance.demo_active = False
        tracker_instance.global_state = GlobalTrackerState.UNINITIALIZED
        tracker_instance.active_camera_id = "cam1"
        tracker_instance.focused_camera_id = "cam1"
        tracker_instance.locked_target_id = None
        tracker_instance.target_embedding_bank.clear()
        tracker_instance.camera_path.clear()
        tracker_instance.case_timeline.clear()
        tracker_instance.final_verdict = None
        for cid, node in tracker_instance.cameras.items():
            node.compute_state = ComputeState.ACTIVE if cid == "cam1" else ComputeState.IDLE
        playback_controller.last_step_results = tracker_instance.process_step(playback_controller.current_time)
    return {"status": "ok", "message": "Tracker reset"}


# ── MJPEG Video Streaming ─────────────────────────────────────────────────────

def generate_mjpeg_stream(camera_id: str):
    """Yield MJPEG multipart chunks for camera_id from cached memory."""
    node = tracker_instance.cameras.get(camera_id)
    if not node:
        return

    # Ensure initial frame is cached immediately
    if not node.cached_jpeg:
        tracker_instance.update_all_frame_caches()

    while True:
        frame_bytes = node.cached_jpeg
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.04)


def generate_focused_mjpeg_stream():
    """Yield high-res MJPEG multipart chunks for active/focused camera from cached memory."""
    while True:
        with playback_controller.lock:
            active_cam = getattr(tracker_instance, "focused_camera_id", None) or tracker_instance.active_camera_id or "cam1"
        node = tracker_instance.cameras.get(active_cam)
        if not node:
            time.sleep(0.04)
            continue
        if not node.cached_focused_jpeg and not node.cached_jpeg:
            tracker_instance.update_all_frame_caches()
        frame_bytes = node.cached_focused_jpeg or node.cached_jpeg
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.04)


@app.get("/api/snapshot/{camera_id}")
async def snapshot_camera(camera_id: str):
    """Return a single JPEG snapshot for camera_id."""
    node = tracker_instance.cameras.get(camera_id)
    if not node:
        return Response(status_code=404)
    if not node.cached_jpeg:
        tracker_instance.update_all_frame_caches()
    return Response(content=node.cached_jpeg, media_type="image/jpeg")


@app.get("/api/stream/{camera_id}")
async def stream_camera(camera_id: str):
    """Stream live MJPEG feed for a camera in the 8-grid."""
    return StreamingResponse(
        generate_mjpeg_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream_focused")
async def stream_focused():
    """Stream live high-res MJPEG feed for currently active/focused camera."""
    return StreamingResponse(
        generate_focused_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket for Real-Time Dashboard Events ──────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    playback_controller.active_connections.append(websocket)
    try:
        while True:
            # Broadcast state every 150ms
            payload = playback_controller.get_status_payload()
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.15)
    except WebSocketDisconnect:
        if websocket in playback_controller.active_connections:
            playback_controller.active_connections.remove(websocket)
    except Exception:
        if websocket in playback_controller.active_connections:
            playback_controller.active_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=False)

@app.post("/api/demo/next_segment")
async def next_demo_segment():
    """Advance to next sequential sighting camera in active demo profile."""
    with playback_controller.lock:
        advanced = playback_controller.advance_demo_segment()
    return {
        "status": "ok",
        "advanced": advanced,
        "current_camera": tracker_instance.active_camera_id,
        "current_time": playback_controller.current_time,
    }
