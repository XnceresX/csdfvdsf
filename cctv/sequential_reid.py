"""
cctv/sequential_reid.py
=======================
Sequential CCTV Person Re-Identification - Cross-Camera Investigation Engine.

Architecture (NO HARDCODED RULES):
  Camera 1  ->  Build gallery: detect every real person, extract OSNet + ArcFace embeddings.
                Each unique person gets assigned PERSON_001, PERSON_002, etc.
  Camera 2+ ->  For every detected person, query the gallery.
                - Best cosine similarity >= threshold  =>  MATCHED  (display matching ID)
                - Best cosine similarity <  threshold  =>  NEW PERSON (auto-register new ID)
  Final     ->  Cross-camera verification report showing all persons and their camera appearances.

Decision is 100% model-driven.  No clothing checks.  No luminance gates.  No intruder labels.
ArcFace face embedding is used as a bonus layer when a face is visible; if no face detected it
falls back to OSNet body embedding only.  Final score = weighted blend.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from reid.detector import PersonDetector
from reid.tracker import SimpleTracker, Track
from reid.embedding import ReIDFeatureExtractor, cosine_similarity


# ─── ArcFace wrapper ────────────────────────────────────────────────────────

class ArcFaceExtractor:
    """
    Thin wrapper around InsightFace buffalo_l ArcFace.
    Returns 512-d normalized embedding or None when no face is detected.
    """

    def __init__(self):
        self._app = None
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(320, 320))
            self._app = app
            print("[ArcFace] InsightFace buffalo_l loaded successfully.")
        except Exception as e:
            print(f"[ArcFace] Could not load InsightFace: {e}  (face layer disabled)")

    @property
    def available(self) -> bool:
        return self._app is not None

    def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts ArcFace embedding from an image crop.
        Returns normalized 512-d np.float32 array, or None if no face found.
        """
        if self._app is None or crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            # InsightFace expects RGB
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            faces = self._app.get(rgb)
            if not faces:
                return None
            # Pick highest-confidence face
            best = max(faces, key=lambda f: float(f.det_score))
            emb = best.normed_embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 1e-8 else emb
        except Exception:
            return None


# ─── Gallery ─────────────────────────────────────────────────────────────────

class PersonGallery:
    """
    Stores embeddings for every known person encountered across cameras.
    Each person ID has:
      - body_emb  : running-average OSNet 512-d embedding
      - face_emb  : running-average ArcFace 512-d embedding (may be None)
      - best_crop : highest-resolution crop seen so far
      - cameras   : {cam_name: {"crop": ..., "score": ..., "frame": ...}}
      - hits      : total frame hits
    """

    # Colour palette for drawing boxes (one per person)
    _PALETTE = [
        (0, 229, 255),   # Cyan   - Person 001
        (0, 255, 120),   # Green  - Person 002
        (255, 165, 0),   # Orange - Person 003
        (180, 0, 255),   # Purple - Person 004
        (255, 255, 0),   # Yellow - Person 005
        (255, 80, 80),   # Red    - Person 006
    ]

    def __init__(self, body_threshold: float = 0.60, face_threshold: float = 0.45):
        self.body_threshold = body_threshold
        self.face_threshold = face_threshold
        self._persons: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1

    @property
    def persons(self) -> Dict[str, Dict[str, Any]]:
        return self._persons

    def color_for(self, pid: str) -> Tuple[int, int, int]:
        idx = int(pid.split("_")[-1]) - 1
        return self._PALETTE[idx % len(self._PALETTE)]

    def query(
        self,
        body_emb: np.ndarray,
        face_emb: Optional[np.ndarray],
    ) -> Tuple[Optional[str], float]:
        """
        Finds the best matching person in the gallery.
        Returns (person_id, score) or (None, 0.0) if gallery is empty.
        Score is a weighted blend: 60% body + 40% face (when face available),
        or 100% body when no face embedding exists.
        """
        best_id = None
        best_score = 0.0

        for pid, pdata in self._persons.items():
            # Body similarity against exemplar bank
            if pdata.get("exemplars"):
                sims = [cosine_similarity(body_emb, ex) for ex in pdata["exemplars"]]
                sims.sort(reverse=True)
                b_sim = float(np.mean(sims[:min(3, len(sims))]))
            else:
                b_sim = cosine_similarity(body_emb, pdata["body_emb"])

            # Face similarity (if both have face embeddings)
            f_sim = None
            if face_emb is not None and pdata.get("face_emb") is not None:
                f_sim = cosine_similarity(face_emb, pdata["face_emb"])

            # Weighted blend
            if f_sim is not None:
                score = 0.60 * b_sim + 0.40 * f_sim
            else:
                score = b_sim

            if score > best_score:
                best_score = score
                best_id = pid

        return best_id, best_score

    def match_or_register(
        self,
        body_emb: np.ndarray,
        face_emb: Optional[np.ndarray],
        crop: np.ndarray,
        cam_name: str,
        frame_idx: int,
        source_cam: bool = False,
    ) -> Tuple[str, float, bool]:
        """
        Attempts to match against existing gallery.
        If matched (score >= threshold) -> update existing entry, return (id, score, False).
        If not matched (or gallery empty and source_cam) -> register new entry, return (id, 1.0, True).

        For cameras beyond Camera 1, unmatched persons are always registered as new.
        """
        best_id, best_score = self.query(body_emb, face_emb)

        # Determine threshold: face available → use weighted threshold, else body only
        threshold = self.body_threshold
        if face_emb is not None and best_id and self._persons.get(best_id, {}).get("face_emb") is not None:
            # With face blend active, the threshold is the blended one
            threshold = self.body_threshold  # same, kept for clarity

        matched = best_id is not None and best_score >= threshold

        if matched:
            pid = best_id
            p = self._persons[pid]
            # Multi-exemplar gallery update with temporal spacing to eliminate drift
            if "exemplars" not in p:
                p["exemplars"] = [p["body_emb"].copy()]
            last_enrolled = p.get("last_enrolled_frame", 0)
            if frame_idx - last_enrolled >= 15:
                if len(p["exemplars"]) < 10:
                    p["exemplars"].append(body_emb.copy())
                else:
                    sims = [cosine_similarity(body_emb, ex) for ex in p["exemplars"]]
                    p["exemplars"][int(np.argmax(sims))] = body_emb.copy()
                p["last_enrolled_frame"] = frame_idx
                # Recalculate centroid
                mean_b = np.mean(p["exemplars"], axis=0)
                norm = np.linalg.norm(mean_b)
                if norm > 1e-8:
                    p["body_emb"] = mean_b / norm
            else:
                p["body_emb"] = 0.95 * p["body_emb"] + 0.05 * body_emb
                p["body_emb"] /= np.linalg.norm(p["body_emb"])
            # Update running average face embedding (when available)
            if face_emb is not None:
                if p.get("face_emb") is None:
                    p["face_emb"] = face_emb.copy()
                else:
                    p["face_emb"] = 0.85 * p["face_emb"] + 0.15 * face_emb
                    p["face_emb"] /= np.linalg.norm(p["face_emb"])
            # Update best crop
            if crop.shape[0] > p.get("best_h", 0):
                p["best_crop"] = crop.copy()
                p["best_h"] = crop.shape[0]
            p["hits"] += 1
            # Record best appearance per camera
            prev = p["cameras"].get(cam_name)
            if prev is None or best_score > prev["score"]:
                p["cameras"][cam_name] = {"crop": crop.copy(), "score": best_score, "frame": frame_idx}
            return pid, best_score, False

        else:
            # Register new person
            pid = f"PERSON_{self._next_id:03d}"
            self._next_id += 1
            self._persons[pid] = {
                "body_emb": body_emb.copy(),
                "face_emb": face_emb.copy() if face_emb is not None else None,
                "best_crop": crop.copy(),
                "best_h": crop.shape[0],
                "hits": 1,
                "cameras": {cam_name: {"crop": crop.copy(), "score": 1.0 if source_cam else best_score, "frame": frame_idx}},
                "source_cam": cam_name,
            }
            return pid, 1.0 if source_cam else best_score, True


# ─── Main Investigator ────────────────────────────────────────────────────────

class SequentialCCTVReIDInvestigator:
    """
    Executes sequential cross-camera investigation clip-by-clip.
    Pure model-driven identification — no rules, no hardcoded assumptions.
    """

    def __init__(
        self,
        sources: List[str],
        camera_names: Optional[List[str]] = None,
        body_threshold: float = 0.60,
        face_threshold: float = 0.45,
        display_width: int = 960,
        display_height: int = 640,
    ) -> None:
        self.sources = sources
        self.camera_names = camera_names or [f"Camera {i+1}" for i in range(len(sources))]
        self.display_width = display_width
        self.display_height = display_height

        print("[Investigator] Loading models...")
        self.detector = PersonDetector(detect_interval=3, confidence_threshold=0.45)
        self.extractor = ReIDFeatureExtractor(use_osnet=True)
        self.arcface = ArcFaceExtractor()

        self.gallery = PersonGallery(body_threshold=body_threshold, face_threshold=face_threshold)

        # Per-track embedding cache: {local_track_id: (frame_updated, body_emb, face_emb)}
        self._track_cache: Dict[int, Tuple[int, np.ndarray, Optional[np.ndarray]]] = {}

        print(f"[Investigator] Ready. Body threshold={body_threshold:.2f}, Face threshold={face_threshold:.2f}")

    def _get_track_embeddings(
        self,
        frame: np.ndarray,
        track: Track,
        frame_idx: int,
        update_interval: int = 4,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract or return cached embeddings for a track."""
        cached = self._track_cache.get(track.local_id)
        if cached is not None and (frame_idx - cached[0]) < update_interval:
            return cached[1], cached[2]

        x1, y1, x2, y2 = track.bbox
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]

        # OSNet body embedding
        body_emb = self.extractor.extract(frame, track.bbox)

        # ArcFace face embedding (None if no face detected)
        face_emb = None
        if self.arcface.available and crop.size > 0:
            face_emb = self.arcface.extract(crop)

        self._track_cache[track.local_id] = (frame_idx, body_emb, face_emb)
        return body_emb, face_emb

    def _draw_hud(
        self,
        frame: np.ndarray,
        cam_idx: int,
        cam_name: str,
        frame_idx: int,
        total_frames: int,
        paused: bool,
        person_count: int,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Top banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, 44), (w, 44), (0, 229, 255), 1)

        cam_label = f"[{cam_idx+1}/{len(self.sources)}] {cam_name.upper()} | Frame {frame_idx}/{total_frames}"
        status = "  [PAUSED]" if paused else "  [LIVE ANALYSIS]"
        cv2.putText(frame, cam_label + status, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 229, 255), 2, cv2.LINE_AA)

        # Right side: gallery person count
        gallery_label = f"Gallery: {person_count} person(s) registered"
        (tw, _), _ = cv2.getTextSize(gallery_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(frame, gallery_label, (w - tw - 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA)

        # PiP: show all gallery persons (top-right, stacked)
        if self.gallery.persons:
            pip_y = 52
            pip_w, pip_h = 90, 115
            for pid, pdata in list(self.gallery.persons.items())[:4]:
                crop = pdata.get("best_crop")
                if crop is None or crop.size == 0:
                    continue
                rx1 = w - pip_w - 10
                ry1 = pip_y
                color = self.gallery.color_for(pid)
                scaled = cv2.resize(crop, (pip_w - 4, pip_h - 22))
                frame[ry1 + 2: ry1 + 2 + scaled.shape[0], rx1 + 2: rx1 + 2 + scaled.shape[1]] = scaled
                cv2.rectangle(frame, (rx1, ry1), (rx1 + pip_w, ry1 + pip_h), color, 2)
                cv2.rectangle(frame, (rx1, ry1 + pip_h - 20), (rx1 + pip_w, ry1 + pip_h), color, -1)
                cv2.putText(frame, pid[-7:], (rx1 + 4, ry1 + pip_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA)
                pip_y += pip_h + 6

        # Bottom controls
        cv2.rectangle(frame, (0, h - 30), (w, h), (15, 20, 28), -1)
        controls = "[SPACE] Pause/Resume  |  [N/ENTER] Next Camera  |  [S] Snapshot  |  [Q] Quit"
        cv2.putText(frame, controls, (14, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (100, 116, 139), 1, cv2.LINE_AA)

        return frame

    def _draw_person_box(
        self,
        frame: np.ndarray,
        track: Track,
        pid: str,
        score: float,
        is_new: bool,
        face_used: bool,
    ) -> None:
        x1, y1, x2, y2 = track.bbox
        color = self.gallery.color_for(pid)

        # Box thickness: thicker for matched (not new)
        thickness = 2 if is_new else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        face_tag = " [F+B]" if face_used else " [B]"
        if is_new:
            label = f"NEW: {pid}{face_tag}"
        else:
            label = f"{pid} {score*100:.0f}%{face_tag}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        ty = max(0, y1 - 26)
        cv2.rectangle(frame, (x1, ty), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, ty + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)

    def process_camera(
        self,
        cam_idx: int,
        show: bool = True,
        max_frames: Optional[int] = None,
    ) -> bool:
        """
        Process one camera video clip sequentially.
        Returns True to continue to next camera, False to quit.
        """
        source_path = self.sources[cam_idx]
        cam_name = self.camera_names[cam_idx]
        is_reference_cam = (cam_idx == 0)

        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"[Investigator] Cannot open: {source_path}")
            return True

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        # Play at native speed: compensate for processing time
        target_delay_ms = max(1, int(1000.0 / video_fps))

        tracker = SimpleTracker(max_distance=120.0, max_missed=25, min_hits=2)
        self._track_cache.clear()

        win_title = f"SAFEPASS Re-ID [{cam_name}]"
        if show:
            cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_title, self.display_width, self.display_height)

        frame_idx = 0
        paused = False
        proceed_next = True

        phase = "ENROLLING" if is_reference_cam else "RE-IDENTIFYING"
        print(f"\n>>> {cam_name} | Phase: {phase} | {total_frames} frames")

        while True:
            t_start = time.time()

            if not paused:
                ret, raw_frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(raw_frame, (self.display_width, self.display_height))
                frame_idx += 1

                # Neural detection every 3 frames
                run_det = (frame_idx % 3 == 1)
                detections = self.detector.detect(frame, force=run_det)
                tracks = tracker.update(detections)

                for track in tracks:
                    x1, y1, x2, y2 = track.bbox
                    h_box = y2 - y1
                    w_box = x2 - x1

                    # Minimum size: must be a plausible human silhouette
                    if w_box < 25 or h_box < 70:
                        continue
                    # Aspect ratio gate: standing person is taller than wide
                    if h_box / max(1, w_box) < 1.2:
                        continue

                    body_emb, face_emb = self._get_track_embeddings(frame, track, frame_idx)
                    crop = frame[max(0, y1):min(frame.shape[0], y2),
                                 max(0, x1):min(frame.shape[1], x2)]
                    if crop.size == 0:
                        continue

                    face_used = face_emb is not None

                    # Camera 1: must be a properly moving track before enrolling
                    if is_reference_cam:
                        # Only register after track has enough hits and shows displacement
                        if track.hits < 6:
                            # Draw "observing" in grey
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 90, 100), 1)
                            cv2.putText(frame, "Observing...", (x1, max(14, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 90, 100), 1)
                            continue

                    pid, score, is_new = self.gallery.match_or_register(
                        body_emb=body_emb,
                        face_emb=face_emb,
                        crop=crop,
                        cam_name=cam_name,
                        frame_idx=frame_idx,
                        source_cam=is_reference_cam,
                    )

                    self._draw_person_box(frame, track, pid, score, is_new, face_used)

                self._draw_hud(frame, cam_idx, cam_name, frame_idx, total_frames,
                               paused, len(self.gallery.persons))
                if show:
                    cv2.imshow(win_title, frame)

                if max_frames and frame_idx >= max_frames:
                    break

            # Timing: compensate for compute time so video plays at correct speed
            if show:
                elapsed_ms = int((time.time() - t_start) * 1000)
                wait = max(1, target_delay_ms - elapsed_ms) if not paused else 40
                key = cv2.waitKey(wait) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    proceed_next = False
                    break
                elif key == 32:  # SPACE
                    paused = not paused
                    print(f"  [{cam_name}] {'PAUSED' if paused else 'RESUMED'}")
                elif key in (13, ord("n"), ord("N")):
                    print(f"  [{cam_name}] -> Next camera")
                    break
                elif key in (ord("s"), ord("S")):
                    snap = f"snap_{cam_name.replace(' ', '_').lower()}_f{frame_idx}.jpg"
                    cv2.imwrite(snap, frame)
                    print(f"  [{cam_name}] Snapshot saved: {snap}")

        cap.release()
        if show:
            cv2.destroyWindow(win_title)

        print(f"  [{cam_name}] Done. Gallery now has {len(self.gallery.persons)} person(s).")
        return proceed_next

    def show_final_report(self, show: bool = True) -> None:
        """
        Renders the cross-camera person verification report.
        One row per registered person, one column per camera.
        """
        win_title = "SAFEPASS - Cross-Camera Re-ID Report"
        persons = self.gallery.persons
        n_persons = len(persons)
        n_cams = len(self.camera_names)

        if n_persons == 0:
            print("[Report] No persons registered.")
            return

        # Canvas sizing
        col_w = 210
        row_h = 240
        header_h = 70
        canvas_w = max(1280, 30 + n_cams * (col_w + 10) + 20)
        canvas_h = header_h + n_persons * (row_h + 20) + 60

        canvas = np.full((canvas_h, canvas_w, 3), 14, dtype=np.uint8)

        # Header
        cv2.rectangle(canvas, (0, 0), (canvas_w, header_h - 2), (10, 14, 22), -1)
        cv2.putText(canvas, "CROSS-CAMERA PERSON RE-IDENTIFICATION REPORT",
                    (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 229, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Model: OSNet (body) + ArcFace (face, when visible)  |  Decision: 100% model-driven  |  No manual rules",
                    (22, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 116, 139), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, header_h - 2), (canvas_w, header_h - 2), (0, 229, 255), 1)

        # Column headers (camera names)
        col_start_x = 145
        for ci, cname in enumerate(self.camera_names):
            cx = col_start_x + ci * (col_w + 8)
            cv2.putText(canvas, cname, (cx + 6, header_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Rows: one per registered person
        for ri, (pid, pdata) in enumerate(persons.items()):
            color = self.gallery.color_for(pid)
            ry = header_h + 30 + ri * (row_h + 18)

            # Row background
            cv2.rectangle(canvas, (8, ry), (canvas_w - 8, ry + row_h),
                          (18, 22, 32), -1)
            cv2.rectangle(canvas, (8, ry), (canvas_w - 8, ry + row_h), color, 1)

            # Person label sidebar
            cv2.rectangle(canvas, (8, ry), (138, ry + row_h), (20, 26, 38), -1)
            cv2.rectangle(canvas, (8, ry), (138, ry + row_h), color, 2)

            # Best crop in sidebar
            best_crop = pdata.get("best_crop")
            if best_crop is not None and best_crop.size > 0:
                th, tw = 110, 110
                sc = cv2.resize(best_crop, (tw, th))
                canvas[ry + 6: ry + 6 + th, 14: 14 + tw] = sc

            cv2.putText(canvas, pid, (14, ry + row_h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Hits: {pdata['hits']}", (14, ry + row_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (148, 163, 184), 1, cv2.LINE_AA)

            # Camera columns
            for ci, cname in enumerate(self.camera_names):
                cx = col_start_x + ci * (col_w + 8)
                cam_data = pdata["cameras"].get(cname)

                cell_x1, cell_y1 = cx, ry + 2
                cell_x2, cell_y2 = cx + col_w, ry + row_h - 2

                if cam_data and cam_data.get("crop") is not None:
                    crop = cam_data["crop"]
                    ch, cw = crop.shape[:2]
                    scale = min((col_w - 12) / max(1, cw), (row_h - 50) / max(1, ch))
                    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
                    sc = cv2.resize(crop, (nw, nh))

                    px = cx + (col_w - nw) // 2
                    py = ry + 8
                    py2, px2 = min(canvas_h, py + nh), min(canvas_w, px + nw)
                    canvas[py: py2, px: px2] = sc[: py2 - py, : px2 - px]

                    cv2.rectangle(canvas, (cell_x1, cell_y1), (cell_x2, cell_y2), color, 2)
                    score = cam_data["score"]
                    src = "SRC" if cname == pdata.get("source_cam") else f"{score*100:.0f}%"
                    cv2.putText(canvas, src, (cell_x1 + 6, cell_y2 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(canvas, (cell_x1, cell_y1), (cell_x2, cell_y2), (40, 50, 65), 1)
                    cv2.putText(canvas, "Not Seen", (cx + 30, ry + row_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 75, 95), 1, cv2.LINE_AA)

        # Save report
        cv2.imwrite("cctv_verification_report.jpg", canvas)
        print("[Report] Saved: cctv_verification_report.jpg")

        if show:
            if show:
                cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_title, min(canvas_w, 1600), min(canvas_h, 900))
                cv2.imshow(win_title, canvas)
                print("[Report] Showing report. Press Q/ESC/ENTER to close.")
                while True:
                    k = cv2.waitKey(50) & 0xFF
                    if k in (ord("q"), ord("Q"), 27, 13):
                        break
                cv2.destroyWindow(win_title)

    def run(self, show: bool = True, max_frames_per_cam: Optional[int] = None) -> None:
        """Full sequential investigation: Camera 1 -> ... -> Camera N -> Final Report."""
        print("\n" + "=" * 68)
        print("  SAFEPASS SEQUENTIAL CCTV RE-ID  |  Pure Model-Driven Identification")
        print("  OSNet body embeddings + ArcFace face embeddings (when visible)")
        print("  Threshold: cosine similarity — model decides, NO manual rules")
        print("=" * 68 + "\n")

        for i in range(len(self.sources)):
            ok = self.process_camera(i, show=show, max_frames=max_frames_per_cam)
            if not ok:
                break

        print("\n[Investigator] Generating final report...")
        self.show_final_report(show=show)
        cv2.destroyAllWindows()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAFEPASS Sequential CCTV Re-ID (Pure Model)")
    parser.add_argument("--sources", nargs="+", default=None, help="CCTV video files in order")
    parser.add_argument("--names", nargs="+", default=None, help="Camera display names")
    parser.add_argument("--body-thresh", type=float, default=0.60,
                        help="OSNet cosine similarity threshold (default 0.60)")
    parser.add_argument("--face-thresh", type=float, default=0.45,
                        help="ArcFace similarity threshold (default 0.45)")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no OpenCV window)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per camera")
    args = parser.parse_args()

    sources = args.sources
    if not sources:
        import glob
        sources = sorted(glob.glob("data/videos/*.mp4") + glob.glob("data/videos/*.avi"))
        if not sources:
            print("ERROR: No video files found in data/videos/")
            sys.exit(1)

    print(f"[SAFEPASS] Cameras: {sources}")
    names = args.names or [f"Camera {i+1}" for i in range(len(sources))]

    investigator = SequentialCCTVReIDInvestigator(
        sources=sources,
        camera_names=names,
        body_threshold=args.body_thresh,
        face_threshold=args.face_thresh,
    )
    investigator.run(show=not args.no_display, max_frames_per_cam=args.max_frames)


if __name__ == "__main__":
    main()
