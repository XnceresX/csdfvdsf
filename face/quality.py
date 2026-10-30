"""
face/quality.py
===============
Quality estimation algorithms for facial crops (Track P1).

Evaluates multiple quality dimensions that directly affect face recognition
accuracy. These scores drive the adaptive fusion weights: when face quality
is low, the system shifts reliance to body appearance and clothing color.

Quality dimensions:
    1. Blur         - Normalised variance of Laplacian
    2. Resolution   - Pixel height of the face crop
    3. Pose         - Absolute yaw and pitch angles (from FaceDetector landmarks)
    4. Occlusion    - Landmark visibility ratio & patch occlusion analysis
    5. Composite    - Bottleneck-aware blend governing multimodal fusion weights
"""

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np

from contract.schema import QualityMetrics


# ---------------------------------------------------------------------------
# Individual quality scorers
# ---------------------------------------------------------------------------

def calculate_blur_score(face_image: np.ndarray) -> float:
    """
    Normalised blur score via variance of the Laplacian operator.

    Returns:
        float in [0.0, 1.0] - 0.0 is severely blurred, 1.0 is sharp.
    """
    if face_image is None or face_image.size == 0:
        return 0.0

    gray = (
        cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        if len(face_image.shape) == 3
        else face_image
    )
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Empirical normalisation: variance >= 200 is considered sharp
    normalised = min(max(variance / 200.0, 0.0), 1.0)
    return float(round(normalised, 4))


def calculate_resolution_score(height: int, min_height: int = 50) -> float:
    """
    Penalises crops below *min_height* pixels.
    Below ~50 px face recognition accuracy degrades sharply.
    """
    if height <= 0:
        return 0.0
    if height >= min_height:
        return 1.0
    return float(round(height / min_height, 4))


def calculate_pose_score(yaw: float, pitch: float) -> float:
    """
    Converts head-pose angles into a quality score in [0.0, 1.0].
    Frontal faces (yaw ≈ 0, pitch ≈ 0) score highest.
    Beyond ±45° the score drops towards zero.
    """
    max_angle = 45.0
    yaw_penalty = max(1.0 - abs(yaw) / max_angle, 0.0)
    pitch_penalty = max(1.0 - abs(pitch) / max_angle, 0.0)
    return float(round(yaw_penalty * pitch_penalty, 4))


def calculate_occlusion_score(landmark_visibility: float) -> float:
    """
    Converts landmark visibility ratio into an occlusion quality score.
    Returns: 1.0 = fully visible, 0.0 = heavily occluded.
    """
    return float(round(max(min(landmark_visibility, 1.0), 0.0), 4))


def calculate_patch_occlusion_ratio(image: np.ndarray) -> float:
    """
    Estimates the fraction of a face crop covered by artificial patches
    (e.g. black masks, occlusion boxes, zero-intensity blocks).
    """
    if image is None or image.size == 0:
        return 1.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Detect near-black pixels (< 8 intensity)
    black_pixels = np.count_nonzero(gray < 8)
    total_pixels = gray.size
    return float(round(black_pixels / total_pixels, 4))


# ---------------------------------------------------------------------------
# Composite quality & dynamic weight recommendation
# ---------------------------------------------------------------------------

def estimate_face_quality(
    face_image: Optional[np.ndarray] = None,
    detection_conf: float = 0.9,
    yaw: float = 0.0,
    pitch: float = 0.0,
    landmark_visibility: float = 1.0,
) -> QualityMetrics:
    """
    Compute a comprehensive face quality assessment and recommend dynamic
    fusion weights for the multimodal matcher.
    """

    # --- No face available --------------------------------------------------
    if face_image is None or face_image.size == 0:
        return QualityMetrics(
            blur_score=0.0,
            occlusion_score=1.0,        # 1.0 in schema means 100% occluded
            face_detected=False,
            face_confidence=0.0,
            recommended_face_weight=0.0,
            recommended_appearance_weight=0.60,
            recommended_color_weight=0.40,
        )

    # --- Individual scores ---------------------------------------------------
    h = face_image.shape[0]
    blur = calculate_blur_score(face_image)
    res = calculate_resolution_score(h)
    pose = calculate_pose_score(yaw, pitch)

    # Occlusion combines landmark visibility with image patch detection
    lm_visibility = calculate_occlusion_score(landmark_visibility)
    patch_occ = calculate_patch_occlusion_ratio(face_image)
    occ_visibility = max(min(lm_visibility * (1.0 - patch_occ), 1.0), 0.0)

    # Bottleneck-aware effective quality:
    # A severe drop in blur or visibility prevents high quality.
    composite_base = (
        0.35 * blur
        + 0.30 * occ_visibility
        + 0.20 * pose
        + 0.15 * res
    )
    bottleneck = min(blur, occ_visibility, pose)
    effective_quality = 0.60 * composite_base + 0.40 * bottleneck

    # --- Dynamic weight recommendation --------------------------------------
    # High quality (sharp, frontal, clear)  -> trust face heavily
    # Medium quality (moderate blur / pose) -> balanced split
    # Low quality (severe blur / occluded)  -> fallback to body appearance & colour
    if effective_quality >= 0.60:
        w_face = 0.70
        w_app = 0.20
        w_color = 0.10
    elif effective_quality >= 0.35:
        w_face = 0.40
        w_app = 0.35
        w_color = 0.25
    else:
        # CCTV quality low / non-frontal / blurry: zero out face weight so it cannot corrupt match
        w_face = 0.0
        w_app = 0.60
        w_color = 0.40

    return QualityMetrics(
        blur_score=blur,
        occlusion_score=float(round(1.0 - occ_visibility, 4)),  # schema: 0=clear, 1=occluded
        face_detected=True,
        face_confidence=detection_conf,
        recommended_face_weight=w_face,
        recommended_appearance_weight=w_app,
        recommended_color_weight=w_color,
    )
