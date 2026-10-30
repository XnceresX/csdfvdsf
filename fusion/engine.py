"""
fusion/engine.py
================
Track P2 · Task 4: Quality-Adaptive Multimodal Fusion Engine.

Implements the core quality-weighted score fusion formula:

    Score_fused = w_face(q_face) · S_face
                + w_app(q_body) · S_app
                + w_color(q_body) · S_color

where weights sum to 1.0 and are dynamically computed from per-sample
quality estimates.

Fusion modes (selectable at construction time):
    "dynamic"     Default. Weights adapt per-sample from quality metrics.
    "face_only"   Baseline: w_face=1.0, ignores appearance/colour.
    "app_only"    Baseline: w_app=1.0, ignores face/colour.
    "fixed"       Fixed weights supplied at construction.

The engine consumes ``QualityMetrics`` (from contract.schema) for face quality
and ``BodyQualityEstimator`` output for body quality, integrating P1 and P2
seamlessly.

Public API
----------
    FusionEngine(mode="dynamic", fixed_weights=None)
        .fuse(face_sim, app_sim, color_sim, q_face, q_body) -> ModalitySimilarities
        .fuse_from_schema(similarities, quality) -> float

    fuse_scores(face_sim, app_sim, color_sim, q_face, q_body, mode) -> float
        Convenience one-shot call.
"""

from __future__ import annotations

from typing import Optional, Tuple, Literal, Dict, Any

import numpy as np

from contract.schema import ModalitySimilarities, QualityMetrics


# ---------------------------------------------------------------------------
# Weight computation helper
# ---------------------------------------------------------------------------

def _quality_to_weights(
    q_face: float,
    q_body: float,
    face_available: bool = True,
    app_available: bool = True,
    color_available: bool = True,
) -> Tuple[float, float, float]:
    """
    Compute dynamic fusion weights from quality estimates.

    Logic:
        1. Derive per-modality confidence from quality.
        2. Zero out unavailable modalities.
        3. L1-normalise to sum = 1.0.

    Returns
    -------
    (w_face, w_app, w_color) all in [0, 1], summing to 1.0.
    """
    # Base weights from quality ranges (mirrors P1 face quality thresholds)
    if q_face >= 0.60:
        base_face = 0.70
    elif q_face >= 0.35:
        base_face = 0.40
    else:
        # If face quality is low or noisy in CCTV views, strictly 0.0 weight
        base_face = 0.0

    if q_body >= 0.65:
        base_app = 0.65
        base_color = 0.35
    elif q_body >= 0.35:
        base_app = 0.40
        base_color = 0.25
    else:
        base_app = 0.15
        base_color = 0.20

    # Combined pre-normalised weights: face weight strictly 0.0 if unavailable or quality < 0.35
    w_face = base_face if (face_available and q_face >= 0.35) else 0.0
    w_app = base_app if app_available else 0.0
    w_color = base_color if color_available else 0.0

    total = w_face + w_app + w_color
    if total < 1e-8:
        # All modalities unavailable — fallback: equal weight among available flags
        n = sum([face_available, app_available, color_available])
        eq = 1.0 / max(n, 1)
        return (
            eq if face_available else 0.0,
            eq if app_available else 0.0,
            eq if color_available else 0.0,
        )

    return (
        round(w_face / total, 4),
        round(w_app / total, 4),
        round(w_color / total, 4),
    )


# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """
    Quality-adaptive multimodal score fusion engine.

    Parameters
    ----------
    mode : "dynamic" | "face_only" | "app_only" | "color_only" | "fixed"
        Fusion strategy.
    fixed_weights : (w_face, w_app, w_color), optional
        Required when mode="fixed". Weights are automatically L1-normalised.
    min_modalities : int
        Minimum number of modalities with non-None scores required to produce
        a valid fused score. If fewer are present, returns 0.0.

    Usage::

        engine = FusionEngine(mode="dynamic")
        result = engine.fuse(
            face_sim=0.82,
            app_sim=0.71,
            color_sim=0.65,
            q_face=0.75,
            q_body=0.68,
        )
        print(result.fused_score)
    """

    MODES = {"dynamic", "face_only", "app_only", "color_only", "fixed"}

    def __init__(
        self,
        mode: Literal["dynamic", "face_only", "app_only", "color_only", "fixed"] = "dynamic",
        fixed_weights: Optional[Tuple[float, float, float]] = None,
        min_modalities: int = 1,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")
        if mode == "fixed" and fixed_weights is None:
            raise ValueError("fixed_weights must be provided when mode='fixed'")

        self.mode = mode
        self.min_modalities = min_modalities

        # Normalise fixed weights
        if fixed_weights is not None:
            total = sum(fixed_weights) + 1e-8
            self.fixed_weights = tuple(w / total for w in fixed_weights)
        else:
            self.fixed_weights = None

    # ------------------------------------------------------------------
    # Internal weight dispatch
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        q_face: float,
        q_body: float,
        face_available: bool,
        app_available: bool,
        color_available: bool,
    ) -> Tuple[float, float, float]:
        """Return (w_face, w_app, w_color) for the current mode."""
        if self.mode == "face_only":
            return (1.0, 0.0, 0.0) if face_available else (0.0, 1.0, 0.0)

        elif self.mode == "app_only":
            return (0.0, 1.0, 0.0) if app_available else (0.0, 0.0, 1.0)

        elif self.mode == "color_only":
            return (0.0, 0.0, 1.0) if color_available else (0.0, 1.0, 0.0)

        elif self.mode == "fixed":
            wf, wa, wc = self.fixed_weights
            return (
                wf if face_available else 0.0,
                wa if app_available else 0.0,
                wc if color_available else 0.0,
            )

        else:  # dynamic
            return _quality_to_weights(
                q_face, q_body,
                face_available, app_available, color_available,
            )

    # ------------------------------------------------------------------
    # Core fusion methods
    # ------------------------------------------------------------------

    def fuse(
        self,
        face_sim: Optional[float] = None,
        app_sim: Optional[float] = None,
        color_sim: Optional[float] = None,
        q_face: float = 0.5,
        q_body: float = 0.5,
    ) -> ModalitySimilarities:
        """
        Compute the quality-weighted fused score.

        Parameters
        ----------
        face_sim : float or None
            Normalised face cosine similarity [0, 1].  None = unavailable.
        app_sim : float or None
            Normalised appearance cosine similarity [0, 1].  None = unavailable.
        color_sim : float or None
            Normalised clothing colour similarity [0, 1].  None = unavailable.
        q_face : float
            Face quality estimate in [0, 1] (from face/quality.py).
        q_body : float
            Body crop quality estimate in [0, 1] (from appearance/quality.py).

        Returns
        -------
        ModalitySimilarities with fused_score set.
        """
        face_available = face_sim is not None
        app_available = app_sim is not None
        color_available = color_sim is not None

        n_available = sum([face_available, app_available, color_available])
        if n_available < self.min_modalities:
            return ModalitySimilarities(
                face_similarity=face_sim,
                appearance_similarity=app_sim,
                color_similarity=color_sim,
                fused_score=0.0,
            )

        w_face, w_app, w_color = self._compute_weights(
            q_face, q_body, face_available, app_available, color_available
        )

        # Re-normalise in case some modalities were zeroed
        total_w = (
            (w_face if face_available else 0.0) +
            (w_app if app_available else 0.0) +
            (w_color if color_available else 0.0)
        )
        if total_w < 1e-8:
            total_w = 1.0

        weighted_sum = (
            (w_face * (face_sim or 0.0)) +
            (w_app * (app_sim or 0.0)) +
            (w_color * (color_sim or 0.0))
        )

        fused = float(round(min(max(weighted_sum / total_w * total_w, 0.0), 1.0), 4))
        # Correct: simple weighted sum (weights already normalised to 1.0 sum)
        fused = float(round(min(max(
            (w_face * (face_sim or 0.0)) +
            (w_app * (app_sim or 0.0)) +
            (w_color * (color_sim or 0.0)),
            0.0), 1.0), 4))

        return ModalitySimilarities(
            face_similarity=round(face_sim, 4) if face_sim is not None else None,
            appearance_similarity=round(app_sim, 4) if app_sim is not None else None,
            color_similarity=round(color_sim, 4) if color_sim is not None else None,
            fused_score=fused,
        )

    def fuse_from_schema(
        self,
        similarities: ModalitySimilarities,
        quality: QualityMetrics,
    ) -> float:
        """
        Fuse scores using pre-populated schema objects.

        Parameters
        ----------
        similarities : ModalitySimilarities
            Contains face_similarity, appearance_similarity, color_similarity.
        quality : QualityMetrics
            Contains recommended fusion weights and quality info from P1.

        Returns
        -------
        float: fused_score in [0, 1].
        """
        q_face = max(
            quality.blur_score * (1.0 - quality.occlusion_score),
            0.0,
        )
        q_body = float(quality.recommended_appearance_weight)

        result = self.fuse(
            face_sim=similarities.face_similarity,
            app_sim=similarities.appearance_similarity,
            color_sim=similarities.color_similarity,
            q_face=q_face,
            q_body=q_body,
        )
        return result.fused_score

    def get_weights(self, q_face: float = 0.5, q_body: float = 0.5) -> Dict[str, float]:
        """
        Return the fusion weights for the given quality inputs (useful for visualisation).

        Returns
        -------
        dict: {"w_face": float, "w_app": float, "w_color": float}
        """
        wf, wa, wc = self._compute_weights(q_face, q_body, True, True, True)
        return {"w_face": wf, "w_app": wa, "w_color": wc}

    # ------------------------------------------------------------------
    # Baseline comparison helpers
    # ------------------------------------------------------------------

    def compare_all_modes(
        self,
        face_sim: Optional[float],
        app_sim: Optional[float],
        color_sim: Optional[float],
        q_face: float = 0.5,
        q_body: float = 0.5,
    ) -> Dict[str, float]:
        """
        Run all fusion modes and return a dict of mode → fused_score.
        Useful for ablation studies.
        """
        results = {}
        for mode in self.MODES - {"fixed"}:
            eng = FusionEngine(mode=mode)
            out = eng.fuse(face_sim, app_sim, color_sim, q_face, q_body)
            results[mode] = out.fused_score
        # Also run fixed with equal weights
        eng_fixed = FusionEngine(mode="fixed", fixed_weights=(1 / 3, 1 / 3, 1 / 3))
        out_fixed = eng_fixed.fuse(face_sim, app_sim, color_sim, q_face, q_body)
        results["fixed_equal"] = out_fixed.fused_score
        return results


# ---------------------------------------------------------------------------
# Convenience one-shot function
# ---------------------------------------------------------------------------

def fuse_scores(
    face_sim: Optional[float] = None,
    app_sim: Optional[float] = None,
    color_sim: Optional[float] = None,
    q_face: float = 0.5,
    q_body: float = 0.5,
    mode: str = "dynamic",
) -> float:
    """
    Quick one-shot quality-adaptive score fusion.

    Parameters
    ----------
    face_sim, app_sim, color_sim : float or None
        Per-modality similarity scores in [0, 1]. None = modality not available.
    q_face : float
        Face crop quality [0, 1].
    q_body : float
        Body crop quality [0, 1].
    mode : str
        Fusion mode: "dynamic" | "face_only" | "app_only" | "color_only".

    Returns
    -------
    float: fused score in [0, 1].
    """
    engine = FusionEngine(mode=mode)
    result = engine.fuse(face_sim, app_sim, color_sim, q_face, q_body)
    return result.fused_score
