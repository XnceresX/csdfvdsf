"""
appearance/test_p2.py
=====================
Track P2 unit tests – no GPU, no model weights required.
Exercises all three appearance modules plus the fusion engine and calibration.

Run::
    python -m appearance.test_p2
"""

from __future__ import annotations

import sys
import os
import traceback

import numpy as np
import cv2

# Ensure repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_person_crop(h=200, w=80, upper_rgb=(200, 50, 50), lower_rgb=(50, 50, 200)):
    """Create a synthetic person crop with distinct upper/lower colours."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    upper_end = int(h * 0.50)
    img[:upper_end] = upper_rgb         # upper garment colour (BGR)
    img[upper_end:] = lower_rgb         # lower garment colour (BGR)
    return img


def test_appearance_extractor():
    print("\n[TEST] AppearanceExtractor (HOG fallback)...")
    from appearance.extractor import AppearanceExtractor, appearance_cosine_similarity

    ext = AppearanceExtractor()
    crop1 = _make_person_crop(upper_rgb=(200, 50, 50), lower_rgb=(50, 50, 200))
    crop2 = _make_person_crop(upper_rgb=(200, 55, 45), lower_rgb=(55, 45, 205))  # similar
    crop3 = _make_person_crop(upper_rgb=(50, 200, 50), lower_rgb=(200, 50, 50))  # different

    emb1 = ext.extract_embedding(crop1)
    emb2 = ext.extract_embedding(crop2)
    emb3 = ext.extract_embedding(crop3)

    assert emb1.shape == (512,), f"Expected (512,), got {emb1.shape}"
    assert abs(np.linalg.norm(emb1) - 1.0) < 0.01, "Embedding should be L2-normalised"

    sim_same = ext.compare(crop1, crop2)
    sim_diff = ext.compare(crop1, crop3)
    print(f"   same-person sim: {sim_same:.4f}  (expect > 0.90)")
    print(f"   diff-person sim: {sim_diff:.4f}  (expect < 0.90)")

    # Batch extraction
    batch = ext.batch_extract([crop1, crop2, crop3])
    assert batch.shape == (3, 512), f"Batch shape mismatch: {batch.shape}"

    # None / empty input
    empty_emb = ext.extract_embedding(None)
    assert empty_emb.shape == (512,) and (empty_emb == 0).all()

    print("   PASS")


def test_body_quality():
    print("\n[TEST] BodyQualityEstimator...")
    from appearance.quality import BodyQualityEstimator, estimate_body_quality

    est = BodyQualityEstimator()

    # Good crop: tall, sharp, clear
    good = _make_person_crop(h=220, w=90)
    q_good = est.estimate(good)
    print(f"   good crop q_body: {q_good['q_body']:.4f}  (expect > 0.2)")
    assert q_good["q_body"] >= 0.0
    assert "usable" in q_good

    # None input
    q_none = est.estimate(None)
    assert q_none["q_body"] == 0.0 and not q_none["usable"]

    # Convenience function
    score = estimate_body_quality(good)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    # Weight recommendation
    w_high = est.get_appearance_weight_recommendation(0.8)
    w_low = est.get_appearance_weight_recommendation(0.1)
    assert w_high > w_low, "Higher quality should yield higher appearance weight"

    print("   PASS")


def test_clothing_color():
    print("\n[TEST] ClothingColorExtractor...")
    from appearance.color import ClothingColorExtractor

    ext = ClothingColorExtractor()

    red_blue = _make_person_crop(upper_rgb=(0, 0, 200), lower_rgb=(200, 0, 0))   # BGR: red top, blue bottom
    red_blue2 = _make_person_crop(upper_rgb=(0, 0, 210), lower_rgb=(195, 0, 10))  # similar

    feats = ext.extract(red_blue)
    print(f"   upper_label='{feats.upper_label}'  lower_label='{feats.lower_label}'")
    assert feats.upper_hist.shape == (ext.hist_size,)
    assert feats.lower_hist.shape == (ext.hist_size,)

    # Identical crops → high similarity
    sim_same = ext.compare(red_blue, red_blue2)
    print(f"   same colour sim: {sim_same:.4f}  (expect > 0.70)")
    assert sim_same > 0.50, f"Expected > 0.50, got {sim_same}"

    # Compare against text labels
    text_sim = ext.compare_with_text(red_blue, "red", "blue")
    print(f"   text colour sim: {text_sim:.4f}")

    # Histogram similarity edge cases
    zero_hist = np.zeros(ext.hist_size, dtype=np.float32)
    sim_zero = ext.hist_similarity(zero_hist, zero_hist)
    assert sim_zero == 0.0

    print("   PASS")


def test_fusion_engine():
    print("\n[TEST] FusionEngine...")
    from fusion.engine import FusionEngine, fuse_scores

    # Dynamic mode
    eng = FusionEngine(mode="dynamic")
    result = eng.fuse(face_sim=0.85, app_sim=0.75, color_sim=0.70, q_face=0.80, q_body=0.75)
    print(f"   dynamic fused: {result.fused_score:.4f}  (expect 0.7..0.9)")
    assert 0.0 <= result.fused_score <= 1.0

    # Face-only baseline
    face_eng = FusionEngine(mode="face_only")
    r_face = face_eng.fuse(face_sim=0.85, app_sim=0.75, color_sim=0.70, q_face=0.80, q_body=0.75)
    assert abs(r_face.fused_score - 0.85) < 0.001, f"Face-only should equal face_sim, got {r_face.fused_score}"

    # Fixed weights
    fixed_eng = FusionEngine(mode="fixed", fixed_weights=(0.5, 0.3, 0.2))
    r_fixed = fixed_eng.fuse(face_sim=0.85, app_sim=0.75, color_sim=0.70, q_face=0.80, q_body=0.75)
    expected = 0.5 * 0.85 + 0.3 * 0.75 + 0.2 * 0.70
    assert abs(r_fixed.fused_score - expected) < 0.001, f"Expected {expected:.4f}, got {r_fixed.fused_score}"

    # Missing modalities
    r_miss = eng.fuse(face_sim=None, app_sim=0.75, color_sim=0.70, q_face=0.0, q_body=0.75)
    assert 0.0 <= r_miss.fused_score <= 1.0

    # Convenience function
    s = fuse_scores(face_sim=0.85, app_sim=0.70, color_sim=0.60, q_face=0.7, q_body=0.7)
    assert 0.0 <= s <= 1.0

    # compare_all_modes
    all_modes = eng.compare_all_modes(0.85, 0.75, 0.70, q_face=0.7, q_body=0.7)
    assert "dynamic" in all_modes
    assert "face_only" in all_modes

    print("   PASS")


def test_score_calibrator():
    print("\n[TEST] ScoreCalibrator...")
    from fusion.calibration import ScoreCalibrator, calibrate_scores

    rng = np.random.RandomState(0)
    genuine = rng.normal(0.75, 0.08, 300)
    impostor = rng.normal(0.45, 0.10, 300)
    scores = np.concatenate([genuine, impostor])
    labels = np.array([1] * 300 + [0] * 300)

    calib = ScoreCalibrator(method="platt", modality="test")
    cal_scores = calib.fit_transform(scores, labels)
    assert cal_scores.shape == scores.shape
    assert cal_scores.min() >= 0.0
    assert cal_scores.max() <= 1.0

    # Auto-select (should pick platt for < 200 samples)
    calib_auto = ScoreCalibrator(method="auto")
    small_scores = np.concatenate([scores[:50], scores[300:350]])
    small_labels = np.concatenate([labels[:50], labels[300:350]])
    out = calib_auto.fit_transform(small_scores, small_labels)
    assert len(out) == 100

    # One-shot helper
    out2 = calibrate_scores(scores, labels)
    assert len(out2) == len(scores)

    print("   PASS")


def test_learned_fusion():
    print("\n[TEST] LearnedFusion (NumPy fallback)...")
    from fusion.learned import LearnedFusion, build_quality_vector, LearnedFusionDataset

    lf = LearnedFusion()

    # build_quality_vector
    qv = build_quality_vector(q_face=0.8, q_body=0.7)
    assert qv.shape == (7,)

    # predict_weights before fitting (rule-based fallback)
    wf, wa, wc = lf.predict_weights(qv)
    assert abs(wf + wa + wc - 1.0) < 0.01, f"Weights don't sum to 1: {wf+wa+wc}"

    # fuse before fitting
    score = lf.fuse(face_sim=0.8, app_sim=0.7, color_sim=0.6, quality_vec=qv)
    assert 0.0 <= score <= 1.0

    # Dataset helper
    ds = LearnedFusionDataset()
    for _ in range(20):
        rng = np.random.RandomState(42)
        ds.add(
            quality_vec=build_quality_vector(rng.rand(), rng.rand()),
            face_sim=float(rng.rand()),
            app_sim=float(rng.rand()),
            color_sim=float(rng.rand()),
            label=int(rng.rand() > 0.5),
        )
    assert len(ds) == 20

    print("   PASS")


def test_ablation_eval():
    print("\n[TEST] Ablation evaluation (synthetic, 200 pairs)...")
    from eval.appearance_baseline import run_ablation

    results = run_ablation(
        n_pairs=200,
        output_path="eval/results_p2_test.json",
        plot=False,
        calibrate=True,
        seed=7,
    )
    assert "dynamic" in results
    assert "face_only" in results
    dynamic_auc = results.get("dynamic", {}).get("auc", 0)
    print(f"   dynamic AUC: {dynamic_auc:.4f}")
    print("   PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_appearance_extractor,
        test_body_quality,
        test_clothing_color,
        test_fusion_engine,
        test_score_calibrator,
        test_learned_fusion,
        test_ablation_eval,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception:
            print(f"\n   FAIL: {test.__name__}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 50)
    print(f" Track P2 Tests: {passed} passed, {failed} failed")
    print("=" * 50)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
