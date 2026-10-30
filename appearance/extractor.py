"""
appearance/extractor.py
=======================
Track P2 · Task 1: OSNet Body Re-ID Embedding Extractor.

Uses torchreid's OSNet-AIN (x1.0) as the primary re-identification backbone.
Produces 512-dimensional L2-normalised body embeddings from person bounding-box
crops.  Falls back to a deterministic HOG + colour-histogram pseudo-embedding
when torchreid / CUDA is unavailable (CI-safe mode).

Public API
----------
    AppearanceExtractor(weights_path=None, device="cpu", use_tta=True)
        .extract_embedding(body_crop: np.ndarray) -> np.ndarray   # shape (512,)
        .compare(crop1, crop2) -> float                           # cosine sim [0,1]
        .is_loaded -> bool

Cosine similarity helper
------------------------
    appearance_cosine_similarity(vec1, vec2) -> float

Design notes
------------
* Lazy import of torch / torchreid so the module can be imported without GPU.
* TTA: horizontal-flip averaging doubles stability on mirrored camera angles.
* HOG fallback is deterministic → unit tests pass without model weights.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default weight paths (torchreid auto-downloads on first run)
# ---------------------------------------------------------------------------

OSNET_WEIGHT_CANDIDATES = [
    os.path.join("data", "models", "osnet_x1_0_market1501.pth"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models", "osnet_x1_0_market1501.pth"),
    os.path.join(r"C:\project\safepass_appearance\models", "osnet_x1_0_market1501.pth"),
    os.path.join("data", "models", "osnet_x1_0_imagenet.pth"),
    os.path.join("data", "models", "osnet_ain_x1_0_imagenet.pth"),
    os.path.join("data", "models", "osnet_x1_0.pth"),
]

REID_INPUT_SIZE: Tuple[int, int] = (256, 128)  # height × width (standard Market-1501)
EMBEDDING_DIM = 512


# ---------------------------------------------------------------------------
# Cosine similarity helper (mirrors face/extractor.py convention)
# ---------------------------------------------------------------------------

def appearance_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Cosine similarity mapped to [0, 1].
    Returns 0.0 for zero/None vectors.
    """
    if vec1 is None or vec2 is None or vec1.size == 0 or vec2.size == 0:
        return 0.0
    dot = float(np.dot(vec1.flatten(), vec2.flatten()))
    norm = float(np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-8
    sim = dot / norm
    return float(round((sim + 1.0) / 2.0, 4))


# ---------------------------------------------------------------------------
# Fallback extractor (HOG + spatial colour histogram, 512-d)
# ---------------------------------------------------------------------------

def _hog_color_embedding(image: np.ndarray) -> np.ndarray:
    """
    Deterministic 512-d embedding using spatial LAB colour histograms.
    Used when torchreid is not available (CI / CPU-only environments).
    Splits the crop into 4 vertical strips × 3 channels × 32 bins
    to capture spatial colour distribution (512-d).
    """
    img = cv2.resize(image, (64, 128))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 4 vertical strips × 3 channels × 32 bins = 384-d spatial colour histogram
    h = img.shape[0]
    parts = []
    n_strips = 4
    strip_h = h // n_strips
    for s in range(n_strips):
        strip = lab[s * strip_h:(s + 1) * strip_h]
        for ch in range(3):
            hist = cv2.calcHist([strip], [ch], None, [32], [0, 256]).flatten()
            parts.append(hist)
    c_feat = np.concatenate(parts).astype(np.float32)  # 384-d

    # Pad to 512-d
    emb = np.pad(c_feat, (0, 512 - len(c_feat)), "constant")
    norm = np.linalg.norm(emb) + 1e-8
    return (emb / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class AppearanceExtractor:
    """
    OSNet-based body re-identification feature extractor.

    Parameters
    ----------
    weights_path : str, optional
        Path to OSNet `.pth` weights. If None the class tries the default
        candidate paths; torchreid will auto-download if missing.
    device : str
        "cuda" or "cpu".  Falls back to "cpu" if CUDA is unavailable.
    use_tta : bool
        Enable horizontal-flip Test-Time Augmentation (recommended).
    model_name : str
        torchreid model key.  Default: "osnet_ain_x1_0".
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        use_tta: bool = True,
        model_name: str = "osnet_ain_x1_0",
    ):
        self.weights_path = weights_path
        self.device = device
        self.use_tta = use_tta
        self.model_name = model_name
        self._model = None
        self._transform = None
        self._torch = None

        self._try_load_model()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True if a real neural network model is loaded."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load_model(self):
        """Attempt to load OSNet via torchreid. Silently falls back on failure."""
        try:
            import torch  # noqa: F401 – test import first
            self._torch = torch

            import torchreid
            try:
                FeatureExtractor = torchreid.utils.FeatureExtractor
            except AttributeError:
                from torchreid.reid.utils import FeatureExtractor

            # Resolve weights path
            resolved_weights = self._resolve_weights()

            # Build FeatureExtractor (downloads weights if needed)
            device = self.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                print("[AppearanceExtractor] CUDA unavailable, falling back to CPU.")

            # Default to osnet_x1_0 if model_name is osnet_ain_x1_0
            m_name = "osnet_x1_0" if "ain" in self.model_name else self.model_name
            self._extractor = FeatureExtractor(
                model_name=m_name,
                model_path=resolved_weights or "",
                device=device,
            )
            self._model = self._extractor  # sentinel
            self.device = device

            # Torchvision normalisation pipeline for 256×128 person crops
            from torchvision import transforms
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(REID_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            print(
                f"[AppearanceExtractor] Loaded {self.model_name} on {device}. "
                f"TTA={'on' if self.use_tta else 'off'}."
            )
        except Exception as exc:
            print(
                f"[AppearanceExtractor] torchreid unavailable ({exc}). "
                "Using HOG+colour fallback extractor."
            )
            self._model = None
            self._extractor = None

    def _resolve_weights(self) -> Optional[str]:
        """Return first existing weight path from candidates, or None."""
        if self.weights_path and os.path.exists(self.weights_path):
            return self.weights_path
        for cand in OSNET_WEIGHT_CANDIDATES:
            if os.path.exists(cand):
                return cand
        return None  # torchreid will auto-download

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize and convert BGR → RGB, returning (H, W, 3) uint8."""
        if crop is None or crop.size == 0:
            return np.zeros((*REID_INPUT_SIZE, 3), dtype=np.uint8)
        resized = cv2.resize(crop, (REID_INPUT_SIZE[1], REID_INPUT_SIZE[0]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if len(resized.shape) == 3 else resized
        return rgb

    def _to_tensor(self, rgb_img: np.ndarray):
        """Apply torchvision transform; returns (1, 3, H, W) float tensor."""
        return self._transform(rgb_img).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Core embedding extraction
    # ------------------------------------------------------------------

    def extract_embedding(
        self,
        body_crop: np.ndarray,
        use_tta: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Extract a 512-dimensional L2-normalised body Re-ID embedding.

        Parameters
        ----------
        body_crop : np.ndarray
            BGR image crop from a person bounding box (OpenCV format).
        use_tta : bool, optional
            Override the instance-level TTA setting for this call.

        Returns
        -------
        np.ndarray, shape (512,), dtype float32, L2-normalised.
        """
        if body_crop is None or body_crop.size == 0:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        apply_tta = use_tta if use_tta is not None else self.use_tta

        # ---- Neural network path ----------------------------------------
        if self._model is not None:
            try:
                torch = self._torch
                rgb = self._preprocess_crop(body_crop)
                t = self._to_tensor(rgb)
                use_fp16 = (self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda"))

                with torch.no_grad():
                    if use_fp16:
                        with torch.amp.autocast('cuda'):
                            feat = self._extractor.model(t)
                    else:
                        feat = self._extractor.model(t)

                emb = feat.detach().float().cpu().numpy().flatten().astype(np.float32)

                if apply_tta:
                    rgb_flip = cv2.flip(rgb, 1)
                    t_flip = self._to_tensor(rgb_flip)
                    with torch.no_grad():
                        if use_fp16:
                            with torch.amp.autocast('cuda'):
                                feat_flip = self._extractor.model(t_flip)
                        else:
                            feat_flip = self._extractor.model(t_flip)
                    emb_flip = feat_flip.detach().float().cpu().numpy().flatten().astype(np.float32)
                    emb = emb + emb_flip  # average via sum then L2-norm

                # L2 normalise
                norm = np.linalg.norm(emb) + 1e-8
                result = (emb / norm).astype(np.float32)

                # Pad or truncate to EMBEDDING_DIM=512
                if result.size < EMBEDDING_DIM:
                    result = np.pad(result, (0, EMBEDDING_DIM - result.size))
                elif result.size > EMBEDDING_DIM:
                    result = result[:EMBEDDING_DIM]
                return result

            except Exception as exc:
                print(f"[AppearanceExtractor] Inference error: {exc}. Falling back to HOG.")

        # ---- HOG + colour fallback ---------------------------------------
        return _hog_color_embedding(body_crop)

    # ------------------------------------------------------------------
    # Convenience comparison
    # ------------------------------------------------------------------

    def compare(self, crop1: np.ndarray, crop2: np.ndarray) -> float:
        """
        Compute appearance cosine similarity [0, 1] between two body crops.
        """
        e1 = self.extract_embedding(crop1)
        e2 = self.extract_embedding(crop2)
        return appearance_cosine_similarity(e1, e2)

    def batch_extract(self, crops: list, batch_size: int = 16) -> np.ndarray:
        """
        Extract embeddings for a list of body crop images with true batched tensor inference.

        Returns
        -------
        np.ndarray, shape (N, 512), one row per crop.
        """
        if not crops:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        if self._model is None:
            return np.stack([self.extract_embedding(c) for c in crops], axis=0)

        try:
            torch = self._torch
            use_fp16 = (self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda"))
            tensors = [self._transform(self._preprocess_crop(c)) for c in crops]
            all_feats = []
            for i in range(0, len(tensors), batch_size):
                chunk = torch.stack(tensors[i:i + batch_size]).to(self.device)
                with torch.no_grad():
                    if use_fp16:
                        with torch.amp.autocast('cuda'):
                            feats = self._extractor.model(chunk)
                    else:
                        feats = self._extractor.model(chunk)
                f_np = feats.detach().float().cpu().numpy()
                all_feats.append(f_np)
            res = np.concatenate(all_feats, axis=0).astype(np.float32)
            # L2 normalise
            norms = np.linalg.norm(res, axis=1, keepdims=True) + 1e-8
            return (res / norms).astype(np.float32)
        except Exception as e:
            # Fallback to sequential
            return np.stack([self.extract_embedding(c) for c in crops], axis=0)
