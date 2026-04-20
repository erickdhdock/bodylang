from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config import CONFIG

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame landmark record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HandRecord:
    handedness: str                     # "Left" or "Right"
    landmarks: np.ndarray               # (21, 3) normalized image coords + z
    world_landmarks: np.ndarray         # (21, 3) metric
    confidence: float


@dataclass
class FrameLandmarks:
    frame_idx: int
    pts_sec: float

    face_detected: bool = False
    face_landmarks: Optional[np.ndarray] = None          # (468, 3) normalized
    face_transform_matrix: Optional[np.ndarray] = None   # (4, 4)
    face_blendshapes: Optional[dict[str, float]] = None  # gaze + blink scores only

    pose_detected: bool = False
    pose_landmarks: Optional[np.ndarray] = None          # (33, 4) x,y,z,visibility normalized
    pose_world_landmarks: Optional[np.ndarray] = None    # (33, 3) metric

    hands: list[HandRecord] = field(default_factory=list)

    # Rec. 601 luma 0-255, face region when face detected, else full frame.
    # Used to gate gaze blendshapes under low light.
    frame_luminance: Optional[float] = None

    low_confidence: bool = False


def _compute_luminance(
    rgb: np.ndarray, face_lms_normalized: Optional[np.ndarray]
) -> float:
    """Rec. 601 luma in [0, 255]. Uses face bbox when available, else full frame."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return 0.0
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    if face_lms_normalized is not None and len(face_lms_normalized) > 0:
        h, w = luma.shape
        xs = face_lms_normalized[:, 0] * w
        ys = face_lms_normalized[:, 1] * h
        x0 = max(0, int(xs.min()))
        x1 = min(w, int(xs.max()) + 1)
        y0 = max(0, int(ys.min()))
        y1 = min(h, int(ys.max()) + 1)
        if x1 > x0 and y1 > y0:
            return float(luma[y0:y1, x0:x1].mean())
    return float(luma.mean())


GAZE_BLENDSHAPE_KEYS: frozenset[str] = frozenset({
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
    "eyeBlinkLeft", "eyeBlinkRight",
})


# ──────────────────────────────────────────────────────────────────────────────
# Hand-pose classifiers (pure geometry on 21 landmarks)
# ──────────────────────────────────────────────────────────────────────────────

# MediaPipe hand landmark indices: wrist=0, thumb 1-4, index 5-8,
# middle 9-12, ring 13-16, pinky 17-20. Each finger: MCP, PIP, DIP, TIP.
_FINGER_EXTENDED_COS = 0.7   # cos of joint-bend angle; >0.7 ≈ <45° bend ≈ extended
_FINGER_FOLDED_COS = 0.3     # <0.3 ≈ >72° bend ≈ clearly folded


def _finger_bend_cos(lms: np.ndarray, mcp: int, pip: int, tip: int) -> float:
    v1 = lms[pip] - lms[mcp]
    v2 = lms[tip] - lms[pip]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.dot(v1, v2)) / (n1 * n2)


def is_middle_finger_flip(lms: np.ndarray) -> bool:
    """Middle finger extended while index, ring, pinky are folded."""
    return (
        _finger_bend_cos(lms, 9, 10, 12) > _FINGER_EXTENDED_COS
        and _finger_bend_cos(lms, 5, 6, 8) < _FINGER_FOLDED_COS
        and _finger_bend_cos(lms, 13, 14, 16) < _FINGER_FOLDED_COS
        and _finger_bend_cos(lms, 17, 18, 20) < _FINGER_FOLDED_COS
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model cache / downloader
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_model(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    logger.info("Downloading MediaPipe model %s from %s", path.name, url)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def _pose_asset() -> tuple[str, str]:
    variant = CONFIG.processing.pose_variant
    m = CONFIG.models
    if variant == "full":
        return m.pose_landmarker_full, m.pose_full_url
    if variant == "heavy":
        return m.pose_landmarker_heavy, m.pose_heavy_url
    return m.pose_landmarker_lite, m.pose_lite_url


# ──────────────────────────────────────────────────────────────────────────────
# Landmarker bundle
# ──────────────────────────────────────────────────────────────────────────────

class LandmarkerBundle:
    """Holds the three MediaPipe Tasks landmarkers in VIDEO mode."""

    def __init__(self) -> None:
        self._face = None
        self._pose = None
        self._hand = None
        self.delegate_used: str = "cpu"
        self.model_versions: dict[str, str] = {}

    def __enter__(self) -> "LandmarkerBundle":
        model_dir = Path(CONFIG.models.model_dir)
        face_path = _ensure_model(
            model_dir / CONFIG.models.face_landmarker, CONFIG.models.face_url
        )
        pose_name, pose_url = _pose_asset()
        pose_path = _ensure_model(model_dir / pose_name, pose_url)
        hand_path = _ensure_model(
            model_dir / CONFIG.models.hand_landmarker, CONFIG.models.hand_url
        )

        self.model_versions = {
            "face": face_path.name,
            "pose": pose_path.name,
            "hand": hand_path.name,
        }

        self._create_all(face_path, pose_path, hand_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _create_all(self, face_path: Path, pose_path: Path, hand_path: Path) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        want_gpu = CONFIG.processing.gpu_delegate
        tried = [("gpu", BaseOptions.Delegate.GPU)] if want_gpu else []
        tried.append(("cpu", BaseOptions.Delegate.CPU))

        last_err: Optional[Exception] = None
        for name, delegate in tried:
            try:
                self._face = vision.FaceLandmarker.create_from_options(
                    vision.FaceLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=str(face_path), delegate=delegate
                        ),
                        running_mode=vision.RunningMode.VIDEO,
                        num_faces=1,
                        output_face_blendshapes=True,
                        output_facial_transformation_matrixes=True,
                        min_face_detection_confidence=CONFIG.quality.min_face_confidence,
                        min_face_presence_confidence=CONFIG.quality.min_face_confidence,
                        min_tracking_confidence=0.5,
                    )
                )
                self._pose = vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=str(pose_path), delegate=delegate
                        ),
                        running_mode=vision.RunningMode.VIDEO,
                        num_poses=1,
                        min_pose_detection_confidence=CONFIG.quality.min_pose_confidence,
                        min_pose_presence_confidence=CONFIG.quality.min_pose_confidence,
                        min_tracking_confidence=0.5,
                    )
                )
                self._hand = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=str(hand_path), delegate=delegate
                        ),
                        running_mode=vision.RunningMode.VIDEO,
                        num_hands=2,
                        min_hand_detection_confidence=CONFIG.quality.min_hand_confidence,
                        min_hand_presence_confidence=CONFIG.quality.min_hand_confidence,
                        min_tracking_confidence=0.5,
                    )
                )
                self.delegate_used = name
                logger.info("MediaPipe landmarkers initialized with delegate=%s", name)
                return
            except Exception as e:  # noqa: BLE001 — MediaPipe raises raw RuntimeError
                last_err = e
                logger.warning("Failed to init MediaPipe with delegate=%s: %s", name, e)
                self.close()
        raise RuntimeError(f"Could not initialize MediaPipe landmarkers: {last_err}")

    def close(self) -> None:
        for lm in (self._face, self._pose, self._hand):
            if lm is not None:
                try:
                    lm.close()
                except Exception:
                    pass
        self._face = self._pose = self._hand = None

    # ── per-frame inference ────────────────────────────────────────────────

    def detect(self, frame_rgb: np.ndarray, frame_idx: int, pts_sec: float) -> FrameLandmarks:
        import mediapipe as mp

        assert self._face and self._pose and self._hand, "Bundle not initialized"
        timestamp_ms = int(pts_sec * 1000)
        # MediaPipe's Metal/GPU path on macOS rejects 3-channel SRGB ImageFrames
        # (ImageCloneCalculator → CVPixelBuffer needs SRGBA). SRGBA works on CPU too.
        if frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3:
            alpha = np.full(frame_rgb.shape[:2] + (1,), 255, dtype=np.uint8)
            frame_rgb = np.ascontiguousarray(np.concatenate([frame_rgb, alpha], axis=2))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgb)

        rec = FrameLandmarks(frame_idx=frame_idx, pts_sec=pts_sec)

        face_res = self._face.detect_for_video(mp_image, timestamp_ms)
        if face_res.face_landmarks:
            rec.face_detected = True
            lms = face_res.face_landmarks[0]
            rec.face_landmarks = np.array(
                [(p.x, p.y, p.z) for p in lms], dtype=np.float32
            )
            if face_res.facial_transformation_matrixes:
                rec.face_transform_matrix = np.asarray(
                    face_res.facial_transformation_matrixes[0], dtype=np.float32
                )
            if face_res.face_blendshapes:
                rec.face_blendshapes = {
                    c.category_name: float(c.score)
                    for c in face_res.face_blendshapes[0]
                    if c.category_name in GAZE_BLENDSHAPE_KEYS
                }

        pose_res = self._pose.detect_for_video(mp_image, timestamp_ms)
        if pose_res.pose_landmarks:
            rec.pose_detected = True
            lms = pose_res.pose_landmarks[0]
            rec.pose_landmarks = np.array(
                [(p.x, p.y, p.z, p.visibility) for p in lms], dtype=np.float32
            )
            if pose_res.pose_world_landmarks:
                wl = pose_res.pose_world_landmarks[0]
                rec.pose_world_landmarks = np.array(
                    [(p.x, p.y, p.z) for p in wl], dtype=np.float32
                )

        hand_res = self._hand.detect_for_video(mp_image, timestamp_ms)
        if hand_res.hand_landmarks:
            for i, hl in enumerate(hand_res.hand_landmarks):
                handedness = (
                    hand_res.handedness[i][0].category_name
                    if hand_res.handedness and i < len(hand_res.handedness)
                    else "Unknown"
                )
                conf = (
                    float(hand_res.handedness[i][0].score)
                    if hand_res.handedness and i < len(hand_res.handedness)
                    else 1.0
                )
                lm_arr = np.array([(p.x, p.y, p.z) for p in hl], dtype=np.float32)
                if hand_res.hand_world_landmarks and i < len(hand_res.hand_world_landmarks):
                    wl_arr = np.array(
                        [(p.x, p.y, p.z) for p in hand_res.hand_world_landmarks[i]],
                        dtype=np.float32,
                    )
                else:
                    wl_arr = lm_arr.copy()
                rec.hands.append(
                    HandRecord(
                        handedness=handedness,
                        landmarks=lm_arr,
                        world_landmarks=wl_arr,
                        confidence=conf,
                    )
                )

        rec.frame_luminance = _compute_luminance(frame_rgb, rec.face_landmarks)

        # Low-confidence heuristic: we require at least face + pose to analyse.
        q = CONFIG.quality
        if q.skip_frame_on_low_confidence:
            rec.low_confidence = not (rec.face_detected and rec.pose_detected)
        return rec
