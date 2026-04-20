from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CONFIG
from landmarks import FrameLandmarks
from schema import BehaviourEvent, PostureMetrics
from smoothing import moving_average, rle_events


# MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass
class PostureResult:
    metrics: PostureMetrics
    posture_shift_events: list[BehaviourEvent]


def _extract_world_series(frames: list[FrameLandmarks]) -> dict[str, np.ndarray]:
    n = len(frames)
    shoulder_mid = np.full((n, 3), np.nan, dtype=np.float32)
    hip_mid = np.full((n, 3), np.nan, dtype=np.float32)
    nose = np.full((n, 3), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        if f.pose_world_landmarks is None:
            continue
        wl = f.pose_world_landmarks
        shoulder_mid[i] = 0.5 * (wl[LEFT_SHOULDER] + wl[RIGHT_SHOULDER])
        hip_mid[i] = 0.5 * (wl[LEFT_HIP] + wl[RIGHT_HIP])
        nose[i] = wl[NOSE]
    return {"shoulder_mid": shoulder_mid, "hip_mid": hip_mid, "nose": nose}


def analyze_posture(frames: list[FrameLandmarks]) -> PostureResult:
    cfg = CONFIG.posture
    smoothing = CONFIG.smoothing
    pts = np.array([f.pts_sec for f in frames], dtype=np.float32)

    series = _extract_world_series(frames)
    shoulder_mid = moving_average(series["shoulder_mid"], smoothing.window_frames)
    hip_mid = moving_average(series["hip_mid"], smoothing.window_frames)
    nose = moving_average(series["nose"], smoothing.window_frames)

    valid = ~np.isnan(shoulder_mid[:, 0])
    n_valid = int(valid.sum())

    # Torso axis: from hip-midpoint up to shoulder-midpoint.
    torso_vec = shoulder_mid - hip_mid
    # Angle from vertical (world +y is down; shoulders above hips → torso_vec.y < 0).
    vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    torso_norm = np.linalg.norm(torso_vec, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = (torso_vec @ vertical) / np.where(torso_norm > 0, torso_norm, np.nan)
    torso_angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    # Sway: std of shoulder_mid horizontal position (x, z) across the clip, in cm.
    sway_cm = 0.0
    if n_valid > 1:
        sm_valid = shoulder_mid[valid]
        sway_cm = float(
            np.sqrt(np.nanvar(sm_valid[:, 0]) + np.nanvar(sm_valid[:, 2])) * 100.0
        )

    # Head drift: std of (nose - shoulder_mid) over the clip, in cm.
    drift_cm = 0.0
    if n_valid > 1:
        rel = (nose - shoulder_mid)[valid]
        drift_cm = float(np.sqrt(np.nansum(np.nanvar(rel, axis=0))) * 100.0)

    # Stability score: inverse of combined normalized variance, clipped to [0, 1].
    sway_component = min(sway_cm / max(cfg.sway_threshold_cm * 2, 1e-3), 1.0)
    drift_component = min(drift_cm / max(cfg.drift_threshold_cm * 2, 1e-3), 1.0)
    stability_score = float(max(0.0, 1.0 - 0.5 * (sway_component + drift_component)))

    # Lean classification from shoulder_mid.z median (relative to hips at origin).
    mean_shoulder_z = float(np.nanmean(shoulder_mid[:, 2])) if n_valid else 0.0
    lean_angle_from_neutral = np.degrees(np.arctan2(mean_shoulder_z, 1.0))
    if lean_angle_from_neutral < -cfg.lean_threshold_deg:
        lean_label = "forward"
    elif lean_angle_from_neutral > cfg.lean_threshold_deg:
        lean_label = "backward"
    else:
        lean_label = "neutral"

    # Positional-shift events: rolling std of shoulder position above threshold.
    shift_flags, shift_values = _rolling_shift_flags(
        shoulder_mid, pts, window_sec=cfg.stability_window_sec,
        threshold_cm=cfg.sway_threshold_cm,
    )
    shift_runs = rle_events(
        shift_flags, pts, min_duration_sec=cfg.stability_window_sec * 0.5, values=shift_values
    )
    posture_shift_events = [
        BehaviourEvent(
            start_sec=round(e.start_sec, 3),
            end_sec=round(e.end_sec, 3),
            duration_sec=round(e.duration_sec, 3),
            event_type="posture_shift",
            peak_value=round(e.peak_value, 2) if e.peak_value else None,
            detail=lean_label if lean_label != "neutral" else None,
        )
        for e in shift_runs
    ]

    # Overall label.
    if stability_score >= 0.85 and sway_cm < cfg.sway_threshold_cm * 0.5:
        label = "overly rigid" if len(posture_shift_events) == 0 and drift_cm < 0.5 else "stable posture"
    elif stability_score >= 0.7:
        label = "stable posture with mild sway"
    else:
        label = "unstable seated movement"

    metrics = PostureMetrics(
        stability_score=round(stability_score, 3),
        torso_sway_cm=round(sway_cm, 2),
        head_drift_cm=round(drift_cm, 2),
        mean_torso_angle_deg=round(
            float(np.nanmean(torso_angle_deg)) if n_valid else 0.0, 2
        ),
        lean_label=lean_label,
        positional_shift_count=len(posture_shift_events),
        label=label,
    )

    return PostureResult(metrics=metrics, posture_shift_events=posture_shift_events)


def _rolling_shift_flags(
    shoulder_mid: np.ndarray,
    pts: np.ndarray,
    window_sec: float,
    threshold_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(shoulder_mid)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)
    dt = float(np.median(np.diff(pts))) if n > 1 else 0.1
    win = max(3, int(round(window_sec / max(dt, 1e-3))))
    flags = np.zeros(n, dtype=bool)
    values = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - win // 2)
        hi = min(n, i + win // 2 + 1)
        w = shoulder_mid[lo:hi]
        w = w[~np.isnan(w[:, 0])]
        if len(w) < 3:
            continue
        local_sway_cm = float(np.sqrt(np.var(w[:, 0]) + np.var(w[:, 2])) * 100.0)
        values[i] = local_sway_cm
        if local_sway_cm > threshold_cm:
            flags[i] = True
    return flags, values
