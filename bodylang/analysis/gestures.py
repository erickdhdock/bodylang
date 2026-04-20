from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from config import CONFIG
from landmarks import FrameLandmarks, is_middle_finger_flip
from schema import BehaviourEvent, GestureMetrics
from smoothing import moving_average, rle_events


HAND_WRIST = 0
INAPPROPRIATE_MIN_DURATION_SEC = 0.2  # suppress single-frame false positives


@dataclass
class GestureResult:
    metrics: GestureMetrics
    gesture_burst_events: list[BehaviourEvent]
    inappropriate_events: list[BehaviourEvent]


def analyze_gestures(
    frames: list[FrameLandmarks],
    pitch_series: Optional[np.ndarray] = None,
) -> GestureResult:
    cfg = CONFIG.gestures
    smoothing_cfg = CONFIG.smoothing
    pts = np.array([f.pts_sec for f in frames], dtype=np.float32)
    n = len(frames)

    # ── Wrist positions per hand (world coords, meters) ────────────────────
    left_wrist = np.full((n, 3), np.nan, dtype=np.float32)
    right_wrist = np.full((n, 3), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        for h in f.hands:
            pos = h.world_landmarks[HAND_WRIST]
            if h.handedness == "Left":
                left_wrist[i] = pos
            elif h.handedness == "Right":
                right_wrist[i] = pos

    left_wrist = moving_average(left_wrist, smoothing_cfg.window_frames)
    right_wrist = moving_average(right_wrist, smoothing_cfg.window_frames)

    # ── Per-frame wrist speed (m/s) → take max across hands ────────────────
    speed = _max_hand_speed(left_wrist, right_wrist, pts)

    active_threshold_m_per_s = cfg.active_wrist_velocity_cm_per_sec / 100.0
    active = speed > active_threshold_m_per_s
    valid_speed = ~np.isnan(speed)

    hand_activity_pct = (
        float(active[valid_speed].sum() / valid_speed.sum())
        if valid_speed.any()
        else 0.0
    )

    # ── Gesture bursts ─────────────────────────────────────────────────────
    burst_runs = rle_events(
        active & valid_speed,
        pts,
        min_duration_sec=0.3,
        values=speed,
    )
    gesture_burst_events = []
    amplitudes_cm = []
    for e in burst_runs:
        amp_cm = _burst_amplitude_cm(left_wrist, right_wrist, e.start_idx, e.end_idx)
        amplitudes_cm.append(amp_cm)
        gesture_burst_events.append(
            BehaviourEvent(
                start_sec=round(e.start_sec, 3),
                end_sec=round(e.end_sec, 3),
                duration_sec=round(e.duration_sec, 3),
                event_type="gesture_burst",
                peak_value=round(amp_cm, 2),
            )
        )
    mean_amp_cm = float(np.mean(amplitudes_cm)) if amplitudes_cm else 0.0

    # ── Nod detection (from head-pitch time series) ────────────────────────
    nod_count = _count_nods(pitch_series, pts) if pitch_series is not None else 0

    # ── Inappropriate-gesture detection (middle finger) ────────────────────
    inappropriate_events = _detect_inappropriate_gestures(frames, pts)

    # ── Overall gesture label ──────────────────────────────────────────────
    if hand_activity_pct < cfg.limited_activity_pct:
        label = "limited body-language usage"
    elif hand_activity_pct > cfg.excessive_activity_pct:
        label = "excessive gesture usage"
    else:
        label = "moderate gesture usage"

    metrics = GestureMetrics(
        hand_activity_pct=round(hand_activity_pct, 4),
        mean_gesture_amplitude_cm=round(mean_amp_cm, 2),
        gesture_burst_count=len(gesture_burst_events),
        nod_count=nod_count,
        inappropriate_gesture_count=len(inappropriate_events),
        label=label,
    )

    return GestureResult(
        metrics=metrics,
        gesture_burst_events=gesture_burst_events,
        inappropriate_events=inappropriate_events,
    )


def _detect_inappropriate_gestures(
    frames: list[FrameLandmarks], pts: np.ndarray
) -> list[BehaviourEvent]:
    n = len(frames)
    flags = np.zeros(n, dtype=bool)
    hand_labels: list[str] = [""] * n
    for i, f in enumerate(frames):
        triggered: list[str] = []
        for h in f.hands:
            if is_middle_finger_flip(h.landmarks):
                triggered.append(h.handedness)
        if triggered:
            flags[i] = True
            hand_labels[i] = "+".join(sorted(set(triggered)))

    runs = rle_events(flags, pts, min_duration_sec=INAPPROPRIATE_MIN_DURATION_SEC)
    events: list[BehaviourEvent] = []
    for e in runs:
        window_labels = [lbl for lbl in hand_labels[e.start_idx : e.end_idx + 1] if lbl]
        detail = max(set(window_labels), key=window_labels.count) if window_labels else None
        events.append(
            BehaviourEvent(
                start_sec=round(e.start_sec, 3),
                end_sec=round(e.end_sec, 3),
                duration_sec=round(e.duration_sec, 3),
                event_type="inappropriate_gesture",
                detail=f"middle_finger({detail})" if detail else "middle_finger",
            )
        )
    return events


def _max_hand_speed(
    left: np.ndarray, right: np.ndarray, pts: np.ndarray
) -> np.ndarray:
    n = len(pts)
    if n < 2:
        return np.full(n, np.nan, dtype=np.float32)
    dt = np.diff(pts, prepend=pts[0])
    dt[dt <= 0] = np.nan

    def speed(arr: np.ndarray) -> np.ndarray:
        diff = np.diff(arr, axis=0, prepend=arr[:1])
        dist = np.linalg.norm(diff, axis=1)
        return dist / dt

    sp_l = speed(left)
    sp_r = speed(right)
    return np.fmax(sp_l, sp_r)


def _burst_amplitude_cm(
    left: np.ndarray, right: np.ndarray, start: int, end: int
) -> float:
    best = 0.0
    for arr in (left, right):
        w = arr[start : end + 1]
        w = w[~np.isnan(w[:, 0])]
        if len(w) < 2:
            continue
        span = np.linalg.norm(w.max(axis=0) - w.min(axis=0))
        best = max(best, float(span) * 100.0)
    return best


def _count_nods(pitch_series: np.ndarray, pts: np.ndarray) -> int:
    cfg = CONFIG.gestures
    if pitch_series is None or len(pitch_series) < 8:
        return 0
    valid = ~np.isnan(pitch_series)
    if valid.sum() < 8:
        return 0
    # Fill NaNs with local mean so filtfilt is well-defined.
    sig = pitch_series.copy()
    sig[~valid] = np.nanmean(sig)

    dt = float(np.median(np.diff(pts))) if len(pts) > 1 else 0.1
    fs = 1.0 / max(dt, 1e-3)
    lo_hz, hi_hz = cfg.nod_freq_range_hz
    nyq = fs * 0.5
    if hi_hz >= nyq:
        hi_hz = nyq * 0.95
    if lo_hz >= hi_hz:
        return 0
    try:
        b, a = butter(2, [lo_hz / nyq, hi_hz / nyq], btype="band")
        filtered = filtfilt(b, a, sig)
    except ValueError:
        return 0

    # Peaks exceeding configured amplitude.
    peaks_pos, _ = find_peaks(filtered, prominence=cfg.nod_min_amplitude_deg)
    peaks_neg, _ = find_peaks(-filtered, prominence=cfg.nod_min_amplitude_deg)
    # A full nod = one down + one up; divide total extrema by 2 (rounded down).
    return int((len(peaks_pos) + len(peaks_neg)) // 2)
