from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CONFIG
from landmarks import FrameLandmarks
from schema import BehaviourEvent, RestlessnessMetrics
from smoothing import moving_average, rle_events


NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


@dataclass
class RestlessnessResult:
    metrics: RestlessnessMetrics
    restless_events: list[BehaviourEvent]


def analyze_restlessness(frames: list[FrameLandmarks]) -> RestlessnessResult:
    cfg = CONFIG.restlessness
    smoothing_cfg = CONFIG.smoothing
    pts = np.array([f.pts_sec for f in frames], dtype=np.float32)
    n = len(frames)

    # Aggregate a "body motion signal": combined speed of upper-body anchors + wrists.
    anchor_pos = _stack_anchors(frames)                # (n, k, 3)
    anchor_pos = moving_average(
        anchor_pos.reshape(n, -1), smoothing_cfg.window_frames
    ).reshape(anchor_pos.shape)

    speed = _combined_speed(anchor_pos, pts)           # (n,) m/s
    valid = ~np.isnan(speed)

    # Rolling RMS over velocity_window_sec.
    dt = float(np.median(np.diff(pts))) if n > 1 else 0.1
    win = max(3, int(round(cfg.velocity_window_sec / max(dt, 1e-3))))
    rms = _rolling_rms(speed, win)

    velocity_rms = float(np.nanmean(rms[valid])) if valid.any() else 0.0

    restless_flags = (rms > cfg.velocity_rms_threshold) & valid
    overly_still_flags = (rms < cfg.overly_still_velocity_rms) & valid

    restless_runs = rle_events(
        restless_flags, pts, min_duration_sec=cfg.velocity_window_sec, values=rms
    )
    restless_events = [
        BehaviourEvent(
            start_sec=round(e.start_sec, 3),
            end_sec=round(e.end_sec, 3),
            duration_sec=round(e.duration_sec, 3),
            event_type="restless_window",
            peak_value=round(e.peak_value, 4),
        )
        for e in restless_runs
    ]

    # Repetitive-motion detection via FFT on the body motion signal.
    repetitive_count = _count_repetitive_events(speed, pts)

    overly_still_pct = (
        float(overly_still_flags.sum() / valid.sum()) if valid.any() else 0.0
    )

    # Overall label.
    if overly_still_pct > 0.6:
        label = "overly still"
    elif repetitive_count >= 2 and velocity_rms > cfg.velocity_rms_threshold:
        label = "repetitive motion"
    elif velocity_rms > cfg.velocity_rms_threshold * 1.5:
        label = "restless movement"
    elif velocity_rms > cfg.velocity_rms_threshold * 0.75:
        label = "occasional fidgeting"
    elif velocity_rms > cfg.velocity_rms_threshold * 0.3:
        label = "natural expressive movement"
    else:
        label = "stable composure"

    metrics = RestlessnessMetrics(
        velocity_rms=round(velocity_rms, 4),
        repetitive_motion_events=repetitive_count,
        overly_still_pct=round(overly_still_pct, 4),
        label=label,
    )
    return RestlessnessResult(metrics=metrics, restless_events=restless_events)


def _stack_anchors(frames: list[FrameLandmarks]) -> np.ndarray:
    """Stack (nose, L-shoulder, R-shoulder, L-wrist, R-wrist) into (n, 5, 3)."""
    n = len(frames)
    out = np.full((n, 5, 3), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        if f.pose_world_landmarks is not None:
            wl = f.pose_world_landmarks
            out[i, 0] = wl[NOSE]
            out[i, 1] = wl[LEFT_SHOULDER]
            out[i, 2] = wl[RIGHT_SHOULDER]
        for h in f.hands:
            target = 3 if h.handedness == "Left" else 4
            out[i, target] = h.world_landmarks[0]
    return out


def _combined_speed(anchors: np.ndarray, pts: np.ndarray) -> np.ndarray:
    n, k, _ = anchors.shape
    if n < 2:
        return np.full(n, np.nan, dtype=np.float32)
    dt = np.diff(pts, prepend=pts[0])
    dt[dt <= 0] = np.nan
    speed = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if i == 0:
            continue
        diff = anchors[i] - anchors[i - 1]
        dist = np.linalg.norm(diff, axis=1)
        valid = ~np.isnan(dist)
        if valid.any():
            speed[i] = float(np.nansum(dist)) / dt[i]
    return speed


def _rolling_rms(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = x[lo:hi]
        w = w[~np.isnan(w)]
        if len(w) >= 3:
            out[i] = float(np.sqrt(np.mean(w * w)))
    return out


def _count_repetitive_events(signal: np.ndarray, pts: np.ndarray) -> int:
    cfg = CONFIG.restlessness
    valid = ~np.isnan(signal)
    if valid.sum() < 32:
        return 0
    sig = signal.copy()
    sig[~valid] = np.nanmean(sig)
    sig = sig - np.mean(sig)

    dt = float(np.median(np.diff(pts))) if len(pts) > 1 else 0.1
    fs = 1.0 / max(dt, 1e-3)

    # Chunk the signal into windows and test each for dominant-peak in repetitive range.
    chunk_len = int(round(8.0 / max(dt, 1e-3)))  # 8-second windows
    if chunk_len < 16:
        return 0
    step = chunk_len // 2
    lo_hz, hi_hz = cfg.repetitive_hz_range

    count = 0
    for s in range(0, len(sig) - chunk_len + 1, step):
        chunk = sig[s : s + chunk_len]
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / fs)
        band_mask = (freqs >= lo_hz) & (freqs <= hi_hz)
        if not band_mask.any():
            continue
        band_power = spectrum[band_mask]
        if band_power.size == 0:
            continue
        band_peak = float(np.max(band_power))
        total_power = float(np.sum(spectrum[1:]) + 1e-9)
        if band_peak / total_power > 0.25:
            count += 1
    return count
