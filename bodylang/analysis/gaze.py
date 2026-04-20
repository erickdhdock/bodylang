from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CONFIG
from landmarks import FrameLandmarks
from schema import BehaviourEvent, GazeMetrics
from smoothing import moving_average, rle_events


@dataclass
class GazeResult:
    metrics: GazeMetrics
    gaze_drift_events: list[BehaviourEvent]


def _bs(d: dict[str, float] | None, k: str) -> float:
    if d is None:
        return float("nan")
    return float(d.get(k, float("nan")))


def analyze_gaze(frames: list[FrameLandmarks]) -> GazeResult:
    cfg = CONFIG.gaze
    smoothing = CONFIG.smoothing
    n = len(frames)
    pts = np.array([f.pts_sec for f in frames], dtype=np.float32)

    gaze_x = np.full(n, np.nan, dtype=np.float32)
    gaze_y = np.full(n, np.nan, dtype=np.float32)
    blink = np.full(n, np.nan, dtype=np.float32)

    # Signs: subject's right = +x (right eye looks outward; left eye looks inward).
    # Subject looking down = +y.
    for i, f in enumerate(frames):
        bs = f.face_blendshapes
        if bs is None:
            continue
        right_side = 0.5 * (_bs(bs, "eyeLookOutRight") + _bs(bs, "eyeLookInLeft"))
        left_side  = 0.5 * (_bs(bs, "eyeLookOutLeft")  + _bs(bs, "eyeLookInRight"))
        down_side  = 0.5 * (_bs(bs, "eyeLookDownLeft") + _bs(bs, "eyeLookDownRight"))
        up_side    = 0.5 * (_bs(bs, "eyeLookUpLeft")   + _bs(bs, "eyeLookUpRight"))
        if not (np.isnan(right_side) or np.isnan(left_side)):
            gaze_x[i] = right_side - left_side
        if not (np.isnan(down_side) or np.isnan(up_side)):
            gaze_y[i] = down_side - up_side
        bl_l = _bs(bs, "eyeBlinkLeft")
        bl_r = _bs(bs, "eyeBlinkRight")
        if not (np.isnan(bl_l) and np.isnan(bl_r)):
            blink[i] = np.nanmax([bl_l, bl_r])

    # Blink gate — drop gaze readings while either eye is closed.
    blink_mask = (blink > cfg.blink_gate) & ~np.isnan(blink)
    gaze_x[blink_mask] = np.nan
    gaze_y[blink_mask] = np.nan

    gaze_x = moving_average(gaze_x, smoothing.window_frames)
    gaze_y = moving_average(gaze_y, smoothing.window_frames)

    valid = ~(np.isnan(gaze_x) | np.isnan(gaze_y))
    n_valid = int(valid.sum())

    h_flag = np.abs(np.nan_to_num(gaze_x, nan=0.0)) > cfg.horizontal_threshold
    v_flag = np.abs(np.nan_to_num(gaze_y, nan=0.0)) > cfg.vertical_threshold
    drift_flags = (h_flag | v_flag) & valid

    # Axis-classification helper for the event `detail` field.
    def _axis_label(start: int, end: int) -> str:
        any_h = bool(h_flag[start : end + 1].any())
        any_v = bool(v_flag[start : end + 1].any())
        if any_h and any_v:
            return "both"
        if any_h:
            return "horizontal"
        return "vertical"

    runs = rle_events(
        drift_flags, pts, min_duration_sec=cfg.min_drift_sec, values=gaze_x
    )
    events: list[BehaviourEvent] = []
    for e in runs:
        if e.duration_sec >= cfg.prolonged_drift_sec:
            severity = "prolonged"
        elif e.duration_sec >= cfg.prolonged_drift_sec * 0.5:
            severity = "moderate"
        else:
            severity = "brief"
        events.append(
            BehaviourEvent(
                start_sec=round(e.start_sec, 3),
                end_sec=round(e.end_sec, 3),
                duration_sec=round(e.duration_sec, 3),
                event_type="gaze_drift",
                severity=severity,
                peak_value=round(e.peak_value, 3) if e.peak_value else None,
                detail=_axis_label(e.start_idx, e.end_idx),
            )
        )

    drift_pct = (drift_flags.sum() / n_valid) if n_valid > 0 else 0.0
    longest = max((e.duration_sec for e in events), default=0.0)
    prolonged_count = sum(
        1 for e in events if e.duration_sec >= cfg.prolonged_drift_sec
    )
    blink_valid = ~np.isnan(blink)
    blink_fraction = (
        float((blink[blink_valid] > cfg.blink_gate).sum() / blink_valid.sum())
        if blink_valid.any()
        else 0.0
    )

    if drift_pct < 0.08 and prolonged_count == 0:
        label = "steady gaze"
    elif drift_pct < 0.20 and prolonged_count <= 2:
        label = "occasional gaze drift"
    else:
        label = "frequent gaze drift"

    metrics = GazeMetrics(
        gaze_drift_pct=round(float(drift_pct), 4),
        gaze_drift_count=len(events),
        prolonged_gaze_drift_count=prolonged_count,
        longest_gaze_drift_sec=round(float(longest), 3),
        mean_horizontal_gaze=round(float(np.nanmean(gaze_x)) if n_valid else 0.0, 4),
        mean_vertical_gaze=round(float(np.nanmean(gaze_y)) if n_valid else 0.0, 4),
        blink_fraction=round(blink_fraction, 4),
        label=label,
    )

    return GazeResult(metrics=metrics, gaze_drift_events=events)
