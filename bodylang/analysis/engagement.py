from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CONFIG
from head_pose import decompose_series
from landmarks import FrameLandmarks
from schema import BehaviourEvent, EngagementMetrics
from smoothing import moving_average, rle_events


@dataclass
class EngagementResult:
    metrics: EngagementMetrics
    look_away_events: list[BehaviourEvent]
    engaged_events: list[BehaviourEvent]
    yaw_series: np.ndarray
    pitch_series: np.ndarray
    roll_series: np.ndarray


def analyze_engagement(frames: list[FrameLandmarks]) -> EngagementResult:
    cfg = CONFIG.engagement
    smoothing = CONFIG.smoothing

    matrices = [f.face_transform_matrix for f in frames]
    ypr = decompose_series(matrices)
    pts = np.array([f.pts_sec for f in frames], dtype=np.float32)

    yaw = moving_average(ypr[:, 0], smoothing.window_frames)
    pitch = moving_average(ypr[:, 1], smoothing.window_frames)
    roll = moving_average(ypr[:, 2], smoothing.window_frames)

    valid = ~np.isnan(yaw)
    n_valid = int(valid.sum())

    look_away_flags = np.zeros(len(frames), dtype=bool)
    if n_valid > 0:
        look_away_flags = (
            (np.abs(np.nan_to_num(yaw, nan=0.0)) > cfg.yaw_threshold_deg)
            | (np.abs(np.nan_to_num(pitch, nan=0.0)) > cfg.pitch_threshold_deg)
        ) & valid
    engaged_flags = valid & ~look_away_flags

    away_events = rle_events(
        look_away_flags, pts, min_duration_sec=cfg.min_look_away_sec, values=yaw
    )
    engaged_runs = rle_events(engaged_flags, pts, min_duration_sec=0.0)

    def to_schema_event(
        evt, event_type: str, prolonged_sec: float | None = None
    ) -> BehaviourEvent:
        severity = None
        if prolonged_sec is not None:
            if evt.duration_sec >= prolonged_sec:
                severity = "prolonged"
            elif evt.duration_sec >= prolonged_sec * 0.5:
                severity = "moderate"
            else:
                severity = "brief"
        return BehaviourEvent(
            start_sec=round(evt.start_sec, 3),
            end_sec=round(evt.end_sec, 3),
            duration_sec=round(evt.duration_sec, 3),
            event_type=event_type,
            severity=severity,
            peak_value=round(evt.peak_value, 2) if evt.peak_value else None,
        )

    look_away_events = [
        to_schema_event(e, "look_away", cfg.prolonged_look_away_sec) for e in away_events
    ]
    engaged_events = [to_schema_event(e, "engaged") for e in engaged_runs]

    look_away_pct = (look_away_flags.sum() / n_valid) if n_valid > 0 else 0.0
    engaged_pct = (engaged_flags.sum() / n_valid) if n_valid > 0 else 0.0
    longest_away = max((e.duration_sec for e in look_away_events), default=0.0)
    prolonged_count = sum(
        1 for e in look_away_events if e.duration_sec >= cfg.prolonged_look_away_sec
    )

    if look_away_pct < 0.08 and prolonged_count == 0:
        label = "visually engaged"
    elif look_away_pct < 0.20 and prolonged_count <= 2:
        label = "occasional glances"
    else:
        label = "frequent gaze avoidance"

    metrics = EngagementMetrics(
        look_away_pct=round(float(look_away_pct), 4),
        engaged_pct=round(float(engaged_pct), 4),
        look_away_count=len(look_away_events),
        prolonged_look_away_count=prolonged_count,
        longest_look_away_sec=round(float(longest_away), 3),
        mean_yaw_deg=round(float(np.nanmean(yaw)) if n_valid else 0.0, 2),
        mean_pitch_deg=round(float(np.nanmean(pitch)) if n_valid else 0.0, 2),
        mean_roll_deg=round(float(np.nanmean(roll)) if n_valid else 0.0, 2),
        label=label,
    )

    return EngagementResult(
        metrics=metrics,
        look_away_events=look_away_events,
        engaged_events=engaged_events,
        yaw_series=yaw,
        pitch_series=pitch,
        roll_series=roll,
    )
