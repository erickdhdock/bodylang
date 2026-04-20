from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class BehaviourEvent(BaseModel):
    start_sec: float
    end_sec: float
    duration_sec: float
    event_type: str
    severity: Optional[Literal["brief", "moderate", "prolonged"]] = None
    peak_value: Optional[float] = None
    detail: Optional[str] = None


class EngagementMetrics(BaseModel):
    look_away_pct: float = Field(ge=0.0, le=1.0)
    engaged_pct: float = Field(ge=0.0, le=1.0)
    look_away_count: int
    prolonged_look_away_count: int
    longest_look_away_sec: float
    mean_yaw_deg: float
    mean_pitch_deg: float
    mean_roll_deg: float
    label: Literal[
        "visually engaged",
        "occasional glances",
        "frequent gaze avoidance",
    ]


class GazeMetrics(BaseModel):
    gaze_drift_pct: float = Field(ge=0.0, le=1.0)
    gaze_drift_count: int
    prolonged_gaze_drift_count: int
    longest_gaze_drift_sec: float
    mean_horizontal_gaze: float   # signed; positive = subject's right
    mean_vertical_gaze: float     # signed; positive = down
    blink_fraction: float = Field(ge=0.0, le=1.0)
    label: Literal[
        "steady gaze",
        "occasional gaze drift",
        "frequent gaze drift",
    ]


class PostureMetrics(BaseModel):
    stability_score: float = Field(ge=0.0, le=1.0)
    torso_sway_cm: float
    head_drift_cm: float
    mean_torso_angle_deg: float
    lean_label: Literal["neutral", "forward", "backward"]
    positional_shift_count: int
    label: Literal[
        "stable posture",
        "stable posture with mild sway",
        "unstable seated movement",
        "overly rigid",
    ]


class GestureMetrics(BaseModel):
    hand_activity_pct: float = Field(ge=0.0, le=1.0)
    mean_gesture_amplitude_cm: float
    gesture_burst_count: int
    nod_count: int
    inappropriate_gesture_count: int = 0
    label: Literal[
        "limited body-language usage",
        "moderate gesture usage",
        "excessive gesture usage",
    ]


class RestlessnessMetrics(BaseModel):
    velocity_rms: float
    repetitive_motion_events: int
    overly_still_pct: float = Field(ge=0.0, le=1.0)
    label: Literal[
        "stable composure",
        "natural expressive movement",
        "occasional fidgeting",
        "restless movement",
        "repetitive motion",
        "overly still",
    ]


class Timeline(BaseModel):
    look_away: list[BehaviourEvent] = Field(default_factory=list)
    engaged: list[BehaviourEvent] = Field(default_factory=list)
    gaze_drift: list[BehaviourEvent] = Field(default_factory=list)
    gesture_bursts: list[BehaviourEvent] = Field(default_factory=list)
    posture_shifts: list[BehaviourEvent] = Field(default_factory=list)
    restless_windows: list[BehaviourEvent] = Field(default_factory=list)
    inappropriate_gestures: list[BehaviourEvent] = Field(default_factory=list)
    low_confidence: list[BehaviourEvent] = Field(default_factory=list)


class BodyLangReport(BaseModel):
    job_id: str
    duration_sec: float
    source_fps: float
    analysis_fps: float
    frames_analyzed: int
    frames_skipped: int
    accuracy_confidence_ai: Literal["low", "mid", "high"]
    accuracy_confidence_reason: str

    engagement: EngagementMetrics
    gaze: GazeMetrics
    posture: PostureMetrics
    gestures: GestureMetrics
    restlessness: RestlessnessMetrics
    timeline: Timeline

    overall_labels: list[str]
    evidence: list[str]
    coaching: list[str]

    model_versions: dict[str, str]
    config_snapshot: dict
