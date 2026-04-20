from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


@dataclass
class IngestionConfig:
    allowed_modes: tuple = ("upload", "url")
    max_size_mb: int = 2048
    max_duration_min: int = 45
    url_download_timeout_sec: int = 300
    temp_dir: str = os.getenv("BL_TEMP_DIR", "/tmp/bodylang")
    allowed_content_types: tuple = (
        "video/mp4",
        "video/quicktime",
        "video/webm",
        "video/x-matroska",
    )


@dataclass
class ProcessingConfig:
    analysis_fps: float = 10.0
    target_input_width: int = 1280
    pose_variant: str = "lite"  # "lite" | "full" | "heavy"
    gpu_delegate: bool = False
    max_workers: int = 1


@dataclass
class QualityConfig:
    min_face_confidence: float = 0.5
    min_pose_confidence: float = 0.5
    min_hand_confidence: float = 0.5
    min_visibility: float = 0.5
    skip_frame_on_low_confidence: bool = True
    max_skipped_ratio: float = 0.30
    low_luminance_threshold: float = 40.0
    mid_luminance_threshold: float = 70.0
    mid_skipped_ratio: float = 0.10
    low_skipped_ratio: float = 0.20


@dataclass
class SmoothingConfig:
    window_frames: int = 5
    max_interp_gap_frames: int = 3
    one_euro_enabled: bool = False


@dataclass
class EngagementConfig:
    yaw_threshold_deg: float = 20.0
    pitch_threshold_deg: float = 15.0
    min_look_away_sec: float = 0.4
    prolonged_look_away_sec: float = 2.0


@dataclass
class GazeConfig:
    # Advisory signal, independent of head-pose engagement. Catches the
    # "head forward, eyes on second monitor" case. Thresholds apply to
    # MediaPipe gaze blendshape scores in [0, 1].
    enabled: bool = True
    horizontal_threshold: float = 0.35
    vertical_threshold: float = 0.35
    blink_gate: float = 0.5
    min_drift_sec: float = 1.0
    prolonged_drift_sec: float = 3.0


@dataclass
class PostureConfig:
    stability_window_sec: float = 2.0
    sway_threshold_cm: float = 4.0
    lean_threshold_deg: float = 10.0
    drift_threshold_cm: float = 3.0


@dataclass
class GestureConfig:
    amplitude_threshold_cm: float = 5.0
    limited_activity_pct: float = 0.10
    excessive_activity_pct: float = 0.60
    nod_min_amplitude_deg: float = 8.0
    nod_freq_range_hz: tuple = (1.0, 3.0)
    active_wrist_velocity_cm_per_sec: float = 8.0


@dataclass
class RestlessnessConfig:
    velocity_window_sec: float = 1.0
    velocity_rms_threshold: float = 0.06
    repetitive_hz_range: tuple = (0.5, 3.0)
    overly_still_velocity_rms: float = 0.005


@dataclass
class ApiConfig:
    host: str = os.getenv("BL_HOST", "0.0.0.0")
    port: int = int(os.getenv("BL_PORT", "8080"))
    auth_token: str = os.getenv("BL_AUTH_TOKEN", "")
    queue_max_size: int = 16
    storage_dir: str = os.getenv("BL_STORAGE_DIR", "/var/bodylang/results")
    result_ttl_hours: int = 72


@dataclass
class WebhookConfig:
    enabled: bool = True
    timeout_sec: int = 15
    retry_count: int = 3
    retry_backoff_sec: float = 2.0


@dataclass
class ModelPaths:
    model_dir: str = os.getenv("BL_MODEL_DIR", "./models")
    face_landmarker: str = "face_landmarker.task"
    pose_landmarker_lite: str = "pose_landmarker_lite.task"
    pose_landmarker_full: str = "pose_landmarker_full.task"
    pose_landmarker_heavy: str = "pose_landmarker_heavy.task"
    hand_landmarker: str = "hand_landmarker.task"

    face_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )
    pose_lite_url: str = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )
    pose_full_url: str = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    )
    pose_heavy_url: str = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )
    hand_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )


@dataclass
class Config:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    restlessness: RestlessnessConfig = field(default_factory=RestlessnessConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    models: ModelPaths = field(default_factory=ModelPaths)

    def snapshot(self) -> dict:
        return asdict(self)


CONFIG = Config()
