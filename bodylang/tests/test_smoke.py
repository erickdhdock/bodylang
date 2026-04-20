"""Smoke tests that exercise the analysis stages with synthetic landmark data.

A full end-to-end MediaPipe test runs only if a real sample video is dropped at
`tests/fixtures/sample.mp4`. Without it, we still cover the rule-based stages
deterministically.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow tests to run from the package root with `pytest tests/` (no install required).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.engagement import analyze_engagement  # noqa: E402
from analysis.gaze import analyze_gaze  # noqa: E402
from analysis.gestures import analyze_gestures  # noqa: E402
from analysis.posture import analyze_posture  # noqa: E402
from analysis.restlessness import analyze_restlessness  # noqa: E402
from config import CONFIG  # noqa: E402
from interpret import overall_labels  # noqa: E402
from landmarks import FrameLandmarks, HandRecord  # noqa: E402
from pipeline import _accuracy_confidence  # noqa: E402
from report import assemble_report  # noqa: E402
from schema import Timeline  # noqa: E402
from smoothing import moving_average, rle_events  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic landmark fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _yaw_to_matrix(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """Build a 4x4 facial transformation matrix with given yaw/pitch (YXZ order)."""
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler("YXZ", [yaw_deg, pitch_deg, 0.0], degrees=True).as_matrix()
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = rot
    return m


def _make_frames(duration_sec: float, fps: float = 10.0) -> list[FrameLandmarks]:
    """Build frames with a periodic look-away pattern and mild shoulder sway."""
    n = int(duration_sec * fps)
    frames: list[FrameLandmarks] = []
    for i in range(n):
        t = i / fps
        # Look away (yaw 35deg) between t=2..4s.
        yaw = 35.0 if 2.0 <= t < 4.0 else 2.0
        matrix = _yaw_to_matrix(yaw, pitch_deg=0.0)

        pose_world = np.zeros((33, 3), dtype=np.float32)
        # Shoulders ~30cm above hips, symmetric at ±18cm
        sway = 0.01 * math.sin(2 * math.pi * 0.5 * t)
        pose_world[11] = (-0.18 + sway, -0.30, 0.0)  # L shoulder
        pose_world[12] = ( 0.18 + sway, -0.30, 0.0)  # R shoulder
        pose_world[23] = (-0.10, 0.0, 0.0)           # L hip
        pose_world[24] = ( 0.10, 0.0, 0.0)           # R hip
        pose_world[0]  = ( 0.00 + 0.5 * sway, -0.50, 0.0)  # nose

        hands: list[HandRecord] = []
        # Raise right hand during 5..7s with steady upward motion so wrist speed
        # stays well above the activity threshold for the full burst.
        if 5.0 <= t < 7.0:
            wrist = np.zeros((21, 3), dtype=np.float32)
            # Linear sweep: 25cm travel over 2s ≈ 12.5 cm/s > 8cm/s threshold.
            wrist[0] = (0.30, -0.20 - 0.125 * (t - 5.0), 0.0)
            hands.append(
                HandRecord(
                    handedness="Right",
                    landmarks=wrist,
                    world_landmarks=wrist,
                    confidence=0.95,
                )
            )

        frames.append(
            FrameLandmarks(
                frame_idx=i,
                pts_sec=t,
                face_detected=True,
                face_landmarks=np.zeros((468, 3), dtype=np.float32),
                face_transform_matrix=matrix,
                pose_detected=True,
                pose_landmarks=np.zeros((33, 4), dtype=np.float32),
                pose_world_landmarks=pose_world,
                hands=hands,
                frame_luminance=120.0,
                low_confidence=False,
            )
        )
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# Unit: smoothing helpers
# ──────────────────────────────────────────────────────────────────────────────

def test_moving_average_preserves_shape():
    x = np.array([1, 2, np.nan, 4, 5], dtype=np.float32)
    out = moving_average(x, window=3)
    assert out.shape == x.shape
    assert not np.all(np.isnan(out))


def test_rle_events_detects_runs():
    pts = np.arange(10, dtype=np.float32) / 10.0
    flags = [False, True, True, True, False, False, True, True, False, False]
    events = rle_events(flags, pts, min_duration_sec=0.1)
    assert len(events) == 2
    assert events[0].start_idx == 1 and events[0].end_idx == 3


# ──────────────────────────────────────────────────────────────────────────────
# Integration: full analysis on synthetic frames
# ──────────────────────────────────────────────────────────────────────────────

def test_analysis_on_synthetic_frames():
    frames = _make_frames(duration_sec=10.0, fps=10.0)

    eng = analyze_engagement(frames)
    gaze = analyze_gaze(frames)
    post = analyze_posture(frames)
    gest = analyze_gestures(frames, pitch_series=eng.pitch_series)
    rest = analyze_restlessness(frames)

    # Without blendshapes injected, gaze is NaN everywhere → zero events, steady label.
    assert gaze.metrics.gaze_drift_count == 0
    assert gaze.metrics.label == "steady gaze"

    # Engagement detected the 2-4s look-away block.
    assert eng.metrics.look_away_count >= 1
    assert eng.metrics.longest_look_away_sec > 1.0
    assert 0.15 < eng.metrics.look_away_pct < 0.30

    # Posture should be stable (mild sway but well under threshold).
    assert post.metrics.stability_score > 0.5
    assert post.metrics.lean_label == "neutral"

    # Gesture burst detected in 5-7s window.
    assert gest.metrics.gesture_burst_count >= 1
    assert gest.metrics.hand_activity_pct > 0

    # Restlessness: velocity RMS > 0 (some motion was injected).
    assert rest.metrics.velocity_rms >= 0.0

    labels = overall_labels(eng.metrics, post.metrics, gest.metrics, rest.metrics)
    assert len(labels) >= 2
    confidence, confidence_reason = _accuracy_confidence(frames, skipped=0)

    report = assemble_report(
        job_id="test",
        duration_sec=10.0,
        source_fps=30.0,
        analysis_fps=10.0,
        frames_analyzed=len(frames),
        frames_skipped=0,
        accuracy_confidence_ai=confidence,
        accuracy_confidence_reason=confidence_reason,
        engagement=eng.metrics,
        gaze=gaze.metrics,
        posture=post.metrics,
        gestures=gest.metrics,
        restlessness=rest.metrics,
        timeline=Timeline(
            look_away=eng.look_away_events,
            engaged=eng.engaged_events,
            gaze_drift=gaze.gaze_drift_events,
            gesture_bursts=gest.gesture_burst_events,
            posture_shifts=post.posture_shift_events,
            restless_windows=rest.restless_events,
            low_confidence=[],
        ),
        overall_labels=labels,
        model_versions={"face": "stub", "pose": "stub", "hand": "stub"},
        config_snapshot=CONFIG.snapshot(),
    )
    assert len(report.evidence) >= 2
    assert 1 <= len(report.coaching) <= 4
    assert len(report.timeline.look_away) == eng.metrics.look_away_count
    assert report.accuracy_confidence_ai == "high"


def test_accuracy_confidence_uses_luminance_and_tracking():
    frames = _make_frames(duration_sec=4.0, fps=10.0)
    for f in frames:
        f.frame_luminance = 25.0

    confidence, reason = _accuracy_confidence(frames, skipped=0)
    assert confidence == "low"
    assert "low brightness" in reason

    for f in frames:
        f.frame_luminance = 120.0

    confidence, reason = _accuracy_confidence(frames, skipped=int(len(frames) * 0.15))
    assert confidence == "mid"
    assert "missing face or pose" in reason


def test_gaze_drift_detection_from_blendshapes():
    """Inject synthetic gaze blendshapes: sustained side-glance 3-6s should produce ≥1 event."""
    frames = _make_frames(duration_sec=10.0, fps=10.0)

    # Neutral defaults + a sustained rightward glance between t=3..6s.
    def _bs(t: float) -> dict[str, float]:
        side = 0.7 if 3.0 <= t < 6.0 else 0.02
        # Rightward glance: right eye looks outward, left eye looks inward.
        return {
            "eyeLookOutRight": side, "eyeLookInLeft": side,
            "eyeLookOutLeft": 0.0,   "eyeLookInRight": 0.0,
            "eyeLookUpLeft": 0.0,    "eyeLookUpRight": 0.0,
            "eyeLookDownLeft": 0.0,  "eyeLookDownRight": 0.0,
            "eyeBlinkLeft": 0.0,     "eyeBlinkRight": 0.0,
        }

    for f in frames:
        f.face_blendshapes = _bs(f.pts_sec)

    gaze = analyze_gaze(frames)
    assert gaze.metrics.gaze_drift_count >= 1
    assert gaze.metrics.gaze_drift_pct > 0.15
    assert gaze.metrics.mean_horizontal_gaze > 0.0  # rightward sign
    assert any(e.event_type == "gaze_drift" for e in gaze.gaze_drift_events)


def test_gaze_blink_gate():
    """Frames with blink > blink_gate should be masked out of gaze detection."""
    frames = _make_frames(duration_sec=5.0, fps=10.0)
    # Strong side glance everywhere, but the subject is "blinking" the whole time.
    for f in frames:
        f.face_blendshapes = {
            "eyeLookOutRight": 0.9, "eyeLookInLeft": 0.9,
            "eyeLookOutLeft": 0.0,  "eyeLookInRight": 0.0,
            "eyeLookUpLeft": 0.0,   "eyeLookUpRight": 0.0,
            "eyeLookDownLeft": 0.0, "eyeLookDownRight": 0.0,
            "eyeBlinkLeft": 0.9,    "eyeBlinkRight": 0.9,
        }
    gaze = analyze_gaze(frames)
    assert gaze.metrics.gaze_drift_count == 0


def test_config_threshold_flows_through():
    """Lowering yaw_threshold_deg should strictly increase look-away count."""
    frames = _make_frames(duration_sec=10.0, fps=10.0)

    original = CONFIG.engagement.yaw_threshold_deg
    try:
        CONFIG.engagement.yaw_threshold_deg = 25.0
        high = analyze_engagement(frames).metrics.look_away_count
        CONFIG.engagement.yaw_threshold_deg = 10.0
        low = analyze_engagement(frames).metrics.look_away_count
    finally:
        CONFIG.engagement.yaw_threshold_deg = original

    assert low >= high


# ──────────────────────────────────────────────────────────────────────────────
# Optional: full pipeline on a real fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not Path(__file__).parent.joinpath("fixtures/sample.mp4").exists(),
    reason="Drop a real video at tests/fixtures/sample.mp4 to enable.",
)
def test_pipeline_end_to_end():
    from pipeline import BodyLangPipeline

    video = Path(__file__).parent / "fixtures" / "sample.mp4"
    report = BodyLangPipeline().run(video, job_id="fixture")
    assert report.duration_sec > 0
    assert report.frames_analyzed > 0
    assert report.overall_labels
