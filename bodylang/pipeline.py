from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from analysis.engagement import analyze_engagement
from analysis.gaze import analyze_gaze
from analysis.gestures import analyze_gestures
from analysis.posture import analyze_posture
from analysis.restlessness import analyze_restlessness
from config import CONFIG
from frames import iter_frames, probe
from interpret import overall_labels
from landmarks import FrameLandmarks, LandmarkerBundle
from report import assemble_report
from schema import BodyLangReport, Timeline

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[float, str], None]


@dataclass
class PipelineError(Exception):
    message: str

    def __str__(self) -> str:  # noqa: D401
        return self.message


def _accuracy_confidence(
    frames: list[FrameLandmarks],
    skipped: int,
) -> tuple[str, str]:
    """Estimate report reliability from lighting and landmark tracking quality."""
    q = CONFIG.quality
    n_total = max(1, len(frames))
    skip_ratio = skipped / n_total

    luminance = np.array(
        [
            f.frame_luminance
            for f in frames
            if f.frame_luminance is not None and np.isfinite(f.frame_luminance)
        ],
        dtype=np.float32,
    )

    confidence = "high"
    reasons: list[str] = []

    def downgrade(level: str) -> None:
        nonlocal confidence
        rank = {"low": 0, "mid": 1, "high": 2}
        if rank[level] < rank[confidence]:
            confidence = level

    if luminance.size == 0:
        downgrade("mid")
        reasons.append("brightness unavailable")
        mean_luma = None
    else:
        mean_luma = float(np.mean(luminance))
        low_luma_pct = float(np.mean(luminance < q.low_luminance_threshold))
        mid_luma_pct = float(np.mean(luminance < q.mid_luminance_threshold))

        if mean_luma < q.low_luminance_threshold or low_luma_pct >= 0.35:
            downgrade("low")
            reasons.append(
                "low brightness "
                f"(mean luminance {mean_luma:.1f}/255, "
                f"{low_luma_pct:.0%} frames below {q.low_luminance_threshold:.0f})"
            )
        elif mean_luma < q.mid_luminance_threshold or mid_luma_pct >= 0.35:
            downgrade("mid")
            reasons.append(
                "dim brightness "
                f"(mean luminance {mean_luma:.1f}/255, "
                f"{mid_luma_pct:.0%} frames below {q.mid_luminance_threshold:.0f})"
            )

    if skip_ratio >= q.low_skipped_ratio:
        downgrade("low")
        reasons.append(f"many frames had missing face or pose ({skip_ratio:.0%})")
    elif skip_ratio >= q.mid_skipped_ratio:
        downgrade("mid")
        reasons.append(f"some frames had missing face or pose ({skip_ratio:.0%})")

    if not reasons:
        if mean_luma is None:
            reasons.append(f"tracking quality was adequate ({skip_ratio:.0%} low-confidence frames)")
        else:
            reasons.append(
                "brightness and tracking quality were adequate "
                f"(mean luminance {mean_luma:.1f}/255, "
                f"{skip_ratio:.0%} low-confidence frames)"
            )

    return confidence, "; ".join(reasons)


class BodyLangPipeline:
    """Staged pipeline: ingest → landmarks → analysis → report."""

    def __init__(self) -> None:
        self.cfg = CONFIG

    def run(
        self,
        video_path: str | Path,
        job_id: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> BodyLangReport:
        t0 = time.time()
        video_path = Path(video_path)

        # ── 1. Probe ───────────────────────────────────────────────────────
        info = probe(video_path)
        logger.info(
            "[%s] video: %.1fs, %.2f fps, %dx%d",
            job_id, info.duration_sec, info.src_fps, info.width, info.height,
        )

        analysis_fps = self.cfg.processing.analysis_fps
        expected_out_frames = max(
            1, int(info.duration_sec * analysis_fps)
        )

        # ── 2-3. Frames + Landmarks ────────────────────────────────────────
        frames: list[FrameLandmarks] = []
        skipped = 0
        with LandmarkerBundle() as bundle:
            for f in iter_frames(
                video_path,
                analysis_fps=analysis_fps,
                target_input_width=self.cfg.processing.target_input_width,
            ):
                rec = bundle.detect(f.rgb, f.frame_idx, f.pts_sec)
                if rec.low_confidence:
                    skipped += 1
                frames.append(rec)

                if on_progress and len(frames) % 50 == 0:
                    pct = min(0.7, 0.05 + 0.65 * (len(frames) / expected_out_frames))
                    on_progress(pct, "landmarks")
            model_versions = dict(bundle.model_versions)
            delegate_used = bundle.delegate_used

        n_total = len(frames)
        if n_total == 0:
            raise PipelineError("No frames were extracted from the video.")

        skip_ratio = skipped / n_total
        max_ratio = self.cfg.quality.max_skipped_ratio
        logger.info(
            "[%s] extracted %d frames, %d low-confidence (%.1f%%, limit %.1f%%)",
            job_id, n_total, skipped, skip_ratio * 100, max_ratio * 100,
        )
        if skip_ratio > max_ratio:
            raise PipelineError(
                f"Too many low-confidence frames ({skip_ratio:.1%} > {max_ratio:.1%}). "
                "Subject may not be visible or well-lit."
            )

        # ── 4-8. Analysis ──────────────────────────────────────────────────
        if on_progress:
            on_progress(0.75, "analysis")
        eng = analyze_engagement(frames)
        gaze = analyze_gaze(frames)
        post = analyze_posture(frames)
        gest = analyze_gestures(frames, pitch_series=eng.pitch_series)
        rest = analyze_restlessness(frames)

        # ── Low-confidence timeline (occlusion / no subject) ──────────────
        from smoothing import rle_events
        pts = np.array([f.pts_sec for f in frames], dtype=np.float32)
        lowconf_flags = np.array([f.low_confidence for f in frames], dtype=bool)
        lowconf_runs = rle_events(lowconf_flags, pts, min_duration_sec=0.3)
        from schema import BehaviourEvent as _E
        lowconf_events = [
            _E(
                start_sec=round(e.start_sec, 3),
                end_sec=round(e.end_sec, 3),
                duration_sec=round(e.duration_sec, 3),
                event_type="low_confidence",
                detail="subject_not_detected",
            )
            for e in lowconf_runs
        ]

        timeline = Timeline(
            look_away=eng.look_away_events,
            engaged=eng.engaged_events,
            gaze_drift=gaze.gaze_drift_events,
            gesture_bursts=gest.gesture_burst_events,
            posture_shifts=post.posture_shift_events,
            restless_windows=rest.restless_events,
            inappropriate_gestures=gest.inappropriate_events,
            low_confidence=lowconf_events,
        )

        # ── 9. Interpretation ─────────────────────────────────────────────
        labels = overall_labels(eng.metrics, post.metrics, gest.metrics, rest.metrics)

        # ── 10. Report ────────────────────────────────────────────────────
        if on_progress:
            on_progress(0.95, "reporting")

        accuracy_confidence_ai, accuracy_confidence_reason = _accuracy_confidence(
            frames, skipped
        )
        config_snapshot = self.cfg.snapshot()
        config_snapshot["_delegate"] = delegate_used

        report = assemble_report(
            job_id=job_id,
            duration_sec=info.duration_sec,
            source_fps=info.src_fps,
            analysis_fps=analysis_fps,
            frames_analyzed=n_total,
            frames_skipped=skipped,
            accuracy_confidence_ai=accuracy_confidence_ai,
            accuracy_confidence_reason=accuracy_confidence_reason,
            engagement=eng.metrics,
            gaze=gaze.metrics,
            posture=post.metrics,
            gestures=gest.metrics,
            restlessness=rest.metrics,
            timeline=timeline,
            overall_labels=labels,
            model_versions=model_versions,
            config_snapshot=config_snapshot,
        )

        elapsed = time.time() - t0
        logger.info("[%s] pipeline complete in %.1fs", job_id, elapsed)
        if on_progress:
            on_progress(1.0, "done")
        return report
