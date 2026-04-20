"""End-to-end CLI runner for the bodylang pipeline.

Runs the full pipeline directly (no FastAPI server needed) on one video or every
video in a directory, and writes a JSON report next to each input.

Examples
--------

    # Single video
    python test_run.py --video /path/to/interview.mp4

    # Whole directory (all mp4/mov/webm)
    python test_run.py --video_dir /path/to/clips/ --out_dir ./reports/

    # Override a few config knobs for this run only
    python test_run.py --video clip.mp4 --analysis_fps 6 --pose_variant full --cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from config import CONFIG
from pipeline import BodyLangPipeline, PipelineError

logger = logging.getLogger("test_run")

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".m4v")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run bodylang pipeline on local video(s).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Path to a single video file.")
    src.add_argument(
        "--video_dir", type=Path, help="Directory containing video files to process."
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory for report JSON files. Defaults to alongside each video.",
    )
    p.add_argument(
        "--analysis_fps",
        type=float,
        default=None,
        help="Override processing.analysis_fps for this run.",
    )
    p.add_argument(
        "--pose_variant",
        choices=("lite", "full", "heavy"),
        default=None,
        help="Override processing.pose_variant for this run.",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (ignore GPU delegate).",
    )
    p.add_argument(
        "--print_report",
        action="store_true",
        help="Print the full JSON report to stdout after each video.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _apply_overrides(args: argparse.Namespace) -> None:
    if args.analysis_fps is not None:
        CONFIG.processing.analysis_fps = args.analysis_fps
    if args.pose_variant is not None:
        CONFIG.processing.pose_variant = args.pose_variant
    if args.cpu:
        CONFIG.processing.gpu_delegate = False


def _collect_videos(args: argparse.Namespace) -> list[Path]:
    if args.video is not None:
        if not args.video.exists():
            raise FileNotFoundError(f"Video not found: {args.video}")
        return [args.video]
    assert args.video_dir is not None
    if not args.video_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {args.video_dir}")
    vids = sorted(
        p for p in args.video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not vids:
        raise FileNotFoundError(
            f"No videos with extensions {VIDEO_EXTS} in {args.video_dir}"
        )
    return vids


def _resolve_out_path(video: Path, out_dir: Path | None) -> Path:
    if out_dir is None:
        return video.with_suffix(".bodylang.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{video.stem}.bodylang.json"


def _on_progress(pct: float, stage: str) -> None:
    sys.stderr.write(f"\r  [{stage:<12}] {pct * 100:5.1f}%")
    sys.stderr.flush()


def _run_one(pipeline: BodyLangPipeline, video: Path, out_path: Path, print_report: bool) -> bool:
    job_id = video.stem
    t0 = time.time()
    print(f"\n▶ {video.name}  →  {out_path.name}")
    try:
        report = pipeline.run(video, job_id=job_id, on_progress=_on_progress)
    except PipelineError as e:
        print(f"\n  ✖ pipeline error: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"\n  ✖ unexpected error: {e!r}")
        return False
    finally:
        sys.stderr.write("\n")
        sys.stderr.flush()

    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    elapsed = time.time() - t0

    e = report.engagement
    gz = report.gaze
    po = report.posture
    g = report.gestures
    r = report.restlessness

    print(f"  ✔ {elapsed:.1f}s  ({report.frames_analyzed} frames, {report.frames_skipped} skipped)")
    print(f"    engagement    : {e.label}  ({e.look_away_pct * 100:.0f}% away, "
          f"{e.look_away_count} events, longest {e.longest_look_away_sec:.1f}s)")
    print(f"    gaze (advisory): {gz.label}  ({gz.gaze_drift_pct * 100:.0f}% drift, "
          f"{gz.gaze_drift_count} events, longest {gz.longest_gaze_drift_sec:.1f}s)")
    print(f"    posture       : {po.label}  (stability {po.stability_score:.2f}, "
          f"sway {po.torso_sway_cm:.1f}cm, lean {po.lean_label})")
    flip_str = f", {g.inappropriate_gesture_count} inappropriate" if g.inappropriate_gesture_count else ""
    print(f"    gestures      : {g.label}  ({g.gesture_burst_count} bursts, "
          f"{g.hand_activity_pct * 100:.0f}% active, {g.nod_count} nods{flip_str})")
    print(f"    restlessness  : {r.label}  (rms {r.velocity_rms:.3f}, "
          f"{r.repetitive_motion_events} repetitive windows)")
    print(f"    overall labels: {', '.join(report.overall_labels)}")
    print(f"    timeline events: look_away={len(report.timeline.look_away)}, "
          f"gaze_drift={len(report.timeline.gaze_drift)}, "
          f"gesture_bursts={len(report.timeline.gesture_bursts)}, "
          f"posture_shifts={len(report.timeline.posture_shifts)}, "
          f"restless={len(report.timeline.restless_windows)}, "
          f"inappropriate={len(report.timeline.inappropriate_gestures)}")

    if print_report:
        print("\n--- full report ---")
        print(json.dumps(report.model_dump(), indent=2))
        print("--- end report ---\n")
    return True


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    _apply_overrides(args)
    videos = _collect_videos(args)

    print(f"bodylang e2e runner — {len(videos)} video(s)")
    print(f"  analysis_fps = {CONFIG.processing.analysis_fps}")
    print(f"  pose_variant = {CONFIG.processing.pose_variant}")
    print(f"  gpu_delegate = {CONFIG.processing.gpu_delegate}")

    pipeline = BodyLangPipeline()
    ok = fail = 0
    for v in videos:
        out = _resolve_out_path(v, args.out_dir)
        if _run_one(pipeline, v, out, args.print_report):
            ok += 1
        else:
            fail += 1

    print(f"\nDone — {ok} succeeded, {fail} failed.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
