"""Realtime webcam inference for bodylang, with visualisation and console output.

Examples
--------
    python realtime.py                          # default webcam, 10 Hz inference
    python realtime.py --infer_fps 5 --cpu
    python realtime.py --camera 1 --mirror --pose_variant full
    python realtime.py --no_display             # headless, console output only

Press q or Esc in the video window to quit.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque

import cv2
import numpy as np

from config import CONFIG
from head_pose import decompose_matrix
from landmarks import FrameLandmarks, LandmarkerBundle, is_middle_finger_flip


POSE_EDGES: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
)

HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

MIDDLE_EDGES: tuple[tuple[int, int], ...] = ((9, 10), (10, 11), (11, 12))

VIS_THRESHOLD = 0.3
ACCURACY_WINDOW_SEC = 3.0


def _gaze_from_blendshapes(bs: dict[str, float] | None) -> tuple[float, float, float]:
    """Return (gaze_x, gaze_y, blink) from FaceLandmarker blendshapes.

    x>0 = subject's right, y>0 = down, blink in [0,1]. NaN if bs is None.
    """
    if bs is None:
        return float("nan"), float("nan"), float("nan")
    g = lambda k: bs.get(k, 0.0)
    gx = 0.5 * (g("eyeLookOutRight") + g("eyeLookInLeft")) \
         - 0.5 * (g("eyeLookOutLeft") + g("eyeLookInRight"))
    gy = 0.5 * (g("eyeLookDownLeft") + g("eyeLookDownRight")) \
         - 0.5 * (g("eyeLookUpLeft") + g("eyeLookUpRight"))
    blink = max(g("eyeBlinkLeft"), g("eyeBlinkRight"))
    return float(gx), float(gy), float(blink)


def _gaze_drift_label(gx: float, gy: float, blink: float) -> str:
    cfg = CONFIG.gaze
    if blink > cfg.blink_gate or np.isnan(gx) or np.isnan(gy):
        return ""
    parts: list[str] = []
    if gx > cfg.horizontal_threshold:
        parts.append("right")
    elif gx < -cfg.horizontal_threshold:
        parts.append("left")
    if gy > cfg.vertical_threshold:
        parts.append("down")
    elif gy < -cfg.vertical_threshold:
        parts.append("up")
    return "+".join(parts)


def _realtime_accuracy_confidence(
    luminance_window: deque[float],
    low_confidence_window: deque[bool],
) -> tuple[str, str]:
    q = CONFIG.quality
    confidence = "high"
    reasons: list[str] = []

    def downgrade(level: str) -> None:
        nonlocal confidence
        rank = {"low": 0, "mid": 1, "high": 2}
        if rank[level] < rank[confidence]:
            confidence = level

    if luminance_window:
        luma = np.asarray(luminance_window, dtype=np.float32)
        mean_luma = float(np.mean(luma))
        low_luma_pct = float(np.mean(luma < q.low_luminance_threshold))
        mid_luma_pct = float(np.mean(luma < q.mid_luminance_threshold))
        if mean_luma < q.low_luminance_threshold or low_luma_pct >= 0.35:
            downgrade("low")
            reasons.append(f"low brightness mean={mean_luma:.0f}/255")
        elif mean_luma < q.mid_luminance_threshold or mid_luma_pct >= 0.35:
            downgrade("mid")
            reasons.append(f"dim brightness mean={mean_luma:.0f}/255")
    else:
        downgrade("mid")
        reasons.append("brightness unavailable")

    if low_confidence_window:
        missing_pct = float(np.mean(np.asarray(low_confidence_window, dtype=bool)))
        if missing_pct >= q.low_skipped_ratio:
            downgrade("low")
            reasons.append(f"missing face/pose {missing_pct:.0%}")
        elif missing_pct >= q.mid_skipped_ratio:
            downgrade("mid")
            reasons.append(f"missing face/pose {missing_pct:.0%}")

    if not reasons:
        reasons.append("brightness and tracking OK")
    return confidence, "; ".join(reasons)


def _draw_pose(img: np.ndarray, lms: np.ndarray) -> None:
    h, w = img.shape[:2]
    pts = np.column_stack([(lms[:, 0] * w).astype(int), (lms[:, 1] * h).astype(int)])
    vis = lms[:, 3] if lms.shape[1] >= 4 else np.ones(len(lms))
    for a, b in POSE_EDGES:
        if vis[a] < VIS_THRESHOLD or vis[b] < VIS_THRESHOLD:
            continue
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        if vis[i] < VIS_THRESHOLD:
            continue
        cv2.circle(img, (x, y), 3, (0, 200, 255), -1)


def _draw_hand(
    img: np.ndarray,
    lms: np.ndarray,
    color: tuple[int, int, int],
    *,
    flip: bool = False,
) -> None:
    h, w = img.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in lms]
    for a, b in HAND_EDGES:
        cv2.line(img, pts[a], pts[b], color, 1)
    for p in pts:
        cv2.circle(img, p, 2, color, -1)
    if flip:
        for a, b in MIDDLE_EDGES:
            cv2.line(img, pts[a], pts[b], (0, 0, 255), 4)
        cv2.circle(img, pts[12], 6, (0, 0, 255), -1)


def _draw_face(img: np.ndarray, lms: np.ndarray, stride: int = 4) -> None:
    h, w = img.shape[:2]
    for p in lms[::stride]:
        cv2.circle(img, (int(p[0] * w), int(p[1] * h)), 1, (255, 180, 0), -1)


def _put_hud(img: np.ndarray, lines: list[str]) -> None:
    for i, s in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def _overlay(
    img: np.ndarray,
    rec: FrameLandmarks,
    infer_hz: float,
    flip_hands: list[str],
    gaze_drift: str,
    accuracy_confidence: str,
    accuracy_reason: str,
) -> None:
    if rec.face_landmarks is not None:
        _draw_face(img, rec.face_landmarks)
    if rec.pose_landmarks is not None:
        _draw_pose(img, rec.pose_landmarks)
    for hand in rec.hands:
        color = (255, 0, 255) if hand.handedness == "Left" else (0, 255, 255)
        _draw_hand(img, hand.landmarks, color, flip=hand.handedness in flip_hands)

    yaw, pitch, roll = decompose_matrix(rec.face_transform_matrix)
    lines = [
        f"face={'Y' if rec.face_detected else '.'} "
        f"pose={'Y' if rec.pose_detected else '.'} "
        f"hands={len(rec.hands)}  infer={infer_hz:.1f}Hz",
    ]
    if not np.isnan(yaw):
        lines.append(f"yaw={yaw:+.1f}  pitch={pitch:+.1f}  roll={roll:+.1f}")
    if accuracy_confidence in {"mid", "low"}:
        lines.append(f"accuracy={accuracy_confidence.upper()}")
        lines.append(accuracy_reason)
    _put_hud(img, lines)

    if accuracy_confidence in {"mid", "low"}:
        h, w = img.shape[:2]
        msg = f"{accuracy_confidence.upper()} ACCURACY: {accuracy_reason}"
        max_chars = 72
        if len(msg) > max_chars:
            msg = msg[: max_chars - 3] + "..."
        color = (0, 0, 255) if accuracy_confidence == "low" else (0, 180, 220)
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        x = max(10, (w - tw) // 2)
        y = h - 72 if not flip_hands else h - 108
        cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8), color, -1)
        cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    if gaze_drift:
        h, w = img.shape[:2]
        msg = f"GAZE DRIFT ({gaze_drift})"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = (w - tw) // 2
        y = 60
        cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 180, 220), -1)
        cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    if flip_hands:
        h, w = img.shape[:2]
        msg = f"FUCK YOU ON {'+'.join(flip_hands)} SIDE"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x = (w - tw) // 2
        y = h - 30
        cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 255), -1)
        cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def _print_summary(
    rec: FrameLandmarks,
    t_sec: float,
    n: int,
    infer_hz: float,
    flip_hands: list[str],
    gaze_drift: str,
    accuracy_confidence: str,
    accuracy_reason: str,
) -> None:
    yaw, pitch, roll = decompose_matrix(rec.face_transform_matrix)
    hands = ",".join(f"{h.handedness[:1]}({h.confidence:.2f})" for h in rec.hands) or "-"
    ypr = (
        f"yaw={yaw:+6.1f} pitch={pitch:+6.1f} roll={roll:+6.1f}"
        if not np.isnan(yaw)
        else "yaw=     -  pitch=     -  roll=     -"
    )
    flip_str = f"  FLIP={'+'.join(flip_hands)}" if flip_hands else ""
    gaze_str = f"  GAZE={gaze_drift}" if gaze_drift else ""
    accuracy_str = (
        f"  ACCURACY={accuracy_confidence.upper()}({accuracy_reason})"
        if accuracy_confidence in {"mid", "low"}
        else ""
    )
    print(
        f"[{t_sec:6.2f}s #{n:04d} {infer_hz:4.1f}Hz] "
        f"face={int(rec.face_detected)} pose={int(rec.pose_detected)} "
        f"hands={hands:<14}  {ypr}{flip_str}{gaze_str}{accuracy_str}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer_fps", type=float, default=10.0, help="Inference rate in Hz (default: 10).")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    p.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    p.add_argument("--pose_variant", choices=("lite", "full", "heavy"), default=None)
    p.add_argument("--cpu", action="store_true", help="Force CPU delegate.")
    p.add_argument("--mirror", action="store_true", help="Flip horizontally (selfie view).")
    p.add_argument("--no_display", action="store_true", help="Headless — console output only.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    if args.pose_variant:
        CONFIG.processing.pose_variant = args.pose_variant
    if args.cpu:
        CONFIG.processing.gpu_delegate = False

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"could not open camera index {args.camera}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)

    infer_period = 1.0 / max(args.infer_fps, 0.1)
    t0 = time.monotonic()
    last_infer_t = -infer_period
    last_rec: FrameLandmarks | None = None
    last_flip_hands: list[str] = []
    last_gaze_drift = ""
    last_accuracy_confidence = "high"
    last_accuracy_reason = "brightness and tracking OK"
    prev_flip_state = False
    prev_gaze_state = False
    infer_hz = 0.0
    frame_idx = 0
    infer_count = 0
    flip_event_count = 0
    gaze_event_count = 0
    ema_alpha = 0.2
    accuracy_window_len = max(1, int(round(max(args.infer_fps, 0.1) * ACCURACY_WINDOW_SEC)))
    luminance_window: deque[float] = deque(maxlen=accuracy_window_len)
    low_confidence_window: deque[bool] = deque(maxlen=accuracy_window_len)

    with LandmarkerBundle() as bundle:
        print(
            f"realtime bodylang — infer_fps={args.infer_fps} "
            f"delegate={bundle.delegate_used} pose={CONFIG.processing.pose_variant}"
        )
        print("press q or Esc in the video window to quit.")
        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    print("camera read failed — exiting", file=sys.stderr)
                    break
                if args.mirror:
                    bgr = cv2.flip(bgr, 1)

                now = time.monotonic() - t0
                if now - last_infer_t >= infer_period:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    t_pre = time.monotonic()
                    rec = bundle.detect(rgb, frame_idx=frame_idx, pts_sec=now)
                    dt = time.monotonic() - t_pre
                    instant_hz = 1.0 / dt if dt > 0 else 0.0
                    infer_hz = (
                        instant_hz if infer_count == 0
                        else (1 - ema_alpha) * infer_hz + ema_alpha * instant_hz
                    )
                    if rec.frame_luminance is not None and np.isfinite(rec.frame_luminance):
                        luminance_window.append(float(rec.frame_luminance))
                    low_confidence_window.append(bool(rec.low_confidence))
                    accuracy_confidence, accuracy_reason = _realtime_accuracy_confidence(
                        luminance_window,
                        low_confidence_window,
                    )
                    last_accuracy_confidence = accuracy_confidence
                    last_accuracy_reason = accuracy_reason

                    flip_hands = [
                        h.handedness for h in rec.hands if is_middle_finger_flip(h.landmarks)
                    ]
                    flip_now = bool(flip_hands)
                    if flip_now and not prev_flip_state:
                        flip_event_count += 1
                        print(
                            f"  ⚠️  MIDDLE FINGER DETECTED at {now:.2f}s "
                            f"({'+'.join(flip_hands)}) — event #{flip_event_count}",
                            flush=True,
                        )
                    prev_flip_state = flip_now
                    last_flip_hands = flip_hands

                    gx, gy, blink = _gaze_from_blendshapes(rec.face_blendshapes)
                    gaze_drift = _gaze_drift_label(gx, gy, blink)
                    gaze_now = bool(gaze_drift)
                    if gaze_now and not prev_gaze_state:
                        gaze_event_count += 1
                        print(
                            f"  👁  GAZE DRIFT at {now:.2f}s "
                            f"({gaze_drift}) — event #{gaze_event_count}",
                            flush=True,
                        )
                    prev_gaze_state = gaze_now
                    last_gaze_drift = gaze_drift

                    last_rec = rec
                    last_infer_t = now
                    infer_count += 1
                    _print_summary(
                        rec,
                        now,
                        infer_count,
                        infer_hz,
                        flip_hands,
                        gaze_drift,
                        accuracy_confidence,
                        accuracy_reason,
                    )

                if last_rec is not None:
                    _overlay(
                        bgr,
                        last_rec,
                        infer_hz,
                        last_flip_hands,
                        last_gaze_drift,
                        last_accuracy_confidence,
                        last_accuracy_reason,
                    )

                if not args.no_display:
                    cv2.imshow("bodylang realtime", bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                frame_idx += 1
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
