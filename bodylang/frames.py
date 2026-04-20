from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoProbe:
    path: str
    src_fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


@dataclass
class ExtractedFrame:
    frame_idx: int           # source-video frame index
    pts_sec: float           # source-video timestamp
    rgb: np.ndarray          # HxWx3 uint8, RGB, possibly downscaled


def probe(video_path: str | Path) -> VideoProbe:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or count <= 0:
        raise RuntimeError(f"Invalid video metadata: fps={fps} frames={count}")
    return VideoProbe(
        path=str(video_path),
        src_fps=fps,
        frame_count=count,
        width=w,
        height=h,
        duration_sec=count / fps,
    )


def _resize_keep_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    new_size = (target_width, int(round(h * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def iter_frames(
    video_path: str | Path,
    analysis_fps: float,
    target_input_width: int,
) -> Iterator[ExtractedFrame]:
    """Yield RGB frames subsampled to ~analysis_fps, preserving source timestamps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if src_fps <= 0:
            raise RuntimeError(f"Invalid source fps: {src_fps}")
        stride = max(1, int(round(src_fps / max(analysis_fps, 0.1))))

        idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                bgr = _resize_keep_aspect(bgr, target_input_width)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield ExtractedFrame(
                    frame_idx=idx,
                    pts_sec=idx / src_fps,
                    rgb=rgb,
                )
            idx += 1
    finally:
        cap.release()
