from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class RunEvent:
    start_idx: int
    end_idx: int            # inclusive last index
    start_sec: float
    end_sec: float
    duration_sec: float
    peak_value: float


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """1D centred moving average that ignores NaNs.

    Shape-preserving along axis 0; other axes are carried through.
    """
    if window <= 1:
        return x.astype(np.float32, copy=True)
    x = np.asarray(x, dtype=np.float32)
    mask = ~np.isnan(x)
    filled = np.where(mask, x, 0.0)
    kernel = np.ones(window, dtype=np.float32)

    def conv_along0(arr: np.ndarray) -> np.ndarray:
        # pad for centred window
        pad = window // 2
        if arr.ndim == 1:
            num = np.convolve(arr, kernel, mode="same")
            return num
        out = np.empty_like(arr, dtype=np.float32)
        for j in range(arr.shape[1]):
            out[:, j] = np.convolve(arr[:, j], kernel, mode="same")
        _ = pad  # currently unused; kept for clarity
        return out

    num = conv_along0(filled)
    den = conv_along0(mask.astype(np.float32))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def interpolate_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear-interpolate NaN runs of length <= max_gap along axis 0."""
    x = np.asarray(x, dtype=np.float32).copy()
    if x.ndim == 1:
        _interp_1d(x, max_gap)
    else:
        for j in range(x.shape[1]):
            _interp_1d(x[:, j], max_gap)
    return x


def _interp_1d(arr: np.ndarray, max_gap: int) -> None:
    n = len(arr)
    i = 0
    while i < n:
        if np.isnan(arr[i]):
            j = i
            while j < n and np.isnan(arr[j]):
                j += 1
            gap = j - i
            if 0 < i and j < n and gap <= max_gap:
                left, right = arr[i - 1], arr[j]
                for k in range(gap):
                    arr[i + k] = left + (right - left) * (k + 1) / (gap + 1)
            i = j
        else:
            i += 1


def rle_events(
    flags: Iterable[bool],
    pts_sec: np.ndarray,
    min_duration_sec: float = 0.0,
    values: Optional[np.ndarray] = None,
) -> list[RunEvent]:
    """Run-length-encode a bool stream into (start, end, duration, peak) events."""
    flags_arr = np.asarray(list(flags), dtype=bool)
    pts = np.asarray(pts_sec, dtype=np.float32)
    if len(flags_arr) != len(pts):
        raise ValueError("flags and pts_sec must have equal length")
    if values is not None and len(values) != len(flags_arr):
        raise ValueError("values must have same length as flags")

    n = len(flags_arr)
    dt = float(np.median(np.diff(pts))) if len(pts) > 1 else 0.0

    events: list[RunEvent] = []
    i = 0
    while i < n:
        if flags_arr[i]:
            j = i
            while j < n and flags_arr[j]:
                j += 1
            start_idx, end_idx = i, j - 1
            start_sec = float(pts[start_idx])
            end_sec = float(pts[end_idx])
            # Each sample represents dt of time; N samples → N*dt duration.
            dur = (end_idx - start_idx + 1) * dt
            end_sec = start_sec + dur
            if values is not None:
                window = values[start_idx : end_idx + 1]
                peak = float(np.nanmax(np.abs(window))) if np.any(~np.isnan(window)) else 0.0
            else:
                peak = 0.0
            # Allow a small epsilon so boundary cases aren't lost to float drift.
            if dur + 1e-6 >= min_duration_sec:
                events.append(
                    RunEvent(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        duration_sec=dur,
                        peak_value=peak,
                    )
                )
            i = j
        else:
            i += 1
    return events
