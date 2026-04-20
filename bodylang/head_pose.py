from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


def decompose_matrix(matrix: Optional[np.ndarray]) -> tuple[float, float, float]:
    """Return (yaw_deg, pitch_deg, roll_deg) from a 4x4 facial transformation matrix.

    Uses the Tait-Bryan YXZ convention (yaw around Y, pitch around X, roll around Z).
    Sign convention is consistent frame-to-frame; thresholds in config are compared
    against absolute values so convention-specific sign does not affect classification.
    Returns (nan, nan, nan) if matrix is None.
    """
    if matrix is None:
        return float("nan"), float("nan"), float("nan")
    rot = matrix[:3, :3]
    try:
        yaw, pitch, roll = Rotation.from_matrix(rot).as_euler("YXZ", degrees=True)
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    return float(yaw), float(pitch), float(roll)


def decompose_series(matrices: list[Optional[np.ndarray]]) -> np.ndarray:
    """Vectorized convenience: returns (N, 3) array of [yaw, pitch, roll] in degrees."""
    out = np.full((len(matrices), 3), np.nan, dtype=np.float32)
    for i, m in enumerate(matrices):
        y, p, r = decompose_matrix(m)
        out[i, 0] = y
        out[i, 1] = p
        out[i, 2] = r
    return out
