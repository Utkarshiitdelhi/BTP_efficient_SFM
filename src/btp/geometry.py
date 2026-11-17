"""Camera geometry shared by rendering, planning and evaluation.

Conventions
-----------
* World frame: metres, Z up (Blender's frame).
* A pose is (position, target): the camera sits at `position` and looks at `target`.
* Blender camera axes: looks along its local -Z, local +Y is image-up.
  COLMAP camera axes: looks along +Z, +Y is image-down.  BLENDER_TO_COLMAP converts.
"""
from __future__ import annotations

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])
BLENDER_TO_COLMAP = np.diag([1.0, -1.0, -1.0])


def look_at_rotation(position, target, up=WORLD_UP) -> np.ndarray:
    """Camera-to-world rotation (Blender convention) for a camera at `position`
    looking at `target`, with image-up as close to world `up` as possible.

    Matches Blender's TRACK_TO constraint (track -Z, up Y), which produced the
    original Kaggle runs, so poses from either route are identical.
    """
    position, target = np.asarray(position, float), np.asarray(target, float)
    f = target - position
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-6:            # looking straight up/down: pick any x
        x = np.cross(f, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.column_stack([x, y, -f])


def pose_matrix(position, target) -> np.ndarray:
    """4x4 camera-to-world matrix (Blender convention)."""
    m = np.eye(4)
    m[:3, :3] = look_at_rotation(position, target)
    m[:3, 3] = position
    return m


def intrinsics(lens_mm: float, sensor_mm: float, res_x: int, res_y: int):
    """Pinhole intrinsics for Blender's default sensor_fit='AUTO', where the
    sensor width spans the LONGER image edge."""
    f_px = lens_mm / sensor_mm * max(res_x, res_y)
    return f_px, res_x / 2.0, res_y / 2.0


def path_length(positions, closed: bool = True) -> float:
    p = np.asarray(positions, float)
    if closed:
        p = np.vstack([p, p[:1]])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
