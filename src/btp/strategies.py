"""Baseline flight strategies. Each returns a list of poses {position, target}.

All baselines aim at the structure's bounding-box centre (what a pilot flying a
standard survey pattern would do). The planner in planner.py instead aims each
camera at the surface patch it was chosen to cover.
"""
from __future__ import annotations

import math

import numpy as np

# Kaggle run_002's geometry: orbit radius = 2.8 x structure radius.
RADIUS_MULT = 2.8


def _ring(ctr, radius, alt, n, phase=0.0):
    return [(ctr[0] + radius * math.cos(phase + 2 * math.pi * i / n),
             ctr[1] + radius * math.sin(phase + 2 * math.pi * i / n),
             alt) for i in range(n)]


def orbit_single(scene, n=24):
    """One ring at mid-height (Kaggle run_002)."""
    ctr, lo, H = scene["centre"], scene["bbox_lo"], scene["height"]
    R = scene["struct_radius"] * RADIUS_MULT
    return [{"position": p, "target": ctr} for p in _ring(ctr, R, lo[2] + 0.5 * H, n)]


def orbit_multi(scene, n=24, rings=3):
    """Three rings at 20/50/80% height, azimuth-staggered so views interleave."""
    ctr, lo, H = scene["centre"], scene["bbox_lo"], scene["height"]
    R = scene["struct_radius"] * RADIUS_MULT
    per = n // rings
    poses = []
    for k in range(rings):
        alt = lo[2] + H * (0.20 + 0.30 * k)
        phase = k * (2 * math.pi / per) / rings
        poses += [{"position": p, "target": ctr} for p in _ring(ctr, R, alt, per, phase)]
    return poses


def orbit_high(scene, n=24):
    """One ring above the roof (+25% of height), looking down at the centre."""
    ctr, hi, H = scene["centre"], scene["bbox_hi"], scene["height"]
    R = scene["struct_radius"] * RADIUS_MULT * 0.8
    return [{"position": p, "target": ctr} for p in _ring(ctr, R, hi[2] + 0.25 * H, n)]


def oblique_grid(scene, n=24, tilt_deg=30.0):
    """Survey-style lawnmower grid above the structure with a tilted camera.

    The standard mapping flight: a boustrophedon grid over the footprint at a
    fixed altitude, camera pitched `tilt_deg` from nadir. Rows alternate heading
    so the camera tilts forward along each pass, as a mapping drone would.
    """
    lo, hi, H = scene["bbox_lo"], scene["bbox_hi"], scene["height"]
    ctr = scene["centre"]
    alt = hi[2] + 0.5 * H
    # grid covers the footprint plus a margin of one building half-width
    mx = (hi[0] - lo[0]) / 2 + scene["struct_radius"]
    my = (hi[1] - lo[1]) / 2 + scene["struct_radius"]
    rows = 4
    cols = n // rows
    xs = np.linspace(ctr[0] - mx, ctr[0] + mx, cols)
    ys = np.linspace(ctr[1] - my, ctr[1] + my, rows)
    reach = (alt - ctr[2]) * math.tan(math.radians(tilt_deg))   # horizontal offset of aim
    poses = []
    for r, y in enumerate(ys):
        direction = 1 if r % 2 == 0 else -1
        for x in (xs if direction == 1 else xs[::-1]):
            pos = (float(x), float(y), alt)
            tgt = (float(x) + direction * reach, float(y), ctr[2])
            poses.append({"position": pos, "target": tgt})
    return poses


BASELINES = {
    "orbit_single": orbit_single,
    "orbit_multi": orbit_multi,
    "orbit_high": orbit_high,
    "oblique_grid": oblique_grid,
}
