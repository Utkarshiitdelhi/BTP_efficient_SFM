"""Fast checks of the geometry that every result depends on (no rendering)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from btp.evaluate import umeyama
from btp.geometry import intrinsics, look_at_rotation
from btp.planner import _segment_hits_box, leg_cost

SCENE = {"centre": [0, 0, 25], "bbox_lo": [-10, -10, 0], "bbox_hi": [10, 10, 50], "height": 50,
         "struct_radius": 14.14}


def test_look_at_points_camera_at_target():
    R = look_at_rotation([50, 0, 25], [0, 0, 25])
    assert np.allclose(-R[:, 2], [-1, 0, 0])           # Blender camera looks along -Z
    assert R[2, 1] > 0.99                               # image-up is world-up
    assert np.allclose(R.T @ R, np.eye(3))


def test_look_at_straight_down_is_valid():
    R = look_at_rotation([0, 0, 100], [0, 0, 0])
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)


def test_intrinsics_sensor_spans_long_edge():
    f, cx, cy = intrinsics(20, 36, 768, 1024)
    assert np.isclose(f, 20 / 36 * 1024) and (cx, cy) == (384, 512)


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    Y = 7.5 * (R @ X.T).T + [1, 2, 3]
    s, Rh, t = umeyama(X, Y)
    assert np.isclose(s, 7.5) and np.allclose(Rh, R) and np.allclose(t, [1, 2, 3])


def test_leg_through_building_detours():
    a, b = np.array([40.0, 0, 25]), np.array([-40.0, 0, 25])
    assert _segment_hits_box(a, b, SCENE, 3)
    assert leg_cost(a, b, SCENE) > np.linalg.norm(b - a)          # detour is longer
    assert np.isclose(leg_cost(a, b, SCENE), 40 * np.pi, rtol=1e-6)  # half circle at r=40


def test_leg_clear_of_building_is_straight():
    a, b = np.array([40.0, 0, 25]), np.array([0, 40.0, 25])
    assert not _segment_hits_box(a, b, SCENE, 3)
    assert np.isclose(leg_cost(a, b, SCENE), np.hypot(40, 40))


def test_leg_over_roof_is_straight():
    a, b = np.array([40.0, 0, 70]), np.array([-40.0, 0, 70])
    assert np.isclose(leg_cost(a, b, SCENE), 80)


def test_two_opt_untangles_a_crossed_square():
    from btp.planner import two_opt
    pts = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], float)       # crossed order
    L = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    tour, length = two_opt([0, 1, 2, 3], L)
    assert np.isclose(length, 4.0)


def test_robust_align_ignores_misplaced_cameras():
    from btp.evaluate import robust_align
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3)) * 10
    Y = 2.0 * X + [5, 0, 0]
    Y[:5] += rng.normal(size=(5, 3)) * 20          # five cameras badly misplaced
    s, R, t, inl = robust_align(X, Y)
    assert inl.sum() == 15 and np.isclose(s, 2.0) and np.allclose(R, np.eye(3), atol=1e-6)
