"""Which surface points can a camera see, and how well?

Used by both the evaluation (to exclude surfaces no camera could ever see from
completeness) and the planner (to score candidate viewpoints).
"""
from __future__ import annotations

import numpy as np
import trimesh

from btp.geometry import intrinsics, look_at_rotation


def load_mesh(path: str) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh")


def sample_surface(mesh: trimesh.Trimesh, n: int, seed: int = 0):
    """Area-uniform surface samples with their face normals."""
    pts, fi = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts), mesh.face_normals[fi], fi


def view_geometry(mesh, points, normals, position, target, cam: dict,
                  max_incidence_deg: float = 85.0, chunk: int = 200_000):
    """For one camera, return (visible mask, distance, incidence angle in degrees).

    Visible = in front of the camera, inside the image, facing it (incidence
    below `max_incidence_deg`) and not occluded by other geometry.
    """
    position = np.asarray(position, float)
    R = look_at_rotation(position, target)                 # cam-to-world, Blender axes
    f, cx, cy = intrinsics(cam["lens_mm"], cam["sensor_mm"], cam["res_x"], cam["res_y"])
    d = points - position
    dist = np.linalg.norm(d, axis=1)
    dirs = d / dist[:, None]
    local = d @ R                                          # world -> camera (Blender axes)
    z = -local[:, 2]                                        # depth along viewing direction
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cx + f * local[:, 0] / z
        v = cy - f * local[:, 1] / z
    in_img = (z > 0.1) & (u >= 0) & (u < cam["res_x"]) & (v >= 0) & (v < cam["res_y"])
    cos_inc = np.einsum("ij,ij->i", normals, -dirs)
    facing = cos_inc > np.cos(np.radians(max_incidence_deg))
    cand = np.where(in_img & facing)[0]

    vis = np.zeros(len(points), bool)
    for s in range(0, len(cand), chunk):
        idx = cand[s:s + chunk]
        # the first surface each ray hits must be (approximately) the point itself
        locs, rays, _ = mesh.ray.intersects_location(
            np.repeat(position[None], len(idx), 0), dirs[idx], multiple_hits=False)
        hit_d = np.full(len(idx), np.inf)
        hit_d[rays] = np.linalg.norm(locs - position, axis=1)
        vis[idx] = np.abs(hit_d - dist[idx]) < 0.05 + 0.002 * dist[idx]
    inc = np.degrees(np.arccos(np.clip(cos_inc, -1, 1)))
    return vis, dist, inc


def visibility_matrix(mesh, points, normals, poses, cam: dict):
    """Stack view_geometry over cameras: (n_cams, n_points) arrays."""
    V, D, I = [], [], []
    for p in poses:
        v, d, i = view_geometry(mesh, points, normals, p["position"], p["target"], cam)
        V.append(v), D.append(d.astype(np.float32)), I.append(i.astype(np.float32))
    return np.array(V), np.array(D), np.array(I)
