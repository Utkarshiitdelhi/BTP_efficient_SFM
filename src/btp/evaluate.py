"""Score a reconstruction against the ground-truth mesh.

1. Align: SfM recovers geometry only up to scale/rotation/translation, so the
   estimated camera centres are mapped onto the true ones (Umeyama similarity).
2. Pose error: how far each recovered camera sits from where it really was.
3. Accuracy / precision: how close reconstructed points are to the true surface.
4. Completeness / recall: how much of the *observable* true surface has a
   reconstructed point nearby. "Observable" is one fixed set for every flight:
   surface that any viewpoint in the study could see. Hidden geometry (the base
   on the ground, interior faces) is excluded; surface a flight merely failed
   to look at is not.
5. Predicted coverage: geometry-only score from the flight plan alone (no
   images) - the fraction of surface seen by >=3 cameras with usable angles.
   This is the quantity a planner can optimise before flying.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

from btp.geometry import BLENDER_TO_COLMAP, intrinsics
from btp.reconstruct import load_cameras, load_points
from btp.visibility import load_mesh, sample_surface, visibility_matrix

TAUS = (0.10, 0.30, 0.50, 1.00)       # distance thresholds in metres
HEADLINE_TAU = 0.30


def umeyama(src, dst):
    """Similarity (s, R, t) minimising |dst - (s R src + t)|^2."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((xs ** 2).sum() / len(src))
    return s, R, md - s * R @ ms


def predicted_coverage(V, D, I, min_views=3, max_inc=70.0):
    """Fraction of visible-able surface seen by >= min_views cameras at incidence < max_inc."""
    good = V & (I < max_inc)
    return float((good.sum(0) >= min_views).mean())


OBSERVABLE = "data/observable_surface.npz"


def observable_surface(root: str, n_samples: int = 30_000) -> np.ndarray:
    """Surface samples that ANY viewpoint in the study could see - every planner
    candidate plus every baseline pose. Recall is measured over this one fixed
    set, so a flight is not let off for surface it simply never looked at, but
    nobody is penalised for geometry no camera can see (base, interior faces)."""
    import os
    path = os.path.join(root, OBSERVABLE)
    if os.path.exists(path):
        z = np.load(path)
        if len(z["observable"]) == n_samples:
            return z["observable"]
    from btp.pipeline import CAMERA
    from btp.planner import candidates
    from btp.strategies import BASELINES
    scene = json.load(open(os.path.join(root, "data/scene.json")))
    mesh = load_mesh(os.path.join(root, "data/gt_mesh.ply"))
    P, Nrm, _ = sample_surface(mesh, n_samples, seed=0)
    poses = candidates(scene) + [p for f in BASELINES.values() for p in f(scene, 24)]
    V, _, _ = visibility_matrix(mesh, P, Nrm, poses, CAMERA)
    obs = V.any(0)
    np.savez_compressed(path, observable=obs, n_poses=len(poses))
    return obs


def evaluate(run_dir: str, tag: str = "seed0", mesh_path: str | None = None,
             n_samples: int = 30_000) -> dict:
    work = f"{run_dir}/sfm/{tag}"
    spec = json.load(open(f"{run_dir}/exports/rendered_poses.json"))
    cam, poses = spec["camera"], spec["poses"]
    mesh = load_mesh(mesh_path or f"{run_dir}/exports/gt_mesh.ply")
    lo, hi = mesh.bounds

    # ---- surface samples and which cameras see them (true poses) ----
    P, Nrm, _ = sample_surface(mesh, n_samples, seed=0)
    V, D, I = visibility_matrix(mesh, P, Nrm, poses, cam)
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(run_dir)))
    obs = observable_surface(root, n_samples)
    seen = V.any(0) & obs                 # seen by this flight, within the observable set
    f_px = intrinsics(cam["lens_mm"], cam["sensor_mm"], cam["res_x"], cam["res_y"])[0]
    med_dist = float(np.median(D[V])) if V.any() else float("nan")
    out = {"n_samples": n_samples,
           "median_view_distance_m": med_dist,
           "gsd_mm_px": med_dist * 1000 / f_px,
           "observable_frac": float(obs.mean()),
           "visible_gt_frac": float(seen[obs].mean()),
           "pred_coverage_3views": predicted_coverage(V[:, obs], D[:, obs], I[:, obs])}

    # ---- align estimated cameras to true ones ----
    est = load_cameras(f"{work}/model")
    names = [p["file"] for p in poses if p["file"] in est]
    if len(names) < 3:
        out.update(status="too_few_registered")
        json.dump(out, open(f"{work}/eval.json", "w"), indent=2)
        return out
    true_by = {p["file"]: np.array(p["matrix_world"]) for p in poses}
    C_true = np.array([true_by[n][:3, 3] for n in names])
    R_true = [true_by[n][:3, :3] @ BLENDER_TO_COLMAP for n in names]
    C_est = np.array([est[n][1] for n in names])
    R_est = [est[n][0] for n in names]
    s, Ra, ta = umeyama(C_est, C_true)
    pos_err = np.linalg.norm((s * (Ra @ C_est.T)).T + ta - C_true, axis=1)
    rot_err = [np.degrees(np.arccos(np.clip((np.trace(Rt.T @ (Ra @ Re)) - 1) / 2, -1, 1)))
               for Rt, Re in zip(R_true, R_est)]

    # ---- reconstructed points, aligned, cropped to the structure ----
    X = (s * (Ra @ load_points(f"{work}/model").T)).T + ta
    keep = np.all((X > lo - 1.0) & (X < hi + 1.0), axis=1) & (X[:, 2] > lo[2] + 0.3)
    Xs = X[keep]
    _, d_r2g, _ = mesh.nearest.on_surface(Xs) if len(Xs) else (None, np.array([]), None)
    # completeness over the whole observable surface (same denominator for every flight)
    d_g2r = cKDTree(Xs).query(P[obs])[0] if len(Xs) else np.full(obs.sum(), np.inf)

    out.update(
        status="ok",
        align_scale=float(s),
        pose_pos_err_m=float(pos_err.mean()), pose_pos_err_max_m=float(pos_err.max()),
        pose_rot_err_deg=float(np.mean(rot_err)),
        points_on_structure=int(keep.sum()), points_total=int(len(X)),
        acc_mean_m=float(d_r2g.mean()), acc_median_m=float(np.median(d_r2g)),
    )
    for tau in TAUS:
        p, r = float((d_r2g < tau).mean()), float((d_g2r < tau).mean())
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[f"precision@{tau:.2f}"], out[f"recall@{tau:.2f}"], out[f"fscore@{tau:.2f}"] = p, r, f
    out["fscore"] = out[f"fscore@{HEADLINE_TAU:.2f}"]
    out["recall"] = out[f"recall@{HEADLINE_TAU:.2f}"]

    # per-sample reconstruction status, for coverage maps and the planner study
    np.savez_compressed(f"{work}/surface_eval.npz", P=P, N=Nrm, seen=seen, observable=obs,
                        n_views=V.sum(0), d_g2r=_full(obs, d_g2r))
    json.dump(out, open(f"{work}/eval.json", "w"), indent=2)
    return out


def _full(seen, d):
    full = np.full(len(seen), np.nan)
    full[seen] = d
    return full
