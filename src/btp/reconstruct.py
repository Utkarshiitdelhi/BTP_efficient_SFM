"""COLMAP structure-from-motion via pycolmap (CPU).

Mirrors the Kaggle command-line pipeline: SIFT extraction -> exhaustive matching
-> incremental mapping, one shared PINHOLE camera.

focal="guess" reproduces the Kaggle runs: COLMAP starts from 1.2 x image width
and refines the focal length in bundle adjustment.
focal="known" gives COLMAP the true calibration and holds it fixed, which is
what a calibrated survey drone would provide and isolates the effect of the
flight path from focal-length drift.
focal="prior" is for real phone photos: a SIMPLE_RADIAL camera (focal + one
lens-distortion term) started from cam["f_px"] (from EXIF) and refined.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time

import numpy as np
import pycolmap

from btp.geometry import intrinsics


# Settings that let the mapper continue round the featureless rear elevation.
# The first two are the values the Kaggle run_002 used; the rest are tuned here.
DEFAULT_SFM = {
    "init_min_num_inliers": 30,       # COLMAP default 100
    "abs_pose_min_num_inliers": 20,   # COLMAP default 30
    "max_num_features": 16384,
    "guided_matching": True,
    "affine_dsp": False,              # estimate_affine_shape + domain_size_pooling (CPU-only)
}


def reconstruct(run_dir: str, cam: dict, focal: str = "known", seed: int = 0,
                threads: int = -1, sfm: dict | None = None, tag: str | None = None) -> dict:
    """Writes everything under {run_dir}/sfm/{tag}/ (tag defaults to seed{seed})."""
    sfm = {**DEFAULT_SFM, **(sfm or {})}
    img_dir = f"{run_dir}/images"
    work = f"{run_dir}/sfm/{tag or f'seed{seed}'}"
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(f"{work}/sparse", exist_ok=True)
    db = f"{work}/database.db"
    t0 = time.time()

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    if focal == "prior":
        reader.camera_model = "SIMPLE_RADIAL"
        reader.camera_params = f"{cam['f_px']},{cam['cx']},{cam['cy']},0"
    if focal == "known":
        f, cx, cy = intrinsics(cam["lens_mm"], cam["sensor_mm"], cam["res_x"], cam["res_y"])
        reader.camera_params = f"{f},{f},{cx},{cy}"
    ext = pycolmap.FeatureExtractionOptions()
    ext.num_threads = threads
    ext.sift.max_num_features = sfm["max_num_features"]
    if sfm["affine_dsp"]:
        ext.sift.estimate_affine_shape = True
        ext.sift.domain_size_pooling = True
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext,
                              device=pycolmap.Device.cpu)
    t_ext = time.time()

    mopt = pycolmap.FeatureMatchingOptions()
    mopt.num_threads = threads
    mopt.guided_matching = sfm["guided_matching"]
    pycolmap.match_exhaustive(db, matching_options=mopt, device=pycolmap.Device.cpu)
    t_match = time.time()

    opts = pycolmap.IncrementalPipelineOptions()
    opts.random_seed = seed
    opts.num_threads = threads
    opts.mapper.init_min_num_inliers = sfm["init_min_num_inliers"]
    opts.mapper.abs_pose_min_num_inliers = sfm["abs_pose_min_num_inliers"]
    if focal == "known":
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.mapper.abs_pose_refine_focal_length = False
    recs = pycolmap.incremental_mapping(db, img_dir, f"{work}/sparse", opts)
    t_map = time.time()

    feats, pair_stats = _db_stats(db)
    out = {"focal_mode": focal, "seed": seed, "sfm_settings": sfm,
           "features_per_image": feats,
           "pairs_matched": pair_stats[0], "pairs_verified": pair_stats[1],
           "n_models": len(recs),
           "time_extract_s": round(t_ext - t0, 1), "time_match_s": round(t_match - t_ext, 1),
           "time_map_s": round(t_map - t_match, 1)}
    if not recs:
        out.update(registered=0, points=0)
        json.dump(out, open(f"{work}/sfm_stats.json", "w"), indent=2)
        return out

    # largest model is the reconstruction; others (if any) are fragments
    best_id = max(recs, key=lambda k: recs[k].num_reg_images())
    rec = recs[best_id]
    rec.export_PLY(f"{work}/sparse.ply")
    os.makedirs(f"{work}/model", exist_ok=True)
    rec.write_text(f"{work}/model")
    cam0 = next(iter(rec.cameras.values()))
    out.update(
        model_id=int(best_id),
        registered=rec.num_reg_images(),
        registered_names=sorted(img.name for img in rec.images.values() if img.has_pose),
        points=rec.num_points3D(),
        observations=int(sum(len(p.track.elements) for p in rec.points3D.values())),
        track_length=float(rec.compute_mean_track_length()),
        reproj_err_px=float(rec.compute_mean_reprojection_error()),
        focal_estimated_px=float(cam0.params[0]),
    )
    json.dump(out, open(f"{work}/sfm_stats.json", "w"), indent=2)
    os.remove(db)                      # regenerable, and the largest file in the run
    return out


def _db_stats(db: str):
    con = sqlite3.connect(db)
    names = dict(con.execute("SELECT image_id, name FROM images"))
    feats = {names[i]: r for i, r in con.execute("SELECT image_id, rows FROM keypoints")}
    matched = sum(1 for (r,) in con.execute("SELECT rows FROM matches") if r and r > 0)
    verified = sum(1 for (r,) in con.execute("SELECT rows FROM two_view_geometries") if r and r > 0)
    con.close()
    return dict(sorted(feats.items())), (matched, verified)


def load_cameras(model_dir: str) -> dict:
    """Estimated camera centres and camera-to-world rotations (COLMAP axes), by image name."""
    rec = pycolmap.Reconstruction(model_dir)
    out = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        cam_from_world = img.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation
        out[img.name] = (R.T, -R.T @ t)
    return out


def load_points(model_dir: str) -> np.ndarray:
    rec = pycolmap.Reconstruction(model_dir)
    return np.array([p.xyz for p in rec.points3D.values()])
