"""One command per experiment:  plan -> render -> reconstruct (x seeds) -> evaluate -> results row.

    python -m btp.pipeline --strategy orbit_multi --run-id run_003
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from btp.evaluate import evaluate
from btp.geometry import path_length
from btp.reconstruct import reconstruct
from btp.strategies import BASELINES

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLEND = os.path.join(ROOT, "assets/Blender/Blender_004.blend")
SCENE = os.path.join(ROOT, "data/scene.json")
RESULTS = os.path.join(ROOT, "results")

# Identical to the final Kaggle run_002 camera.
CAMERA = dict(lens_mm=20.0, sensor_mm=36.0, res_x=768, res_y=1024, samples=32)
SPEED_MPS, DWELL_S = 3.0, 2.0         # cruise speed; hover time per photo
SEEDS = (0, 1, 2)

# metrics averaged over seeds in the results table
KEY_METRICS = ["registered", "points", "track_length", "reproj_err_px",
               "pose_pos_err_m", "pose_rot_err_deg", "acc_median_m",
               "precision@0.30", "recall@0.30", "fscore@0.30",
               "recall@0.50", "fscore@0.50", "recall@1.00", "fscore@1.00"]


def flight_cost(positions, scene: dict | None = None) -> dict:
    """Closed tour in the given order; legs through the building become detours round it."""
    if scene is not None:
        from btp.planner import tour_length
        L = tour_length(positions, scene)
    else:
        L = path_length(positions, closed=True)
    return {"n_images": len(positions), "path_length_m": L,
            "flight_time_s": L / SPEED_MPS + DWELL_S * len(positions)}


def plan(strategy: str, scene: dict, n: int):
    if strategy in BASELINES:
        return BASELINES[strategy](scene, n)
    if strategy.startswith("planner"):
        from btp.planner import plan_from_spec
        return plan_from_spec(strategy, scene, n, CAMERA)
    raise ValueError(f"unknown strategy {strategy}")


def run(strategy: str, run_id: str, n: int = 24, seeds=SEEDS, rerender: bool = False,
        poses: list | None = None, note: str = "") -> dict:
    scene = json.load(open(SCENE))
    out = os.path.join(ROOT, "runs", run_id)
    os.makedirs(f"{out}/exports", exist_ok=True)
    if poses is None:
        poses = plan(strategy, scene, n)
    spec = {"run": run_id, "strategy": strategy, "camera": CAMERA, "poses": poses}
    json.dump(spec, open(f"{out}/poses.json", "w"), indent=2)

    have = len(os.listdir(f"{out}/images")) if os.path.isdir(f"{out}/images") else 0
    if rerender or have != len(poses) or not os.path.exists(f"{out}/exports/rendered_poses.json"):
        t0 = time.time()
        env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
        with open(f"{out}/render.log", "w") as log:
            subprocess.run([sys.executable, "-m", "btp.render", "--blend", BLEND,
                            "--poses", f"{out}/poses.json", "--out", out],
                           check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"[{run_id}] rendered {len(poses)} images in {time.time() - t0:.0f}s", flush=True)

    per_seed = []
    for seed in seeds:
        sfm = reconstruct(out, CAMERA, focal="known", seed=seed)
        ev = evaluate(out, tag=f"seed{seed}") if sfm["registered"] >= 3 else {"status": "failed"}
        row = {"run": run_id, "strategy": strategy, "seed": seed,
               **{k: v for k, v in sfm.items() if not isinstance(v, (dict, list))},
               **ev}
        row["min_features"] = int(min(sfm["features_per_image"].values()))
        per_seed.append(row)
        print(f"[{run_id}] seed {seed}: registered {sfm['registered']}/{len(poses)}, "
              f"points {sfm.get('points')}, F@0.3 {ev.get('fscore@0.30', float('nan')):.3f}", flush=True)

    cost = flight_cost([p["position"] for p in poses], scene)
    df = pd.DataFrame(per_seed)
    summary = {"run": run_id, "strategy": strategy, **cost,
               "gsd_mm_px": per_seed[0].get("gsd_mm_px"),
               "visible_gt_frac": per_seed[0].get("visible_gt_frac"),
               "pred_coverage_3views": per_seed[0].get("pred_coverage_3views"),
               "seeds": len(seeds), "note": note}
    for m in KEY_METRICS:
        vals = pd.to_numeric(df.get(m), errors="coerce") if m in df else pd.Series(dtype=float)
        summary[m] = float(vals.mean()) if len(vals) else float("nan")
        summary[m + "_sd"] = float(vals.std(ddof=0)) if len(vals) else float("nan")
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
    _upsert(os.path.join(RESULTS, "results.csv"), pd.DataFrame([summary]), ["run"])
    _upsert(os.path.join(RESULTS, "results_by_seed.csv"), df, ["run", "seed"])
    return summary


def _upsert(path: str, new: pd.DataFrame, keys: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old.merge(new[keys], on=keys, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--rerender", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    s = run(a.strategy, a.run_id, a.n, rerender=a.rerender, note=a.note)
    print(json.dumps({k: s[k] for k in ("run", "registered", "points", "fscore@0.30",
                                         "path_length_m", "flight_time_s")}, indent=1))
