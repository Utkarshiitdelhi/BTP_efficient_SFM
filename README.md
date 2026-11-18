# Drone Path Optimisation for 3D Reconstruction of Civil Structures

B.Tech Project (CVD 411), Department of Civil Engineering, IIT Delhi.
Utkarsh (2023CE10878) · Supervisor: Prof. Saikat Sarkar

A drone photographs a structure and photogrammetry rebuilds it in 3D. **Where the drone
flies decides how good that model is** — and flight time is limited by the battery. This
project builds a simulation harness that measures reconstruction quality against exact
ground truth, compares standard survey flights, and tests a planner that chooses
viewpoints from the structure's own geometry.

Simulation is what makes the comparison possible: on a real building the true geometry is
unknown, so reconstruction error can't be measured. Here the 3D model *is* the ground truth.

<!-- RESULTS -->

## Pipeline

```
plan viewpoints ──► render images ──► structure-from-motion ──► compare with ground truth
 strategies.py        render.py          reconstruct.py            evaluate.py
 planner.py          (Blender/bpy)       (COLMAP/pycolmap)
```

One command runs an experiment end to end and appends a row to `results/results.csv`:

```bash
PYTHONPATH=src python -m btp.pipeline --strategy orbit_high --run-id run_004
```

| Stage | What happens |
|---|---|
| **Scene** | 12-storey RC residential block (photogrammetry-textured, 28.9 × 21.3 m plan, 48.7 m tall), textured ground plane, sun **plus sky light** |
| **Render** | Cycles, 20 mm lens on a 36 mm (full-frame) sensor, 768 × 1024 portrait, 32 samples |
| **Reconstruct** | SIFT → exhaustive matching → incremental SfM, one calibrated pinhole camera; repeated with **3 random seeds** |
| **Evaluate** | Align recovered cameras to true ones (Umeyama similarity) → pose error, point accuracy, and completeness of the *visible* surface |

### How the reconstruction works (COLMAP SfM)

1. **Feature extraction** — SIFT finds distinctive patches (corners, texture specks) in each image.
2. **Matching** — every image pair is compared; similar descriptors are proposed as the same physical point.
3. **Geometric verification** — RANSAC keeps only matches consistent with one camera geometry; wrong matches die here.
4. **Incremental mapping** — start from the best image pair, triangulate points, then register one camera at a time against points already built.
5. **Bundle adjustment** — jointly refine every camera pose and 3D point to minimise reprojection error.

The output is the camera poses plus a **sparse** point cloud. Dense multi-view stereo needs a
CUDA GPU, which this project did not have, so all quality numbers are on the sparse cloud — the
cheap tier of the evaluation. Comparisons between strategies are like-for-like.

### Metrics

| Metric | Meaning |
|---|---|
| Registered | images COLMAP could place; anything below all of them is a flight-path failure |
| Track length | mean number of images that see each 3D point (≥3 for reliable triangulation) |
| Pose error | distance between each recovered camera and its true position, after alignment |
| Median error | median distance from reconstructed points to the true surface |
| Precision@τ | share of reconstructed points within τ of the true surface |
| Recall@τ | share of the **visible** true surface with a reconstructed point within τ |
| F@τ | harmonic mean of the two; τ = 0.30 m headline, 1.0 m for coarse coverage |
| Predicted coverage | from the flight plan alone (no images): share of surface seen by ≥3 cameras at < 70° incidence |

Surfaces no camera can see — the base resting on the ground, interior faces — are excluded
from recall, so no strategy is penalised for geometry that was never observable.

### Flight cost

Every flight is a closed tour back to its start at 3 m/s, plus 2 s hover per photo. A leg that
would cut through the building is replaced by a detour round it — the same rule for every
strategy.

### The planner

1. **Candidates** — 540 poses on rings at six heights and two standoffs, plus a ring above
   the roof. Each position is offered twice: aimed level at its own height (a facade view) and
   aimed at the structure's centre (an oblique view taking in a second facade or the roof).
2. **Scoring by ray casting** — which surface samples each candidate sees, at what distance and
   incidence angle. Pure geometry, no images.
3. **Greedy selection** — a surface point's utility rises with each useful view up to three,
   and a view counts only if it is ≥5° from the views already counted for that point (parallax).
   This is a concave function of a coverage count, i.e. submodular, so greedy selection is within
   1 − 1/e of the best set of that size. A new view must also overlap ≥25% with a view already
   chosen, or SfM could not register it.
4. **Routing** — shortest closed tour through the chosen views (OR-Tools TSP).

Two variants: **planner** picks the best 24 views (equal image count with the baselines);
**planner_budget153** maximises coverage under a 153 s flight-time budget (equal flight time with
the single orbit), adding at each step the view with the best gain per extra second of flight —
the orienteering formulation.

## Reproduce

```bash
uv venv -p 3.13 .venv && VIRTUAL_ENV=.venv uv pip install -e . pytest
# model: "Arabic Neoclassical Tall Residential Complex Building" (Carthage Studio, Fab, free
# personal licence) -> assets/Blender/Blender_004.blend with its Textures/ folder
./scripts/run_all.sh          # ~25 min per strategy on 2 CPUs
python -m pytest tests
```

Runs on CPU only; peak memory stays under 7 GB because Cycles caps textures at 2K
(`texture_limit_render`), which is still finer than a 768 × 1024 render can resolve.

## Repository

```
src/btp/       pipeline code (geometry, render, strategies, planner, reconstruct, evaluate, report)
runs/run_*/    poses.json, rendered_poses, SfM stats and evaluation per seed
results/       results.csv, results_by_seed.csv, results_table.md, figures/
notebooks/     the original Kaggle notebooks (outputs stripped)
tests/         geometry, alignment and routing checks
```

## References

- Smith et al., *Aerial Path Planning for Urban Scene Reconstruction: A Continuous Optimization
  Method and Benchmark*, SIGGRAPH Asia 2018 — reconstructability heuristic.
- Maboudi et al., *A Review on Viewpoints and Path Planning for UAV-Based 3-D Reconstruction*,
  IEEE JSTARS 2023.
- Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016 — COLMAP.
- Khuller, Moss & Naor, *The Budgeted Maximum Coverage Problem*, IPL 1999.
