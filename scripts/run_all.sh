#!/bin/bash
# Reproduce every experiment. Needs assets/Blender/Blender_004.blend (+ Textures/).
# Each run: plan -> render 24 images (~20 min on 2 CPUs) -> SfM x3 seeds -> evaluate.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-.venv/bin/python}
$PY -m btp.render --blend assets/Blender/Blender_004.blend --scene-info data/scene.json
for pair in "orbit_single run_002" "orbit_multi run_003" "orbit_high run_004" \
            "oblique_grid run_005" "planner run_006" "planner_budget153 run_007" \
            "planner_res run_008" "planner_res_budget153 run_009" \
            "planner_chain run_010" "planner_chain_budget153 run_011"; do
  set -- $pair
  $PY -m btp.pipeline --strategy $1 --run-id $2
done
# illumination ablation: the single orbit with sun only (no sky light)
$PY -m btp.pipeline --strategy orbit_single --run-id run_012 --no-sky
$PY -m btp.report
