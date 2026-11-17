#!/bin/bash
# Reproduce every experiment. Needs assets/Blender/Blender_004.blend (+ Textures/).
# Each run: plan -> render 24 images (~15 min on 2 CPUs) -> SfM x3 seeds -> evaluate.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-.venv/bin/python}
$PY -m btp.render --blend assets/Blender/Blender_004.blend --scene-info data/scene.json
for pair in "orbit_single run_002" "orbit_multi run_003" "orbit_high run_004" \
            "oblique_grid run_005" "planner run_006" "planner_budget153 run_007"; do
  set -- $pair
  $PY -m btp.pipeline --strategy $1 --run-id $2
done
$PY -m btp.report
