#!/usr/bin/env bash
# Reproduce the evaluation of the shipped policy in plain MuJoCo (CPU): metrics, causality test, video.
#   ./run.sh                 -> evaluation/results.json, evaluation/episodes.csv, evaluation/trained.mp4, evaluation/neutral.mp4
# Training from scratch (GPU, ~1 h on an A40):  python build_model.py && python calibrate_ball.py --timeconst 0.03 --mjx \
#   && python build_model.py --ball_solref "0.03 <dampratio from assets/ball_calibration.json>" && python train.py
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -q -r requirements-train.txt
export MUJOCO_GL="${MUJOCO_GL:-egl}"
python evaluate_mjx.py --episodes "${EPISODES:-20}" --video_episodes 2 "$@"
