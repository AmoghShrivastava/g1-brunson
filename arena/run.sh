#!/usr/bin/env bash
# Reproduce the evaluation of the shipped controller (CPU is fine):
#   evaluation/results.json, episodes.csv, trained.mp4, neutral.mp4   dribble test (MJX, arena gains, clamps on)
#   evaluation/arena_screen_cpu.json                                  arena safety screen replica (MuJoCo 3.3.7)
# Training: see README "Reproduce the results".
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -q -r requirements-train.txt
export MUJOCO_GL="${MUJOCO_GL:-egl}"
SAFE="--override safety_config.enable=true --override safety_config.clamp_action_to_range=true \
  --override safety_config.handover_clamp=true --override safety_config.torque_limit_to_profile=true"
python evaluate_mjx.py --episodes "${EPISODES:-12}" --video_episodes 1 $SAFE "$@"
# the arena screens with MuJoCo 3.3.7: run the replica in its own environment
python -m venv .venv_screen && .venv_screen/bin/pip install -q mujoco==3.3.7 onnxruntime onnx numpy
.venv_screen/bin/python arena_screen_cpu.py --onnx policy.onnx --episodes "${EPISODES:-12}"
