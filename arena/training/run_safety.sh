#!/usr/bin/env bash
# Hardware-safety fine-tune of G1 Brunson on the HIM Arena joint profile (two stages, back to back).
set -uo pipefail
cd /workspace/g1dribble
source /workspace/venv/bin/activate
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
VIS="--override vision_config.enable=true --override vision_config.kalman=true --override vision_config.dropout=0.12 --override vision_config.pos_noise=0.02 --override vision_config.vel_noise=0.5 --override vision_config.tan_half_h=0.949 --override vision_config.tan_half_v=0.554"
SAFE="--override safety_config.enable=true --override reward_config.scales.dof_pos_limits=-10.0"
echo "== S1 $(date -u)"
python train.py --out runs/S1 --restore params_G236.pkl --num_timesteps 80000000 --num_envs 4096 --num_evals 9 \
  --lr 1e-4 --entropy 0.002 $VIS $SAFE --override 'safety_config.gains_alpha_range=[0.3,1.0]' \
  --override reward_config.scales.track_limit=-2.0
echo "== S2 $(date -u)"
python train.py --out runs/S2 --restore runs/S1/params.pkl --num_timesteps 60000000 --num_envs 4096 --num_evals 7 \
  --lr 5e-5 --entropy 0.002 $VIS $SAFE --override 'safety_config.gains_alpha_range=[0.8,1.0]' \
  --override reward_config.scales.track_limit=-5.0
echo "== DONE $(date -u)"
