#!/usr/bin/env bash
# Stage 6: dribble quality under the arena gains. From stage 3 @ 20M (best safe dribbler). Keeps every
# structural clamp, the torque cap and the wrist speed charge; tracking charge relaxed; long chains rewarded.
set -uo pipefail
cd /workspace/g1dribble
source /workspace/venv/bin/activate
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
VIS="--override vision_config.enable=true --override vision_config.kalman=true --override vision_config.dropout=0.12 --override vision_config.pos_noise=0.02 --override vision_config.vel_noise=0.5 --override vision_config.tan_half_h=0.949 --override vision_config.tan_half_v=0.554"
echo "== S6 $(date -u)"
python train.py --out runs/S6 --restore params_S3_20M.pkl --num_timesteps 100000000 --num_envs 4096 --num_evals 11 \
  --lr 5e-5 --entropy 0.002 $VIS --override safety_config.enable=true --override safety_config.clamp_action_to_range=true \
  --override safety_config.handover_clamp=true --override safety_config.torque_limit_to_profile=true \
  --override 'safety_config.gains_alpha_range=[0.9,1.0]' --override safety_config.wrist_weight=10.0 \
  --override reward_config.scales.dof_pos_limits=-10.0 --override reward_config.scales.track_limit=-1.0 \
  --override reward_config.scales.vel_limit=-2.0 --override reward_config.scales.torque_limit=-1.0 \
  --override reward_config.scales.init_target=-50.0 \
  --override reward_config.scales.chain=4.0 --override reward_config.scales.ball_reset=-8.0
echo "== DONE $(date -u)"
