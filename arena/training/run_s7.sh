#!/usr/bin/env bash
# Stage 7: one-hour polish of stage 6 @ 61M for longer unbroken dribble runs. Same safety settings;
# chains and catches pay more, a lost ball or a mis-hit bounce costs more.
set -uo pipefail
cd /workspace/g1dribble
source /workspace/venv/bin/activate
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
VIS="--override vision_config.enable=true --override vision_config.kalman=true --override vision_config.dropout=0.12 --override vision_config.pos_noise=0.02 --override vision_config.vel_noise=0.5 --override vision_config.tan_half_h=0.949 --override vision_config.tan_half_v=0.554"
echo "== S7 $(date -u)"
python train.py --out runs/S7 --restore params_S6_61M.pkl --num_timesteps 36000000 --num_envs 4096 --num_evals 5 \
  --lr 3e-5 --entropy 0.001 $VIS --override safety_config.enable=true --override safety_config.clamp_action_to_range=true \
  --override safety_config.handover_clamp=true --override safety_config.torque_limit_to_profile=true \
  --override 'safety_config.gains_alpha_range=[0.9,1.0]' --override safety_config.wrist_weight=10.0 \
  --override reward_config.scales.dof_pos_limits=-10.0 --override reward_config.scales.track_limit=-1.0 \
  --override reward_config.scales.vel_limit=-2.0 --override reward_config.scales.torque_limit=-1.0 \
  --override reward_config.scales.init_target=-50.0 \
  --override reward_config.scales.chain=6.0 --override reward_config.scales.ball_reset=-12.0 \
  --override reward_config.scales.catch=3.0 --override reward_config.scales.bad_bounce=-2.0
echo "== DONE $(date -u)"
