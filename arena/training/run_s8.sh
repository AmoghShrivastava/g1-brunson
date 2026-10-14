#!/usr/bin/env bash
# Stage 8: train against the arena screen as it actually runs (HIM report 2026-10-13).
# Grounded upright start (assets: make_stance_arena.py), torque DEMAND before clamping, per-axis body
# rotation, joint oscillation (thrashing), measured joint range, joint speed against the screen's
# per-joint limits, and episodes in which a lost ball is never replaced (the robot must stand quietly).
# usage: run_s8.sh <run name> <restore params.pkl> [extra train.py args...]
set -uo pipefail
cd /workspace/g1dribble
source /workspace/venv/bin/activate
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN=$1; RESTORE=$2; shift 2
VIS="--override vision_config.enable=true --override vision_config.kalman=true --override vision_config.dropout=0.12 --override vision_config.pos_noise=0.02 --override vision_config.vel_noise=0.5 --override vision_config.tan_half_h=0.949 --override vision_config.tan_half_v=0.554"
echo "== $RUN $(date -u) from $RESTORE $*"
python train.py --out runs/$RUN --restore $RESTORE --num_timesteps ${STEPS:-100000000} --num_envs 4096 --num_evals ${EVALS:-11} \
  --lr ${LR:-5e-5} --entropy 0.002 $VIS --override safety_config.enable=true --override safety_config.clamp_action_to_range=true \
  --override safety_config.handover_clamp=true --override safety_config.torque_limit_to_profile=true \
  --override 'safety_config.gains_alpha_range=[0.9,1.0]' --override safety_config.wrist_weight=3.0 \
  --override 'safety_config.vel_limit_source="screen_velocity_rad_s"' --override safety_config.vel_limit_scale=0.8 \
  --override safety_config.no_respawn_prob=0.5 \
  --override reward_config.scales.dof_pos_limits=-10.0 --override reward_config.scales.track_limit=-0.2 \
  --override reward_config.scales.vel_limit=-2.0 --override reward_config.scales.torque_limit=-1.0 \
  --override reward_config.scales.init_target=-50.0 \
  --override reward_config.scales.demand_limit=-0.2 --override reward_config.scales.thrash=-5.0 \
  --override reward_config.scales.pos_limit=-40.0 --override reward_config.scales.gyro_limit=-2.0 \
  --override reward_config.scales.parked_still=-0.05 \
  --override reward_config.scales.chain=6.0 --override reward_config.scales.ball_reset=-12.0 \
  --override reward_config.scales.catch=3.0 --override reward_config.scales.bad_bounce=-2.0 "$@"
echo "== DONE $(date -u)"
