"""Teacher-forced observation check: arena_screen_cpu.py's observation vs the MJX environment's.

Rolls the policy out in MJX (no noise, no randomisation), and at every control step puts the same
MuJoCo state into the CPU screen and builds its observation (its own cue tracker runs alongside).
Reports the largest difference per observation block and the first step each block diverges.

usage: python check_obs_cpu.py --weights w.npz --onnx p.onnx [--steps 100] [--seed 1001]
"""
import argparse

import jax
import jax.numpy as jp
import mujoco
import numpy as np

import arena_screen_cpu as A
import dribble_env
from policy import MLPPolicy

BLOCKS = [("gyro", 0, 3), ("gravity", 3, 6), ("joint_pos", 6, 35), ("joint_vel", 35, 64), ("last_act", 64, 93),
          ("ball_hist", 93, 111), ("cue", 111, 115)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1001)
    args = ap.parse_args()
    ov = {"safety_config.enable": True, "safety_config.clamp_action_to_range": True, "safety_config.handover_clamp": True,
          "safety_config.torque_limit_to_profile": True, "noise_config.level": 0.0, "noise_config.ball_delay_max": 1,
          "noise_config.action_delay_max": 0, "reset_config.hold_prob": 0.0}
    env = dribble_env.G1Dribble(config_overrides=ov)
    pol = MLPPolicy(args.weights)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    st = reset(jax.random.PRNGKey(args.seed))
    scr = A.Screen(argparse.Namespace(scene="g1_dribble_scene.xml", onnx=args.onnx, neutral=False, seconds=1))
    scr.reset(np.random.default_rng(0))
    d = scr.d
    stance = int(st.info["stance"])
    scr.stance = stance
    scr.stance_q = np.array(st.info["stance_qpos"])
    scr.targets0 = np.array(st.info["ctrl0"])
    scr.tracker = A.DribbleTracker(0.02, int(st.info["next_hand"]))
    scr.tracker.expect = int(st.info["expect"])
    first = {b[0]: None for b in BLOCKS}
    worst = {b[0]: 0.0 for b in BLOCKS}
    for t in range(args.steps):
        o_mjx = np.array(st.obs["state"])
        d.qpos[:] = np.array(st.data.qpos); d.qvel[:] = np.array(st.data.qvel)
        mujoco.mj_forward(scr.m, d)
        scr.last_action = o_mjx[64:93].astype(np.float32)
        scr.hc = [bool(x) for x in np.array(st.info["hand_contact"])]
        if t > 0:
            scr.ball_hist = o_mjx[93:111].reshape(3, 6)[[1, 2, 2]] if scr.ball_hist is None else scr.ball_hist
        o_cpu, _ = scr.observe()
        for name, a, b in BLOCKS:
            e = float(np.abs(o_cpu[a:b] - o_mjx[a:b]).max())
            worst[name] = max(worst[name], e)
            if e > 1e-3 and first[name] is None:
                first[name] = (t, np.round(o_cpu[a:b], 3).tolist()[:6], np.round(o_mjx[a:b], 3).tolist()[:6])
        a_ = pol.forward(o_mjx)
        last = o_mjx[64:93]
        lim = np.asarray(env._sf_rate) / np.asarray(dribble_env.ACTION_SCALE)
        a_ = np.clip(a_, last - lim, last + lim) if np.any(last) else a_
        st = step(st, jp.array(a_))
        if float(st.done) > 0:
            break
    for name, _, _ in BLOCKS:
        print(f"{name:10s} worst |cpu - mjx| {worst[name]:.4f}   first divergence: {first[name]}")


if __name__ == "__main__":
    main()
