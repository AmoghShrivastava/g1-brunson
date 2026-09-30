"""Safety evidence for the controller in the packaged simulator: joint-limit margins, torque
utilisation against Unitree's actuator limits, joint speeds, base height and tilt, action rate,
falls.  Same episodes and seeds as evaluate_mjx.py.  Writes <out>/safety.json and prints a table.

usage: python safety_check.py [--episodes 12] [--seconds 20] [--seed 1000]
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jp
import numpy as np

import dribble_env
from policy import MLPPolicy

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "policy_weights.npz"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "evaluation"))
    args = ap.parse_args()
    env = dribble_env.G1Dribble()
    m = env.mj_model
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    steps = int(args.seconds / env.dt)
    pol = MLPPolicy(args.weights)
    nq_free = 7
    nj = m.nu  # 29 hinge joints follow the free joint
    lo, hi = m.jnt_range[1:1 + nj, 0], m.jnt_range[1:1 + nj, 1]
    span = hi - lo
    fmax = m.actuator_forcerange[:, 1]
    vmax_spec = np.array([32.0] * nj)  # rad/s, Unitree G1 rated no-load speed (conservative common value)
    names = [m.actuator(i).name for i in range(nj)]
    tot = dict(steps=0, falls=0, episodes=0, limit_hits=0, torque_sat_steps=0)
    peak_torque_util = np.zeros(nj)
    peak_vel = np.zeros(nj)
    min_margin = np.full(nj, 1.0)  # fraction of joint span from the nearest limit
    viol_steps = np.zeros(nj, int)
    max_over_rad = np.zeros(nj)
    falls_ep = []
    torque_util_sum = np.zeros(nj)
    rms_torque = np.zeros(nj)
    base_z, tilt, act_rate, act_abs = [], [], [], []
    t0 = time.time()
    for e in range(args.episodes):
        state = reset(jax.random.PRNGKey(args.seed + e))
        prev_a = np.zeros(nj, np.float32)
        for t in range(steps):
            obs = np.array(state.obs["state"])
            a = pol.forward(obs)
            act_rate.append(float(np.abs(a - prev_a).max()))
            act_abs.append(float(np.abs(a).max()))
            prev_a = a
            state = step(state, jp.array(a))
            d = state.data
            q = np.array(d.qpos[nq_free:nq_free + nj])
            qd = np.array(d.qvel[6:6 + nj])
            f = np.array(d.actuator_force)
            margin = np.minimum(q - lo, hi - q) / span
            min_margin = np.minimum(min_margin, margin)
            tot["limit_hits"] += int((margin < 0.0).any())
            over = np.maximum(np.maximum(lo - q, q - hi), 0.0)
            viol_steps += over > 0
            max_over_rad = np.maximum(max_over_rad, over)
            util = np.abs(f) / fmax
            peak_torque_util = np.maximum(peak_torque_util, util)
            torque_util_sum += util
            rms_torque += f ** 2
            tot["torque_sat_steps"] += int((util > 0.98).any())
            peak_vel = np.maximum(peak_vel, np.abs(qd))
            base_z.append(float(d.qpos[2]))
            g = obs[3:6]  # gravity direction in the base frame
            tilt.append(float(np.degrees(np.arccos(np.clip(-g[2] / (np.linalg.norm(g) + 1e-9), -1, 1)))))
            tot["steps"] += 1
            if float(state.done) > 0:
                tot["falls"] += int(float(state.metrics["term/fall"]) > 0)
                break
        tot["episodes"] += 1
        falls_ep.append(int(float(state.done) > 0 and float(state.metrics["term/fall"]) > 0))
        print(f"episode {e + 1}/{args.episodes} done ({time.time() - t0:.0f}s)", flush=True)
    n = tot["steps"]
    per_joint = [dict(joint=names[i], torque_limit_Nm=float(fmax[i]), peak_torque_pct=round(100 * peak_torque_util[i], 1),
                      mean_torque_pct=round(100 * torque_util_sum[i] / n, 1), rms_torque_Nm=round(float(np.sqrt(rms_torque[i] / n)), 2),
                      peak_joint_speed_rad_s=round(float(peak_vel[i]), 2), min_margin_to_joint_limit_pct=round(100 * float(min_margin[i]), 1),
                      range_deg=[round(float(np.degrees(lo[i])), 1), round(float(np.degrees(hi[i])), 1)],
                      steps_beyond_range_pct=round(100 * int(viol_steps[i]) / n, 1), max_beyond_range_deg=round(float(np.degrees(max_over_rad[i])), 2))
                 for i in range(nj)]
    summary = dict(
        episodes=tot["episodes"], control_steps=n, seconds_simulated=round(n * env.dt, 1), falls=tot["falls"], falls_by_episode=falls_ep,
        joint_limit_violations_steps=tot["limit_hits"], torque_saturation_steps=tot["torque_sat_steps"],
        peak_torque_utilisation_pct=round(100 * float(peak_torque_util.max()), 1),
        mean_torque_utilisation_pct=round(100 * float((torque_util_sum / n).mean()), 1),
        peak_joint_speed_rad_s=round(float(peak_vel.max()), 2),
        min_margin_to_any_joint_limit_pct=round(100 * float(min_margin.min()), 1),
        base_height_m=dict(min=round(min(base_z), 3), mean=round(float(np.mean(base_z)), 3), max=round(max(base_z), 3)),
        base_tilt_deg=dict(max=round(max(tilt), 1), mean=round(float(np.mean(tilt)), 1)),
        action=dict(max_abs=round(max(act_abs), 3), max_step_change=round(max(act_rate), 3), mean_step_change=round(float(np.mean(act_rate)), 3),
                    note="actions are tanh-bounded to [-1,1]; joint target = stance + action * ACTION_SCALE (<= 0.8 rad legs, 0.25 rad ankles)"),
        limits_source="Unitree g1_29dof_rev_1_0 joint ranges and actuator torque limits (assets/g1_robot.xml)",
        per_joint=per_joint)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "safety.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_joint"}, indent=1))
    print("worst joints by peak torque:", sorted(per_joint, key=lambda r: -r["peak_torque_pct"])[:5])


if __name__ == "__main__":
    main()
