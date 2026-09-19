"""Evaluate the exported numpy policy inside the MJX simulator (the training engine), with the same
observation pipeline the policy was trained on (tracker noise, latency), physics randomisation off,
and render an MP4 with MuJoCo's renderer.  Trained vs neutral (zero-action) policy on identical seeds.

Outputs: <out>/results.json, <out>/episodes.csv, <out>/trained.mp4, <out>/neutral.mp4
"""
import argparse
import csv
import json
import os
import time

import imageio
import jax
import jax.numpy as jp
import mujoco
import numpy as np

import dribble_env
from policy import MLPPolicy

HERE = os.path.dirname(os.path.abspath(__file__))


def annotate(img, text):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((12, 12), text, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception:  # noqa
        return img


def run_episode(env, reset, step, pol, seed, steps, renderer=None, frames=None, label="", camera="side", traj=None):
    state = reset(jax.random.PRNGKey(seed))
    m = env.mj_model
    d = mujoco.MjData(m)
    st = dict(seed=seed, stance=int(state.info["stance"]), crossings=0, touches=0, catches=0, chains=0, resets=0, bad_bounces=0,
              wrong_hand=0, fell=0, steps=0, feet_off_steps=0, ball_apex_mean=0.0, pelvis_z_mean=0.0,
              max_abs_action=0.0, mean_dribble_period_s=0.0)
    pelvis, apex, cur_apex, bounce_t = [], [], 0.0, []
    for t in range(steps):
        obs = np.array(state.obs["state"])
        a = np.zeros(29, np.float32) if pol is None else pol.forward(obs)
        st["max_abs_action"] = max(st["max_abs_action"], float(np.abs(a).max()))
        state = step(state, jp.array(a))
        mt = state.metrics
        st["crossings"] += int(mt["task/crossings"])
        st["touches"] += int(mt["task/touches"])
        st["catches"] += int(mt["task/catches"])
        st["chains"] += int(mt["task/chains"])
        st["resets"] += int(mt["task/ball_resets"])
        st["bad_bounces"] += int(mt["task/bad_bounces"])
        st["wrong_hand"] += int(mt["task/wrong_hand"])
        bz = float(mt["task/ball_z_mean"])
        if float(state.info["since_bounce"]) == 0.0:
            bounce_t.append(t * env.dt)
            if cur_apex > 0:
                apex.append(cur_apex)
            cur_apex = 0.0
        cur_apex = max(cur_apex, bz)
        pelvis.append(float(mt["task/pelvis_z"]))
        st["feet_off_steps"] += int(float(mt["reward/feet_contact"]) < 0)
        st["steps"] = t + 1
        if renderer is not None and t % 2 == 0:
            d.qpos[:] = np.array(state.data.qpos)
            d.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=camera)
            frames.append(annotate(renderer.render().copy(),
                                   f"{label}   t={t * env.dt:4.1f}s   between-the-legs crossings: {st['crossings']}   chained: {st['chains']}   hand contacts: {st['touches']}"))
        if traj is not None:
            traj.append(np.array(state.data.qpos))
        if float(state.done) > 0:
            st["fell"] = int(float(mt["term/fall"]) > 0)
            break
    st["ball_apex_mean"] = float(np.mean(apex)) if apex else 0.0
    st["pelvis_z_mean"] = float(np.mean(pelvis))
    st["mean_dribble_period_s"] = float(np.mean(np.diff(bounce_t))) if len(bounce_t) > 2 else 0.0
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "policy_weights.npz"))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--video_episodes", type=int, default=2)
    ap.add_argument("--camera", default="side")
    ap.add_argument("--success_crossings", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "evaluation"))
    ap.add_argument("--save_traj", type=int, default=1, help="save qpos trajectories of the video episodes (for replay_viewer.py)")
    ap.add_argument("--xml", default=dribble_env.SCENE_XML, help="scene file (e.g. the no-self-collision scene for older policies)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    env = dribble_env.G1Dribble(xml_path=args.xml)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    steps = int(args.seconds / env.dt)
    pols = {"trained": MLPPolicy(args.weights), "neutral": None}
    results, rows = {}, []
    for name, pol in pols.items():
        renderer = mujoco.Renderer(env.mj_model, height=720, width=1280) if args.video_episodes > 0 else None
        frames, eps = [], []
        t0 = time.time()
        for e in range(args.episodes):
            r = renderer if e < args.video_episodes else None
            traj = [] if (args.save_traj and e < args.video_episodes) else None
            st = run_episode(env, reset, step, pol, args.seed + e, steps, r, frames, f"{name} policy", args.camera, traj)
            if traj:
                np.savez_compressed(os.path.join(args.out, f"traj_{name}_ep{e}.npz"), qpos=np.stack(traj), dt=env.dt,
                                    crossings=st["crossings"], chains=st["chains"])
            st["policy"] = name
            st["success"] = int(st["crossings"] >= args.success_crossings and not st["fell"])
            eps.append(st)
            rows.append(st)
        keys = [k for k in eps[0] if isinstance(eps[0][k], (int, float)) and k != "seed"]
        agg = {k: float(np.mean([s[k] for s in eps])) for k in keys}
        agg["episodes"] = len(eps)
        results[name] = agg
        print(f"[{name}] {len(eps)} eps ({time.time() - t0:.0f}s): success={agg['success']:.2f} crossings/ep={agg['crossings']:.1f} "
              f"touches/ep={agg['touches']:.1f} catches/ep={agg['catches']:.1f} chains/ep={agg['chains']:.1f} resets/ep={agg['resets']:.1f} fell={agg['fell']:.2f} steps={agg['steps']:.0f} "
              f"apex={agg['ball_apex_mean']:.2f}m period={agg['mean_dribble_period_s']:.2f}s pelvis={agg['pelvis_z_mean']:.2f}m |a|max={agg['max_abs_action']:.2f}", flush=True)
        if frames:
            path = os.path.join(args.out, f"{name}.mp4")
            imageio.mimsave(path, frames, fps=25, macro_block_size=1)
            print("video", path, flush=True)
        if renderer is not None:
            renderer.close()
    results["settings"] = vars(args)
    results["causality"] = {"trained_success_rate": results["trained"]["success"],
                            "neutral_success_rate": results["neutral"]["success"],
                            "passed": bool(results["trained"]["success"] > 0.5 and results["neutral"]["success"] == 0.0)}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("causality:", results["causality"])


if __name__ == "__main__":
    main()
