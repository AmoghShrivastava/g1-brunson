"""Vision-in-the-loop evaluation: the policy's ball information comes ONLY from the head D435 camera
(rendered RGB + depth -> BallPerception), never from the simulator's ball state.  Physics/task
bookkeeping still runs in the MJX environment for ground-truth metrics.

Outputs: <out>/results.json, <out>/episodes.csv, <out>/trained.mp4 (side view with the camera view
and detection mask inset), <out>/traj_trained_ep0.npz
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
import vision
from policy import BALL_R, DribbleTracker, MLPPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = None


def yaw_matrix(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def annotate(img, text):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((12, 12), text, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception:  # noqa
        return img


def run_episode(env, reset, step, pol, seed, steps, m, d, perc, renderer=None, frames=None, label="", traj=None):
    state = reset(jax.random.PRNGKey(seed))
    stance = int(state.info["stance"])
    st = dict(seed=seed, stance=stance, crossings=0, touches=0, catches=0, chains=0, resets=0, fell=0, steps=0,
              detection_rate=0.0, detection_rate_depth=0.0, detection_rate_colour=0.0, mean_detection_error_m=0.0,
              mean_estimate_error_m=0.0, mean_estimate_vel_error_mps=0.0, max_abs_action=0.0,
              cue_hand_mismatch=0.0, cue_expect_mismatch=0.0, cue_bounce_mismatch=0.0)
    sid = {n: m.site(n).id for n in ("imu_in_pelvis", "left_foot", "right_foot")}
    hand_geoms = [m.geom("left_hand_collision").id, m.geom("right_hand_collision").id]
    ball_geom = m.geom("ball").id
    tracker = DribbleTracker(env.dt, int(state.info["next_hand"]))
    tracker.expect = int(state.info["expect"])
    ball_hist = None
    perc.stats = dict(frames=0, detections=0, det_depth=0, det_colour=0, err_sum=0.0, err_n=0,
                      est_err_sum=0.0, est_verr_sum=0.0, est_n=0)
    perc.kf = vision.BallKalman(env.dt)
    for t in range(steps):
        # sync the C model with the MJX state (for rendering + kinematics; this is what encoders/FK give)
        d.qpos[:] = np.array(state.data.qpos)
        d.qvel[:] = np.array(state.data.qvel)
        mujoco.mj_forward(m, d)
        true_ball = d.qpos[36:39].copy()
        est_p, est_v, seen = perc.observe(d, true_ball, d.qvel[35:38].copy())
        pelvis = d.site_xpos[sid["imu_in_pelvis"]].copy()
        pelvis_v = d.sensor("global_linvel_pelvis").data.copy()
        R = yaw_matrix(d.sensor("pelvis_quat").data)
        if est_p is None:  # nothing seen yet: assume the ball is where it spawned relative to the pelvis
            est_p, est_v = true_ball * 0 + pelvis + R @ np.array([0.3, 0.0, -0.3]), np.zeros(3)
        rel = np.concatenate([R.T @ (est_p - pelvis), R.T @ (est_v - pelvis_v)]).astype(np.float32)
        ball_hist = np.tile(rel[None], (3, 1)) if ball_hist is None else np.concatenate([rel[None], ball_hist[:-1]], 0)
        # hand-contact signal (on hardware: ball-velocity discontinuity at the hand / contact sensor)
        hc = [False, False]
        if ARGS.env_contact:  # contact flag ORed over the physics substeps (what a wrist force sensor / velocity-jump detector gives)
            hc = [bool(x) for x in np.array(state.info["hand_contact"])]
        for i in range(0 if ARGS.env_contact else d.ncon):
            c = d.contact[i]
            for k, g in enumerate(hand_geoms):
                if (c.geom1 == ball_geom and c.geom2 == g) or (c.geom2 == ball_geom and c.geom1 == g):
                    hc[k] = True
        feet = np.stack([d.site_xpos[sid["left_foot"]], d.site_xpos[sid["right_foot"]]])
        tracker.update(est_p, est_v, feet, hc, R[:, 1])
        cue = tracker.cue(stance)
        obs = np.array(state.obs["state"])
        # diagnostic: how often the onboard cue disagrees with the simulator's task state
        st["cue_hand_mismatch"] += float(cue[0] != obs[111]); st["cue_expect_mismatch"] += float(cue[1] != obs[112])
        st["cue_bounce_mismatch"] += float(abs(cue[2] - obs[113]) > 0.05)
        if ARGS.true_cue:  # ablation: ground-truth task cue, vision ball
            cue = obs[111:115]
        if ARGS.true_ball:  # ablation: ground-truth ball history, tracker cue
            ball_hist = obs[93:111].reshape(3, 6)
        obs_v = np.concatenate([obs[:93], ball_hist.ravel(), cue]).astype(np.float32)
        a = np.zeros(29, np.float32) if pol is None else pol.forward(obs_v)
        st["max_abs_action"] = max(st["max_abs_action"], float(np.abs(a).max()))
        state = step(state, jp.array(a))
        mt = state.metrics
        st["crossings"] += int(mt["task/crossings"]); st["touches"] += int(mt["task/touches"])
        st["catches"] += int(mt["task/catches"]); st["chains"] += int(mt["task/chains"])
        st["resets"] += int(mt["task/ball_resets"]); st["steps"] = t + 1
        if int(mt["task/ball_resets"]):  # simulator respawn = operator hands the ball back: restart the bookkeeping
            tracker.resync(int(state.info["next_hand"]), int(state.info["expect"]))
        if traj is not None:
            traj.append(np.array(state.data.qpos))
        if renderer is not None and t % 2 == 0:
            renderer.update_scene(d, camera="side")
            img = renderer.render().copy()
            cam = perc.last_rgb.copy()
            cam[perc.last_mask] = (cam[perc.last_mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
            ch, cw = cam.shape[:2]
            img[16:16 + ch, img.shape[1] - cw - 16:img.shape[1] - 16] = cam
            if perc.last_depth_mask is not None:  # depth-stream detection mask (wider FOV), below the colour inset
                dm = np.zeros((ch, cw, 3), np.uint8) + 40
                dm[perc.last_depth_mask] = (255, 255, 0)
                img[32 + ch:32 + 2 * ch, img.shape[1] - cw - 16:img.shape[1] - 16] = dm
            src = perc.last_source or "predicted"
            frames.append(annotate(img, f"{label}  t={t * env.dt:4.1f}s  crossings: {st['crossings']}  chained: {st['chains']}  "
                                        f"ball: {src} (inset: colour stream / depth-stream mask)"))
        if float(state.done) > 0:
            st["fell"] = int(float(mt["term/fall"]) > 0)
            break
    for k in ("cue_hand_mismatch", "cue_expect_mismatch", "cue_bounce_mismatch"):
        st[k] /= max(st["steps"], 1)
    st.update(perc.summary())
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "policy_weights.npz"))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--video_episodes", type=int, default=1)
    ap.add_argument("--success_crossings", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "evaluation_vision"))
    ap.add_argument("--xml", default=dribble_env.SCENE_XML)
    ap.add_argument("--mode", default="depth_first", choices=["depth_first", "colour", "depth"])
    ap.add_argument("--no_depth", action="store_true", help="colour mode: monocular range instead of aligned depth")
    ap.add_argument("--true_cue", action="store_true", help="ablation: use the simulator's task cue instead of the tracker's")
    ap.add_argument("--true_ball", action="store_true", help="ablation: use the simulator's ball history instead of vision")
    ap.add_argument("--env_contact", action="store_true", help="hand-contact flag from the physics substeps instead of the instantaneous C-engine contact list")
    args = ap.parse_args()
    global ARGS
    ARGS = args
    os.makedirs(args.out, exist_ok=True)
    env = dribble_env.G1Dribble(xml_path=args.xml)
    m = env.mj_model
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, vision.CAM_NAME) < 0:
        raise SystemExit("robot xml has no d435 camera: run vision.add_camera_to_robot_xml on assets/g1_robot.xml")
    d = mujoco.MjData(m)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    steps = int(args.seconds / env.dt)
    perc = vision.BallPerception(m, env.dt, use_depth=not args.no_depth, mode=args.mode)
    pols = {"trained": MLPPolicy(args.weights), "neutral": None}
    results, rows = {}, []
    for name, pol in pols.items():
        renderer = mujoco.Renderer(m, height=720, width=1280) if args.video_episodes > 0 else None
        frames, eps = [], []
        t0 = time.time()
        for e in range(args.episodes):
            r = renderer if e < args.video_episodes else None
            traj = [] if e < args.video_episodes else None
            stt = run_episode(env, reset, step, pol, args.seed + e, steps, m, d, perc, r, frames, f"{name} policy (vision)", traj)
            stt["policy"] = name
            stt["success"] = int(stt["crossings"] >= args.success_crossings and not stt["fell"])
            eps.append(stt); rows.append(stt)
            if traj:
                np.savez_compressed(os.path.join(args.out, f"traj_{name}_ep{e}.npz"), qpos=np.stack(traj), dt=env.dt,
                                    crossings=stt["crossings"], chains=stt["chains"])
        keys = [k for k in eps[0] if isinstance(eps[0][k], (int, float)) and k != "seed"]
        agg = {k: float(np.mean([s[k] for s in eps])) for k in keys}
        agg["episodes"] = len(eps)
        results[name] = agg
        print(f"[{name}] {len(eps)} eps ({time.time() - t0:.0f}s): success={agg['success']:.2f} crossings/ep={agg['crossings']:.1f} "
              f"chains/ep={agg['chains']:.1f} catches/ep={agg['catches']:.1f} resets/ep={agg['resets']:.1f} fell={agg['fell']:.2f} "
              f"steps={agg['steps']:.0f} detection_rate={agg['detection_rate']:.2f} (depth {agg['detection_rate_depth']:.2f} / colour {agg['detection_rate_colour']:.2f}) "
              f"det_err={agg['mean_detection_error_m']:.3f}m est_err={agg['mean_estimate_error_m']:.3f}m "
              f"est_verr={agg['mean_estimate_vel_error_mps']:.2f}m/s cue_mismatch hand={agg['cue_hand_mismatch']:.2f} "
              f"expect={agg['cue_expect_mismatch']:.2f} since_bounce={agg['cue_bounce_mismatch']:.2f}", flush=True)
        if frames:
            imageio.mimsave(os.path.join(args.out, f"{name}.mp4"), frames, fps=25, macro_block_size=1)
        if renderer is not None:
            renderer.close()
    results["settings"] = vars(args)
    results["causality"] = {"trained_success_rate": results["trained"]["success"],
                            "neutral_success_rate": results["neutral"]["success"],
                            "passed": bool(results["trained"]["success"] > 0.5 and results["neutral"]["success"] == 0.0)}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("causality:", results["causality"])


if __name__ == "__main__":
    main()
