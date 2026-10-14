"""Arena-faithful safety screen: plain MuJoCo (CPU, C engine), the exported ONNX controller, arena checks.

Mirrors the HIM Arena screening bridge as described in its report: MuJoCo C engine (the arena uses 3.3.7),
PD gains from assets/arena_g1_profile.json without randomisation, torque clamped at 98 % of the profile
limit, joints starting on the re-solved stance targets (+-0.005 rad) with the grounded keyframe pelvis,
ball observation one control step late, no action delay, no observation noise, no ball respawn.

Checks (first time and worst value per episode):
  joint_torque         demand before clamping kp*(target-q) - kd*qd, and applied torque, vs max_torque_nm
  joint_velocity       |qd| vs the screening speed per joint (screen_velocity_rad_s)
  joint_position       measured q beyond the joint range by more than 0.05 rad
  base_angular_speed   any pelvis angular-velocity component above 6 rad/s
  thrashing_energy     weighted RMS of each joint's velocity oscillation (std over a 0.5 s window) above
                       1.5 for 0.5 s; weights legs/waist 4, ankles 2, shoulders 3, elbows 1, wrists 0.25
  init_mismatch        first command vs measured pose above the 0.03 rad handover tolerance
  fall                 pelvis below 0.28 m or torso up-vector z below 0.54

usage: python arena_screen_cpu.py [--onnx policy.onnx] [--episodes 12] [--out evaluation/arena_screen_cpu.json]
"""
import argparse
import json
import os
import time

import mujoco
import numpy as np
import onnxruntime as ort

from export_onnx import arena_bounds
from policy import ACTION_SCALE, BALL_R, DribbleTracker

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = ["init_mismatch", "joint_torque", "joint_velocity", "joint_position", "base_angular_speed",
          "thrashing_energy", "fall"]


def yaw_matrix(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Screen:
    def __init__(self, args):
        self.args = args
        self.m = m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "assets", args.scene))
        self.d = mujoco.MjData(m)
        prof = json.load(open(os.path.join(HERE, "assets", "arena_g1_profile.json")))
        J = prof["joints"]
        self.sc = prof["screening"]
        self.names = [j["name"] for j in J]
        self.kp = np.array([j["kp"] for j in J]); self.kd = np.array([j["kd"] for j in J])
        self.tmax = np.array([j["max_torque_nm"] for j in J])
        self.vmax = np.array([j["screen_velocity_rad_s"] for j in J])
        self.w = np.array([j["thrash_weight"] for j in J])
        self.lo, self.hi = m.jnt_range[1:30, 0].copy(), m.jnt_range[1:30, 1].copy()
        self.init_tol = prof["initial_position_tolerance_rad"]
        m.actuator_gainprm[:, 0] = self.kp
        m.actuator_biasprm[:, 1] = -self.kp
        m.actuator_biasprm[:, 2] = -self.kd
        tl = np.minimum(m.actuator_forcerange[:, 1], 0.98 * self.tmax)
        m.actuator_forcerange[:] = np.c_[-tl, tl]
        m.actuator_forcelimited[:] = 1
        self.bounds = arena_bounds(os.path.join(HERE, "assets", "arena_g1_profile.json"))
        self.keys = {0: m.keyframe("stance_left_front"), 1: m.keyframe("stance_right_front")}
        self.sid = {n: m.site(n).id for n in ("imu_in_pelvis", "left_foot", "right_foot")}
        self.hand_geoms = [m.geom("left_hand_collision").id, m.geom("right_hand_collision").id]
        self.ball_geom = m.geom("ball").id
        self.n_sub = int(round(0.02 / m.opt.timestep))
        self.sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    def sensor(self, n):
        return self.d.sensor(n).data.copy()

    def hand_contact(self):
        f = [False, False]
        for c in self.d.contact[:self.d.ncon]:
            for k, g in enumerate(self.hand_geoms):
                if (c.geom1 == self.ball_geom and c.geom2 == g) or (c.geom2 == self.ball_geom and c.geom1 == g):
                    f[k] = True
        return f

    def reset(self, rng):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        stance = int(rng.integers(0, 2))
        key = self.keys[stance]
        side = "left" if stance == 0 else "right"
        self.targets0 = self.bounds[side]["default"].copy() if self.args.scene == "g1_dribble_scene.xml" else arena_bounds(os.path.join(HERE, "assets", "arena_g1_profile.json"), os.path.join(HERE, "assets", self.args.scene))[side]["default"].copy()
        d.qpos[:] = key.qpos
        d.qpos[7:36] = self.targets0 + rng.uniform(-0.005, 0.005, 29)
        yaw = rng.uniform(-np.pi, np.pi)
        r = np.zeros(4)
        mujoco.mju_mulQuat(r, key.qpos[3:7].copy(), np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        d.qpos[3:7] = r
        d.ctrl[:] = self.targets0
        mujoco.mj_forward(m, d)
        next_hand = 1 - stance
        feet = np.stack([d.site_xpos[self.sid["left_foot"]], d.site_xpos[self.sid["right_foot"]]])
        gate = 0.5 * (feet[0] + feet[1])
        pelvis = d.site_xpos[self.sid["imu_in_pelvis"]]
        R = yaw_matrix(self.sensor("pelvis_quat"))
        sgn = -1.0 if next_hand == 1 else 1.0
        if rng.uniform() < 0.3:  # released towards the gate
            pos = pelvis + R @ np.array([0.32, 0.30 * sgn, 0.0]); pos[2] = 0.38
            pos = pos + rng.uniform(-0.03, 0.03, size=3)
            t_f = rng.uniform(0.15, 0.28)
            v_xy = (gate[:2] - pos[:2]) / t_f
            vel = np.array([v_xy[0], v_xy[1], min(-((pos[2] - BALL_R) / t_f - 0.5 * 9.81 * t_f), 0.0)])
        else:  # held up beside the hand
            pos = pelvis + R @ np.array([0.22, 0.30 * sgn, 0.0]); pos[2] = 0.50
            pos = pos + rng.uniform(-0.04, 0.04, size=3)
            vel = np.zeros(3)
        d.qpos[36:39] = pos; d.qpos[39:43] = [1, 0, 0, 0]
        d.qvel[35:38] = vel; d.qvel[38:41] = 0
        mujoco.mj_forward(m, d)
        self.stance = stance
        self.stance_q = key.qpos[7:36].copy()  # observation reference: the authored stance pose
        self.tracker = DribbleTracker(0.02, next_hand)
        self.tracker.expect = 0
        self.ball_buf, self.ball_hist = [], None
        self.last_action = np.zeros(29, np.float32)
        self.hc = self.hand_contact()

    def observe(self):
        d = self.d
        gyro = self.sensor("gyro_pelvis")
        gravity = d.site_xmat[self.sid["imu_in_pelvis"]].reshape(3, 3).T @ np.array([0, 0, -1.0])
        pelvis = d.site_xpos[self.sid["imu_in_pelvis"]].copy()
        R = yaw_matrix(self.sensor("pelvis_quat"))
        bp, bv = self.sensor("ball_pos"), self.sensor("ball_linvel")
        rel = np.concatenate([R.T @ (bp - pelvis), R.T @ (bv - self.sensor("global_linvel_pelvis"))])
        self.ball_buf = [rel] + self.ball_buf[:3]
        ball = self.ball_buf[min(1, len(self.ball_buf) - 1)]  # one control step late
        self.ball_hist = np.tile(ball[None], (3, 1)) if self.ball_hist is None else np.concatenate([ball[None], self.ball_hist[:-1]])
        feet = np.stack([d.site_xpos[self.sid["left_foot"]], d.site_xpos[self.sid["right_foot"]]])
        crossing = self.tracker.update(bp, bv, feet, self.hc, R[:, 1])  # hand contact OR-ed over the last step
        self.hc = [False, False]
        x = np.concatenate([gyro, gravity, d.qpos[7:36] - self.stance_q, d.qvel[6:35], self.last_action,
                            self.ball_hist.ravel(), self.tracker.cue(self.stance)]).astype(np.float32)
        return x, crossing

    def episode(self, seed):
        a = self.args
        rng = np.random.default_rng(seed)
        self.reset(rng)
        d, sc = self.d, self.sc
        first = {c: None for c in CHECKS}
        worst = {c: 0.0 for c in CHECKS}
        where = {c: "" for c in CHECKS}
        steps = int(a.seconds / 0.02)
        win = int(round(0.5 / 0.02))
        vbuf = np.zeros((win, 29))
        run_thrash = 0
        crossings = 0
        max_wrms = 0.0
        t_end = steps
        lost_t, dead = None, 0
        for t in range(steps):
            tt = t * 0.02

            def hit(c, ratio, j=None, tsub=0.0):
                if ratio > 1.0:
                    if first[c] is None:
                        first[c] = round(tt + tsub, 3)
                    if ratio > worst[c]:
                        worst[c] = float(ratio)
                        where[c] = self.names[j] if j is not None else ""
            obs, crossing = self.observe()
            crossings += int(crossing)
            act = np.zeros(29, np.float32) if self.args.neutral else self.sess.run(["action"], {"obs": obs[None]})[0][0]
            self.last_action = act.astype(np.float32)
            tgt = self.targets0 + act * ACTION_SCALE
            if t == 0:
                e = np.abs(tgt - d.qpos[7:36]) / self.init_tol
                hit("init_mismatch", e.max(), int(e.argmax()))
            d.ctrl[:] = tgt
            for k in range(self.n_sub):
                mujoco.mj_step(self.m, d)
                self.hc = [a or b for a, b in zip(self.hc, self.hand_contact())]
                q, qd = d.qpos[7:36], d.qvel[6:35]
                demand = np.abs(self.kp * (tgt - q) - self.kd * qd)
                r = np.maximum(demand, np.abs(d.actuator_force)) / self.tmax
                hit("joint_torque", r.max(), int(r.argmax()), k * self.m.opt.timestep)
                r = np.abs(qd) / self.vmax
                hit("joint_velocity", r.max(), int(r.argmax()), k * self.m.opt.timestep)
                w = np.abs(self.sensor("gyro_pelvis")) / sc["base_angular_speed_per_axis_rad_s"]
                hit("base_angular_speed", w.max(), None, k * self.m.opt.timestep)
            q = d.qpos[7:36]
            ex = np.maximum(self.lo - q, q - self.hi) / sc["joint_position_tolerance_rad"]
            hit("joint_position", ex.max(), int(ex.argmax()))
            vbuf = np.roll(vbuf, 1, axis=0); vbuf[0] = d.qvel[6:35]
            if t + 1 >= win:
                osc = vbuf.std(axis=0)
                wrms = float(np.sqrt(np.sum(self.w * osc ** 2) / self.w.sum()))
                max_wrms = max(max_wrms, wrms)
                run_thrash = run_thrash + 1 if wrms > sc["thrash_weighted_rms_limit"] else 0
                if run_thrash * 0.02 >= sc["thrash_persistence_s"]:
                    hit("thrashing_energy", wrms / sc["thrash_weighted_rms_limit"], int(np.argmax(self.w * osc ** 2)))
            bp, pel = self.sensor("ball_pos"), d.site_xpos[self.sid["imu_in_pelvis"]]
            dead = dead + 1 if (bp[2] < BALL_R + 0.03 and abs(self.sensor("ball_linvel")[2]) < 0.4) else 0
            if lost_t is None and (np.linalg.norm((bp - pel)[:2]) > 1.0 or dead >= 25):
                lost_t = round(tt, 2)
            up = self.sensor("upvector_torso")
            if d.qpos[2] < sc["fall_height_m"] or up[2] < sc["fall_upright_min"]:
                hit("fall", 2.0)
                t_end = t + 1
                break
        return dict(seed=seed, stance=self.stance, crossings=crossings, seconds=round(t_end * 0.02, 2), ball_lost_at=lost_t,
                    clean=all(v is None for v in first.values()), max_thrash_rms=round(max_wrms, 3),
                    first_violation={c: v for c, v in first.items() if v is not None},
                    worst_ratio={c: round(worst[c], 3) for c in CHECKS if first[c] is not None},
                    worst_joint={c: where[c] for c in CHECKS if first[c] is not None and where[c]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(HERE, "policy.onnx"))
    ap.add_argument("--scene", default="g1_dribble_scene.xml")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "evaluation", "arena_screen_cpu.json"))
    ap.add_argument("--neutral", action="store_true", help="hold the stance targets (zero action) instead of the policy")
    args = ap.parse_args()
    s = Screen(args)
    eps, t0 = [], time.time()
    for e in range(args.episodes):
        r = s.episode(args.seed + e)
        eps.append(r)
        print(f"episode {e + 1}/{args.episodes} ({time.time() - t0:.0f}s): seed {r['seed']} crossings={r['crossings']} "
              f"t={r['seconds']}s lost_at={r['ball_lost_at']} thrash_rms_max={r['max_thrash_rms']} violations={r['first_violation']} "
              f"worst={r['worst_ratio']} {r['worst_joint']}", flush=True)
    summary = dict(episodes=len(eps), clean_episodes=sum(e["clean"] for e in eps),
                   falls=sum("fall" in e["first_violation"] for e in eps),
                   mean_crossings=round(float(np.mean([e["crossings"] for e in eps])), 1),
                   episodes_with={c: sum(c in e["first_violation"] for e in eps) for c in CHECKS},
                   mujoco=mujoco.__version__, onnx=os.path.basename(args.onnx), per_episode=eps)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}))


if __name__ == "__main__":
    main()
