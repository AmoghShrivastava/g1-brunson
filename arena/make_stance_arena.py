"""Write the arena stances: assets/stance.json and the stance keyframes of the scene files.

The authored stances (Unitree gains, kept under "*_unitree_gains") hold their pose with PD offsets that
scale with 1/kp, so under the profile's stiffer gains the default targets are re-solved:
target = q + (target_unitree - q) * kp_unitree / kp_profile.  Starting the joints on those targets
straightens the legs, which with the authored pelvis pose would put the feet up to 4 cm into the floor
(MuJoCo's CPU engine resolves that with a violent push-out).  So the start pose is grounded: the
pelvis keeps its authored (upright) orientation, and the pelvis height and the four ankle angles are
solved so both soles rest flat on the floor.

The keyframes then carry that grounded pose as both qpos and ctrl (start pose = default targets), so
re-solving them for any gains is the identity, the pelvis is upright (a random heading about the world
vertical keeps the feet flat), and the first command equals the measured pose up to the policy's first,
handover-limited action.  Gravity load is carried by the policy's action from the first steps.

usage: python make_stance_arena.py [--profile assets/arena_g1_profile.json]
"""
import argparse
import json
import os
import re

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = ("g1_dribble_scene.xml", "g1_dribble_scene_noself.xml")
ANKLES = (4, 5, 10, 11)  # left/right ankle pitch, roll (Unitree order)


def sole_heights(m, d, q):
    """Height above the floor of the 4 bottom corners of each foot box."""
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    out = []
    for g in (m.geom("left_foot").id, m.geom("right_foot").id):
        R, p, s = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                out.append((p + R @ np.array([sx * s[0], sy * s[1], -s[2]]))[2])
    return np.array(out)


def ground(m, q):
    """Solve pelvis height + ankle pitch/roll so the soles are flat on the floor (orientation kept)."""
    d = mujoco.MjData(m)

    def pose(x):
        qq = q.copy()
        qq[2] += x[0]
        qq[[7 + j for j in ANKLES]] += x[1:]
        return qq

    x = np.zeros(5)
    for _ in range(40):
        r = sole_heights(m, d, pose(x))
        J = np.stack([(sole_heights(m, d, pose(x + e)) - r) / 1e-6 for e in np.eye(5) * 1e-6], axis=1)
        x = x - np.linalg.lstsq(J, r, rcond=None)[0]
    qq = pose(x)
    return qq, sole_heights(m, d, qq), x


def write_keyframe(path, key, qpos, ctrl):
    s = open(path).read()
    pat = re.compile(r'<key name="%s" qpos="([^"]*)" ctrl="([^"]*)"' % key)
    mt = pat.search(s)
    assert mt, f"keyframe {key} not found in {path}"
    n_q = len(mt.group(1).split())
    qs = " ".join(f"{v:.5f}" for v in qpos[:n_q])
    cs = " ".join(f"{v:.5f}" for v in ctrl)
    s = s[:mt.start(1)] + qs + s[mt.end(1):mt.start(2)] + cs + s[mt.end(2):]
    open(path, "w").write(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.path.join(HERE, "assets", "arena_g1_profile.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "assets", "stance.json"))
    args = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "assets", "g1_dribble_scene.xml"))
    kp_u = m.actuator_gainprm[:, 0].copy()
    kp_a = np.array([j["kp"] for j in json.load(open(args.profile))["joints"]])
    old = json.load(open(args.out))
    lo, hi = m.jnt_range[1:30, 0], m.jnt_range[1:30, 1]
    out = {"gains": os.path.basename(args.profile),
           "note": "start pose = default targets: authored stance re-solved for the profile PD gains, then pelvis "
                   "height and ankle pitch/roll solved so both soles rest flat on the floor (pelvis upright)"}
    for side in ("left", "right"):
        key = f"stance_{side}_front"
        src = old[f"{key}_unitree_gains"]  # the authored stance (Unitree gains)
        q_a, c_a = np.array(src["qpos"]), np.array(src["ctrl"])
        q = q_a.copy()
        q[7:36] = q_a[7:36] + (c_a - q_a[7:36]) * (kp_u / kp_a)
        q, soles, x = ground(m, q)
        q[7:36] = np.clip(q[7:36], lo + 0.02, hi - 0.02)
        print(f"{key}: pelvis z {q_a[2]:.4f} -> {q[2]:.4f}; ankle corrections {np.round(x[1:], 4)} rad; "
              f"soles {soles.min():+.4f} .. {soles.max():+.4f} m")
        out[key] = {"qpos": q.tolist(), "ctrl": q[7:36].tolist()}
        out[f"{key}_unitree_gains"] = src
        for sc in SCENES:
            write_keyframe(os.path.join(HERE, "assets", sc), key, q, q[7:36])
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out, "and the stance keyframes of", ", ".join(SCENES))


if __name__ == "__main__":
    main()
