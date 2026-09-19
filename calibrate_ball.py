"""Calibrate the basketball's contact parameters to the official bounce test.

Rule (FIBA Official Basketball Rules, Basketball Equipment; NBA/NFHS use the same 6 ft / 49-54 in
test): a ball dropped from 1800 mm (measured from the BOTTOM of the ball) onto the playing floor must
rebound to between 1200 mm and 1400 mm (measured to the TOP of the ball).  Inflation 7.5-8.5 psi is
exactly what this test controls, so we calibrate the sim to the test, not to psi.

MuJoCo has no restitution coefficient; the bounce comes from the contact spring-damper `solref`
[timeconst, dampratio].  We bisect dampratio until the rebound hits the middle of the legal band
(1300 mm) under the *training* solver settings and check it in MJX (jax + warp) and in a high
accuracy MuJoCo run.  We also report the dampratio values that map to the band edges so training can
randomize the ball within the legal range.
"""
import argparse
import json
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DROP_BOTTOM = 1.8
LEGAL_TOP = (1.2, 1.4)


def make_model(scene, timeconst, dampratio):
    m = mujoco.MjModel.from_xml_path(scene)
    for p in range(m.npair):
        if m.geom(m.pair_geom2[p]).name == "ball" or m.geom(m.pair_geom1[p]).name == "ball":
            m.pair_solref[p] = [timeconst, dampratio]
    return m


def drop_test(m, iterations=None, seconds=2.0):
    if iterations is not None:
        m.opt.iterations = iterations
        m.opt.ls_iterations = 50
    d = mujoco.MjData(m)
    d.qpos[:] = m.key("home").qpos
    d.qpos[0] = 5.0  # robot far away
    r = m.geom("ball").size[0]
    bq = m.jnt_qposadr[m.joint("ball_freejoint").id]
    d.qpos[bq:bq + 3] = [0, 0, DROP_BOTTOM + r]
    d.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    zs = []
    bounced = False
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        z = d.qpos[bq + 2]
        if d.sensor("ball_floor_found").data[0] > 0:
            bounced = True
        if bounced:
            zs.append(z)
    zs = np.array(zs)
    top = zs.max() + r
    return float(top)


def drop_test_mjx(m, impl):
    import jax
    from mujoco import mjx
    mx = mjx.put_model(m, impl=impl)
    d = mujoco.MjData(m)
    d.qpos[:] = m.key("home").qpos
    d.qpos[0] = 5.0
    r = m.geom("ball").size[0]
    bq = m.jnt_qposadr[m.joint("ball_freejoint").id]
    d.qpos[bq:bq + 3] = [0, 0, DROP_BOTTOM + r]
    d.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
    dx = mjx.put_data(m, d, impl=impl, naconmax=64, njmax=256)
    step = jax.jit(lambda dd: mjx.step(mx, dd))
    n = int(2.0 / m.opt.timestep)
    zs = []
    for _ in range(n):
        dx = step(dx)
        zs.append(float(dx.qpos[bq + 2]))
    zs = np.array(zs)
    # after the first bounce: find the first local minimum then max after it
    i_min = int(np.argmin(zs[: n // 2]))
    top = zs[i_min:].max() + r
    return float(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.path.join(HERE, "assets", "g1_dribble_scene.xml"))
    ap.add_argument("--timeconst", type=float, default=0.015)
    ap.add_argument("--mjx", action="store_true")
    args = ap.parse_args()

    def rebound(dr):
        return drop_test(make_model(args.scene, args.timeconst, dr))

    def bisect(target, lo=0.05, hi=1.5):
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if rebound(mid) > target:
                lo = mid  # more damping needed
            else:
                hi = mid
        return 0.5 * (lo + hi)

    print("sensitivity sweep (dampratio -> rebound top):")
    for dr in (0.05, 0.1, 0.12, 0.14, 0.16, 0.2, 0.3, 0.5, 1.0):
        print(f"   {dr:.3f} -> {rebound(dr):.3f} m")
    d_mid = bisect(1.3)
    d_hi = bisect(LEGAL_TOP[1])  # bounciest legal ball (lowest dampratio)
    d_lo = bisect(LEGAL_TOP[0])  # deadest legal ball
    top_train = rebound(d_mid)
    top_accurate = drop_test(make_model(args.scene, args.timeconst, d_mid), iterations=100)
    cor = np.sqrt((top_train - 2 * make_model(args.scene, 0.01, 1).geom("ball").size[0]) / DROP_BOTTOM)
    out = {
        "drop_bottom_m": DROP_BOTTOM, "legal_rebound_top_m": LEGAL_TOP,
        "solref_timeconst": args.timeconst, "solref_dampratio": d_mid,
        "dampratio_legal_range": [d_hi, d_lo],
        "rebound_top_training_solver_m": top_train, "rebound_top_accurate_solver_m": top_accurate,
        "effective_cor": float(cor),
    }
    if args.mjx:
        m = make_model(args.scene, args.timeconst, d_mid)
        for impl in ("jax", "warp"):
            try:
                out[f"rebound_top_mjx_{impl}_m"] = drop_test_mjx(m, impl)
            except Exception as e:  # noqa
                out[f"rebound_top_mjx_{impl}_m"] = f"failed: {e}"[:200]
    print(json.dumps(out, indent=1))
    with open(os.path.join(HERE, "assets", "ball_calibration.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
