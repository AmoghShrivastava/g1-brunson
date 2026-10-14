"""MJX vs MuJoCo C engine on the same states and commands (arena gains, no randomisation).

Rolls the policy out in the MJX training environment, records the state and the motor targets of every
control step, then
  * teacher-forced: puts each recorded MJX state into the C engine, applies the same targets for one
    control step and compares the result with MJX's next state (ball velocity, pelvis, joints);
  * open loop: replays the recorded targets in the C engine from the first state.
Errors are split by whether the ball touched a hand during that step.

usage: python sim2sim.py --weights policy_weights.npz [--steps 300] [--seed 1000] [--override key=value ...]
"""
import argparse

import jax
import jax.numpy as jp
import mujoco
import numpy as np

import dribble_env
from policy import MLPPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--c_iterations", type=int, default=0, help="override the C engine solver iterations")
    ap.add_argument("--dump", default="", help="write the MJX rollout (states, targets, actuator params) to this .npz and stop")
    args = ap.parse_args()
    ov = {"safety_config.enable": True, "safety_config.clamp_action_to_range": True,
          "safety_config.handover_clamp": True, "safety_config.torque_limit_to_profile": True}
    for o in args.override:
        k, v = o.split("=", 1)
        ov[k] = eval(v) if v[:1] in "0123456789-[(" or v in ("True", "False") else v
    env = dribble_env.G1Dribble(config_overrides=ov)
    pol = MLPPolicy(args.weights)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    st = reset(jax.random.PRNGKey(args.seed))
    qs, vs, tg, touch = [np.array(st.data.qpos)], [np.array(st.data.qvel)], [], []
    x = 0
    for t in range(args.steps):
        a = pol.forward(np.array(st.obs["state"]))
        last = np.array(st.obs["state"])[64:93]
        lim = np.asarray(env._sf_rate) / np.asarray(dribble_env.ACTION_SCALE)
        a = np.clip(a, last - lim, last + lim) if np.any(last) else a
        st = step(st, jp.array(a))
        tg.append(np.array(st.info["motor_targets"]))
        qs.append(np.array(st.data.qpos)); vs.append(np.array(st.data.qvel))
        touch.append(float(st.metrics["task/touches"]) > 0 or bool(np.any(np.array(st.info.get("hand_contact", [0])))))
        x += int(st.metrics["task/crossings"])
        if float(st.done) > 0:
            break
    n = len(tg)
    print(f"MJX rollout: {n} steps, crossings {x}")
    if args.dump:
        mm = env.mj_model
        np.savez(args.dump, qs=np.array(qs), vs=np.array(vs), tg=np.array(tg), kp=mm.actuator_gainprm[:, 0], b1=mm.actuator_biasprm[:, 1],
                 b2=mm.actuator_biasprm[:, 2], fr=mm.actuator_forcerange, dt=env.dt)
        return
    m = env.mj_model
    if args.c_iterations:
        m.opt.iterations = args.c_iterations
    d = mujoco.MjData(m)
    n_sub = int(round(env.dt / m.opt.timestep))
    hand = [m.geom("left_hand_collision").id, m.geom("right_hand_collision").id]
    ball = m.geom("ball").id
    errs = []
    for t in range(n):
        d.qpos[:] = qs[t]; d.qvel[:] = vs[t]; d.ctrl[:] = tg[t]
        mujoco.mj_forward(m, d)
        contact = False
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
            contact |= any((c.geom1 == ball and c.geom2 in hand) or (c.geom2 == ball and c.geom1 in hand) for c in d.contact[:d.ncon])
        dv_ball = np.linalg.norm(d.qvel[35:38] - vs[t + 1][35:38])
        dq = np.abs(d.qpos[7:36] - qs[t + 1][7:36]).max()
        dz = abs(d.qpos[2] - qs[t + 1][2])
        mjx_dv = np.linalg.norm(vs[t + 1][35:38] - vs[t][35:38])
        errs.append((contact, dv_ball, dq, dz, mjx_dv))
    e = np.array(errs, dtype=float)
    for lab, sel in (("hand contact", e[:, 0] > 0), ("no hand contact", e[:, 0] == 0)):
        if sel.any():
            print(f"teacher-forced one-step, {lab:15s} ({int(sel.sum())} steps): ball dv err mean {e[sel, 1].mean():.3f} "
                  f"max {e[sel, 1].max():.3f} m/s (MJX |dv| mean {e[sel, 4].mean():.3f}); joint err max {e[sel, 2].max():.4f} rad; "
                  f"pelvis z err max {e[sel, 3].max() * 1000:.2f} mm")
    worst = np.argsort(-e[:, 1])[:5]
    print("worst steps (t, contact, ball dv err, MJX |dv|):", [(int(i), int(e[i, 0]), round(e[i, 1], 3), round(e[i, 4], 3)) for i in worst])
    for i in worst[:2]:
        print(f"contact probe t={int(i)}:")
        contact_probe(env, qs[i], vs[i], tg[i], n_sub)
    d.qpos[:] = qs[0]; d.qvel[:] = vs[0]
    mujoco.mj_forward(m, d)
    for t in range(n):
        d.ctrl[:] = tg[t]
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        if t in (10, 25, 50, 100, 150, 200, 299) and t < n:
            print(f"open loop t={(t + 1) * env.dt:.2f}s: ball pos err {np.linalg.norm(d.qpos[36:39] - qs[t + 1][36:39]):.3f} m, "
                  f"pelvis err {np.linalg.norm(d.qpos[:3] - qs[t + 1][:3]):.3f} m")


def contact_probe(env, q, v, ctrl, n_sub):
    """List ball contacts per substep for one control step, in MJX and in the C engine."""
    from mujoco import mjx
    m = env.mj_model
    name = lambda g: m.geom(int(g)).name or m.body(m.geom_bodyid[int(g)]).name  # noqa: E731
    d = mujoco.MjData(m); d.qpos[:] = q; d.qvel[:] = v; d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d)
    bg = m.geom("ball").id
    dx = mjx.put_data(m, d, impl=env.mjx_model.impl.value)
    stepx = jax.jit(lambda dd: mjx.step(env.mjx_model, dd))
    for s in range(n_sub):
        cx = [(name(g[0]), name(g[1]), round(float(dd), 4)) for g, dd in zip(np.array(dx._impl.contact.geom), np.array(dx._impl.contact.dist))
              if bg in g and dd < 0.01]
        cc = [(name(c.geom1), name(c.geom2), round(c.dist, 4)) for c in d.contact[:d.ncon] if bg in (c.geom1, c.geom2)]
        if cx or cc:
            print(f"   substep {s}: MJX {cx}  C {cc}  ball vz MJX {float(dx.qvel[37]):+.2f} C {d.qvel[37]:+.2f}")
        dx = stepx(dx)
        mujoco.mj_step(m, d)


if __name__ == "__main__":
    main()
