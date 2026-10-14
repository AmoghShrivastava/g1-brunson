"""Export the controller network to ONNX (the HIM Arena "Preferred" submission format).

Graph: obs[N,115] -> normalise (Brax running statistics) -> 512 -> 256 -> 128 (SiLU) -> 58 -> take the
first 29 (mean of the tanh-normal head) -> tanh -> action[N,29] in [-1, 1].  Joint targets are
stance_targets + action * ACTION_SCALE (see policy.py); the cue bookkeeping and the head-camera
perception stay outside the network.  Verified numerically against policy.MLPPolicy.

usage: python export_onnx.py [--weights policy_weights.npz] [--out policy.onnx]
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))


def arena_bounds(profile_path, scene_xml=os.path.join(HERE, "assets", "g1_dribble_scene.xml"), margin=0.01):
    """Per-stance action bounds that keep stance_target + a * ACTION_SCALE inside the joint range, with
    the stance targets re-solved for the profile's gains (as dribble_env does in safety mode)."""
    import json
    import mujoco
    from policy import ACTION_SCALE
    prof = json.load(open(profile_path))["joints"]
    m = mujoco.MjModel.from_xml_path(scene_xml)
    kp_a = np.array([j["kp"] for j in prof])
    lo = np.maximum([j["position_min_rad"] for j in prof], m.jnt_range[1:30, 0]) + margin
    hi = np.minimum([j["position_max_rad"] for j in prof], m.jnt_range[1:30, 1]) - margin
    out = {}
    for side in ("left", "right"):
        k = m.keyframe(f"stance_{side}_front")
        q, c = k.qpos[7:36], k.ctrl
        d = q + (c - q) * (m.actuator_gainprm[:, 0] / kp_a)
        out[side] = dict(default=d, a_lo=(lo - d) / ACTION_SCALE, a_hi=(hi - d) / ACTION_SCALE)
    return out


def _mlp(w, p, inits, nodes):
    """Normalise obs and run one tanh-normal actor MLP; returns the name of its raw output."""
    n = int(w["n_layers"])
    inits += [numpy_helper.from_array(w["obs_mean"].astype(np.float32), f"{p}obs_mean"),
              numpy_helper.from_array(w["obs_std"].astype(np.float32), f"{p}obs_std")]
    nodes += [helper.make_node("Sub", ["obs_in", f"{p}obs_mean"], [f"{p}x_c"]),
              helper.make_node("Div", [f"{p}x_c", f"{p}obs_std"], [f"{p}h0"])]
    x = f"{p}h0"
    for i in range(n):
        inits += [numpy_helper.from_array(w[f"w{i}"].astype(np.float32), f"{p}W{i}"),
                  numpy_helper.from_array(w[f"b{i}"].astype(np.float32), f"{p}b{i}")]
        nodes.append(helper.make_node("MatMul", [x, f"{p}W{i}"], [f"{p}mm{i}"]))
        nodes.append(helper.make_node("Add", [f"{p}mm{i}", f"{p}b{i}"], [f"{p}z{i}"]))
        if i < n - 1:  # SiLU = x * sigmoid(x)
            nodes.append(helper.make_node("Sigmoid", [f"{p}z{i}"], [f"{p}s{i}"]))
            nodes.append(helper.make_node("Mul", [f"{p}z{i}", f"{p}s{i}"], [f"{p}h{i + 1}"]))
            x = f"{p}h{i + 1}"
        else:
            x = f"{p}z{i}"
    nodes.append(helper.make_node("Slice", [x, "sl_start", "sl_end", "sl_axes"], [f"{p}mlp_out"]))
    return f"{p}mlp_out"


def build(weights_path, rate=None, bounds=None, init_step=None, wrist_mask=None, stand_path=None, stand_dist=1.0):
    """stand_path: a second (stand) controller used while the newest ball position is farther than
    stand_dist (m, horizontal, pelvis frame) from the pelvis, i.e. once the ball is out of play."""
    w = np.load(weights_path)
    n = int(w["n_layers"])
    obs_dim = int(w["obs_mean"].shape[0])
    act_dim = int(w[f"b{n - 1}"].shape[0]) // 2
    inits = [numpy_helper.from_array(np.array([0], np.int64), "sl_start"),
             numpy_helper.from_array(np.array([act_dim], np.int64), "sl_end"),
             numpy_helper.from_array(np.array([1], np.int64), "sl_axes")]
    # the ball readings are clipped to the range seen in training (BALL_OBS_LIMIT): a ball that rolled
    # 10 m away must look like one that is out of play, not like an input the networks never saw
    from policy import BALL_OBS_LIMIT
    lim = np.tile(BALL_OBS_LIMIT, 3).astype(np.float32)[None]
    inits += [numpy_helper.from_array(np.array([93], np.int64), "b_start"),
              numpy_helper.from_array(np.array([111], np.int64), "b_end"),
              numpy_helper.from_array(np.array([0], np.int64), "o_start"),
              numpy_helper.from_array(np.array([obs_dim], np.int64), "o_end"),
              numpy_helper.from_array(lim, "b_hi"), numpy_helper.from_array(-lim, "b_lo")]
    nodes = [helper.make_node("Slice", ["obs", "o_start", "b_start", "sl_axes"], ["obs_a"]),
             helper.make_node("Slice", ["obs", "b_start", "b_end", "sl_axes"], ["obs_b"]),
             helper.make_node("Slice", ["obs", "b_end", "o_end", "sl_axes"], ["obs_c"]),
             helper.make_node("Max", ["obs_b", "b_lo"], ["obs_b1"]),
             helper.make_node("Min", ["obs_b1", "b_hi"], ["obs_b2"]),
             helper.make_node("Concat", ["obs_a", "obs_b2", "obs_c"], ["obs_in"], axis=1)]
    main = _mlp(w, "", inits, nodes)
    if stand_path is None:
        nodes.append(helper.make_node("Identity", [main], ["loc_full"]))
    else:
        stand = _mlp(np.load(stand_path), "st_", inits, nodes)
        inits += [numpy_helper.from_array(np.array([93], np.int64), "bxy_start"),
                  numpy_helper.from_array(np.array([95], np.int64), "bxy_end"),
                  numpy_helper.from_array(np.array(stand_dist ** 2, np.float32), "stand_d2"),
                  numpy_helper.from_array(np.array([1], np.int64), "sum_axes")]
        nodes += [helper.make_node("Slice", ["obs", "bxy_start", "bxy_end", "sl_axes"], ["ball_xy"]),
                  helper.make_node("Mul", ["ball_xy", "ball_xy"], ["ball_xy2"]),
                  helper.make_node("ReduceSum", ["ball_xy2", "sum_axes"], ["ball_d2"], keepdims=1),
                  helper.make_node("Greater", ["ball_d2", "stand_d2"], ["ball_out"]),
                  helper.make_node("Where", ["ball_out", stand, main], ["loc_full"])]
    if wrist_mask is not None:
        # wrist pitch/yaw held at their default targets (mean 0 -> tanh 0)
        inits.append(numpy_helper.from_array(wrist_mask.astype(np.float32), "wrist_mask"))
        nodes.append(helper.make_node("Mul", ["loc_full", "wrist_mask"], ["loc"]))
    else:
        nodes.append(helper.make_node("Identity", ["loc_full"], ["loc"]))
    if rate is None:
        nodes.append(helper.make_node("Tanh", ["loc"], ["action"]))
    else:
        # safety layer: each action may move at most rate (action units) from the previous action,
        # which is part of the observation (obs[:, 64:93])
        nodes.append(helper.make_node("Tanh", ["loc"], ["a_raw"]))
        inits += [numpy_helper.from_array(np.array([64], np.int64), "pa_start"),
                  numpy_helper.from_array(np.array([93], np.int64), "pa_end"),
                  numpy_helper.from_array(rate.astype(np.float32), "rate")]
        nodes.append(helper.make_node("Slice", ["obs", "pa_start", "pa_end", "sl_axes"], ["prev_a"]))
        # handover: on the first call (previous action all zero) the step is limited to 80 % of the
        # initial-position tolerance instead of the rate limit
        inits += [numpy_helper.from_array(init_step.astype(np.float32), "init_step"),
                  numpy_helper.from_array(np.array(0.0, np.float32), "zero_f")]
        nodes.append(helper.make_node("Abs", ["prev_a"], ["prev_abs"]))
        nodes.append(helper.make_node("ReduceMax", ["prev_abs"], ["prev_max"], axes=[1], keepdims=1))
        nodes.append(helper.make_node("Equal", ["prev_max", "zero_f"], ["is_first"]))
        nodes.append(helper.make_node("Where", ["is_first", "init_step", "rate"], ["step_lim"]))
        nodes.append(helper.make_node("Sub", ["prev_a", "step_lim"], ["a_lo"]))
        nodes.append(helper.make_node("Add", ["prev_a", "step_lim"], ["a_hi"]))
        nodes.append(helper.make_node("Max", ["a_raw", "a_lo"], ["a_c1"]))
        nodes.append(helper.make_node("Min", ["a_c1", "a_hi"], ["a_rc" if bounds else "action"]))
        if bounds:
            # keep the decoded target inside the joint range; the stance cue obs[:, 114] (+1 = left
            # foot forward) selects which stance's default targets apply
            for side in ("left", "right"):
                inits += [numpy_helper.from_array(bounds[side]["a_lo"].astype(np.float32), f"rlo_{side}"),
                          numpy_helper.from_array(bounds[side]["a_hi"].astype(np.float32), f"rhi_{side}")]
            inits += [numpy_helper.from_array(np.array([114], np.int64), "cue_start"),
                      numpy_helper.from_array(np.array([115], np.int64), "cue_end"),
                      numpy_helper.from_array(np.array(0.0, np.float32), "zero")]
            nodes.append(helper.make_node("Slice", ["obs", "cue_start", "cue_end", "sl_axes"], ["stance_cue"]))
            nodes.append(helper.make_node("Greater", ["stance_cue", "zero"], ["is_left"]))
            nodes.append(helper.make_node("Where", ["is_left", "rlo_left", "rlo_right"], ["r_lo"]))
            nodes.append(helper.make_node("Where", ["is_left", "rhi_left", "rhi_right"], ["r_hi"]))
            nodes.append(helper.make_node("Max", ["a_rc", "r_lo"], ["a_c2"]))
            nodes.append(helper.make_node("Min", ["a_c2", "r_hi"], ["action"]))
    graph = helper.make_graph(
        nodes, "g1_brunson",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["N", obs_dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["N", act_dim])],
        initializer=inits,
        doc_string="G1 Brunson: between-the-legs dribble controller for the Unitree G1 (29 joint targets in [-1,1] "
                   "action units; obs = gyro(3), gravity(3), joint_pos_rel(29), joint_vel(29), prev_action(29), "
                   "ball_rel history 3x6, task cues(4)).")
    model = helper.make_model(graph, producer_name="g1-brunson", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model.metadata_props.append(onnx.StringStringEntryProto(key="policy_type", value="trained"))
    model.metadata_props.append(onnx.StringStringEntryProto(key="robot", value="unitree-g1-29dof"))
    model.metadata_props.append(onnx.StringStringEntryProto(key="control_hz", value="50"))
    onnx.checker.check_model(model)
    return model, obs_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "policy_weights.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "policy.onnx"))
    ap.add_argument("--freeze_wrists", action="store_true", help="hold wrist pitch/yaw at their default targets")
    ap.add_argument("--waist_rate", type=float, default=0.0, help="rad/step rate limit for the waist targets (0 = profile)")
    ap.add_argument("--safety_profile", default=os.path.join(HERE, "assets", "arena_g1_profile.json"),
                    help="joint profile; adds the per-step target rate limit to the graph ('' to disable)")
    ap.add_argument("--stand_weights", default="", help="stand controller used while the ball is out of play")
    ap.add_argument("--stand_dist", type=float, default=1.0, help="m; ball farther than this from the pelvis = out of play")
    args = ap.parse_args()
    rate = init_step = None
    if args.safety_profile:
        import json
        from policy import ACTION_SCALE
        prof = json.load(open(args.safety_profile))["joints"]
        rate_rad = np.array([j["max_velocity_rad_s"] * 0.02 for j in prof], np.float32)
        if args.waist_rate > 0:
            rate_rad[12:15] = np.minimum(rate_rad[12:15], args.waist_rate)  # waist yaw, roll, pitch
        rate = rate_rad / ACTION_SCALE
        init_step = 0.8 * json.load(open(args.safety_profile))["initial_position_tolerance_rad"] / ACTION_SCALE
    bounds = arena_bounds(args.safety_profile) if args.safety_profile else None
    wrist_mask = None
    if args.freeze_wrists:
        from policy import WRIST_MASK
        wrist_mask = WRIST_MASK
    model, obs_dim = build(args.weights, rate, bounds, init_step, wrist_mask, args.stand_weights or None, args.stand_dist)
    onnx.save(model, args.out)
    # verify against the numpy controller
    import onnxruntime as ort
    from policy import MLPPolicy
    ref = MLPPolicy(args.weights)
    ref_stand = MLPPolicy(args.stand_weights) if args.stand_weights else None
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in range(50):
        obs = (rng.normal(size=(1, obs_dim)) * 2).astype(np.float32)
        if i % 5 == 0:
            obs[0, 64:93] = 0.0  # first-call case
        if i % 2 == 0:
            obs[0, 93:95] *= 0.2  # ball in play (near)
        a = sess.run(["action"], {"obs": obs})[0][0]
        far = ref_stand is not None and float(np.sum(obs[0, 93:95] ** 2)) > args.stand_dist ** 2
        r = (ref_stand if far else ref).forward(obs[0])
        if wrist_mask is not None:
            r = r * wrist_mask
        if rate is not None:
            lim = init_step if not np.any(obs[0, 64:93]) else rate
            r = np.clip(r, obs[0, 64:93] - lim, obs[0, 64:93] + lim)
        if bounds is not None:
            b = bounds["left" if obs[0, 114] > 0 else "right"]
            r = np.clip(r, b["a_lo"], b["a_hi"])
        worst = max(worst, float(np.abs(a - r).max()))
    print(f"wrote {args.out} ({os.path.getsize(args.out) // 1024} KB), max |onnx - numpy| = {worst:.2e}")
    assert worst < 5e-5  # float32 rounding through the saturated tanh head


if __name__ == "__main__":
    main()
