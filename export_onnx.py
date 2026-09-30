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


def build(weights_path):
    w = np.load(weights_path)
    n = int(w["n_layers"])
    obs_dim = int(w["obs_mean"].shape[0])
    act_dim = int(w[f"b{n - 1}"].shape[0]) // 2
    inits = [numpy_helper.from_array(w["obs_mean"].astype(np.float32), "obs_mean"),
             numpy_helper.from_array(w["obs_std"].astype(np.float32), "obs_std")]
    nodes = [helper.make_node("Sub", ["obs", "obs_mean"], ["x_c"]),
             helper.make_node("Div", ["x_c", "obs_std"], ["h0"])]
    x = "h0"
    for i in range(n):
        inits += [numpy_helper.from_array(w[f"w{i}"].astype(np.float32), f"W{i}"),
                  numpy_helper.from_array(w[f"b{i}"].astype(np.float32), f"b{i}")]
        nodes.append(helper.make_node("MatMul", [x, f"W{i}"], [f"mm{i}"]))
        nodes.append(helper.make_node("Add", [f"mm{i}", f"b{i}"], [f"z{i}"]))
        if i < n - 1:  # SiLU = x * sigmoid(x)
            nodes.append(helper.make_node("Sigmoid", [f"z{i}"], [f"s{i}"]))
            nodes.append(helper.make_node("Mul", [f"z{i}", f"s{i}"], [f"h{i + 1}"]))
            x = f"h{i + 1}"
        else:
            x = f"z{i}"
    inits += [numpy_helper.from_array(np.array([0], np.int64), "sl_start"),
              numpy_helper.from_array(np.array([act_dim], np.int64), "sl_end"),
              numpy_helper.from_array(np.array([1], np.int64), "sl_axes")]
    nodes.append(helper.make_node("Slice", [x, "sl_start", "sl_end", "sl_axes"], ["loc"]))
    nodes.append(helper.make_node("Tanh", ["loc"], ["action"]))
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
    args = ap.parse_args()
    model, obs_dim = build(args.weights)
    onnx.save(model, args.out)
    # verify against the numpy controller
    import onnxruntime as ort
    from policy import MLPPolicy
    ref = MLPPolicy(args.weights)
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        obs = (rng.normal(size=(1, obs_dim)) * 2).astype(np.float32)
        a = sess.run(["action"], {"obs": obs})[0][0]
        worst = max(worst, float(np.abs(a - ref.forward(obs[0])).max()))
    print(f"wrote {args.out} ({os.path.getsize(args.out) // 1024} KB), max |onnx - numpy| = {worst:.2e}")
    assert worst < 1e-5


if __name__ == "__main__":
    main()
