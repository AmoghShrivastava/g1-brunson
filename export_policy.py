"""Export Brax PPO params to a plain numpy .npz usable by policy.py (no JAX at inference).

Verifies numerically that the numpy forward pass equals the Brax/JAX deterministic policy.
"""
import argparse
import os
import pickle

import jax
import jax.numpy as jp
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params", help="params.pkl from train.py")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_weights.npz"))
    args = ap.parse_args()
    with open(args.params, "rb") as f:
        params = pickle.load(f)
    normalizer, policy = params[0], params[1]
    # Brax running statistics: mean / std per observation key (dict obs -> only 'state' used by the policy).
    mean = normalizer.mean["state"] if isinstance(normalizer.mean, dict) else normalizer.mean
    std = normalizer.std["state"] if isinstance(normalizer.std, dict) else normalizer.std
    layers = policy["params"]
    names = sorted(layers.keys(), key=lambda n: int(n.split("_")[1]))
    weights = {"obs_mean": np.asarray(mean, np.float32), "obs_std": np.asarray(std, np.float32), "n_layers": len(names)}
    for i, n in enumerate(names):
        weights[f"w{i}"] = np.asarray(layers[n]["kernel"], np.float32)
        weights[f"b{i}"] = np.asarray(layers[n]["bias"], np.float32)
    np.savez(args.out, **weights)

    # --- verification against brax ---
    import functools
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    obs_size = {"state": int(mean.shape[0]), "privileged_state": int(normalizer.mean["privileged_state"].shape[0])}
    nets = ppo_networks.make_ppo_networks(
        obs_size, 29, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state", value_obs_key="privileged_state")
    make_policy = ppo_networks.make_inference_fn(nets)
    policy_fn = make_policy((normalizer, policy), deterministic=True)
    import policy as np_policy
    p = np_policy.MLPPolicy(args.out)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        obs = rng.normal(size=(obs_size["state"],)).astype(np.float32) * 2
        a_jax, _ = policy_fn({"state": jp.array(obs), "privileged_state": jp.zeros(obs_size["privileged_state"])}, jax.random.PRNGKey(0))
        a_np = p.forward(obs)
        worst = max(worst, float(np.abs(np.asarray(a_jax) - a_np).max()))
    print(f"exported {args.out}: layers={len(names)} obs={obs_size['state']} max|jax-numpy|={worst:.2e}")
    assert worst < 1e-4


if __name__ == "__main__":
    main()
