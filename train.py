"""PPO training (Brax PPO on MJX) for the G1 between-the-legs dribble.

Hyper-parameters are MuJoCo Playground's G1 hardware-transferred joystick defaults (asymmetric
actor-critic, 512-256-128 MLPs, 8192 envs, unroll 20, 32 minibatches, lr 3e-4, gamma 0.97).
"""
import argparse
import functools
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp

if not hasattr(jax, "device_put_replicated"):  # removed in JAX 0.10; Brax 0.14 still calls it
    def _device_put_replicated(x, devices):
        return jax.device_put(jax.tree_util.tree_map(lambda a: jnp.broadcast_to(jnp.asarray(a), (len(devices),) + jnp.shape(a)), x), devices[0])
    jax.device_put_replicated = _device_put_replicated
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

import dribble_env

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "runs", time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--num_timesteps", type=int, default=200_000_000)
    ap.add_argument("--num_envs", type=int, default=8192)
    ap.add_argument("--num_evals", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restore", default=None, help="params.pkl to warm start from")
    ap.add_argument("--no_dr", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--override", action="append", default=[], help="env config override key=value (json)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    overrides = {}
    for o in args.override:
        k, v = o.split("=", 1)
        overrides[k] = json.loads(v)
    env = dribble_env.G1Dribble(config_overrides=overrides)
    eval_env = dribble_env.G1Dribble(config_overrides=overrides)
    with open(os.path.join(args.out, "env_config.json"), "w") as f:
        json.dump(env._config.to_dict(), f, indent=1, default=str)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    log = open(os.path.join(args.out, "progress.jsonl"), "a")
    t0 = time.time()

    def progress(num_steps, metrics):
        rec = {"steps": int(num_steps), "time": time.time() - t0}
        rec.update({k: float(v) for k, v in metrics.items()})
        log.write(json.dumps(rec) + "\n")
        log.flush()
        keys = ["eval/episode_reward", "eval/avg_episode_length", "eval/episode_task/crossings",
                "eval/episode_term/fall", "eval/episode_term/ball_lost", "eval/episode_term/dead_ball",
                "eval/episode_task/ball_resets", "eval/episode_task/touches", "eval/episode_task/catches", "eval/episode_task/chains", "eval/episode_task/bad_bounces", "eval/episode_task/wrong_hand", "eval/episode_task/recoveries", "eval/episode_task/kicks"]
        print(f"[{rec['time'] / 60:.1f} min] steps={num_steps:,} " +
              " ".join(f"{k.split('/')[-1]}={metrics[k]:.3f}" for k in keys if k in metrics), flush=True)

    def policy_params_fn(current_step, make_policy, params):
        with open(os.path.join(args.out, "params.pkl"), "wb") as f:
            pickle.dump(params, f)
        with open(os.path.join(args.out, f"params_{int(current_step)}.pkl"), "wb") as f:
            pickle.dump(params, f)

    restore_params = None
    if args.restore:
        with open(args.restore, "rb") as f:
            restore_params = pickle.load(f)

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        episode_length=env._config.episode_length,
        action_repeat=1,
        batch_size=256,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=args.lr,
        entropy_cost=args.entropy,
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        max_grad_norm=1.0,
        num_evals=args.num_evals,
        num_resets_per_eval=1,
        seed=args.seed,
        network_factory=network_factory,
        randomization_fn=None if args.no_dr else dribble_env.domain_randomize,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        restore_params=restore_params,
    )
    make_inference_fn, params, metrics = train_fn(environment=env, eval_env=eval_env)
    with open(os.path.join(args.out, "params.pkl"), "wb") as f:
        pickle.dump(params, f)
    print("final", {k: float(v) for k, v in metrics.items() if k.startswith("eval/episode_task") or k == "eval/episode_reward"})
    print("saved", args.out)


if __name__ == "__main__":
    main()
