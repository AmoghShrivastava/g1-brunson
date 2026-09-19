"""Observation-in, joint-target-out controller for the Unitree G1 between-the-legs dribble.

`Policy.reset()` / `Policy.act(obs)` follow the HIM Open Build contract.  Inference is plain numpy
(no JAX/torch at run time).  The observation is exactly what a real G1 + a ball tracker provide:

    gyro (3)                    pelvis IMU angular velocity, body frame
    projected gravity (3)       from the pelvis IMU orientation
    joint pos - stance (29)     Unitree hardware joint order
    joint vel (29)
    last action (29)
    ball state history (3 x 6)  ball position & velocity relative to the pelvis, in the pelvis
                                yaw-aligned frame (newest first; each frame is the tracker's
                                latest estimate, 0-40 ms old)
    cue (4)                     [next hand (+1 right / -1 left), ball released (+1) or not (-1),
                                 time since last floor bounce (s, clipped to 1), stance (+1 left
                                 foot forward / -1 right)]

The cue is derived on-robot from the same ball tracker (bounce = floor contact after flight, hand
switch after a bounce between the feet); `DribbleTracker` below implements that logic from ball
position/velocity and foot positions only, and is what `evaluate_vision.py` uses -- the simulator's
contact flags are not needed.

Output: 29 joint position targets = stance targets + tanh(mlp) * per-joint action scale, sent to the
PD loop (kp/kd = Unitree deployment gains) at 50 Hz.
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BALL_R = 0.11926

ACTION_SCALE = np.array(
    [0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
     0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
     0.55, 0.44, 0.44,
     0.8, 0.8, 0.8, 0.8, 0.8, 0.25, 0.25,
     0.8, 0.8, 0.8, 0.8, 0.8, 0.25, 0.25], dtype=np.float32)


def _silu(x):
    return x / (1.0 + np.exp(-x))


class MLPPolicy:
    def __init__(self, path):
        w = np.load(path)
        self.mean = w["obs_mean"]
        self.std = w["obs_std"]
        self.n = int(w["n_layers"])
        self.layers = [(w[f"w{i}"], w[f"b{i}"]) for i in range(self.n)]

    def forward(self, obs):
        x = (obs - self.mean) / self.std  # brax running_statistics.normalize
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < self.n - 1:
                x = _silu(x)
        loc = x[: x.shape[0] // 2]
        return np.tanh(loc)  # deterministic mode of the tanh-normal policy


class DribbleTracker:
    """Ball-only bookkeeping of the dribble phase (runs on the robot next to the ball tracker).

    It must reproduce the training environment's task state machine from the ball estimate alone:
      * bounce = floor contact after flight.  At 50 Hz the ball is on the floor for less than one
        control step, so a height threshold alone misses bounces; a bounce is therefore detected
        from the vertical-velocity reversal (falling -> rising) close to the floor, or the height
        threshold, whichever fires first.
      * crossing = bounce inside the gate between the feet, travelling towards the receiving hand,
        after a flight whose apex reached min_apex (the training rule): the next hand flips.
      * a bounce that fails the crossing test (bad bounce) clears the "released" flag without
        flipping the hand; any hand contact sets "released" and, if it is the other hand, hands
        the ball over to it.
    """

    def __init__(self, dt, first_hand, min_apex=0.30):
        self.dt = dt
        self.min_apex = min_apex
        self.next_hand = int(first_hand)  # 0 left, 1 right
        self.expect = 1  # 1: ball released, waiting for a bounce between the feet; 0: waiting for the hand
        self.since_bounce = 0.0
        self.airborne = True
        self.prev_vz = 0.0
        self.apex = 0.0
        self.crossings = 0
        self.bounces = 0
        self.bad_bounces = 0

    def resync(self, next_hand, expect):
        """Operator hands the ball back / simulator respawn: restart the phase bookkeeping."""
        self.next_hand, self.expect = int(next_hand), int(expect)
        self.since_bounce, self.airborne, self.prev_vz, self.apex = 0.0, True, 0.0, 0.0

    def update(self, ball_pos_w, ball_vel_w, feet_w, hand_contact, body_left_w):
        """ball/feet in world frame; hand_contact = (left, right) physical hand-ball contact flags
        (simulator contacts here; on hardware a ball-velocity discontinuity at the hand);
        body_left_w = robot's +y axis in world (xy)."""
        z, vz = float(ball_pos_w[2]), float(ball_vel_w[2])
        on_floor = z < BALL_R + 0.02
        reversal = (self.prev_vz < -0.3) and (vz > 0.2) and (z < BALL_R + 0.12)
        bounce = (on_floor and self.airborne) or (reversal and self.airborne)
        self.airborne = not on_floor
        self.prev_vz = vz
        self.since_bounce = 0.0 if bounce else self.since_bounce + self.dt
        touched = None
        for h in (0, 1):
            if hand_contact[h]:
                touched = h
        gate = 0.5 * (feet_w[0][:2] + feet_w[1][:2])
        u = feet_w[0][:2] - feet_w[1][:2]
        half = 0.5 * np.linalg.norm(u) + 1e-6
        u = u / (2 * half)
        n_left = np.array([-u[1], u[0]])
        n_left *= np.sign(np.dot(n_left, body_left_w[:2]) + 1e-9)
        crossing = False
        if bounce:
            self.bounces += 1
            along = np.dot(ball_pos_w[:2] - gate, u)
            perp = np.dot(ball_pos_w[:2] - gate, n_left)
            in_gate = abs(along) < 0.8 * half and abs(perp) < 0.15
            want = 1.0 if self.next_hand == 1 else -1.0
            dir_ok = want * np.dot(ball_vel_w[:2], n_left) > 0.05
            apex_ok = self.apex >= self.min_apex
            if self.expect == 1 and in_gate and dir_ok and apex_ok:
                crossing = True
                self.crossings += 1
                self.next_hand = 1 - self.next_hand
                self.expect = 0
            elif self.expect == 1:
                self.bad_bounces += 1
                self.expect = 0
            self.apex = 0.0
        else:
            self.apex = max(self.apex, z)
        if touched is not None:
            if touched != self.next_hand:
                self.next_hand = touched
            self.expect = 1
        return crossing

    def cue(self, stance):
        return np.array([1.0 if self.next_hand == 1 else -1.0, 1.0 if self.expect == 1 else -1.0,
                         min(self.since_bounce, 1.0), 1.0 if stance == 0 else -1.0], dtype=np.float32)


class Policy:
    """HIM contract: reset() then act(obs) -> 29 joint position targets (rad)."""

    def __init__(self, weights=os.path.join(HERE, "policy_weights.npz"), stance_targets=None):
        self.mlp = MLPPolicy(weights)
        self.stance_targets = stance_targets
        self.reset()

    def reset(self):
        self.last_action = np.zeros(29, np.float32)
        self.ball_hist = None

    def act(self, obs):
        """obs: dict with gyro(3), gravity(3), joint_pos_rel(29), joint_vel(29), ball_rel(6), cue(4).

        Returns joint position targets (29) in Unitree joint order.
        """
        ball = np.asarray(obs["ball_rel"], np.float32)
        if self.ball_hist is None:
            self.ball_hist = np.tile(ball[None], (3, 1))
        else:
            self.ball_hist = np.concatenate([ball[None], self.ball_hist[:-1]], axis=0)
        x = np.concatenate([
            np.asarray(obs["gyro"], np.float32), np.asarray(obs["gravity"], np.float32),
            np.asarray(obs["joint_pos_rel"], np.float32), np.asarray(obs["joint_vel"], np.float32),
            self.last_action, self.ball_hist.ravel(), np.asarray(obs["cue"], np.float32)])
        a = self.mlp.forward(x).astype(np.float32)
        self.last_action = a
        targets = np.asarray(obs["stance_targets"], np.float32) + a * ACTION_SCALE
        return targets


class NeutralPolicy(Policy):
    """Causality control: same interface, zero action (holds the stance targets)."""

    def __init__(self):
        self.reset()

    def act(self, obs):
        self.last_action = np.zeros(29, np.float32)
        return np.asarray(obs["stance_targets"], np.float32)
