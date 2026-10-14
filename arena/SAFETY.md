# Safety evidence

This release is trained and tested against the HIM Arena G1 screening setup: the `g1_type5` joint
profile (`assets/arena_g1_profile.json`: PD gains, torque, speed and range limits) and the arena's
basketball screen as described in its reports (MuJoCo 3.3.7 on CPU, one-step ball delay, no action
delay, no ball replacement, 20-second episodes). Everything below is simulation; it does not claim
hardware readiness.

## Result

`arena_screen_cpu.py` replicates that screen with the exported `policy.onnx`. It was run on 36 episodes
(seeds 1000–1011 and 2000–2023). Raw numbers: `evaluation/arena_screen_cpu.json`,
`evaluation/arena_screen_cpu_seed2000.json`.

| check | limit | episodes with a finding |
|---|---|---|
| fall | pelvis below 0.28 m or torso tilted past ~57° | **0 / 36** |
| motor torque | demand before clamping and applied torque vs. 88–139 / 35 / 25 / 5 Nm | **0 / 36** |
| joint speed | 20–37 rad/s per joint (22 wrists) | **0 / 36** |
| joint position | 0.05 rad past the joint range | **0 / 36** |
| body rotation | 6 rad/s on any axis | **0 / 36** |
| thrashing | weighted joint oscillation above 1.5 for 0.5 s | **0 / 36** |
| first command | within 0.03 rad of the measured pose | **0 / 36** |

## Built-in safeguards

| safeguard | where |
|---|---|
| joint targets clamped to the joint range (1 cm margin) | `policy.onnx`, `policy.py` |
| each target moves at most 9 rad/s × 20 ms per step | `policy.onnx`, `policy.py` |
| first command within 80 % of the 0.03 rad handover tolerance of the measured pose | `policy.onnx`, `policy.py` |
| ball readings clipped to the range seen in training (±3 m, ±3 m/s), so a ball that rolled far away reads as out of play | `policy.onnx`, `policy.py` |
| motor torque capped at 98 % of the profile limit | `assets/g1_robot.xml` |

## Simulation setup

* **Start pose.** The stance targets are re-solved for the profile's gains. The pelvis height and the
  four ankle angles are solved so both soles rest flat on the floor with the pelvis upright, and the
  start pose equals the default targets (`make_stance_arena.py`). With the joints on re-solved targets
  and the previously authored pelvis height, the feet started up to 3.8 cm inside the floor, which
  MuJoCo 3.3.7 resolves with a violent push-out in the first milliseconds.
* **Solver.** The scene uses 10 Newton iterations. With 4, MuJoCo 3.3.7 does not converge on the foot
  contacts and the ankles spike to 10–28 rad/s; 10, 30 and 100 iterations give identical results.
* **Training.** The final stages charged every screened limit: torque demand before clamping, per-axis
  body rotation, weighted joint oscillation, joint position past the range, joint speed against the
  screen's per-joint limits, foot drift, and falls. Half or more of the training episodes never replaced
  a lost ball, as in the screen, so the robot learned to keep its stance once the ball is gone.

## Task performance

`evaluation/results.json` (12 × 20 s, MJX, same clamps and gains, a lost ball is replaced):
17.0 crossovers and 13.4 catches per test. A controller that holds the stance without acting falls in
every episode, so the motion is produced by the policy.

## How to reproduce

```bash
pip install mujoco==3.3.7 onnxruntime numpy onnx     # the arena's simulator version
python arena_screen_cpu.py --onnx policy.onnx --episodes 12 --seed 1000
python arena_screen_cpu.py --onnx policy.onnx --episodes 24 --seed 2000
python check_obs_cpu.py --weights policy_weights.npz --onnx policy.onnx   # observation parity with training
```
