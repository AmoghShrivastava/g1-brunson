# Safety evidence

HIM Arena's simulation safety screening returned **unsupported** for this package: the screener currently
has one adapter, `g1-walker-99-v1`, whose observation contract is a locomotion policy's (IMU, joints,
velocity command). G1 Brunson's observation includes the tracked basketball and four task cues, so it
cannot be run through that adapter, and we do not declare it. The policy was therefore **not screened**;
nothing about its behaviour was tested by the arena. This file is the evidence we can offer instead,
measured in the packaged simulator with `safety_check.py` (same 12 episodes and seeds as
`evaluation/results.json`, physics randomisation off). Raw numbers: `evaluation/safety.json`.

**This report does not claim hardware readiness.** Everything below is simulation.

## Summary

| | |
|---|---|
| episodes × length | 12 × 20 s (240.0 s simulated, 12000 control steps at 50 Hz) |
| falls | **0 of 12** |
| base height | 0.614 m min · 0.679 m mean (standing 0.75 m) |
| base tilt from vertical | 13.5° mean · 30.6° max |
| action range | tanh-bounded to [-1, 1]; max |a| = 1.0; targets = stance + a × per-joint scale (≤ 0.8 rad legs, 0.25 rad ankles) |
| action change per step | 0.854 mean · 1.992 max (in action units) |
| torque utilisation (all motors) | 11.0 % mean of the Unitree limit |
| steps where any motor sits at its torque limit | 705 of 12000 (5.9 %) |
| steps where any joint is beyond its range | 10557 of 12000 (88 %) |
| peak joint speed | 30.29 rad/s |

Limits are Unitree's `g1_29dof_rev_1_0` joint ranges and actuator torque limits (`assets/g1_robot.xml`).
Torques are clamped to those limits in simulation, so "at its torque limit" means the motor is asking for
more than it has.

## Findings a hardware operator should know

### 1. Waist pitch and ankle roll lean on their joint stops

| joint | range | steps beyond range | max beyond |
|---|---|---|---|
| waist_pitch | -29.8° to 29.8° | 63.1 % | 2.62° |
| left_ankle_roll | -15.0° to 15.0° | 54.2 % | 6.45° |
| right_ankle_roll | -15.0° to 15.0° | 45.0 % | 3.59° |
| waist_roll | -29.8° to 29.8° | 0.2 % | 1.37° |

The dribbling posture is a deep crouch with the torso pitched forward over the ball and a staggered,
slightly wide stance with both feet planted. That posture loads the waist pitch joint against its 29.8°
stop and the ankle roll joints against their ±15° stops for about half of every episode. In MuJoCo a
joint limit is a soft constraint, so the joint reads a few degrees past it under load. The position
targets themselves are clamped to the joint range (`ctrlrange` = `jnt_range`), so the motor is never
commanded beyond the range; the excursion is external load on the stop. On hardware the mechanical stop
carries this load instead. Before running the stance on a robot, confirm the waist pitch and ankle roll
stops are rated for sustained load, or add a small margin to the stance targets and re-test.

### 2. Small motors saturate on ball impacts

| motor | limit (Nm) | peak | mean | RMS (Nm) |
|---|---|---|---|---|
| right_shoulder_pitch | 25 | 100 % | 21.3 % | 6.55 |
| left_shoulder_pitch | 25 | 100 % | 20.8 % | 6.36 |
| right_wrist_pitch | 5 | 100 % | 15.7 % | 1.19 |
| left_wrist_pitch | 5 | 100 % | 13.7 % | 1.0 |
| left_wrist_yaw | 5 | 100 % | 9.5 % | 0.93 |
| right_wrist_yaw | 5 | 100 % | 7.6 % | 0.74 |

The 5 Nm wrist motors and the 25 Nm shoulder pitch motors reach their torque limit briefly when the palm
strikes the ball; mean utilisation stays low (8 to 21 %). On hardware this means the wrist will be
back-driven by the ball at contact rather than holding its target, which is acceptable for a palm push
but should be checked against the wrist gearbox's impact rating.

### 3. Joint speeds

Fastest joints: right_wrist_yaw 30.29 rad/s, right_wrist_pitch 28.58 rad/s, left_wrist_yaw 25.69 rad/s, left_wrist_pitch 24.09 rad/s. These are wrist
flicks at the ball. Confirm them against the rated joint speeds on Unitree's current G1 datasheet before deployment.

### 4. Falls

0 falls in 12 episodes, matching `evaluation/results.json` on the same seeds. Under the training-time
physics randomisation (masses, gains, friction, pushes) about 3 % of episodes ended in a fall, so the
honest estimate is a low but non-zero fall rate. A gantry or harness is required for any hardware attempt.

## What was randomised in training

Link masses, motor gains, floor and hand friction, ball mass and bounce, sensor offsets and noise, tracker
delay up to 40 ms, and periodic pushes to the hips. The head-camera tracker, with its dropouts and
noise, was replicated inside training.

## How to reproduce

```bash
python safety_check.py --episodes 12          # writes evaluation/safety.json
```
