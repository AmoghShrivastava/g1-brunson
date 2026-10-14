# Sensor contract: G1 Brunson

`required_sensors`: `joint_state`, `imu`, `camera`.

The dribble controller needs the basketball's position and velocity. On the robot these come from the
G1's own head camera through the tracker in `vision.py`. The standard runner supplies joints and IMU
only; until a runner qualifies and connects the camera input below, the image runs the controller's
stand-by mode (holds the pose measured at session start with the profile gains) and does not dribble.

## Camera

| item | value |
|---|---|
| camera | Intel RealSense D435i, Unitree G1 stock head mount (`d435_joint` in `g1_29dof_rev_1_0.urdf`), tilted 47.6° down |
| depth stream (primary) | 424 × 240, 16-bit millimetres, 30 Hz, 87° × 58° field of view |
| colour stream (fallback) | 424 × 240 RGB8, 30 Hz, 69° × 42° field of view |
| timestamps | per-frame capture time, same clock as the joint state |
| intrinsics | from the camera's factory calibration (`fx, fy, cx, cy` per stream) |
| extrinsics | head-camera pose from the URDF `d435_joint` chain, evaluated with the measured joint state |

## Ball state delivered to the controller

| item | value |
|---|---|
| quantity | ball centre position (m) and velocity (m/s) |
| frame | relative to the pelvis IMU, expressed in the pelvis yaw frame (x forward, y left, z up); velocity minus pelvis velocity |
| rate | 50 Hz (the controller rate); the tracker predicts between camera frames |
| latency | up to 40 ms end to end (the controller was trained with 0–40 ms) |
| detection | depth-first: the expected depth of the robot's own body is rendered from the joint state and subtracted; the remaining near pixels are fitted with a sphere of radius 0.11926 m. Colour (orange) detection is the fallback |
| lost track | a Kalman filter with a ballistic and floor-bounce model carries the estimate while the ball is hidden; beyond the trained range (±3 m, ±3 m/s per axis) the controller treats the ball as out of play |

## Other inputs the dribble mode needs

* **Foot positions** for the crossover gate: forward kinematics of the measured joint state.
* **Hand–ball contact** within 20 ms: a wrist joint torque or current transient, or the velocity
  discontinuity of the tracked ball at the palm.

Both feed `policy.DribbleTracker`, which produces the four task cues of the observation.
