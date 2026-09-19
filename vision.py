"""Vision pipeline: head-mounted Intel RealSense D435i (the G1 EDU head camera: RGB + depth) -> ball
detection in RGB -> 3-D position from depth (fallback: monocular size of the known-radius ball) -> Kalman
tracker.  This is the perception module the state-based policy needs on hardware; in simulation the
camera frames are rendered from the same MuJoCo model, and the policy only receives this module's
output (never the simulator's ball state).

Camera: Unitree's URDF places the D435 on the torso at (0.0576235, 0.01753, 0.42987) m with a pitch of
0.83078 rad (47.6 deg) downward (`d435_joint`, g1_29dof_rev_1_0.urdf).

Two streams, two detectors (Intel RealSense D435 datasheet):
  * PRIMARY = depth stream, 87 x 58 deg field of view (the stereo IR imagers), rendered 424 x 240 with
    fovy = 58 deg.  The ball is segmented geometrically: every depth pixel that is nearer than the
    robot's own body would be at that pixel (robot-only depth image rendered from the URDF at the
    encoder pose = the same "expected self-depth" mask a real deployment computes from FK) is a
    foreign object; the floor and the robot cancel out, the ball is what remains.  This does not depend
    on lighting/colour and has the wider field of view, so it sees the ball longer around the legs.
  * BACKUP = colour stream, 69.4 x 42.5 deg, 424 x 240, fovy = 42.5 deg: orange-ball segmentation, range
    from the aligned depth (front surface + one radius) or, if depth is unavailable, from the apparent
    size of the ball (known radius).  Used whenever the depth detector has no blob.
Both give the ball centre in the camera frame; the camera pose comes from forward kinematics.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

BALL_R = 0.11926
CAM_NAME = "d435"
CAM_POS = (0.0576235, 0.01753, 0.42987)  # torso_link frame (Unitree URDF d435_joint)
CAM_PITCH = 0.8307767239493009  # rad, downward (Unitree URDF d435_joint rpy)
FOVY_DEG = 42.5  # D435 colour vertical FOV (69.4 x 42.5)
IMG_W, IMG_H = 424, 240
DEPTH_CAM_NAME = "d435_depth"
DEPTH_FOVY_DEG = 58.0  # D435 depth/IR vertical FOV (87 x 58)
DEPTH_MIN, DEPTH_MAX = 0.15, 6.0  # m, D435 operating range (short-range preset)


def camera_xml(name=CAM_NAME, fovy=FOVY_DEG):
    """MuJoCo <camera> element for the D435 in the torso_link body (MuJoCo cameras look along -z, y up)."""
    p = CAM_PITCH
    fwd = np.array([math.cos(p), 0.0, -math.sin(p)])  # URDF x axis pitched down
    right = np.array([0.0, -1.0, 0.0])
    up = np.cross(-fwd, right)  # camera y = (-fwd) x right
    return (f'<camera name="{name}" pos="{CAM_POS[0]} {CAM_POS[1]} {CAM_POS[2]}" '
            f'xyaxes="{right[0]:.6f} {right[1]:.6f} {right[2]:.6f} {up[0]:.6f} {up[1]:.6f} {up[2]:.6f}" fovy="{fovy}"/>')


def add_camera_to_robot_xml(path):
    """Adds the colour and the depth camera of the D435 (same pose, different FOV) to the robot MJCF."""
    s = open(path, encoding="utf-8").read()
    changed = False
    anchor = '<site name="imu_in_torso"'
    assert anchor in s
    for name, fovy in ((CAM_NAME, FOVY_DEG), (DEPTH_CAM_NAME, DEPTH_FOVY_DEG)):
        if f'name="{name}"' in s:
            continue
        s = s.replace(anchor, camera_xml(name, fovy) + "\n            " + anchor, 1)
        changed = True
    if changed:
        open(path, "w", encoding="utf-8").write(s)
    return changed


class BallDetector:
    """Ball localisation: colour segmentation of the orange ball in RGB, direction from the pixel ray,
    range from the aligned depth image (front surface + one radius); fallback without depth: range
    from the apparent size of a sphere of known radius.

    A blob cut by the image border (the ball is often at the bottom edge of a head camera looking at
    the feet) is handled with circle geometry: the ball's pixel radius is half of the widest row that
    is fully inside the image, and the centre is (row-centre of that row, top edge + radius).
    """

    def __init__(self, w=IMG_W, h=IMG_H, fovy_deg=FOVY_DEG, min_pixels=12):
        self.w, self.h = w, h
        self.fy = 0.5 * h / math.tan(math.radians(fovy_deg) / 2)
        self.fx = self.fy
        self.cx, self.cy = 0.5 * w, 0.5 * h
        self.min_pixels = min_pixels

    def detect(self, rgb, depth=None):
        """Colour detector (backup stream)."""
        r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
        mask = (r > 120) & (r - g > 35) & (r - b > 70) & (g > 30) & (g < 190) & (b < 120)
        return self.from_mask(mask, depth)

    def detect_depth(self, depth, depth_self):
        """Depth detector (primary stream): pixels nearer than the robot's own expected depth
        (URDF rendered at the encoder pose) and inside the sensor range are foreign objects.  The floor
        and the robot are in both images and cancel; the ball is the remaining blob."""
        valid = np.isfinite(depth) & (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
        mask = valid & (depth < depth_self - 0.02)
        if int(mask.sum()) < self.min_pixels:
            return None, mask
        # Sphere of known radius fitted to the depth points (works from a partial cap, e.g. when a hand or
        # a leg hides part of the ball): Gauss-Newton on sum_i (|p_i - c| - R)^2.
        vv, uu = np.nonzero(mask)
        zc = depth[vv, uu]
        pts = np.stack([(uu - self.cx) / self.fx * zc, -(vv - self.cy) / self.fy * zc, -zc], 1)
        if pts.shape[0] > 400:
            pts = pts[np.random.default_rng(0).choice(pts.shape[0], 400, replace=False)]
        centroid = pts.mean(0)
        c = centroid + BALL_R * centroid / np.linalg.norm(centroid)  # behind the visible cap, along the view ray
        for _ in range(8):
            dvec = pts - c
            dist = np.linalg.norm(dvec, axis=1)
            J = -dvec / np.maximum(dist[:, None], 1e-6)
            r = dist - BALL_R
            step, *_ = np.linalg.lstsq(J, -r, rcond=None)
            c = c + step
            if np.linalg.norm(step) < 1e-4:
                break
        resid = np.abs(np.linalg.norm(pts - c, axis=1) - BALL_R)
        if np.median(resid) > 0.03:  # not a sphere of this radius (e.g. a foreign object/noise)
            return None, mask
        return c, mask

    def from_mask(self, mask, depth=None):
        n = int(mask.sum())
        if n < self.min_pixels:
            return None, mask
        rows = np.nonzero(mask.any(axis=1))[0]
        top, bottom = int(rows[0]), int(rows[-1])
        widths = mask.sum(axis=1)
        # widest row: its half-width is the pixel radius (valid even if the bottom of the ball is cut)
        row_w = int(np.argmax(widths))
        r_px = 0.5 * float(widths[row_w])
        cols = np.nonzero(mask[row_w])[0]
        u = 0.5 * (cols[0] + cols[-1])
        cut_bottom = bottom >= self.h - 1
        cut_top = top <= 0
        if cut_top or r_px < 2.5:
            return None, mask
        if cut_bottom and (bottom - top) < r_px:  # less than the top half visible: radius unreliable
            return None, mask
        v = top + r_px  # centre row from the top edge (robust to a cut bottom)
        ray = np.array([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        ray /= np.linalg.norm(ray)
        d = None
        if depth is not None:  # D435i depth: front surface of the ball = closest valid depths in the blob
            dd = depth[mask]
            dd = dd[np.isfinite(dd) & (dd > DEPTH_MIN) & (dd < DEPTH_MAX)]
            if dd.size >= self.min_pixels // 2:
                z_front = float(np.percentile(dd, 10))
                d = z_front / -ray[2] + BALL_R  # along the centre ray: front surface + one radius
        if d is None:  # monocular fallback: half-angle theta of the sphere satisfies sin(theta) = R / d
            theta = math.atan(r_px / self.fy)
            d = BALL_R / math.sin(theta)
        return ray * d, mask


class BallKalman:
    """Constant-velocity + gravity Kalman filter on the ball centre with a bounce model for
    prediction through occlusions (the ball spends much of a between-the-legs dribble out of view)."""

    def __init__(self, dt, cor=0.77, r_meas=0.008, q_vel=1.5):
        self.dt = dt
        self.cor = cor
        self.x = None  # [px py pz vx vy vz]
        self.P = None
        # process noise per step: the hand impulses change the velocity abruptly, so keep q_vel large
        # (little smoothing lag); r_meas = measurement std (depth-stream sphere fit: ~3 mm in sim, a few
        # mm on the D435 at < 1 m; 8 mm is a conservative hardware value)
        self.q_pos, self.q_vel = 0.01 ** 2, q_vel ** 2
        self.r_meas = r_meas ** 2
        self.misses = 0

    def predict(self):
        dt = self.dt
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        self.x = F @ self.x + np.array([0, 0, -0.5 * 9.81 * dt * dt, 0, 0, -9.81 * dt])
        if self.x[2] < BALL_R and self.x[5] < 0:  # ballistic bounce on the floor
            self.x[2] = BALL_R + (BALL_R - self.x[2])
            self.x[5] = -self.cor * self.x[5]
            self.x[3:5] *= 0.9
        Q = np.diag([self.q_pos] * 3 + [self.q_vel] * 3)
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        if self.x is None:
            if z is None:
                return
            self.x = np.concatenate([z, np.zeros(3)])
            self.P = np.diag([self.r_meas] * 3 + [4.0] * 3)
            self.misses = 0
            return
        self.predict()
        if z is None:
            self.misses += 1
            return
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        S = H @ self.P @ H.T + np.eye(3) * self.r_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P
        self.misses = 0

    @property
    def pos(self):
        return None if self.x is None else self.x[:3].copy()

    @property
    def vel(self):
        return None if self.x is None else self.x[3:].copy()


class BallPerception:
    """Renders the head camera, detects the ball, tracks it; returns world-frame position/velocity."""

    BALL_GROUP = 4  # render group used to drop the ball from the robot-only ("self") depth render

    def __init__(self, model, dt, w=IMG_W, h=IMG_H, use_depth=True, mode="depth_first"):
        """mode: 'depth_first' (primary depth-stream detector, colour backup), 'colour' (colour only,
        range from aligned depth if use_depth else monocular), 'depth' (depth stream only)."""
        self.model = model
        self.mode = mode
        self.cam_id = model.camera(CAM_NAME).id
        self.rgb_r = mujoco.Renderer(model, height=h, width=w)
        self.use_depth = use_depth
        if use_depth:
            self.dep_r = mujoco.Renderer(model, height=h, width=w)  # depth aligned to colour
            self.dep_r.enable_depth_rendering()
        self.det = BallDetector(w, h, FOVY_DEG)
        self.depth_cam_id = -1
        self.opt_full = self.opt_self = None
        if mode != "colour":
            self.depth_cam_id = model.camera(DEPTH_CAM_NAME).id
            model.geom_group[model.geom("ball").id] = self.BALL_GROUP
            self.opt_full = mujoco.MjvOption()
            self.opt_full.geomgroup[self.BALL_GROUP] = 1
            self.opt_self = mujoco.MjvOption()
            self.opt_self.geomgroup[self.BALL_GROUP] = 0
            self.dstream_r = mujoco.Renderer(model, height=h, width=w)
            self.dstream_r.enable_depth_rendering()
            self.dself_r = mujoco.Renderer(model, height=h, width=w)
            self.dself_r.enable_depth_rendering()
            self.det_depth = BallDetector(w, h, DEPTH_FOVY_DEG)
        self.kf = BallKalman(dt)
        self.last_rgb = None
        self.last_mask = None
        self.last_depth_mask = None
        self.last_source = None
        self.stats = dict(frames=0, detections=0, det_depth=0, det_colour=0, err_sum=0.0, err_n=0,
                          est_err_sum=0.0, est_verr_sum=0.0, est_n=0)

    def _world(self, data, cam_id, p_cam):
        R = data.cam_xmat[cam_id].reshape(3, 3)  # camera pose = forward kinematics (encoders)
        return data.cam_xpos[cam_id] + R @ p_cam

    def observe(self, data, true_ball_pos=None, true_ball_vel=None):
        self.stats["frames"] += 1
        z, src = None, None
        if self.mode != "colour":  # PRIMARY: depth stream, geometric segmentation
            self.dstream_r.update_scene(data, camera=self.depth_cam_id, scene_option=self.opt_full)
            depth = self.dstream_r.render()
            self.dself_r.update_scene(data, camera=self.depth_cam_id, scene_option=self.opt_self)
            depth_self = self.dself_r.render()
            p_cam, dmask = self.det_depth.detect_depth(depth, depth_self)
            self.last_depth_mask = dmask
            if p_cam is not None:
                z, src = self._world(data, self.depth_cam_id, p_cam), "depth"
        if self.mode != "depth":
            self.rgb_r.update_scene(data, camera=self.cam_id, scene_option=self.opt_full)
            rgb = self.rgb_r.render()
            self.last_rgb = rgb
            if z is None:  # BACKUP: colour segmentation (+ aligned depth or monocular range)
                depth_c = None
                if self.use_depth:
                    self.dep_r.update_scene(data, camera=self.cam_id, scene_option=self.opt_full)
                    depth_c = self.dep_r.render()
                p_cam, mask = self.det.detect(rgb, depth_c)
                self.last_mask = mask
                if p_cam is not None:
                    z, src = self._world(data, self.cam_id, p_cam), "colour"
            else:
                self.last_mask = np.zeros(rgb.shape[:2], bool)
        self.last_source = src
        if z is not None:
            self.stats["detections"] += 1
            self.stats["det_depth" if src == "depth" else "det_colour"] += 1
            if true_ball_pos is not None:
                self.stats["err_sum"] += float(np.linalg.norm(z - true_ball_pos))
                self.stats["err_n"] += 1
        self.kf.update(z)
        if true_ball_pos is not None and self.kf.x is not None:  # tracker output error (what the policy sees)
            self.stats["est_err_sum"] += float(np.linalg.norm(self.kf.pos - true_ball_pos))
            if true_ball_vel is not None:
                self.stats["est_verr_sum"] += float(np.linalg.norm(self.kf.vel - true_ball_vel))
            self.stats["est_n"] += 1
        return self.kf.pos, self.kf.vel, z is not None

    def summary(self):
        s = self.stats
        return {"detection_rate": s["detections"] / max(s["frames"], 1),
                "detection_rate_depth": s["det_depth"] / max(s["frames"], 1),
                "detection_rate_colour": s["det_colour"] / max(s["frames"], 1),
                "mean_detection_error_m": s["err_sum"] / max(s["err_n"], 1),
                "mean_estimate_error_m": s["est_err_sum"] / max(s["est_n"], 1),
                "mean_estimate_vel_error_mps": s["est_verr_sum"] / max(s["est_n"], 1)}
