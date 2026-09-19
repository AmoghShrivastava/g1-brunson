"""Unitree G1 (29 DoF, rubber hands) stationary between-the-legs crossover dribble, MJX environment.

Task (from coaching sources cited in README.md): athletic stance with one foot forward, knees bent,
back straight, feet planted; the ball is pushed by one hand so that it bounces ONCE on the floor
between the feet ("the gate") and comes up into the other hand, which pushes it back through the gate
-- continuous alternating between-the-legs crossovers with the ball kept low ("no higher than the
knees") and under control.

Every reward term is tied either to that description, to a physical failure mode (falling, losing
the ball, hitting the ball with a leg), or to the standard sim-to-real regularisers used by
Unitree/MuJoCo-Playground for G1 hardware policies (action rate, torque, joint acceleration, joint
limits, feet slip).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
import numpy as np

from mujoco_playground._src import mjx_env

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(HERE, "assets", "g1_dribble_scene.xml")

JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# Action scale [rad] per joint.  Legs/waist: Unitree's hardware rule 0.25 * torque_limit / kp
# (deploy.yaml): hip pitch/yaw 0.55, hip roll/knee 0.35, ankles 0.44, waist yaw 0.55, waist r/p 0.44.
# Arms: the same rule gives 0.44 rad (25 Nm, kp 14.3) which cannot reach from a hanging arm down to a
# knee-high ball, so shoulder/elbow/wrist-roll use 0.8 rad (a full-scale action commands 11.4 Nm =
# 46 % of the 25 Nm limit).  Wrist pitch/yaw (4010 motor, 5 Nm, kp 16.8) use 0.25 rad (4.2 Nm = 84 %).
ACTION_SCALE = np.array(
    [0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
     0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
     0.55, 0.44, 0.44,
     0.8, 0.8, 0.8, 0.8, 0.8, 0.25, 0.25,
     0.8, 0.8, 0.8, 0.8, 0.8, 0.25, 0.25], dtype=np.float32)

BALL_R = 0.11926


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,  # 20 s
        action_repeat=1,
        soft_joint_pos_limit_factor=0.95,
        impl="jax",
        naconmax=6 * 8192,
        njmax=41 * 2 + 64,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.01, joint_vel=1.5, gravity=0.05, gyro=0.2,  # Unitree mjlab / playground
                ball_pos=0.015, ball_vel=0.15,  # ball tracker noise (m, m/s)
            ),
            ball_delay_max=2,  # control steps (0..40 ms tracker latency)
            action_delay_max=1,  # control steps (0..20 ms)
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Task (events, not scaled by dt).
                crossing=3.0,
                touch=1.0,  # first hand contact after a floor bounce (a dribble)
                catch=2.0,  # the receiving hand plays the ball after a crossing (the catch)
                chain=2.0,  # a crossing that follows a catch: consecutive between-the-legs dribbles
                # (a lost ball must cost more than one crossing earns, otherwise "push it through
                # and let it roll away" is a paying strategy: crossings == respawns was observed)
                bad_bounce=-1.0,  # cancels the touch reward: pound dribbling alone earns nothing
                wrong_hand=-3.0,  # alternation is part of the task: reaching across must not pay
                free_bounce=-0.3,
                ball_reset=-5.0,
                recovery=2.0,  # first hand contact after the ball strayed past the control radius
                hold=2.0,  # dense (x dt): two-hand hold with the ball up during the hold phase
                hold_timeout=-1.0,  # dense (x dt): still holding after hold_max_steps
                alive=1.0,
                # Task shaping (dense, scaled by dt).
                ball_to_goal=1.0,
                hand_to_ball=5.0,
                ball_low=-2.0,
                apex_low=-12.0,  # at each bounce: (apex_target - apex of the last flight)+ in metres
                ball_near=-1.0,
                leg_touch=-2.0,
                trunk_touch=-2.0,
                carry=-1.0,
                # Posture (coaching) and stationarity.
                pelvis_height=-4.0,
                torso_orientation=-2.0,
                feet_contact=-4.0,
                feet_slip=-1.0,
                feet_hold=-2.0,
                pose=-0.05,
                # Sim2real regularisers.
                action_rate=-0.02,
                torques=-1e-4,
                dof_acc=-1e-7,
                dof_vel=-1e-4,
                dof_pos_limits=-2.0,
                ang_vel_xy=-0.05,
                lin_vel_z=-0.5,
                termination=-50.0,
            ),
            pelvis_height_range=(0.60, 0.76),  # crouch band (standing 0.79)
            torso_pitch_range=(0.0, 0.45),  # forward lean of torso z-axis, rad
            ball_max_center_z=0.65,  # keep the dribble low (centre; top of ball ~0.77 m = crouched hip)
            min_crossing_apex_z=0.30,  # a crossing only counts if the ball rose to >= 0.30 m before it
            apex_target_z=0.40,  # ball centre at the top of each flight >= 0.40 m (top of ball ~0.52 m,
                                 # G1 mid-thigh): a visibly clean dribble, still "below the knees"
                                 # of a human-scale player
            control_radius=0.55,  # ball may not drift beyond this (xy from pelvis)
            carry_steps=8,  # continuous hand contact beyond 0.16 s = carrying
        ),
        term_config=config_dict.create(
            min_pelvis_z=0.45,
            min_torso_up=0.5,
            ball_lost_radius=1.0,
            ball_max_z=1.3,
            dead_ball_steps=25,  # ball resting/rolling on floor for 0.5 s
            max_free_bounces=2,
        ),
        dr_config=config_dict.create(
            enable=True,
            push_interval_range=[4.0, 8.0],
            push_magnitude_range=[0.05, 0.4],
        ),
        perturb_config=config_dict.create(
            # Recoverability training: random velocity kicks on the BALL (a mis-hit, a defender's
            # tap, a bad bounce off a seam).  The policy must reach out and regain the dribble; the
            # first hand contact after the ball has strayed beyond the control radius pays `recovery`.
            enable=False,
            interval_range=[3.0, 6.0],  # s between kicks
            speed_range=[0.4, 1.6],  # m/s horizontal kick, random direction
            up_range=[0.0, 1.0],  # m/s added vertical component
        ),
        vision_config=config_dict.create(
            # Emulates the head-camera perception (vision.py) inside MJX training: the actor only gets
            # a ball estimate that is (a) the true ball + noise while the ball is inside the D435
            # frustum and detected, (b) a ballistic-with-bounce prediction (like the Kalman tracker)
            # while it is out of view / dropped.  Ground truth is still used for rewards/critic.
            enable=False,
            pos_noise=0.03,  # m, matches the measured 3.8 cm detection error
            vel_noise=0.3,  # m/s
            dropout=0.15,  # per-frame miss probability inside the frustum (occlusion by legs/hands)
            min_range=0.15,  # m, D435 minimum depth
            tan_half_h=0.692,  # tan(69.4/2 deg) horizontal half-FOV
            tan_half_v=0.389,  # tan(42.5/2 deg) vertical half-FOV
            cor=0.77,  # bounce model of the predictor
            # Kalman tracker (same constants as vision.BallKalman): the actor sees the FILTER output,
            # including its lag after hand impacts and its drift while the ball is unseen.
            kalman=True,
            r_meas=0.008,  # m, measurement std
            q_pos=0.01,  # m per step
            q_vel=1.5,  # m/s per step
        ),
        reset_config=config_dict.create(
            joint_noise=0.03,
            ball_release_prob=0.3,
            # "Hold" start: the ball rests in both hands at the chest (cradle arm pose); the robot must
            # bring it down, pound it once and then start the between-the-legs sequence.
            hold_prob=0.0,
            hold_min_steps=25,  # 0.5 s of two-hand hold rewarded
            hold_max_steps=75,  # after 1.5 s a hold costs (must release)
            blend_steps=50,  # arm PD set-point blends from the cradle pose back to the stance over 1 s  # else (70 %): ball dropped beside the pushing hand (pound-dribble start)
        ),
    )


class G1Dribble(mjx_env.MjxEnv):
    """Between-the-legs crossover dribble for the Unitree G1."""

    def __init__(self, config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
                 xml_path: str = SCENE_XML):
        super().__init__(config, config_overrides)
        self._xml_path = xml_path
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    # ------------------------------------------------------------------------------------------
    def _post_init(self):
        m = self._mj_model
        assert m.nu == 29
        for i, n in enumerate(JOINT_NAMES):
            assert m.actuator(i).name == n, (i, n, m.actuator(i).name)
        self._action_scale = jp.array(ACTION_SCALE)
        cradle_path = os.path.join(os.path.dirname(self._xml_path), "cradle_pose.json")
        with open(cradle_path) as f:
            cradle = json.load(f)
        self._cradle_arm = jp.array(cradle["arm_qpos"])  # 14 arm joints (qpos[22:36])
        self._cradle_ball = jp.array(cradle["ball_offset_pelvis"])  # ball centre offset in the pelvis yaw frame
        keys = {}
        for side in ("left", "right"):
            k = m.keyframe(f"stance_{side}_front")
            keys[side] = (jp.array(k.qpos), jp.array(k.ctrl))
        self._stance_qpos = jp.stack([keys["left"][0], keys["right"][0]])  # index 0 = left front
        self._stance_ctrl = jp.stack([keys["left"][1], keys["right"][1]])
        self._nq_robot = 36
        self._ball_qadr = m.jnt_qposadr[m.joint("ball_freejoint").id]
        self._ball_dofadr = m.jnt_dofadr[m.joint("ball_freejoint").id]
        assert self._ball_qadr == 36 and self._ball_dofadr == 35

        lowers, uppers = m.jnt_range[1:30].T
        c = (lowers + uppers) / 2
        r = uppers - lowers
        self._soft_lowers = jp.array(c - 0.5 * r * self._config.soft_joint_pos_limit_factor)
        self._soft_uppers = jp.array(c + 0.5 * r * self._config.soft_joint_pos_limit_factor)

        self._pelvis_imu_site = m.site("imu_in_pelvis").id
        self._torso_imu_site = m.site("imu_in_torso").id
        self._feet_site = jp.array([m.site("left_foot").id, m.site("right_foot").id])
        self._palm_site = jp.array([m.site("left_palm").id, m.site("right_palm").id])
        self._hand_geom = jp.array([m.geom("left_hand_collision").id, m.geom("right_hand_collision").id])
        self._hand_touch_dist = BALL_R + 0.035 + 0.05  # ball surface within 5 cm of the hand capsule
        self._ball_body = m.body("ball").id
        self._torso_body = m.body("torso_link").id
        self._pelvis_body = m.body("pelvis").id

        def sadr(name):
            s = m.sensor(name)
            return int(m.sensor_adr[s.id]), int(m.sensor_dim[s.id])

        self._sens = {n: sadr(n) for n in [
            "gyro_pelvis", "upvector_pelvis", "global_linvel_pelvis", "global_angvel_pelvis",
            "local_linvel_pelvis", "upvector_torso", "global_angvel_torso", "global_linvel_torso",
            "ball_pos", "ball_linvel", "ball_angvel", "left_palm_pos", "right_palm_pos",
            "left_palm_linvel", "right_palm_linvel", "left_foot_pos", "right_foot_pos",
            "left_foot_global_linvel", "right_foot_global_linvel", "pelvis_pos", "pelvis_quat",
        ]}
        # Contact "found" sensors, grouped.  Order of the aggregated flag vector:
        # 0 ball-floor, 1 ball-left hand, 2 ball-right hand, 3 ball-left arm, 4 ball-right arm,
        # 5 ball-left leg, 6 ball-right leg, 7 ball-trunk (pelvis/torso/head), 8 left foot-floor,
        # 9 right foot-floor.
        def cs(name):
            return int(m.sensor_adr[m.sensor(name).id])
        groups = {
            0: ["ball_floor_found"],
            1: ["ball_left_hand_collision_found"],
            2: ["ball_right_hand_collision_found"],
            3: [f"ball_left_{g}_found" for g in ("wrist_collision", "elbow_yaw_collision", "shoulder_yaw_collision")],
            4: [f"ball_right_{g}_found" for g in ("wrist_collision", "elbow_yaw_collision", "shoulder_yaw_collision")],
            5: [f"ball_left_{g}_found" for g in ("thigh_collision", "shin_collision", "linkage_brace_collision", "hip_collision", "foot")],
            6: [f"ball_right_{g}_found" for g in ("thigh_collision", "shin_collision", "linkage_brace_collision", "hip_collision", "foot")],
            7: ["ball_pelvis_collision_found", "ball_torso_collision_found", "ball_head_collision_found"],
            8: ["left_foot_floor_found"],
            9: ["right_foot_floor_found"],
        }
        adrs, gid = [], []
        for k, names in groups.items():
            for n in names:
                adrs.append(cs(n))
                gid.append(k)
        self._found_adr = jp.array(adrs)
        self._found_group = jp.array(gid)
        self._n_groups = 10
        # Joint groups for the pose regulariser (legs+waist track the stance; arms free).
        w = np.ones(29, dtype=np.float32)
        w[15:] = 0.0
        self._pose_weights = jp.array(w)

    def _s(self, data, name):
        a, d = self._sens[name]
        return data.sensordata[a:a + d]

    # ------------------------------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        cfg = self._config
        rng, k_st, k_jn, k_bs, k_bp, k_bv, k_dl, k_ad, k_pi, k_mode, k_yaw = jax.random.split(rng, 11)
        stance = jax.random.bernoulli(k_st, 0.5).astype(jp.int32)  # 0 left-front, 1 right-front
        qpos = self._stance_qpos[stance]
        ctrl0 = self._stance_ctrl[stance]
        hold = jax.random.bernoulli(jax.random.fold_in(k_mode, 7), cfg.reset_config.hold_prob)
        qpos = jp.where(hold, qpos.at[22:36].set(self._cradle_arm), qpos)
        qpos = qpos.at[7:36].add(jax.random.uniform(k_jn, (29,), minval=-cfg.reset_config.joint_noise,
                                                    maxval=cfg.reset_config.joint_noise))
        yaw = jax.random.uniform(k_yaw, (), minval=-jp.pi, maxval=jp.pi)
        quat = mjx_math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        qpos = qpos.at[3:7].set(mjx_math.quat_mul(qpos[3:7], quat))
        qvel = jp.zeros(self.mjx_model.nv)
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl0, impl=self.mjx_model.impl.value,
                                 naconmax=cfg.naconmax, njmax=cfg.njmax)
        data = mjx.forward(self.mjx_model, data)

        # First hand = hand on the back-foot side (left-front stance -> right hand).  0 = left, 1 = right.
        next_hand = 1 - stance
        palm = data.geom_xpos[self._hand_geom[next_hand]]  # centre of the hand capsule
        pelvis = data.site_xpos[self._pelvis_imu_site]
        pelvis_quat = self._s(data, "pelvis_quat")
        yaw_mat = self._yaw_matrix(pelvis_quat)
        feet = data.site_xpos[self._feet_site]
        gate = 0.5 * (feet[0] + feet[1])
        release = jax.random.bernoulli(k_mode, cfg.reset_config.ball_release_prob)
        # Mode A: ball just released below the hand, pushed so that it lands on the gate centre
        # after t_f seconds (ballistic), i.e. the first half of a between-the-legs dribble.
        side = jp.where(next_hand == 1, -1.0, 1.0)  # right hand -> robot's -y side
        pos_a, vel_a = self._release_spawn(pelvis, yaw_mat, gate, side, k_bp, k_bv)
        # Mode B: ball dropped from ~0.5 m beside the first hand, outside the legs (pound dribble start).
        pos_b, vel_b = self._drop_spawn(pelvis, yaw_mat, side, k_bp)
        ball_pos = jp.where(release, pos_a, pos_b)
        ball_vel = jp.where(release, vel_a, vel_b)
        # Mode C (hold): ball resting between the two palms at the chest, at rest.
        ball_pos = jp.where(hold, pelvis + yaw_mat @ self._cradle_ball, ball_pos)
        ball_vel = jp.where(hold, jp.zeros(3), ball_vel)
        # A spawned ball never scores by itself: a hand must play it first (expect = 0).
        expect = jp.zeros((), jp.int32)
        qpos = qpos.at[36:39].set(ball_pos)
        qpos = qpos.at[39:43].set(jp.array([1.0, 0.0, 0.0, 0.0]))
        qvel = qvel.at[35:38].set(ball_vel)
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl0, impl=self.mjx_model.impl.value,
                                 naconmax=cfg.naconmax, njmax=cfg.njmax)
        data = mjx.forward(self.mjx_model, data)

        ball_delay = jax.random.randint(k_dl, (), 0, cfg.noise_config.ball_delay_max + 1)
        action_delay = jax.random.randint(k_ad, (), 0, cfg.noise_config.action_delay_max + 1)
        push_interval = jax.random.uniform(k_pi, (), minval=cfg.dr_config.push_interval_range[0],
                                           maxval=cfg.dr_config.push_interval_range[1])
        kick_interval = jax.random.uniform(jax.random.fold_in(k_pi, 1), (), minval=cfg.perturb_config.interval_range[0],
                                           maxval=cfg.perturb_config.interval_range[1])
        ball_rel = self._ball_rel(data, yaw_mat)  # (6,)
        info = {
            "rng": rng,
            "step": 0,
            "stance": stance,
            "default_ctrl": ctrl0,
            "stance_qpos": self._stance_qpos[stance][7:36],
            "feet_init_xy": feet[:, :2],
            "next_hand": next_hand,
            "expect": expect,
            "free_bounces": jp.zeros((), jp.int32),
            "dead_steps": jp.zeros((), jp.int32),
            "hand_contact_steps": jp.zeros((), jp.int32),
            "since_bounce": jp.zeros(()),
            "floor_contact_prev": jp.zeros((), bool),
            "crossings": jp.zeros(()),
            "last_act": jp.zeros(29),
            "last_last_act": jp.zeros(29),
            "act_buffer": jp.zeros((2, 29)),
            "action_delay": action_delay,
            "ball_est_w": jp.concatenate([self._s(data, "ball_pos"), self._s(data, "ball_linvel")]),  # vision estimate (world)
            "ball_est_P": jp.diag(jp.array([0.008 ** 2] * 3 + [4.0] * 3)),  # Kalman covariance of the estimate
            "ball_buffer": jp.tile(ball_rel[None], (3, 1)),  # ring buffer of recent ball obs (delay)
            "ball_delay": ball_delay,
            "ball_hist": jp.tile(ball_rel[None], (3, 1)),  # observed history for the actor
            "push_interval_steps": jp.round(push_interval / self.dt).astype(jp.int32),
            "push_step": 0,
            "kick_interval_steps": jp.round(kick_interval / self.dt).astype(jp.int32),
            "hold_phase": hold,  # ball has not bounced yet since the hold start
            "hold_steps": jp.zeros((), jp.int32),
            "since_release": jp.full((), 10_000, jp.int32),  # steps since the hold ended (large = never held)
            "ctrl0": ctrl0,
            "hand_contact": jp.zeros((2,), bool),
            "stray": jp.zeros((), bool),  # ball has left the control radius since the last hand contact
            "motor_targets": ctrl0,
            "respawn": jp.zeros((), bool),
            "awaiting_catch": jp.zeros((), bool),
            "flight_apex": jp.zeros(()),
            "chained": jp.zeros((), bool),
            "yaw_mat": yaw_mat,
            "resets": jp.zeros(()),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in cfg.reward_config.scales.keys()}
        metrics["task/crossings"] = jp.zeros(())
        metrics["task/ball_resets"] = jp.zeros(())
        metrics["task/recoveries"] = jp.zeros(())
        metrics["task/hold_releases"] = jp.zeros(())
        metrics["task/hold_ok_steps"] = jp.zeros(())
        metrics["task/kicks"] = jp.zeros(())
        metrics["task/touches"] = jp.zeros(())
        metrics["task/catches"] = jp.zeros(())
        metrics["task/chains"] = jp.zeros(())
        metrics["task/bad_bounces"] = jp.zeros(())
        metrics["task/wrong_hand"] = jp.zeros(())
        metrics["task/ball_z_mean"] = jp.zeros(())
        metrics["task/apex_deficit"] = jp.zeros(())
        metrics["task/pelvis_z"] = jp.zeros(())
        metrics["term/fall"] = jp.zeros(())
        metrics["term/ball_lost"] = jp.zeros(())
        metrics["term/dead_ball"] = jp.zeros(())
        metrics["term/free_bounce"] = jp.zeros(())
        found = self._found(data)
        obs = self._get_obs(data, info, found, ball_rel)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ------------------------------------------------------------------------------------------
    def _release_spawn(self, pelvis, yaw_mat, gate, side, k_bp, k_bv):
        """Ball in front and to the side of the pushing hand at knee height, clear of the legs, about
        to travel diagonally back through the gap between the staggered feet (the between-the-legs
        path: front-right -> under the pelvis -> back-left for a right-hand push)."""
        pos = pelvis + yaw_mat @ jp.array([0.32, 0.30 * side, 0.0])
        pos = pos.at[2].set(0.38) + jax.random.uniform(k_bp, (3,), minval=-0.03, maxval=0.03)
        t_f = jax.random.uniform(k_bv, (), minval=0.15, maxval=0.28)
        h = pos[2] - BALL_R
        v_xy = (gate[:2] - pos[:2]) / t_f
        v_z = jp.minimum(-(h / t_f - 0.5 * 9.81 * t_f), 0.0)
        return pos, jp.concatenate([v_xy, v_z[None]])

    def _drop_spawn(self, pelvis, yaw_mat, side, k_bp):
        """Ball dropped from hand height beside and slightly in front of the pushing hand; it keeps
        bouncing within reach (apex 0.45 -> 0.34 -> 0.26 m) so the hand can start a pound dribble."""
        pos = pelvis + yaw_mat @ jp.array([0.22, 0.30 * side, 0.0])
        pos = pos.at[2].set(0.50) + jax.random.uniform(k_bp, (3,), minval=-0.04, maxval=0.04)
        return pos, jp.zeros(3)

    def _found(self, data):
        raw = (data.sensordata[self._found_adr] > 0).astype(jp.float32)
        grouped = jax.ops.segment_sum(raw, self._found_group, num_segments=self._n_groups)
        return grouped > 0

    def _yaw_matrix(self, quat):
        # rotation matrix of the yaw-only part of a (w,x,y,z) quaternion
        w, x, y, z = quat
        yaw = jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = jp.cos(yaw), jp.sin(yaw)
        return jp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _ball_rel(self, data, yaw_mat):
        pelvis = data.site_xpos[self._pelvis_imu_site]
        pelvis_v = self._s(data, "global_linvel_pelvis")
        bp = self._s(data, "ball_pos")
        bv = self._s(data, "ball_linvel")
        rel_p = yaw_mat.T @ (bp - pelvis)
        rel_v = yaw_mat.T @ (bv - pelvis_v)
        return jp.concatenate([rel_p, rel_v])

    # Head camera (Intel RealSense D435i) pose in torso_link, from Unitree's g1_29dof_rev_1_0.urdf d435_joint.
    _CAM_POS = jp.array([0.0576235, 0.01753, 0.42987])
    _CAM_PITCH = 0.8307767239493009

    def _ball_visible(self, data, key):
        """Frustum test of the ball centre against the head camera + random detection dropout."""
        vc = self._config.vision_config
        R = data.xmat[self._torso_body].reshape(3, 3)
        cam_w = data.xpos[self._torso_body] + R @ self._CAM_POS
        d = R.T @ (self._s(data, "ball_pos") - cam_w)  # torso frame
        fwd = jp.array([jp.cos(self._CAM_PITCH), 0.0, -jp.sin(self._CAM_PITCH)])
        right = jp.array([0.0, -1.0, 0.0])
        up = jp.cross(-fwd, right)
        depth = d @ fwd
        inside = (depth > vc.min_range) & (jp.abs(d @ right) < vc.tan_half_h * depth) & (jp.abs(d @ up) < vc.tan_half_v * depth)
        return inside & (jax.random.uniform(key) > vc.dropout)

    def _vision_estimate(self, data, info, yaw_mat, respawned):
        """Ball observation for the actor.  With vision_config.enable: emulated camera perception
        (seen -> noisy measurement, unseen -> ballistic prediction with a floor bounce, like the Kalman
        tracker in vision.py); otherwise the true ball state (tracker noise/latency are added in _get_obs)."""
        vc = self._config.vision_config
        rel_true = self._ball_rel(data, yaw_mat)
        if not vc.enable:
            return rel_true
        info["rng"], k_vis, k_p, k_v = jax.random.split(info["rng"], 4)
        bp = self._s(data, "ball_pos")
        bv = self._s(data, "ball_linvel")
        est = info["ball_est_w"]
        # predict (unseen): constant velocity + gravity + bounce
        p = est[:3] + est[3:] * self.dt
        v = est[3:] + jp.array([0.0, 0.0, -9.81 * self.dt])
        bounce = (p[2] < BALL_R) & (v[2] < 0)
        p = jp.where(bounce, p.at[2].set(2 * BALL_R - p[2]), p)
        v = jp.where(bounce, jp.array([0.9 * v[0], 0.9 * v[1], -vc.cor * v[2]]), v)
        pred = jp.concatenate([p, v])
        seen = self._ball_visible(data, k_vis) | respawned  # a respawned ball is a fresh, seen ball
        if vc.kalman:
            # Exact replica of vision.BallKalman: position-only measurements, velocity inferred by the
            # filter (this is where the lag after every hand impact comes from).
            Fm = jp.eye(6).at[0, 3].set(self.dt).at[1, 4].set(self.dt).at[2, 5].set(self.dt)
            Q = jp.diag(jp.array([vc.q_pos ** 2] * 3 + [vc.q_vel ** 2] * 3))
            P = Fm @ info["ball_est_P"] @ Fm.T + Q
            z = bp + jax.random.normal(k_p, (3,)) * vc.r_meas
            H = jp.zeros((3, 6)).at[0, 0].set(1.0).at[1, 1].set(1.0).at[2, 2].set(1.0)
            S = H @ P @ H.T + jp.eye(3) * vc.r_meas ** 2
            K = P @ H.T @ jp.linalg.inv(S)
            upd = pred + K @ (z - H @ pred)
            P_upd = (jp.eye(6) - K @ H) @ P
            # a respawned ball re-initialises the filter at the new ball (as the real tracker would after
            # a long miss); otherwise seen -> update, unseen -> prediction only
            fresh = jp.concatenate([bp, jp.zeros(3)])
            est = jp.where(respawned, fresh, jp.where(seen, upd, pred))
            info["ball_est_P"] = jp.where(respawned, jp.diag(jp.array([vc.r_meas ** 2] * 3 + [4.0] * 3)),
                                          jp.where(seen, P_upd, P))
        else:
            meas = jp.concatenate([bp + (2 * jax.random.uniform(k_p, (3,)) - 1) * vc.pos_noise,
                                   bv + (2 * jax.random.uniform(k_v, (3,)) - 1) * vc.vel_noise])
            est = jp.where(seen, meas, pred)
        info["ball_est_w"] = est
        pelvis = data.site_xpos[self._pelvis_imu_site]
        pelvis_v = self._s(data, "global_linvel_pelvis")
        return jp.concatenate([yaw_mat.T @ (est[:3] - pelvis), yaw_mat.T @ (est[3:] - pelvis_v)])

    def _physics_step(self, data, motor_targets):
        """Substep the physics, OR-ing the contact flags and tracking the ball over all substeps."""
        def f(d, _):
            d = d.replace(ctrl=motor_targets)
            d = mjx.step(self.mjx_model, d)
            found = self._found(d)
            bp = self._s(d, "ball_pos")
            bv = self._s(d, "ball_linvel")
            return d, (found, bp, bv)
        data, (founds, bps, bvs) = jax.lax.scan(f, data, None, self.n_substeps)
        return data, founds, bps, bvs

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config
        info = state.info
        info["rng"], k_push1, k_push2 = jax.random.split(info["rng"], 3)

        # Perturbation pushes on the pelvis (sim2real robustness, playground-style).
        push_theta = jax.random.uniform(k_push1, maxval=2 * jp.pi)
        push_mag = jax.random.uniform(k_push2, minval=cfg.dr_config.push_magnitude_range[0],
                                      maxval=cfg.dr_config.push_magnitude_range[1])
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)]) * push_mag
        push *= (jp.mod(info["push_step"] + 1, info["push_interval_steps"]) == 0) * cfg.dr_config.enable
        qvel = state.data.qvel.at[:2].add(push)
        # Ball kicks (recoverability): every kick_interval a random horizontal (+ upward) velocity is
        # added to the ball; offset by half an interval from the pelvis pushes.
        info["rng"], k_k1, k_k2, k_k3 = jax.random.split(info["rng"], 4)
        pc = cfg.perturb_config
        kick_theta = jax.random.uniform(k_k1, maxval=2 * jp.pi)
        kick_mag = jax.random.uniform(k_k2, minval=pc.speed_range[0], maxval=pc.speed_range[1])
        kick_up = jax.random.uniform(k_k3, minval=pc.up_range[0], maxval=pc.up_range[1])
        kick = jp.array([jp.cos(kick_theta) * kick_mag, jp.sin(kick_theta) * kick_mag, kick_up])
        kick_now = (jp.mod(info["push_step"] + 1 + info["kick_interval_steps"] // 2, info["kick_interval_steps"]) == 0) & pc.enable
        kick *= kick_now
        qvel = qvel.at[35:38].add(kick)
        data = state.data.replace(qvel=qvel)

        # Ball respawn (ball was lost / dead / bounced twice uncontrolled in the previous step): put
        # a new ball in the release position of the back-foot-side hand.  Cheaper than terminating,
        # and gives many dribble attempts per episode.
        info["rng"], k_bp, k_bv = jax.random.split(info["rng"], 3)
        feet0 = data.site_xpos[self._feet_site]
        gate0 = 0.5 * (feet0[0] + feet0[1])
        pelvis0 = data.site_xpos[self._pelvis_imu_site]
        hand0 = 1 - info["stance"]
        side0 = jp.where(hand0 == 1, -1.0, 1.0)
        info["rng"], k_mode = jax.random.split(info["rng"])
        rp_a, rv_a = self._release_spawn(pelvis0, info["yaw_mat"], gate0, side0, k_bp, k_bv)
        rp_b, rv_b = self._drop_spawn(pelvis0, info["yaw_mat"], side0, k_bp)
        use_a = jax.random.bernoulli(k_mode, cfg.reset_config.ball_release_prob)
        rp = jp.where(use_a, rp_a, rp_b)
        rv = jp.where(use_a, rv_a, rv_b)
        do_rs = info["respawn"]
        qpos = jp.where(do_rs, data.qpos.at[36:39].set(rp).at[39:43].set(jp.array([1.0, 0, 0, 0])), data.qpos)
        qvel = jp.where(do_rs, data.qvel.at[35:38].set(rv).at[38:41].set(0.0), data.qvel)
        data = data.replace(qpos=qpos, qvel=qvel)
        info["next_hand"] = jp.where(do_rs, hand0, info["next_hand"])
        info["expect"] = jp.where(do_rs, 0, info["expect"])  # hand must play the new ball first
        info["free_bounces"] = jp.where(do_rs, 0, info["free_bounces"])
        info["dead_steps"] = jp.where(do_rs, 0, info["dead_steps"])
        info["hand_contact_steps"] = jp.where(do_rs, 0, info["hand_contact_steps"])
        info["floor_contact_prev"] = jp.where(do_rs, False, info["floor_contact_prev"])
        info["since_bounce"] = jp.where(do_rs, 0.0, info["since_bounce"])
        info["awaiting_catch"] = jp.where(do_rs, False, info["awaiting_catch"])
        info["chained"] = jp.where(do_rs, False, info["chained"])

        # Action delay (0..1 control steps).
        act_buffer = jp.concatenate([action[None], info["act_buffer"][:-1]], axis=0)
        info["act_buffer"] = act_buffer
        applied = act_buffer[info["action_delay"]]
        # Hold start: the arm PD set-point is the cradle pose during the hold and blends back to the
        # stance over blend_steps after the release (the action range alone cannot reach the cradle).
        blend = jp.where(info["hold_phase"], 1.0,
                         jp.clip(1.0 - info["since_release"] / cfg.reset_config.blend_steps, 0.0, 1.0))
        arm_delta = (self._cradle_arm - info["ctrl0"][15:29]) * blend
        info["default_ctrl"] = info["ctrl0"].at[15:29].add(arm_delta)
        motor_targets = info["default_ctrl"] + applied * self._action_scale
        info["motor_targets"] = motor_targets
        data, founds, bps, bvs = self._physics_step(data, motor_targets)

        # ---- task state machine ----
        # Uses only quantities a ball tracker + kinematics provide (identical logic in policy.py's
        # DribbleTracker): floor bounce = ball centre within 2 cm of the floor after being airborne;
        # hand touch = palm within 6 cm of the ball surface.  Contact sensors are used for costs only.
        found_any = founds.any(axis=0)
        lhand_sens, rhand_sens = found_any[1], found_any[2]
        larm_c, rarm_c = found_any[3], found_any[4]
        lleg_c, rleg_c = found_any[5], found_any[6]
        trunk_c = found_any[7]
        feet_c = found_any[8:10]
        on_floor_sub = bps[:, 2] < BALL_R + 0.02
        floor_c = on_floor_sub.any()
        # Hand "touch" = physical hand-ball contact during this control step (OR over substeps).
        # (A proximity criterion was exploitable: the policy hovered the hand next to a ball that
        # bounced on its own.)  On hardware this is a ball-velocity discontinuity at the hand.
        hand_c = jp.stack([lhand_sens, rhand_sens])
        lhand_c, rhand_c = hand_c[0], hand_c[1]
        info["hand_contact"] = hand_c  # exposed for the onboard cue tracker (OR over the 10 physics substeps)
        next_hand = info["next_hand"]
        expect = info["expect"]
        correct_hand = hand_c[next_hand]
        other_hand = hand_c[1 - next_hand]

        # Gate geometry from the feet (world xy).
        feet = data.site_xpos[self._feet_site]
        gate_c = 0.5 * (feet[0, :2] + feet[1, :2])
        u = feet[0, :2] - feet[1, :2]  # left foot - right foot
        half_len = 0.5 * jp.linalg.norm(u) + 1e-6
        u = u / (2 * half_len)
        n_left = jp.array([-u[1], u[0]])  # perpendicular; sign fixed below using pelvis frame
        yaw_mat = self._yaw_matrix(self._s(data, "pelvis_quat"))
        body_left = yaw_mat[:2, 1]
        n_left = n_left * jp.sign(jp.dot(n_left, body_left) + 1e-9)
        # Bounce location: ball xy at the lowest substep.
        contact_idx = jp.argmin(bps[:, 2])
        bounce_xy = bps[contact_idx, :2]
        bounce_v = bvs[contact_idx, :2]
        along = jp.dot(bounce_xy - gate_c, u)
        perp = jp.dot(bounce_xy - gate_c, n_left)
        in_gate = (jp.abs(along) < 0.8 * half_len) & (jp.abs(perp) < 0.15)
        # Direction: right hand pushes towards the robot's left (+n_left), left hand towards -n_left.
        want_dir = jp.where(next_hand == 1, 1.0, -1.0)
        dir_ok = want_dir * jp.dot(bounce_v, n_left) > 0.05

        # A bounce = floor contact after the ball was airborne (no floor contact in the previous
        # control step).  A rolling/resting ball therefore counts once, then the dead-ball timer runs.
        bounce = floor_c & ~info["floor_contact_prev"]
        apex_ok = info["flight_apex"] >= cfg.reward_config.min_crossing_apex_z
        holding = info["hold_phase"]
        crossing = bounce & (expect == 1) & in_gate & dir_ok & apex_ok & ~holding
        bad_bounce = bounce & (expect == 1) & ~(in_gate & dir_ok & apex_ok) & ~holding
        free_bounce = bounce & (expect == 0) & ~holding
        wrong_hand = other_hand & ~correct_hand & ~holding
        # The first bounce after a hold is the free "pound" dribble: it ends the hold phase; the hand
        # that last played the ball becomes the dribbling hand and must play it again after the bounce.
        hold_release = holding & bounce
        info["hold_phase"] = holding & ~bounce
        info["hold_steps"] = jp.where(holding, info["hold_steps"] + 1, info["hold_steps"])
        info["since_release"] = jp.where(hold_release, 0, info["since_release"] + 1)

        touch = correct_hand & (expect == 0) & ~holding  # first contact of the right hand after a bounce
        catch = touch & info["awaiting_catch"]  # receiving hand plays the ball after a crossing
        chain = crossing & info["chained"]  # crossing that follows a catch
        awaiting_catch = jp.where(crossing, True, jp.where(touch | wrong_hand | bad_bounce | free_bounce, False, info["awaiting_catch"]))
        chained = jp.where(catch, True, jp.where(crossing | bad_bounce | free_bounce | wrong_hand, False, info["chained"]))
        info["awaiting_catch"] = awaiting_catch
        info["chained"] = chained
        new_next_hand = jp.where(crossing, 1 - next_hand, next_hand)
        new_next_hand = jp.where(wrong_hand, 1 - next_hand, new_next_hand)  # that hand is now dribbling
        new_next_hand = jp.where(holding & rhand_c & ~lhand_c, 1, new_next_hand)  # hold: follow the pushing hand
        new_next_hand = jp.where(holding & lhand_c & ~rhand_c, 0, new_next_hand)
        new_expect = jp.where(crossing | bad_bounce | hold_release, 0, expect)
        new_expect = jp.where((correct_hand | wrong_hand) & ~hold_release, 1, new_expect)
        free_bounces = jp.where(free_bounce, info["free_bounces"] + 1, info["free_bounces"])
        free_bounces = jp.where(correct_hand | wrong_hand, 0, free_bounces)
        since_bounce = jp.where(bounce, 0.0, info["since_bounce"] + self.dt)
        hand_steps = jp.where(lhand_sens | rhand_sens, info["hand_contact_steps"] + 1, 0)

        # Dead ball: resting/rolling on the floor.
        ball_p = self._s(data, "ball_pos")
        ball_v = self._s(data, "ball_linvel")
        dead = (ball_p[2] < BALL_R + 0.03) & (jp.abs(ball_v[2]) < 0.4)
        dead_steps = jp.where(dead, info["dead_steps"] + 1, 0)

        # ---- termination ----
        pelvis_z = data.site_xpos[self._pelvis_imu_site][2]
        torso_up = self._s(data, "upvector_torso")[2]
        pelvis_up = self._s(data, "upvector_pelvis")[2]
        fall = (pelvis_z < cfg.term_config.min_pelvis_z) | (torso_up < cfg.term_config.min_torso_up) | (pelvis_up < cfg.term_config.min_torso_up)
        rel = self._ball_rel(data, yaw_mat)
        ball_dist_xy = jp.linalg.norm(rel[:2])
        ball_lost = (ball_dist_xy > cfg.term_config.ball_lost_radius) | (ball_p[2] > cfg.term_config.ball_max_z)
        dead_ball = dead_steps >= cfg.term_config.dead_ball_steps
        too_many_free = free_bounces >= cfg.term_config.max_free_bounces
        nan = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = fall | nan
        ball_reset = (ball_lost | dead_ball | too_many_free) & ~done
        # Recovery: the ball strayed beyond the control radius and a hand brought it back into play.
        strayed = ball_dist_xy > cfg.reward_config.control_radius
        any_hand = lhand_c | rhand_c
        recovery = info["stray"] & any_hand & ~ball_reset
        info["stray"] = jp.where(any_hand | ball_reset | do_rs, False, info["stray"] | strayed)
        info["respawn"] = ball_reset
        info["yaw_mat"] = yaw_mat
        info["resets"] = info["resets"] + ball_reset

        # ---- rewards ----
        palms = data.geom_xpos[self._hand_geom]
        palm_next = palms[new_next_hand]
        gate_floor = jp.concatenate([gate_c, jp.array([BALL_R])])
        goal = jp.where(new_expect == 1, gate_floor, palm_next)
        to_goal = goal - ball_p
        to_goal_dir = to_goal / (jp.linalg.norm(to_goal) + 1e-6)
        ball_to_goal = jp.clip(jp.dot(ball_v, to_goal_dir), -1.0, 1.5)
        # Hand-to-ball: only while waiting for the hand (expect 0), and only the correct hand.
        hand_dist = jp.linalg.norm(palm_next - ball_p) - BALL_R - 0.035
        hand_to_ball = jp.exp(-jp.square(jp.clip(hand_dist, 0.0, None)) / 0.04) * (new_expect == 0)
        ball_low = jp.clip(ball_p[2] - cfg.reward_config.ball_max_center_z, 0.0, None)
        # apex bookkeeping: track the max ball height since the last bounce; charge the deficit at the bounce
        apex_prev = info["flight_apex"]
        apex_low_ev = bounce.astype(jp.float32) * jp.clip(cfg.reward_config.apex_target_z - apex_prev, 0.0, None)
        info["flight_apex"] = jp.where(bounce | do_rs, ball_p[2], jp.maximum(apex_prev, ball_p[2]))
        ball_near = jp.clip(ball_dist_xy - cfg.reward_config.control_radius, 0.0, None)
        leg_touch = (lleg_c | rleg_c).astype(jp.float32)
        trunk_touch = trunk_c.astype(jp.float32)
        carry = ((hand_steps > cfg.reward_config.carry_steps) & ~holding).astype(jp.float32)
        ball_up = ball_p[2] > pelvis_z + 0.12
        hold_ok = holding & lhand_c & rhand_c & ball_up & (info["hold_steps"] <= cfg.reset_config.hold_min_steps)
        hold_timeout = holding & (info["hold_steps"] > cfg.reset_config.hold_max_steps)

        lo, hi = cfg.reward_config.pelvis_height_range
        pelvis_height = jp.clip(lo - pelvis_z, 0.0, None) + jp.clip(pelvis_z - hi, 0.0, None)
        up_t = self._s(data, "upvector_torso")  # torso z-axis in world
        up_local = yaw_mat.T @ up_t
        # forward lean: torso z-axis tips forward -> positive x component; roll: y component.
        pitch = jp.arcsin(jp.clip(up_local[0], -1, 1))
        plo, phi = cfg.reward_config.torso_pitch_range
        torso_orientation = jp.square(up_local[1]) * 4.0 + jp.square(jp.clip(plo - pitch, 0.0, None)) + jp.square(jp.clip(pitch - phi, 0.0, None))
        feet_contact = (~feet_c).astype(jp.float32).sum()
        feet_v = jp.stack([self._s(data, "left_foot_global_linvel")[:2], self._s(data, "right_foot_global_linvel")[:2]])
        feet_slip = jp.sum(jp.linalg.norm(feet_v, axis=-1) * feet_c)
        feet_hold = jp.sum(jp.linalg.norm(feet[:, :2] - info["feet_init_xy"], axis=-1))
        qj = data.qpos[7:36]
        pose = jp.sum(jp.square(qj - info["stance_qpos"]) * self._pose_weights)
        action_rate = jp.sum(jp.square(action - info["last_act"]))
        torques = jp.sum(jp.square(data.actuator_force))
        dof_acc = jp.sum(jp.square(data.qacc[6:35]))
        dof_vel = jp.sum(jp.square(data.qvel[6:35]))
        out_of_limits = -jp.clip(qj - self._soft_lowers, None, 0.0) + jp.clip(qj - self._soft_uppers, 0.0, None)
        dof_pos_limits = jp.sum(out_of_limits)
        ang_vel_xy = jp.sum(jp.square(self._s(data, "global_angvel_torso")[:2]))
        lin_vel_z = jp.square(self._s(data, "global_linvel_pelvis")[2])

        dense = {
            "ball_to_goal": ball_to_goal, "hand_to_ball": hand_to_ball, "ball_low": ball_low,
            "ball_near": ball_near, "leg_touch": leg_touch, "trunk_touch": trunk_touch, "carry": carry,
            "pelvis_height": pelvis_height, "torso_orientation": torso_orientation,
            "feet_contact": feet_contact, "feet_slip": feet_slip, "feet_hold": feet_hold, "pose": pose,
            "action_rate": action_rate, "torques": torques, "dof_acc": dof_acc, "dof_vel": dof_vel,
            "dof_pos_limits": dof_pos_limits, "ang_vel_xy": ang_vel_xy, "lin_vel_z": lin_vel_z,
        }
        dense["alive"] = jp.ones(())
        dense["hold"] = hold_ok.astype(jp.float32)
        dense["hold_timeout"] = hold_timeout.astype(jp.float32)
        events = {
            "crossing": crossing.astype(jp.float32), "touch": touch.astype(jp.float32),
            "catch": catch.astype(jp.float32), "chain": chain.astype(jp.float32),
            "bad_bounce": bad_bounce.astype(jp.float32),
            "wrong_hand": wrong_hand.astype(jp.float32), "free_bounce": free_bounce.astype(jp.float32),
            "ball_reset": ball_reset.astype(jp.float32),
            "recovery": recovery.astype(jp.float32),
            "apex_low": apex_low_ev,
            "termination": (done & ~nan).astype(jp.float32),
        }
        scales = cfg.reward_config.scales
        reward = sum(v * scales[k] for k, v in dense.items()) * self.dt + sum(v * scales[k] for k, v in events.items())

        # ---- observations ----
        info["next_hand"] = new_next_hand
        info["expect"] = new_expect
        info["free_bounces"] = free_bounces
        info["since_bounce"] = since_bounce
        info["hand_contact_steps"] = hand_steps
        info["dead_steps"] = dead_steps
        info["floor_contact_prev"] = floor_c
        info["crossings"] = info["crossings"] + crossing
        rel_obs = self._vision_estimate(data, info, yaw_mat, do_rs)
        info["ball_buffer"] = jp.concatenate([rel_obs[None], info["ball_buffer"][:-1]], axis=0)
        found = self._found(data)
        obs = self._get_obs(data, info, found, rel)
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["step"] = info["step"] + 1
        info["push_step"] = info["push_step"] + 1

        for k, v in dense.items():
            state.metrics[f"reward/{k}"] = v * scales[k] * self.dt
        for k, v in events.items():
            state.metrics[f"reward/{k}"] = v * scales[k]
        state.metrics["task/crossings"] = crossing.astype(jp.float32)  # per-step event; Brax sums per episode
        state.metrics["task/bad_bounces"] = bad_bounce.astype(jp.float32)
        state.metrics["task/wrong_hand"] = wrong_hand.astype(jp.float32)
        state.metrics["task/ball_z_mean"] = ball_p[2]
        state.metrics["task/apex_deficit"] = apex_low_ev
        state.metrics["task/pelvis_z"] = pelvis_z
        state.metrics["task/ball_resets"] = ball_reset.astype(jp.float32)
        state.metrics["task/recoveries"] = recovery.astype(jp.float32)
        state.metrics["task/hold_releases"] = hold_release.astype(jp.float32)
        state.metrics["task/hold_ok_steps"] = hold_ok.astype(jp.float32)
        state.metrics["task/kicks"] = kick_now.astype(jp.float32)
        state.metrics["task/touches"] = touch.astype(jp.float32)
        state.metrics["task/catches"] = catch.astype(jp.float32)
        state.metrics["task/chains"] = chain.astype(jp.float32)
        state.metrics["term/fall"] = fall.astype(jp.float32)
        state.metrics["term/ball_lost"] = ball_lost.astype(jp.float32)
        state.metrics["term/dead_ball"] = dead_ball.astype(jp.float32)
        state.metrics["term/free_bounce"] = too_many_free.astype(jp.float32)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(jp.float32))

    # ------------------------------------------------------------------------------------------
    def _get_obs(self, data, info, found, ball_rel_true):
        cfg = self._config
        nz = cfg.noise_config
        info["rng"], k1, k2, k3, k4, k5, k6 = jax.random.split(info["rng"], 7)

        def noisy(x, key, scale):
            return x + (2 * jax.random.uniform(key, x.shape) - 1) * nz.level * scale

        gyro = self._s(data, "gyro_pelvis")
        gravity = data.site_xmat[self._pelvis_imu_site].T @ jp.array([0.0, 0.0, -1.0])
        qj = data.qpos[7:36]
        dqj = data.qvel[6:35]
        # Ball tracker: delayed by 0..2 control steps, noisy, expressed in the pelvis yaw frame.
        ball_obs = info["ball_buffer"][info["ball_delay"]]
        ball_obs = jp.concatenate([noisy(ball_obs[:3], k5, nz.scales.ball_pos), noisy(ball_obs[3:], k6, nz.scales.ball_vel)])
        info["ball_hist"] = jp.concatenate([ball_obs[None], info["ball_hist"][:-1]], axis=0)
        cue = jp.array([
            jp.where(info["next_hand"] == 1, 1.0, -1.0),  # which hand should play the ball next
            jp.where(info["expect"] == 1, 1.0, -1.0),  # ball released (waiting for gate bounce) or not
            jp.clip(info["since_bounce"], 0.0, 1.0),
            jp.where(info["stance"] == 0, 1.0, -1.0),  # left foot forward or right
        ])
        state = jp.concatenate([
            noisy(gyro, k1, nz.scales.gyro),  # 3
            noisy(gravity, k2, nz.scales.gravity),  # 3
            noisy(qj, k3, nz.scales.joint_pos) - info["stance_qpos"],  # 29
            noisy(dqj, k4, nz.scales.joint_vel),  # 29
            info["last_act"],  # 29
            info["ball_hist"].ravel(),  # 18
            cue,  # 4
        ])
        pelvis = data.site_xpos[self._pelvis_imu_site]
        palms = data.site_xpos[self._palm_site]
        feet = data.site_xpos[self._feet_site]
        privileged = jp.concatenate([
            state,
            gyro, gravity, qj - info["stance_qpos"], dqj,
            self._s(data, "local_linvel_pelvis"),
            ball_rel_true,  # 6 true ball state
            self._s(data, "ball_angvel"),
            (palms - pelvis).ravel(), (feet - pelvis).ravel(),
            (palms - self._s(data, "ball_pos")).ravel(),
            self._s(data, "left_palm_linvel"), self._s(data, "right_palm_linvel"),
            found.astype(jp.float32),  # 10 contact flags
            jp.array([pelvis[2]]),
            data.actuator_force / 100.0,
        ])
        return {"state": state, "privileged_state": privileged}

    # ------------------------------------------------------------------------------------------
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 29

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


# ----------------------------------------------------------------------------------------------
def load_ball_calibration():
    p = os.path.join(HERE, "assets", "ball_calibration.json")
    with open(p) as f:
        return json.load(f)


def domain_randomize(model: mjx.Model, rng: jax.Array):
    """Per-environment physics randomisation (evaluated once per training env).

    Ranges: floor/foot friction U(0.4,1.0) (playground G1 hardware policy); ball-floor friction
    U(0.4,0.9) (FIBA floor 0.4-0.7 dry, court finish varies); ball-body friction U(0.5,1.0);
    ball bounce inside the legal drop-test band (calibrate_ball.py); ball mass 22 +- 0.5 oz (legal);
    link masses x U(0.9,1.1) and torso +-1 kg (payload/battery); joint friction x U(0.5,2) and
    armature x U(1,1.05) (playground); PD gains x U(0.9,1.1) (motor gain tolerance); encoder offset
    +-0.015 rad (Unitree mjlab encoder_bias).
    """
    calib = load_ball_calibration()
    d_lo, d_hi = calib["dampratio_legal_range"]
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    pair_names = [m.pair(i).name for i in range(m.npair)]
    foot_pairs = np.array([i for i, n in enumerate(pair_names) if n in ("floor_left_foot", "floor_right_foot")])
    ball_floor = np.array([i for i, n in enumerate(pair_names) if n == "floor_ball"])
    ball_body = np.array([i for i, n in enumerate(pair_names) if n.startswith("ball_")])
    ball_pairs = np.concatenate([ball_floor, ball_body])
    torso_id = m.body("torso_link").id
    ball_id = m.body("ball").id

    @jax.vmap
    def rand(rng):
        keys = jax.random.split(rng, 12)
        pf = model.pair_friction
        f_foot = jax.random.uniform(keys[0], minval=0.4, maxval=1.0)
        pf = pf.at[foot_pairs, 0:2].set(f_foot)
        f_bf = jax.random.uniform(keys[1], minval=0.4, maxval=0.9)
        pf = pf.at[ball_floor, 0:2].set(f_bf)
        f_bb = jax.random.uniform(keys[2], minval=0.5, maxval=1.0)
        pf = pf.at[ball_body, 0:2].set(f_bb)
        damp = jax.random.uniform(keys[3], minval=min(d_lo, d_hi), maxval=max(d_lo, d_hi))
        ps = model.pair_solref.at[ball_pairs, 1].set(damp)
        body_mass = model.body_mass * jax.random.uniform(keys[4], (model.nbody,), minval=0.9, maxval=1.1)
        body_mass = body_mass.at[torso_id].add(jax.random.uniform(keys[5], minval=-1.0, maxval=1.0))
        ball_mass = jax.random.uniform(keys[6], minval=21.5 * 0.028349523, maxval=22.5 * 0.028349523)
        body_mass = body_mass.at[ball_id].set(ball_mass)
        body_inertia = model.body_inertia.at[ball_id].set(model.body_inertia[ball_id] * ball_mass / model.body_mass[ball_id])
        fl = model.dof_frictionloss.at[6:35].set(model.dof_frictionloss[6:35] * jax.random.uniform(keys[7], (29,), minval=0.5, maxval=2.0))
        arm = model.dof_armature.at[6:35].set(model.dof_armature[6:35] * jax.random.uniform(keys[8], (29,), minval=1.0, maxval=1.05))
        g = jax.random.uniform(keys[9], (29,), minval=0.9, maxval=1.1)
        gain = model.actuator_gainprm.at[:, 0].set(model.actuator_gainprm[:, 0] * g)
        bias = model.actuator_biasprm.at[:, 1].set(model.actuator_biasprm[:, 1] * g)
        bias = bias.at[:, 2].set(model.actuator_biasprm[:, 2] * jax.random.uniform(keys[10], (29,), minval=0.9, maxval=1.1))
        qpos0 = model.qpos0.at[7:36].add(jax.random.uniform(keys[11], (29,), minval=-0.015, maxval=0.015))
        return pf, ps, body_mass, body_inertia, fl, arm, gain, bias, qpos0

    pf, ps, body_mass, body_inertia, fl, arm, gain, bias, qpos0 = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "pair_friction": 0, "pair_solref": 0, "body_mass": 0, "body_inertia": 0, "dof_frictionloss": 0,
        "dof_armature": 0, "actuator_gainprm": 0, "actuator_biasprm": 0, "qpos0": 0,
    })
    model = model.tree_replace({
        "pair_friction": pf, "pair_solref": ps, "body_mass": body_mass, "body_inertia": body_inertia,
        "dof_frictionloss": fl, "dof_armature": arm, "actuator_gainprm": gain, "actuator_biasprm": bias,
        "qpos0": qpos0,
    })
    return model, in_axes
