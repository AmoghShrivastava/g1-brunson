"""Build the G1 + basketball MuJoCo scene from Unitree's official robot description.

Ground truth used (all committed in ./assets and cited in README.md):
  * Robot kinematics/inertials/joint limits: Unitree `g1_29dof_rev_1_0` (URDF + MJCF) from
    https://github.com/unitreerobotics/unitree_ros (robots/g1_description).  The URDF is loaded by
    MuJoCo and compared body-by-body against the MJCF that we actually simulate, so the simulated
    robot is provably the URDF robot (masses, inertias, joint axes/ranges, link offsets).
  * Collision primitives (capsules) and IMU site: Unitree's own MuJoCo RL model
    `unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml` (same rev_1_0 robot; Unitree deploys
    policies trained on it to hardware).  Only change: each foot's 7 thin capsules are replaced by one
    box with the same footprint so MJX gets 4 stable contact points per foot.
  * Actuators: Unitree's hardware deployment PD gains, torque limits, velocity limits and reflected
    rotor inertias (armature) from `unitree_rl_mjlab` deploy.yaml / g1_constants.py.
  * Ball: NBA/FIBA size-7 game ball: circumference 29.5 in -> r = 0.11926 m, mass 22 oz = 0.6237 kg,
    hollow shell inertia.  Bounce (pressure 7.5-8.5 psi) is calibrated in calibrate_ball.py to the
    official drop test (1.8 m -> 1.2..1.4 m).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------------------------
# Unitree hardware deployment constants (unitree_rl_mjlab: deploy/robots/g1/config/policy/velocity/
# v0/params/deploy.yaml and src/assets/robots/unitree_g1/g1_constants.py).
# ----------------------------------------------------------------------------------------------
JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# Motor type per joint (Unitree g1_constants.py).  Ankle and waist roll/pitch are 4-bar linkages
# driven by two 5020 motors -> Unitree models them as 2x a 5020.
MOTOR = {}
for s in ("left", "right"):
    MOTOR[f"{s}_hip_pitch_joint"] = "7520_14"
    MOTOR[f"{s}_hip_roll_joint"] = "7520_22"
    MOTOR[f"{s}_hip_yaw_joint"] = "7520_14"
    MOTOR[f"{s}_knee_joint"] = "7520_22"
    MOTOR[f"{s}_ankle_pitch_joint"] = "2x5020"
    MOTOR[f"{s}_ankle_roll_joint"] = "2x5020"
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"):
        MOTOR[f"{s}_{j}_joint"] = "5020"
    MOTOR[f"{s}_wrist_pitch_joint"] = "4010"
    MOTOR[f"{s}_wrist_yaw_joint"] = "4010"
MOTOR["waist_yaw_joint"] = "7520_14"
MOTOR["waist_roll_joint"] = "2x5020"
MOTOR["waist_pitch_joint"] = "2x5020"

# reflected inertia (armature), torque limit [Nm], velocity limit [rad/s]  (Unitree g1_constants.py)
MOTOR_SPEC = {
    "5020": dict(armature=0.003609725, effort=25.0, vel=37.0),
    "2x5020": dict(armature=2 * 0.003609725, effort=50.0, vel=37.0),
    "7520_14": dict(armature=0.010177520, effort=88.0, vel=32.0),
    "7520_22": dict(armature=0.025101925, effort=139.0, vel=20.0),
    "4010": dict(armature=0.00425, effort=5.0, vel=22.0),
}
NATURAL_FREQ = 10 * 2.0 * np.pi  # 10 Hz  (Unitree)
DAMPING_RATIO = 2.0  # (Unitree)


def pd_gains(joint):
    a = MOTOR_SPEC[MOTOR[joint]]["armature"]
    kp = a * NATURAL_FREQ ** 2
    kd = 2.0 * DAMPING_RATIO * a * NATURAL_FREQ
    return kp, kd


# Unitree's deployed values (deploy.yaml) -- used as an assertion on the formula above.
DEPLOY_KP = [40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 28.5, 28.5,
             14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8, 14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8]
DEPLOY_KD = [2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 1.8, 1.8,
             0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1]

# Unitree "home" joint targets (deploy.yaml default_joint_pos / FixStand qs).
UNITREE_DEFAULT = [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0,
                   0.35, 0.18, 0, 0.87, 0, 0, 0, 0.35, -0.18, 0, 0.87, 0, 0, 0]

# NBA / FIBA size-7 game ball.
BALL_CIRCUMFERENCE_M = 29.5 * 0.0254
BALL_RADIUS = BALL_CIRCUMFERENCE_M / (2 * np.pi)  # 0.11926 m
BALL_MASS = 22.0 * 0.028349523  # 0.6237 kg


# ----------------------------------------------------------------------------------------------
def load_urdf_model(urdf_path, meshdir):
    txt = open(urdf_path).read()
    # Unitree's URDF already carries a <mujoco><compiler meshdir="meshes"/></mujoco> block; point it
    # at the absolute mesh directory (MuJoCo reads compiler settings from that element).
    txt = re.sub(r"<mujoco>.*?</mujoco>",
                 f'<mujoco><compiler meshdir="{meshdir}" balanceinertia="false" discardvisual="false"/></mujoco>',
                 txt, count=1, flags=re.S)
    # URDF has no floating base: MuJoCo would weld the root link to the world (and drop its mass).
    # Add a floating joint from a dummy world link to the pelvis so the root link is a real body.
    txt = txt.replace("</robot>", '<link name="world"/><joint name="floating_base_joint" type="floating">'
                      '<parent link="world"/><child link="pelvis"/></joint></robot>')
    return mujoco.MjModel.from_xml_string(txt)


def compare_models(m_urdf, m_sim, tol=1e-6):
    """Compare bodies/joints of the simulated model against the URDF model."""
    report = {}
    worst = 0.0
    for b in range(1, m_sim.nbody):
        name = mujoco.mj_id2name(m_sim, mujoco.mjtObj.mjOBJ_BODY, b)
        if name in ("ball",):
            continue
        bu = mujoco.mj_name2id(m_urdf, mujoco.mjtObj.mjOBJ_BODY, name)
        if bu < 0:
            raise RuntimeError(f"body {name} not in URDF")
        d = {}
        d["mass"] = abs(m_sim.body_mass[b] - m_urdf.body_mass[bu])
        d["pos"] = 0.0 if name == "pelvis" else np.abs(m_sim.body_pos[b] - m_urdf.body_pos[bu]).max()
        d["quat"] = min(np.abs(m_sim.body_quat[b] - m_urdf.body_quat[bu]).max(),
                        np.abs(m_sim.body_quat[b] + m_urdf.body_quat[bu]).max())
        d["ipos"] = np.abs(m_sim.body_ipos[b] - m_urdf.body_ipos[bu]).max()
        d["inertia"] = np.abs(np.sort(m_sim.body_inertia[b]) - np.sort(m_urdf.body_inertia[bu])).max()
        for k, v in d.items():
            worst = max(worst, v)
        report[name] = {k: float(v) for k, v in d.items()}
    for j in range(m_sim.njnt):
        name = mujoco.mj_id2name(m_sim, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in ("floating_base_joint", "ball_freejoint"):
            continue
        ju = mujoco.mj_name2id(m_urdf, mujoco.mjtObj.mjOBJ_JOINT, name)
        if ju < 0:
            raise RuntimeError(f"joint {name} not in URDF")
        d = {
            "axis": np.abs(m_sim.jnt_axis[j] - m_urdf.jnt_axis[ju]).max(),
            "range": np.abs(m_sim.jnt_range[j] - m_urdf.jnt_range[ju]).max(),
            "pos": np.abs(m_sim.jnt_pos[j] - m_urdf.jnt_pos[ju]).max(),
        }
        for k, v in d.items():
            worst = max(worst, v)
        report[name] = {k: float(v) for k, v in d.items()}
    return worst, report


# ----------------------------------------------------------------------------------------------
def build_robot_xml(mjlab_xml, out_path, mesh_src, mesh_dst, hand_box=False):
    """Take Unitree's RL MJCF, make it MJX friendly, and copy the meshes it needs."""
    tree = ET.parse(mjlab_xml)
    root = tree.getroot()
    root.set("model", "g1_29dof_rev_1_0_dribble")
    comp = root.find("compiler")
    comp.set("meshdir", "assets/g1")
    # Meshes.
    os.makedirs(mesh_dst, exist_ok=True)
    for mesh in root.find("asset").findall("mesh"):
        f = mesh.get("file")
        shutil.copy(os.path.join(mesh_src, f), os.path.join(mesh_dst, f))
    # Collision class: explicit contact pairs only (contype/conaffinity 0).  Keep condim/priority.
    for dflt in root.iter("default"):
        if dflt.get("class") == "collision":
            g = dflt.find("geom")
            g.set("contype", "0")
            g.set("conaffinity", "0")
            g.set("condim", "3")
    # Feet: replace the 7 capsules by one box with the same footprint.
    #   capsules span x in [-0.054, 0.132] (+-0.01 radius), y in [-0.026, 0.026] (+-0.01), z=-0.025 r=0.01
    for body in root.iter("body"):
        if body.get("name") in ("left_ankle_roll_link", "right_ankle_roll_link"):
            side = body.get("name").split("_")[0]
            for g in list(body.findall("geom")):
                if g.get("class") == "foot_capsule":
                    body.remove(g)
            ET.SubElement(body, "geom", name=f"{side}_foot", type="box", size="0.103 0.036 0.010",
                          pos="0.039 0 -0.025", **{"class": "collision"})
    # Hands: Unitree's RL model uses a capsule; the real rubber hand is a flat paddle (mesh AABB in the
    # wrist_yaw frame: x 0.0415..0.1733, y -0.045..0.022 (left), z -0.043..0.063).  A box fitted to
    # the mesh gives the ball a flat palm face (normal = +-y of the wrist), so "dribble with the palm"
    # is physically meaningful.  Enabled with hand_box=True.
    if hand_box:
        for g in root.iter("geom"):
            if g.get("name") in ("left_hand_collision", "right_hand_collision"):
                side = g.get("name").split("_")[0]
                for k in ("size", "fromto", "rgba"):
                    if k in g.attrib:
                        del g.attrib[k]
                g.set("type", "box")
                g.set("size", "0.066 0.033 0.053")
                g.set("pos", f"0.107 {-0.012 if side == 'left' else 0.012} 0.010")
    # Joint armature (reflected rotor inertia) from Unitree motor constants.
    for j in root.iter("joint"):
        n = j.get("name")
        if n in MOTOR:
            j.set("armature", f"{MOTOR_SPEC[MOTOR[n]]['armature']:.9f}")
            j.set("actuatorfrcrange", f"-{MOTOR_SPEC[MOTOR[n]]['effort']} {MOTOR_SPEC[MOTOR[n]]['effort']}")
    # Ball-facing sites on the palms already exist (left_palm/right_palm).  Add a site for the
    # pelvis "gate" reference and for the head (posture metrics).
    for body in root.iter("body"):
        if body.get("name") == "pelvis":
            body.remove(body.find("light"))
            body.remove(body.find("camera"))
    # Strip Unitree's sensor block (we add our own) and contact excludes (explicit pairs are used).
    for tag in ("sensor", "contact"):
        e = root.find(tag)
        if e is not None:
            root.remove(e)
    tree.write(out_path, encoding="unicode")


def actuator_xml():
    lines = ["  <actuator>"]
    for i, j in enumerate(JOINT_ORDER):
        kp, kd = pd_gains(j)
        assert abs(kp - DEPLOY_KP[i]) < 0.15, (j, kp, DEPLOY_KP[i])
        assert abs(kd - DEPLOY_KD[i]) < 0.1, (j, kd, DEPLOY_KD[i])
        eff = MOTOR_SPEC[MOTOR[j]]["effort"]
        lines.append(
            f'    <position name="{j.replace("_joint", "")}" joint="{j}" kp="{kp:.4f}" kv="{kd:.4f}" '
            f'inheritrange="1" forcerange="-{eff} {eff}"/>')
    lines.append("  </actuator>")
    return "\n".join(lines)


BALL_TOUCH_GEOMS = [
    "left_hand_collision", "right_hand_collision", "left_wrist_collision", "right_wrist_collision",
    "left_elbow_yaw_collision", "right_elbow_yaw_collision",
    "left_shoulder_yaw_collision", "right_shoulder_yaw_collision",
    "left_thigh_collision", "right_thigh_collision", "left_shin_collision", "right_shin_collision",
    "left_linkage_brace_collision", "right_linkage_brace_collision",
    "left_hip_collision", "right_hip_collision", "left_foot", "right_foot",
    "pelvis_collision", "torso_collision", "head_collision",
]


def scene_xml(ball_solref, floor_friction, ball_floor_friction, ball_body_friction, timestep):
    ball_r = BALL_RADIUS
    pairs = [f'    <pair name="floor_{g}" geom1="floor" geom2="{g}" condim="3" friction="{floor_friction} {floor_friction} 0.005 0.0001 0.0001"/>'
             for g in ("left_foot", "right_foot")]
    pairs.append(f'    <pair name="floor_ball" geom1="floor" geom2="ball" condim="3" '
                 f'friction="{ball_floor_friction} {ball_floor_friction} 0.005 0.0001 0.0001" solref="{ball_solref[0]} {ball_solref[1]}"/>')
    for g in BALL_TOUCH_GEOMS:
        pairs.append(f'    <pair name="ball_{g}" geom1="ball" geom2="{g}" condim="3" '
                     f'friction="{ball_body_friction} {ball_body_friction} 0.005 0.0001 0.0001" solref="{ball_solref[0]} {ball_solref[1]}"/>')
    # Robot self-collision: arms (hand, forearm, upper arm) against legs and trunk, frictionless
    # (condim 1) like Unitree's mjlab self-collision setup.  Prevents the policy from swinging the
    # arm through the thigh/torso.
    for s in ("left", "right"):
        for a in ("hand_collision", "wrist_collision", "elbow_yaw_collision"):
            for side in ("left", "right"):
                for b in ("thigh_collision", "shin_collision", "hip_collision"):
                    pairs.append(f'    <pair name="self_{s}_{a}_{side}_{b}" geom1="{s}_{a}" geom2="{side}_{b}" condim="1"/>')
            for t in ("pelvis_collision", "torso_collision"):
                pairs.append(f'    <pair name="self_{s}_{a}_{t}" geom1="{s}_{a}" geom2="{t}" condim="1"/>')
    # Contact sensors (geom-geom only: MJX's JAX backend does not support body/subtree matching).
    contact_sensors = [
        '    <contact name="ball_floor_found" geom1="ball" geom2="floor" reduce="mindist" num="1" data="found"/>',
        '    <contact name="left_foot_floor_found" geom1="left_foot" geom2="floor" reduce="mindist" num="1" data="found"/>',
        '    <contact name="right_foot_floor_found" geom1="right_foot" geom2="floor" reduce="mindist" num="1" data="found"/>',
    ]
    for g in BALL_TOUCH_GEOMS:
        contact_sensors.append(f'    <contact name="ball_{g}_found" geom1="ball" geom2="{g}" reduce="mindist" num="1" data="found"/>')
    return f"""<mujoco model="g1_dribble_scene">
  <include file="g1_robot.xml"/>

  <option timestep="{timestep}" iterations="4" ls_iterations="6" integrator="implicitfast" solver="Newton">
    <flag eulerdamp="disable"/>
  </option>

  <statistic center="0 0 0.6" extent="1.6" meansize="0.04"/>
  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".3 .3 .3" specular="0.6 0.6 0.6"/>
    <global azimuth="150" elevation="-15" offwidth="1920" offheight="1080"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.95 0.95 0.97" rgb2="0.75 0.78 0.85" width="512" height="512"/>
    <texture type="2d" name="court" builtin="checker" mark="edge" rgb1="0.22 0.40 0.80" rgb2="0.15 0.30 0.66" markrgb="0.75 0.85 1.0" width="300" height="300"/>
    <material name="court" texture="court" texuniform="true" texrepeat="6 6" reflectance="0.15"/>
    <texture name="basketball" type="2d" file="assets/basketball.png"/>
    <material name="ball_mat" texture="basketball" specular="0.2" shininess="0.3" rgba="1 1 1 1"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="court" contype="1" conaffinity="0" friction="{floor_friction}"/>
    <light pos="1 1 3" dir="-0.3 -0.3 -1" directional="true" castshadow="true"/>
    <camera name="side" pos="2.4 -2.0 1.1" xyaxes="0.64 0.77 0 -0.22 0.18 0.96" mode="fixed"/>
    <camera name="front" pos="2.6 0.0 0.9" xyaxes="0 1 0 -0.25 0 0.97" mode="fixed"/>
    <body name="ball" pos="0.3 0 {ball_r + 0.5}">
      <freejoint name="ball_freejoint"/>
      <geom name="ball" type="sphere" size="{ball_r:.5f}" mass="{BALL_MASS:.5f}" shellinertia="true"
            material="ball_mat" contype="0" conaffinity="0" condim="3" priority="2"/>
      <site name="ball_site" size="0.01" group="5"/>
    </body>
  </worldbody>

  <contact>
{chr(10).join(pairs)}
  </contact>

{actuator_xml()}

  <sensor>
    <gyro name="gyro_pelvis" site="imu_in_pelvis"/>
    <accelerometer name="accelerometer_pelvis" site="imu_in_pelvis"/>
    <framezaxis name="upvector_pelvis" objtype="site" objname="imu_in_pelvis"/>
    <framelinvel name="global_linvel_pelvis" objtype="site" objname="imu_in_pelvis"/>
    <frameangvel name="global_angvel_pelvis" objtype="site" objname="imu_in_pelvis"/>
    <velocimeter name="local_linvel_pelvis" site="imu_in_pelvis"/>
    <gyro name="gyro_torso" site="imu_in_torso"/>
    <framezaxis name="upvector_torso" objtype="site" objname="imu_in_torso"/>
    <framelinvel name="global_linvel_torso" objtype="site" objname="imu_in_torso"/>
    <frameangvel name="global_angvel_torso" objtype="site" objname="imu_in_torso"/>
    <framepos name="ball_pos" objtype="site" objname="ball_site"/>
    <framelinvel name="ball_linvel" objtype="site" objname="ball_site"/>
    <frameangvel name="ball_angvel" objtype="site" objname="ball_site"/>
    <framepos name="left_palm_pos" objtype="site" objname="left_palm"/>
    <framepos name="right_palm_pos" objtype="site" objname="right_palm"/>
    <framelinvel name="left_palm_linvel" objtype="site" objname="left_palm"/>
    <framelinvel name="right_palm_linvel" objtype="site" objname="right_palm"/>
    <framepos name="left_foot_pos" objtype="site" objname="left_foot"/>
    <framepos name="right_foot_pos" objtype="site" objname="right_foot"/>
    <framelinvel name="left_foot_global_linvel" objtype="site" objname="left_foot"/>
    <framelinvel name="right_foot_global_linvel" objtype="site" objname="right_foot"/>
    <framepos name="pelvis_pos" objtype="site" objname="imu_in_pelvis"/>
    <framequat name="pelvis_quat" objtype="site" objname="imu_in_pelvis"/>
{chr(10).join(contact_sensors)}
  </sensor>
</mujoco>
"""


# ----------------------------------------------------------------------------------------------
def solve_stance(model, front="left", stagger=0.08, half_width=0.18, pelvis_z=0.74):
    """Solve leg joint angles for a staggered, knees-bent athletic stance with flat feet.

    Coaching source: feet about shoulder width or slightly wider, one foot forward, knees bent
    ("Ball Handling Skills & Drills", basketballarmy.com, xbotgo.com).  Numbers scaled to the G1's
    0.6 m legs.  Unknowns per leg: hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll.
    """
    data = mujoco.MjData(model)
    jid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(model.njnt)}
    qadr = {n: model.jnt_qposadr[jid[n]] for n in JOINT_ORDER}
    site_l = model.site("left_foot").id
    site_r = model.site("right_foot").id
    q0 = np.zeros(model.nq)
    q0[3] = 1.0
    q0[2] = pelvis_z
    for i, n in enumerate(JOINT_ORDER):
        q0[qadr[n]] = UNITREE_DEFAULT[i]
    xs = {"left": stagger if front == "left" else -stagger,
          "right": stagger if front == "right" else -stagger}
    ys = {"left": half_width, "right": -half_width}
    legs = ["left", "right"]
    names = [f"{s}_{j}_joint" for s in legs for j in ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")]

    def residual(x):
        q = q0.copy()
        for n, v in zip(names, x):
            q[qadr[n]] = v
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        res = []
        for s, sid in (("left", site_l), ("right", site_r)):
            p = data.site_xpos[sid]
            R = data.site_xmat[sid].reshape(3, 3)
            res += [p[0] - xs[s], p[1] - ys[s], p[2] - 0.0]  # site sits at sole level
            res += [R[0, 2] * 3, R[1, 2] * 3]  # sole flat: site z-axis == world z
        return np.array(res)

    x0 = np.array([-0.35, 0.05, 0.75, -0.4, -0.05, -0.35, -0.05, 0.75, -0.4, 0.05])
    lo = np.array([model.jnt_range[jid[n]][0] for n in names])
    hi = np.array([model.jnt_range[jid[n]][1] for n in names])
    # Solve, then shift the feet so that the whole-body centre of mass sits over the middle of the
    # support polygon (balance), and re-solve.  A couple of iterations converge.
    com_shift = np.zeros(2)
    for _ in range(4):
        sol = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
        q = q0.copy()
        for n, v in zip(names, sol.x):
            q[qadr[n]] = v
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        com = data.subtree_com[model.body("pelvis").id][:2]
        feet_mid = 0.5 * (data.site_xpos[site_l][:2] + data.site_xpos[site_r][:2])
        err = com - feet_mid
        com_shift += err
        for s in legs:
            xs[s] += err[0]
            ys[s] += err[1]
        x0 = sol.x
    return q, sol.cost, dict(zip(names, sol.x.tolist()), com_shift_xy=com_shift.tolist())


def gravity_compensated_ctrl(model, qpos, hold_gain=25.0, seconds=2.0):
    """Joint position targets whose PD torque statically holds the stance.

    The real controller runs tau = kp (q_target - q) - kd qdot with no gravity feed-forward, so a
    deployed "default pose" is always such a pre-compensated target.  We find it physically: hold
    the IK pose with a stiff PD (x`hold_gain`), let contacts settle, read the equilibrium actuator
    torques tau_eq and map them to the real gains: q_target = q_eq + tau_eq / kp.
    """
    gain = model.actuator_gainprm.copy()
    bias = model.actuator_biasprm.copy()
    frc = model.actuator_forcerange.copy()
    model.actuator_gainprm[:, 0] *= hold_gain
    model.actuator_biasprm[:, 1:3] *= hold_gain
    model.actuator_forcerange[:] *= hold_gain
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = qpos[7:7 + model.nu]
    bq = model.jnt_qposadr[model.joint("ball_freejoint").id]
    data.qpos[bq:bq + 3] = [3.0, 3.0, 2.0]
    mujoco.mj_forward(model, data)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    tau = data.actuator_force.copy()
    q_eq = data.qpos.copy()
    model.actuator_gainprm[:] = gain
    model.actuator_biasprm[:] = bias
    model.actuator_forcerange[:] = frc
    ctrl = np.zeros(model.nu)
    for i in range(model.nu):
        j = model.actuator_trnid[i, 0]
        kp = model.actuator_gainprm[i, 0]
        t = float(np.clip(tau[i], frc[i, 0], frc[i, 1]))
        ctrl[i] = q_eq[model.jnt_qposadr[j]] + t / kp
    return ctrl, q_eq


def settle(model, qpos, seconds=1.5, ctrl=None):
    """Run the PD controlled robot from qpos and return the settled state + contact check."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = qpos[7:7 + model.nu] if ctrl is None else ctrl
    # park the ball far away during settling
    bq = model.jnt_qposadr[model.joint("ball_freejoint").id]
    data.qpos[bq:bq + 3] = [3.0, 3.0, 2.0]
    mujoco.mj_forward(model, data)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    lf = data.sensor("left_foot_floor_found").data[0] > 0
    rf = data.sensor("right_foot_floor_found").data[0] > 0
    return data.qpos.copy(), data.qvel.copy(), bool(lf), bool(rf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unitree_ros", default="/workspace/assets/unitree_ros/robots/g1_description")
    ap.add_argument("--mjlab_xml", default="/workspace/assets/mjlab_g1/g1.xml")
    ap.add_argument("--out", default=os.path.join(HERE, "assets"))
    ap.add_argument("--ball_solref", default="0.015 1.0", help="dampratio overridden by calibrate_ball.py")
    ap.add_argument("--timestep", type=float, default=0.002)
    ap.add_argument("--floor_friction", type=float, default=0.6)  # Unitree mjlab foot friction
    ap.add_argument("--ball_floor_friction", type=float, default=0.6)
    ap.add_argument("--ball_body_friction", type=float, default=0.8)
    ap.add_argument("--hand_box", action="store_true", help="flat palm box instead of Unitree hand capsule")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    robot_xml = os.path.join(args.out, "g1_robot.xml")
    build_robot_xml(args.mjlab_xml, robot_xml, os.path.join(args.unitree_ros, "meshes"),
                    os.path.join(args.out, "assets", "g1"), hand_box=args.hand_box)
    # Provenance copy of the URDF + MJCF we validated against.
    prov = os.path.join(args.out, "provenance")
    os.makedirs(prov, exist_ok=True)
    for f in ("g1_29dof_rev_1_0.urdf", "g1_29dof_rev_1_0.xml", "README.md"):
        shutil.copy(os.path.join(args.unitree_ros, f), os.path.join(prov, f))
    shutil.copy(args.mjlab_xml, os.path.join(prov, "unitree_rl_mjlab_g1.xml"))

    solref = [float(x) for x in args.ball_solref.split()]
    scene_path = os.path.join(args.out, "g1_dribble_scene.xml")
    with open(scene_path, "w") as f:
        f.write(scene_xml(solref, args.floor_friction, args.ball_floor_friction,
                          args.ball_body_friction, args.timestep))

    # ---- verification against the URDF ----
    model = mujoco.MjModel.from_xml_path(scene_path)
    m_urdf = load_urdf_model(os.path.join(args.unitree_ros, "g1_29dof_rev_1_0.urdf"), args.unitree_ros)
    m_unitree_mjcf = mujoco.MjModel.from_xml_path(os.path.join(args.unitree_ros, "g1_29dof_rev_1_0.xml"))
    worst_sim, rep_sim = compare_models(m_urdf, model)
    worst_u, _ = compare_models(m_urdf, m_unitree_mjcf)
    print(f"[verify] max |sim - URDF| over masses/inertia/offsets/joint axes+ranges: {worst_sim:.2e}")
    print(f"[verify] max |Unitree MJCF - URDF|: {worst_u:.2e}")
    # The only differences are the 1 g dummy fixed links of the URDF (imu/logo/head/pelvis_contour)
    # that MuJoCo fuses into pelvis/torso, giving 1e-3 kg mass and ~5e-5 kg m^2 inertia rounding in
    # Unitree's own MJCF conversion.  Everything else (offsets, axes, ranges, masses) is identical.
    assert worst_sim < 2e-3, "simulated robot deviates from the Unitree URDF"
    assert model.nu == 29 and model.nq == 7 + 29 + 7
    # Actuator order == JOINT_ORDER (Unitree hardware order).
    for i, j in enumerate(JOINT_ORDER):
        assert model.actuator(i).name == j.replace("_joint", ""), (i, j, model.actuator(i).name)
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0]) == j
    total_mass = float(model.body_subtreemass[model.body("pelvis").id])
    print(f"[verify] robot mass {total_mass:.3f} kg (Unitree spec: ~35 kg with battery)")
    print(f"[verify] ball: r={model.geom('ball').size[0]:.5f} m mass={model.body('ball').mass[0]:.4f} kg "
          f"inertia={model.body('ball').inertia}")
    assert abs(model.body('ball').inertia[0] - (2 / 3) * BALL_MASS * BALL_RADIUS ** 2) < 1e-6

    # ---- stance keyframes ----
    keys = {}
    stance_info = {}
    for front in ("left", "right"):
        q, cost, angles = solve_stance(model, front=front)
        ctrl, q = gravity_compensated_ctrl(model, q)
        qs, vs, lf, rf = settle(model, q, ctrl=ctrl)
        pelvis_drop = q[2] - qs[2]
        print(f"[stance] front={front} ik_cost={cost:.2e} settled pelvis z={qs[2]:.3f} (drop {pelvis_drop:.3f}) "
              f"feet contact L={lf} R={rf} knees=({angles['left_knee_joint']:.2f},{angles['right_knee_joint']:.2f}) "
              f"com_shift={angles['com_shift_xy']}")
        # With Unitree's hardware PD gains the crouch sags under gravity at zero action (the policy
        # has to compensate, exactly like on the real robot); we only require both feet to stay on
        # the floor.  The keyframe stores the IK pose; the sag is reported for the README.
        assert lf and rf and abs(pelvis_drop) < 0.06, "stance did not settle on both feet"
        keys[f"stance_{front}_front"] = (q, ctrl)
        stance_info[front] = angles
    with open(os.path.join(args.out, "stance.json"), "w") as f:
        json.dump({k: {"qpos": v[0].tolist(), "ctrl": v[1].tolist()} for k, v in keys.items()}, f, indent=1)

    # Write keyframes into the scene file.
    kf = ["  <keyframe>"]
    home = np.zeros(model.nq)
    home[2] = 0.79
    home[3] = 1
    home[7:36] = UNITREE_DEFAULT
    home[36:39] = [0.3, 0, BALL_RADIUS]
    home[39] = 1
    kf.append(f'    <key name="home" qpos="{" ".join(f"{x:.5f}" for x in home)}" ctrl="{" ".join(f"{x:.5f}" for x in UNITREE_DEFAULT)}"/>')
    for name, (qs, ctrl) in keys.items():
        qs = qs.copy()
        qs[36:39] = [0.3, 0, BALL_RADIUS]
        qs[39:43] = [1, 0, 0, 0]
        kf.append(f'    <key name="{name}" qpos="{" ".join(f"{x:.5f}" for x in qs)}" ctrl="{" ".join(f"{x:.5f}" for x in ctrl)}"/>')
    kf.append("  </keyframe>\n</mujoco>\n")
    txt = open(scene_path).read().rstrip()
    assert txt.endswith("</mujoco>")
    txt = txt[: -len("</mujoco>")] + "\n".join(kf)
    with open(scene_path, "w") as f:
        f.write(txt)
    model = mujoco.MjModel.from_xml_path(scene_path)
    print(f"[done] wrote {scene_path}: nq={model.nq} nv={model.nv} nu={model.nu} ngeom={model.ngeom} npair={model.npair} nkey={model.nkey}")
    with open(os.path.join(args.out, "urdf_verification.json"), "w") as f:
        json.dump({"max_abs_diff_sim_vs_urdf": worst_sim, "max_abs_diff_unitree_mjcf_vs_urdf": worst_u,
                   "per_element": rep_sim}, f, indent=1)


if __name__ == "__main__":
    main()
