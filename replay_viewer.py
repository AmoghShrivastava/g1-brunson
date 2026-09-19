"""Replay a recorded trajectory (traj_*.npz from evaluate_mjx.py) in an interactive MuJoCo viewer window.
usage: python replay_viewer.py <traj.npz> [scene.xml]   (loops; mouse to rotate/zoom, space pauses)"""
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
traj = np.load(sys.argv[1])
xml = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "assets", "g1_dribble_scene.xml")
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
qpos, dt = traj["qpos"], float(traj["dt"])
print(f"{qpos.shape[0]} frames at {1/dt:.0f} Hz; crossings={int(traj['crossings'])} chained={int(traj['chains'])}")
with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.azimuth, v.cam.elevation, v.cam.distance = 150, -15, 2.6
    v.cam.lookat[:] = [0, 0, 0.5]
    i = 0
    while v.is_running():
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        v.sync()
        time.sleep(dt)
        i = (i + 1) % qpos.shape[0]
