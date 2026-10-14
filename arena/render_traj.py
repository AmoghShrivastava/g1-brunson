"""Render a saved trajectory (traj_*.npz from evaluate_mjx.py / evaluate_vision.py) to an MP4 with the
plain MuJoCo renderer (no JAX needed).  usage: python render_traj.py traj.npz out.mp4 [--camera side]"""
import argparse
import os

import imageio
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--xml", default=os.path.join(HERE, "assets", "g1_dribble_scene.xml"))
    ap.add_argument("--camera", default="side")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--label", default="")
    ap.add_argument("--free", nargs=5, type=float, metavar=("LOOKAT_Z", "DIST", "AZIMUTH", "ELEV", "LOOKAT_X"),
                    help="free camera instead of a named one: lookat z, distance, azimuth, elevation, lookat x")
    args = ap.parse_args()
    tr = np.load(args.traj)
    q, dt = tr["qpos"], float(tr["dt"])
    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=720, width=1280)
    cam = args.camera
    if args.free:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [args.free[4], 0.0, args.free[0]]
        cam.distance, cam.azimuth, cam.elevation = args.free[1], args.free[2], args.free[3]
    every = max(1, int(round(1.0 / (dt * args.fps))))
    frames = []
    for t in range(0, len(q), every):
        d.qpos[:] = q[t]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        img = r.render().copy()
        if args.label:
            try:
                from PIL import Image, ImageDraw
                im = Image.fromarray(img)
                ImageDraw.Draw(im).text((12, 12), f"{args.label}  t={t * dt:4.1f}s", fill=(255, 255, 255))
                img = np.asarray(im)
            except Exception:  # noqa
                pass
        frames.append(img)
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print("wrote", args.out, len(frames), "frames")


if __name__ == "__main__":
    main()
