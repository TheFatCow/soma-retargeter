# SPDX-License-Identifier: Apache-2.0

"""Render a Hugo retarget next to the SOMA motion it came from.

    python tools/render_hugo_retarget.py BVH CSV OUT.mp4 [--sheet OUT.png] [--frames 0,200,400]

Hugo is drawn from the CSV through MuJoCo forward kinematics. The IK targets
(the scaled, offset SOMA effectors the solver was asked to hit) are drawn as
small spheres ON the robot, so a gap between sphere and link is a tracking
error you can see. The SOMA skeleton itself is drawn 0.9 m to the robot's left
so the two can be compared pose-for-pose. Needs ffmpeg on PATH.

This project has been fooled by healthy-looking numbers before; watch the clip.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

import soma_retargeter.assets.hugo as hugo  # noqa: E402
import evaluate_hugo_retarget as ev  # noqa: E402

W, H = 640, 480
BONES = [("Hips", "Chest"), ("Chest", "LeftArm"), ("LeftArm", "LeftForeArm"),
         ("LeftForeArm", "LeftHand"), ("Chest", "RightArm"), ("RightArm", "RightForeArm"),
         ("RightForeArm", "RightHand"), ("Hips", "LeftLeg"), ("LeftLeg", "LeftShin"),
         ("LeftShin", "LeftFoot"), ("Hips", "RightLeg"), ("RightLeg", "RightShin"),
         ("RightShin", "RightFoot")]


def build_model():
    spec = mujoco.MjSpec.from_file(str(hugo.resolve_hugo_mjcf_path()))
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.82, 0.84, 0.86], rgb2=[0.68, 0.70, 0.73], width=256, height=256)
    mat = spec.add_material(name="grid", texrepeat=[8, 8])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1],
                            material="grid", contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    return spec.compile()


def _add_sphere(scene, p, rgba, size=0.025):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]),
                        np.asarray(p, float), np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def _add_bone(scene, a, b, rgba, width=0.018):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bvh"); ap.add_argument("csv"); ap.add_argument("out")
    ap.add_argument("--sheet", default=None, help="also write a PNG contact sheet")
    ap.add_argument("--frames", default=None, help="comma-separated frames for the sheet")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    names, targets, _raw, ik_map, src_fps = ev.human_effectors(args.bvh)
    idx = {k: i for i, k in enumerate(names)}
    d = np.loadtxt(args.csv, delimiter=",", skiprows=1, ndmin=2)
    header = open(args.csv).readline().strip().split(",")
    jn = [h[:-4] for h in header[7:]]
    m = build_model(); data = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in jn]
    rot = R.from_euler("xyz", d[:, 4:7], degrees=True)

    renderer = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = 3.2, -12.0
    n = min(len(d), len(targets))
    stride = max(1, int(round(src_fps / args.fps)))
    sheet_frames = ([int(x) for x in args.frames.split(",")] if args.frames
                    else list(np.linspace(0, n - 1, 6).astype(int)))

    def frame_image(f):
        data.qpos[:] = 0
        data.qpos[0:3] = d[f, 1:4] * 0.01
        data.qpos[3:7] = rot[f].as_quat(scalar_first=True)
        data.qpos[qadr] = np.deg2rad(d[f, 7:])
        mujoco.mj_forward(m, data)
        # Robot heading, so the camera looks at its front and "left" is its left.
        fwd = rot[f].apply([1, 0, 0]); fwd[2] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
        left = np.array([-fwd[1], fwd[0], 0.0])
        cam.lookat[:] = data.qpos[0:3] + 0.45 * left - [0, 0, 0.35]
        cam.azimuth = float(np.degrees(np.arctan2(-fwd[1], -fwd[0]))) + 25.0
        renderer.update_scene(data, cam)
        sc = renderer.scene
        for j, e in ik_map.items():
            _add_sphere(sc, targets[f, idx[j], 0:3], [1.0, 0.25, 0.1, 0.9])
        shift = 0.9 * left
        for a, b in BONES:
            _add_bone(sc, targets[f, idx[a], 0:3] + shift, targets[f, idx[b], 0:3] + shift,
                      [0.15, 0.45, 0.95, 1.0])
        return renderer.render().copy()

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", f"{args.fps:g}", "-i", "-", "-pix_fmt", "yuv420p",
                           "-vcodec", "libx264", args.out], stdin=subprocess.PIPE)
    for f in range(0, n, stride):
        ff.stdin.write(frame_image(f).tobytes())
    ff.stdin.close(); ff.wait()
    print(f"[INFO] wrote {args.out} ({len(range(0, n, stride))} frames)")

    if args.sheet:
        tiles = [frame_image(f) for f in sheet_frames]
        cols = 3; rows = (len(tiles) + cols - 1) // cols
        tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
        img = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{img.shape[1]}x{img.shape[0]}", "-i", "-", args.sheet],
                       input=img.tobytes(), check=True)
        print(f"[INFO] wrote {args.sheet} (frames {sheet_frames})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
