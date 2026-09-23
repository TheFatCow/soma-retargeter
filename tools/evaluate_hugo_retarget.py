# SPDX-License-Identifier: Apache-2.0

"""Measure a Hugo retarget against the SOMA motion it came from.

    python tools/evaluate_hugo_retarget.py BVH CSV [--self-test]

Everything is measured from the robot's forward kinematics in MuJoCo, never
from the retargeter's own internals, so a pipeline bug cannot mark its own
homework:

  tracking   robot body vs the effector target the IK was given (same scaler
             config the pipeline used), position cm / rotation deg
  direction  angle between the robot's SEGMENT directions and those of the IK
             targets -- scale-free, so it measures whether the pose is right,
             not whether the sizes match. Measured against the targets rather
             than raw human joints because Hugo's link origins are not on the
             limb axes (hip_roll_link->knee_link is 11.8 deg off vertical with
             the leg straight); the calibrated offsets make target and robot
             coincide at the reference pose, so a correct pose scores ~0.
  feet       sole penetration below z=0, and skating: horizontal sole speed
             while the sole is within 1 cm of the floor
  limits     % of frames each joint spends within 1 deg of its range
  jumps      largest frame-to-frame joint step (branch flips show up here)

--self-test re-scores the same CSV with two joint columns swapped and with the
root heading rotated; both MUST score clearly worse, or the metrics are not
able to see a broken retarget.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import warp as wp  # noqa: E402

import soma_retargeter.assets.bvh as bvh_utils  # noqa: E402
import soma_retargeter.assets.hugo as hugo  # noqa: E402
import soma_retargeter.utils.io_utils as io_utils  # noqa: E402
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler  # noqa: E402
from soma_retargeter.utils.space_conversion_utils import FacingDirectionType, SpaceConverter  # noqa: E402

RETARGET_CFG = "hugo/soma_to_hugo_retargeter_config.json"
SEGMENTS = {  # name: (human from, human to, robot from, robot to)
    "L upper arm": ("LeftArm", "LeftForeArm", "left_shoulder_roll_link", "left_elbow_link"),
    "L forearm": ("LeftForeArm", "LeftHand", "left_elbow_link", "left_hand_tcp"),
    "R upper arm": ("RightArm", "RightForeArm", "right_shoulder_roll_link", "right_elbow_link"),
    "R forearm": ("RightForeArm", "RightHand", "right_elbow_link", "right_hand_tcp"),
    "L thigh": ("LeftLeg", "LeftShin", "left_hip_roll_link", "left_knee_link"),
    "L shin": ("LeftShin", "LeftFoot", "left_knee_link", "left_ankle_link"),
    "R thigh": ("RightLeg", "RightShin", "right_hip_roll_link", "right_knee_link"),
    "R shin": ("RightShin", "RightFoot", "right_knee_link", "right_ankle_link"),
    "pelvis L-R": ("RightLeg", "LeftLeg", "right_hip_roll_link", "left_hip_roll_link"),
    "shoulders L-R": ("RightArm", "LeftArm", "right_shoulder_roll_link", "left_shoulder_roll_link"),
}
# Bodies carrying the foot collision boxes in the raw MJCF (toe.py splits them
# at training time; here the foot is rigid, so the ankle links carry them all).
FEET = ("left_ankle_link", "right_ankle_link")


def human_effectors(bvh_path: str):
    cfg = io_utils.load_json(io_utils.get_config_file(RETARGET_CFG))
    skel, anim = bvh_utils.load_bvh(bvh_path)
    scaler = HumanToRobotScaler(skel, cfg["model_height"],
                                io_utils.get_config_file(cfg["human_robot_scaler_config"]))
    xform = SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity())
    targets = scaler.compute_effectors_from_buffer(anim, True, xform)
    # Raw (unscaled) joint positions, for scale-free segment directions.
    probe = io_utils.load_json(io_utils.get_config_file(cfg["human_robot_scaler_config"]))
    probe["joint_scales"] = {k: 1.0 for k in probe["joint_scales"]}
    probe["joint_offsets"] = {k: [[0, 0, 0], [0, 0, 0, 1]] for k in probe["joint_offsets"]}
    probe["human_height_assumption"] = 1.0
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(probe, f)
    tmp = Path(f.name)
    try:
        raw = HumanToRobotScaler(skel, 1.0, str(tmp)).compute_effectors_from_buffer(anim, True, xform)
    finally:
        tmp.unlink()
    return scaler.effector_names(), targets, raw, cfg["ik_map"], anim.sample_rate


def robot_fk(csv_path: str, swap=None, yaw_deg=0.0):
    d = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    header = open(csv_path).readline().strip().split(",")
    names = [h[:-4] for h in header[7:]]
    if names != hugo.HUGO_JOINT_NAMES:
        raise ValueError(f"CSV joint columns {names} != HUGO_JOINT_NAMES")
    q = np.deg2rad(d[:, 7:])
    if swap:
        i, j = (names.index(s) for s in swap)
        q[:, [i, j]] = q[:, [j, i]]
    m = mujoco.MjModel.from_xml_path(str(hugo.resolve_hugo_mjcf_path()))
    data = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in names]
    rot = R.from_euler("xyz", d[:, 4:7], degrees=True)
    if yaw_deg:
        rot = R.from_euler("z", yaw_deg, degrees=True) * rot
    bodies = {m.body(b).name: m.body(b).id for b in range(m.nbody)}
    pos = np.zeros((len(d), m.nbody, 3)); quat = np.zeros((len(d), m.nbody, 4))
    sole = np.zeros((len(d), len(FEET), 2, 3))
    for f in range(len(d)):
        data.qpos[:] = 0
        data.qpos[0:3] = d[f, 1:4] * 0.01
        data.qpos[3:7] = rot[f].as_quat(scalar_first=True)
        data.qpos[qadr] = q[f]
        mujoco.mj_kinematics(m, data)
        pos[f] = data.xpos; quat[f] = data.xquat
        for k, b in enumerate(FEET):
            sole[f, k] = _lowest_box_point(m, data, bodies[b])
    lo, hi = m.jnt_range[[m.joint(n).id for n in names]].T
    return dict(pos=pos, quat=quat, bodies=bodies, q=q, lo=lo, hi=hi, names=names,
                sole=sole, fps=None)


def _lowest_box_point(m, data, body):
    best = np.array([np.inf, np.inf, np.inf]); prev = np.zeros(3)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_size[g]
        w = c @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        low = w[np.argmin(w[:, 2])]
        if low[2] < best[2]:
            best = low
    return np.stack([best, prev])


def _angle_deg(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, -1), -1, 1)))


def score(bvh_path, csv_path, swap=None, yaw_deg=0.0, verbose=True):
    names, targets, raw, ik_map, fps = human_effectors(bvh_path)
    rb = robot_fk(csv_path, swap=swap, yaw_deg=yaw_deg)
    n = min(len(targets), len(rb["pos"]))
    idx = {k: i for i, k in enumerate(names)}
    out = {"tracking": {}, "direction": {}, "limits": {}, "feet": {}}

    for joint, e in ik_map.items():
        b = rb["bodies"][e["t_body"]]
        tp = targets[:n, idx[joint], 0:3]
        dp = np.linalg.norm(rb["pos"][:n, b] - tp, axis=-1) * 100
        tq = R.from_quat(targets[:n, idx[joint], 3:7])
        rq = R.from_quat(rb["quat"][:n, b], scalar_first=True)
        dr = np.degrees((tq.inv() * rq).magnitude())
        out["tracking"][joint] = (float(np.median(dp)), float(np.percentile(dp, 95)),
                                  float(np.median(dr)), float(np.percentile(dr, 95)))
    for seg, (h0, h1, r0, r1) in SEGMENTS.items():
        hv = targets[:n, idx[h1], 0:3] - targets[:n, idx[h0], 0:3]
        rv = rb["pos"][:n, rb["bodies"][r1]] - rb["pos"][:n, rb["bodies"][r0]]
        a = _angle_deg(hv, rv)
        out["direction"][seg] = (float(np.median(a)), float(np.percentile(a, 95)))

    q = rb["q"][:n]
    near = (np.minimum(q - rb["lo"], rb["hi"] - q) < np.radians(1.0)).mean(0) * 100
    out["limits"] = {nm: float(v) for nm, v in zip(rb["names"], near) if v > 0.5}
    step = (np.degrees(np.abs(np.diff(q, axis=0))).max(0) if len(q) > 1
            else np.zeros(q.shape[1]))
    out["max_step_deg"] = (rb["names"][int(np.argmax(step))], float(step.max()))

    sole = rb["sole"][:n, :, 0]          # (frames, feet, xyz)
    lowest = sole[:, :, 2].min(1)
    out["feet"]["penetration_cm_p99"] = float(max(0.0, -np.percentile(lowest, 1)) * 100)
    dt = 1.0 / fps
    v = np.linalg.norm(np.diff(sole[:, :, 0:2], axis=0), axis=-1) / dt
    contact = (sole[1:, :, 2] < 0.01)
    out["feet"]["skate_cm_s_median"] = float(np.median(v[contact]) * 100) if contact.any() else float("nan")
    out["feet"]["contact_pct"] = float(contact.mean() * 100)

    if verbose:
        print(f"[{Path(csv_path).name}] {n} frames @ {fps:g} fps")
        print("  tracking (robot body vs IK target):     pos cm med/p95    rot deg med/p95")
        for k, (a, b, c, d) in out["tracking"].items():
            print(f"    {k:13s} {a:6.1f} {b:6.1f}      {c:6.1f} {d:6.1f}")
        print("  segment direction vs human (deg):  med / p95")
        for k, (a, b) in out["direction"].items():
            print(f"    {k:14s} {a:6.1f} {b:6.1f}")
        print(f"  feet: penetration p99 {out['feet']['penetration_cm_p99']:.1f} cm, "
              f"skate median {out['feet']['skate_cm_s_median']:.1f} cm/s in contact "
              f"({out['feet']['contact_pct']:.0f}% of foot-frames in contact)")
        print(f"  joints >0.5% of frames within 1 deg of a limit: "
              + (", ".join(f"{k} {v:.0f}%" for k, v in out["limits"].items()) or "none"))
        print(f"  largest single-frame joint step: {out['max_step_deg'][0]} {out['max_step_deg'][1]:.1f} deg")
    return out


def _summary(o):
    return float(np.mean([v[0] for v in o["direction"].values()]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bvh"); ap.add_argument("csv")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    base = score(args.bvh, args.csv)
    if not args.self_test:
        return 0
    good = _summary(base)
    # Judge each corruption on the segments it touches, not on a mean that a
    # bad arm elsewhere can dominate.
    sw = score(args.bvh, args.csv, swap=("left_knee", "left_hip_pitch"), verbose=False)
    yawed = _summary(score(args.bvh, args.csv, yaw_deg=90.0, verbose=False))
    leg_before = max(base["direction"]["L thigh"][0], base["direction"]["L shin"][0])
    leg_after = max(sw["direction"]["L thigh"][0], sw["direction"]["L shin"][0])
    print(f"[self-test] L thigh/shin direction error: real {leg_before:.1f} deg, "
          f"swapped L knee/hip-pitch {leg_after:.1f} deg")
    print(f"[self-test] mean segment-direction error: real {good:.1f} deg, root yawed 90 deg {yawed:.1f} deg")
    ok = leg_after > leg_before + 10 and yawed > good + 20
    print("[self-test] PASS: the metrics see a broken retarget" if ok else
          "[self-test] FAIL: a corrupted retarget did not score clearly worse")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
