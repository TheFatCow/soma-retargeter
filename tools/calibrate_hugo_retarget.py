# SPDX-License-Identifier: Apache-2.0

"""Derive Hugo's SOMA->robot scaler config by calibration, not by hand-tuning.

    python tools/calibrate_hugo_retarget.py            # write configs/hugo/soma_to_hugo_scaler_config.json
    python tools/calibrate_hugo_retarget.py --check    # exit 1 if the committed file is stale

The G1 scaler's `joint_offsets` are hand-tuned numbers. Hugo's body frames are
rotated so every joint axis is local +z, which makes hand-tuning hopeless, so
this derives them from a reference correspondence instead:

1. The SOMA reference pose is `soma/soma_zero_frame0.bvh` (the retargeter's own
   initialization pose): standing, legs straight, feet flat, upper arms down,
   elbows bent 90 deg forward. It is read through HumanToRobotScaler with unit
   scales and identity offsets, i.e. through the exact code path the pipeline
   uses, so frame conventions (y-up cm BVH -> z-up m) cannot drift.
2. Hugo is posed to match. Legs and waist are already in that configuration at
   qpos=0; the arms are FITTED (shoulder pitch/roll/yaw + elbow) so the upper
   arm and forearm point where SOMA's do. The robot is yawed -90 deg because
   SOMA faces -y in the retarget world, and dropped so its sole is on z=0.
3. Per mapped joint j with robot body b(j):
     scale      s_j = |R_b - R_root| / |H_j - H_root|    (s_root = z ratio)
     rot offset q_j = inv(Hq_j) * Rq_b
     pos offset p_j = inv(Hq_j * q_j) * (R_b - (s_j (H_j - H_root) + s_root H_root))
   which makes every effector coincide EXACTLY with its robot body at the
   reference pose; the report prints that residual as a self-check.
4. The fitted arm angles are also written to the retargeter config as
   `initial_joint_q`, so the IK starts from the reference pose rather than
   qpos=0 (from zero, the right arm fell into a raised mirror branch).
"""

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import warp as wp  # noqa: E402

import soma_retargeter.assets.bvh as bvh_utils  # noqa: E402
import soma_retargeter.assets.hugo as hugo  # noqa: E402
import soma_retargeter.utils.io_utils as io_utils  # noqa: E402
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler  # noqa: E402
from soma_retargeter.utils.space_conversion_utils import FacingDirectionType, SpaceConverter  # noqa: E402

OUTPUT = io_utils.get_config_file("hugo", "soma_to_hugo_scaler_config.json")
RETARGETER_CONFIG = io_utils.get_config_file("hugo", "soma_to_hugo_retargeter_config.json")
REFERENCE_BVH = "soma/soma_zero_frame0.bvh"

# SOMA joint -> Hugo body. The ik_map in soma_to_hugo_retargeter_config.json
# must target these same bodies, or the offsets calibrate the wrong frame.
BODY_MAP = {
    "Hips": "waist_yaw_link",
    "Chest": "torso",
    "Neck1": "torso",
    "LeftArm": "left_shoulder_roll_link",
    "LeftForeArm": "left_elbow_link",
    "LeftHand": "left_hand_tcp",
    "RightArm": "right_shoulder_roll_link",
    "RightForeArm": "right_elbow_link",
    "RightHand": "right_hand_tcp",
    "LeftLeg": "left_hip_roll_link",
    "LeftShin": "left_knee_link",
    "LeftFoot": "left_ankle_link",
    "LeftToeBase": "left_toe_pitch_link",
    "RightLeg": "right_hip_roll_link",
    "RightShin": "right_knee_link",
    "RightFoot": "right_ankle_link",
    "RightToeBase": "right_toe_pitch_link",
}
PARENTS = {
    "Hips": "", "Chest": "Hips", "Neck1": "Chest",
    "LeftArm": "Chest", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "RightArm": "Chest", "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "LeftLeg": "Hips", "LeftShin": "LeftLeg", "LeftFoot": "LeftShin", "LeftToeBase": "LeftFoot",
    "RightLeg": "Hips", "RightShin": "RightLeg", "RightFoot": "RightShin", "RightToeBase": "RightFoot",
}
ARM_JOINTS = {side: [f"{side}_shoulder_pitch", f"{side}_shoulder_roll",
                     f"{side}_shoulder_yaw", f"{side}_elbow"] for side in ("left", "right")}
ROOT_YAW_DEG = -90.0
# The arm fit stays this far inside every joint range so the reference is a
# pose the IK can hold rather than one balanced exactly on a hard stop. Kept
# small: 5 deg doubled the forearm fit residual (5.2 -> 9.8 deg). The drift
# first blamed on the limit was IKSmoothJointFilter -- see the retargeter
# config's smooth_joint_filter_objective_body_masks.
FIT_LIMIT_MARGIN_RAD = math.radians(1.0)


def human_reference() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """SOMA reference frames in the retarget world: {joint: (pos, quat_xyzw)}."""
    skel, anim = bvh_utils.load_bvh(io_utils.get_config_file(REFERENCE_BVH))
    probe = {
        "robot_type": "probe", "human_root_name": "Hips", "human_height_assumption": 1.0,
        "joint_scales": {n: 1.0 for n in BODY_MAP}, "joint_parents": PARENTS,
        "joint_offsets": {n: [[0, 0, 0], [0, 0, 0, 1]] for n in [*BODY_MAP, "LeftToe", "RightToe"]},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(probe, f)
    scaler = HumanToRobotScaler(skel, 1.0, f.name)
    xform = SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity())
    eff = scaler.compute_effectors_from_buffer(anim, True, xform)[0]
    return {n: (np.array(e[0:3], float), np.array(e[3:7], float))
            for n, e in zip(scaler.effector_names(), eff)}


class Robot:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(hugo.resolve_hugo_mjcf_path()))
        self.d = mujoco.MjData(self.m)
        self.qadr = {self.m.joint(j).name: self.m.jnt_qposadr[self.m.joint(j).id]
                     for j in range(self.m.njnt) if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}

    def pose(self, q: dict[str, float], root_z: float = 0.0):
        self.d.qpos[:] = 0.0
        self.d.qpos[2] = root_z
        self.d.qpos[3:7] = R.from_euler("z", ROOT_YAW_DEG, degrees=True).as_quat(scalar_first=True)
        for name, v in q.items():
            self.d.qpos[self.qadr[name]] = v
        mujoco.mj_kinematics(self.m, self.d)

    def pos(self, body: str) -> np.ndarray:
        return self.d.xpos[self.m.body(body).id].copy()

    def quat_xyzw(self, body: str) -> np.ndarray:
        return R.from_quat(self.d.xquat[self.m.body(body).id], scalar_first=True).as_quat()

    def limits(self, name: str) -> tuple[float, float]:
        lo, hi = self.m.jnt_range[self.m.joint(name).id]
        return float(lo), float(hi)

    def sole_z(self) -> float:
        """Lowest point of the foot collision geometry (boxes; see toe.py)."""
        lowest = math.inf
        feet = {self.m.body(b).id for b in ("left_ankle_link", "right_ankle_link",
                                            "left_toe_pitch_link", "right_toe_pitch_link")}
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] not in feet or self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            half = self.m.geom_size[g]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
            world = corners @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
            lowest = min(lowest, float(world[:, 2].min()))
        if not math.isfinite(lowest):
            raise ValueError("no foot box geoms found to place the sole on the floor")
        return lowest


def _unit(v):
    return v / np.linalg.norm(v)


def fit_arms(robot: Robot, H) -> tuple[dict[str, float], dict[str, float]]:
    """Fit each arm so upper arm and forearm directions match SOMA's.

    Multi-start, because shoulder yaw vs elbow sign has two mirror solutions and
    the wrong one bends the forearm backwards. Returns (angles, residual_deg).
    """
    q, residual = {}, {}
    for side, Side in (("left", "Left"), ("right", "Right")):
        names = ARM_JOINTS[side]
        lo = np.array([robot.limits(n)[0] for n in names]) + FIT_LIMIT_MARGIN_RAD
        hi = np.array([robot.limits(n)[1] for n in names]) - FIT_LIMIT_MARGIN_RAD
        up_h = _unit(H[f"{Side}ForeArm"][0] - H[f"{Side}Arm"][0])
        fore_h = _unit(H[f"{Side}Hand"][0] - H[f"{Side}ForeArm"][0])

        def err(x):
            robot.pose(dict(zip(names, x)))
            up_r = _unit(robot.pos(f"{side}_elbow_link") - robot.pos(f"{side}_shoulder_roll_link"))
            fore_r = _unit(robot.pos(f"{side}_hand_tcp") - robot.pos(f"{side}_elbow_link"))
            return np.concatenate([up_r - up_h, fore_r - fore_h])

        best = None
        for yaw in np.linspace(lo[2], hi[2], 7):
            for elbow in np.linspace(lo[3], hi[3], 7):
                x0 = np.clip([0.0, 0.0, yaw, elbow], lo + 1e-6, hi - 1e-6)
                r = least_squares(err, x0, bounds=(lo, hi))
                if best is None or r.cost < best.cost:
                    best = r
        q.update(dict(zip(names, best.x)))
        e = err(best.x)
        residual[f"{side} upper arm"] = math.degrees(2 * math.asin(min(1.0, np.linalg.norm(e[:3]) / 2)))
        residual[f"{side} forearm"] = math.degrees(2 * math.asin(min(1.0, np.linalg.norm(e[3:]) / 2)))
    return q, residual


def calibrate(scale_mode: str = "uniform"):
    H = human_reference()
    robot = Robot()
    arm_q, fit_residual = fit_arms(robot, H)
    robot.pose(arm_q)
    robot.pose(arm_q, root_z=-robot.sole_z())

    root_h, root_r = H["Hips"][0], robot.pos(BODY_MAP["Hips"])
    s_root = root_r[2] / root_h[2]
    scales, offsets, check = {}, {}, {}
    for joint, body in BODY_MAP.items():
        hp, hq = H[joint]
        rp, rq = robot.pos(body), robot.quat_xyzw(body)
        if joint == "Hips" or scale_mode == "uniform":
            s = s_root
        else:
            dh = np.linalg.norm(hp - root_h)
            s = float(np.linalg.norm(rp - root_r) / dh) if dh > 1e-6 else s_root
        q_off = R.from_quat(hq).inv() * R.from_quat(rq)
        q_eff = R.from_quat(hq) * q_off
        base = (np.zeros(3) if joint == "Hips" else s * (hp - root_h)) + s_root * root_h
        p_off = q_eff.inv().apply(rp - base)
        scales[joint] = round(float(s), 6)
        offsets[joint] = [[round(float(x), 6) for x in p_off],
                          [round(float(x), 6) for x in q_off.as_quat()]]
        # Re-evaluate the scaler's own formula at the reference pose.
        t = base + q_eff.apply(p_off)
        check[joint] = (float(np.linalg.norm(t - rp)), float(np.linalg.norm(p_off)), body)

    # HumanToRobotScaler overwrites the ToeBase offsets with the "Toe" entries.
    offsets["LeftToe"] = offsets.pop("LeftToeBase")
    offsets["RightToe"] = offsets.pop("RightToeBase")
    config = {
        "robot_type": "hugo",
        "_generated_by": "tools/calibrate_hugo_retarget.py -- do not hand-edit; re-run it",
        "_scale_mode": scale_mode,
        "_reference_pose": {"bvh": REFERENCE_BVH, "root_yaw_deg": ROOT_YAW_DEG,
                            "arm_joint_angles_rad": {k: round(float(v), 6) for k, v in arm_q.items()}},
        "human_root_name": "Hips",
        "human_height_assumption": 1.8,
        "joint_scales": scales,
        "joint_parents": PARENTS,
        "joint_offsets": offsets,
    }
    return config, fit_residual, check


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(OUTPUT))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--scale-mode", choices=("uniform", "per-joint"), default="uniform",
                    help="uniform: one scale (pelvis-height ratio) for every joint, so limb "
                         "shape is preserved and the offsets absorb Hugo's proportions. "
                         "per-joint: each joint's distance-from-root ratio (G1-style); "
                         "measured worse on Hugo, see the docstring.")
    args = ap.parse_args()

    config, fit_residual, check = calibrate(args.scale_mode)
    print("[INFO] arm fit residual (angle between robot and SOMA segment):")
    for k, v in fit_residual.items():
        print(f"         {k:16s} {v:6.2f} deg")
    print("[INFO] per-joint calibration (effector-vs-body residual at reference must be ~0):")
    for joint, (res, off, body) in check.items():
        print(f"         {joint:13s} -> {body:26s} scale {config['joint_scales'][joint]:.3f}  "
              f"|pos offset| {off*100:5.1f} cm  residual {res*1000:.3f} mm")
    worst = max(r for r, _, _ in check.values())
    if worst > 1e-4:
        print(f"[ERROR] calibration does not reproduce the reference pose ({worst*1000:.3f} mm)")
        return 1

    text = json.dumps(config, indent=2) + "\n"
    retarget = json.loads(RETARGETER_CONFIG.read_text())
    retarget["initial_joint_q"] = config["_reference_pose"]["arm_joint_angles_rad"]
    retarget_text = json.dumps(retarget, indent=4) + "\n"
    outputs = [(Path(args.output), text), (RETARGETER_CONFIG, retarget_text)]
    if args.check:
        stale = [p for p, t in outputs if not p.is_file() or p.read_text() != t]
        for p in stale:
            print(f"[STALE] {p}; re-run without --check.")
        if not stale:
            print("[OK] scaler config and initial_joint_q are current.")
        return 1 if stale else 0
    for p, t in outputs:
        p.write_text(t)
        print(f"[INFO] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
