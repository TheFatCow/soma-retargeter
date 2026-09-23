# SPDX-License-Identifier: Apache-2.0

"""Hugo humanoid robot asset constants and the retarget-MJCF generator.

Hugo (26.6 kg, 19 actuated DoF) is deliberately poorer than a G1: no wrists or
hands, no ankle roll, no torso pitch, waist yaw only. Two consequences shape
this target:

* There is no hand body, so the SOMA ``LeftHand``/``RightHand`` effectors track
  a massless proxy body at the forearm tip (``*_hand_tcp``) added by
  :func:`build_retarget_spec`. It has no joint, so the robot's DoF set and the
  CSV layout are unchanged.
* The waist yaw joint sits BELOW the torso: the legs hang from
  ``waist_yaw_link`` and the floating base is on ``torso``. SOMA ``Hips`` maps to
  ``waist_yaw_link`` and ``Chest`` to ``torso``.

The source of truth for the robot is hugo-mjlab's ``robot.mjcf`` (itself
generated from hugo-humanoid). The retarget MJCF committed under
``configs/hugo/`` is derived from it and checked for staleness by
``tools/generate_hugo_retarget_mjcf.py --check``. The passive toe hinge that
hugo-mjlab adds at load time (``toe.py``) is NOT applied here: it is unactuated
and unobserved on the robot, so kinematic retargeting treats the foot as rigid.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import soma_retargeter.utils.io_utils as io_utils


# Actuated hinge joints in MJCF depth-first order, which is the order Newton's
# ModelBuilder assigns joint coordinates and therefore the order of the joint
# columns in an exported CSV. tests/test_hugo_target.py asserts this against
# the builder, so a regenerated MJCF with a different tree fails loudly.
HUGO_JOINT_NAMES: list[str] = [
    "waist_yaw",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle",
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
]

HAND_TCP_BODIES = {"left": "left_hand_tcp", "right": "right_hand_tcp"}
_ELBOW_BODIES = {"left": "left_elbow_link", "right": "right_elbow_link"}

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = io_utils.get_config_file("hugo", "hugo_retarget.xml")
# robot_mjlab.xml is hugo-mjlab's .xml symlink to robot.mjcf: MjSpec.from_file
# picks its decoder by extension and refuses ".mjcf".
DEFAULT_SOURCE_MJCF = (_REPO_ROOT.parent / "hugo-mjlab" / "hugo_mjlab" / "assets"
                       / "hugo" / "robot_mjlab.xml")
SOURCE_MJCF_ENV = "SOMA_RETARGETER_HUGO_SOURCE_MJCF"
MJCF_ENV = "SOMA_RETARGETER_HUGO_MJCF_PATH"


def resolve_source_mjcf_path(value: str | Path | None = None) -> Path:
    """hugo-mjlab's robot MJCF (.xml): explicit value, then env, then the sibling checkout."""
    path = Path(value or os.environ.get(SOURCE_MJCF_ENV) or DEFAULT_SOURCE_MJCF)
    if not path.is_file():
        raise FileNotFoundError(
            f"Hugo source MJCF not found at {path}. Check out hugo-mjlab next to "
            f"soma-retargeter or set {SOURCE_MJCF_ENV}.")
    return path


def resolve_hugo_mjcf_path(config_value: str | Path | None = None) -> Path:
    """The retarget MJCF: explicit config value (relative to configs/), then env,
    then the committed ``configs/hugo/hugo_retarget.xml``."""
    if config_value:
        path = Path(config_value)
        if not path.is_absolute():
            path = io_utils.get_config_file(str(path))
    else:
        path = Path(os.environ.get(MJCF_ENV) or DEFAULT_OUTPUT_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"Hugo retarget MJCF not found at {path}. Generate it with "
            "tools/generate_hugo_retarget_mjcf.py.")
    return path


def _forearm_tip_local(model, data, side: str) -> np.ndarray:
    """Forearm tip in the elbow link's frame, measured from the mesh.

    At qpos=0 Hugo's arms hang straight down, so the tip is the lowest band of
    forearm mesh vertices. The centroid of the lowest 2 cm is used rather than
    a single vertex, so one stray vertex cannot move the effector.
    """
    import mujoco

    body = model.body(_ELBOW_BODIES[side]).id
    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        pts.append(v.astype(np.float64) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    if not pts:
        raise ValueError(f"{_ELBOW_BODIES[side]} has no mesh geoms to measure a forearm tip from")
    p = np.vstack(pts)
    tip_w = p[p[:, 2] < p[:, 2].min() + 0.02].mean(axis=0)
    R = data.xmat[body].reshape(3, 3)
    return R.T @ (tip_w - data.xpos[body])


def build_retarget_spec(source_mjcf: str | Path | None = None, out_dir: str | Path | None = None):
    """Return an MjSpec: hugo-mjlab's robot plus the two hand-TCP proxy bodies.

    ``meshdir`` is rewritten relative to ``out_dir`` so the generated file loads
    the same meshes in place instead of copying them.
    """
    import mujoco

    src = resolve_source_mjcf_path(source_mjcf)
    out_dir = Path(out_dir or DEFAULT_OUTPUT_PATH.parent)

    model = mujoco.MjModel.from_xml_path(str(src))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tips = {side: _forearm_tip_local(model, data, side) for side in ("left", "right")}

    spec = mujoco.MjSpec.from_file(str(src))
    spec.meshdir = os.path.relpath(src.parent / (spec.meshdir or ""), out_dir)
    for side, tip in tips.items():
        spec.body(_ELBOW_BODIES[side]).add_body(
            name=HAND_TCP_BODIES[side], pos=[float(x) for x in np.round(tip, 6)])
    return spec
