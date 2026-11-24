"""Evaluation-only MuJoCo scene and right-arm IK/servo controller.

Builds directly from the pinned ``openarm-mujoco`` OpenArm v2 Pedestal XML via
``mujoco.MjSpec`` without duplicating meshes or modifying existing TouchDesigner,
OSC, hardware, or Dora runtime paths.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import openarm_mujoco.v2 as openarm_mujoco
from openarm_control import ArmSetup, IKParams, Kinematics

CAMERA_WRIST_RIGHT = "camera_wrist_right"
CAMERA_HEAD_RIGHT = "camera_head_right"
ARM_ORIGIN_SITE = "arm_origin"
RIGHT_EE_SITE = "right_ee_control_point"
LEFT_EE_SITE = "left_ee_control_point"
LEFT_FINGER_COLLISION_GEOMS = (
    "finger_inner_right_collision_00",
    "finger_inner_right_collision_01",
    "finger_inner_right_collision_02",
    "finger_inner_right_collision_03",
)
RIGHT_FINGER_COLLISION_GEOMS = (
    "finger_outer_right_collision_00",
    "finger_outer_right_collision_01",
    "finger_outer_right_collision_02",
    "finger_outer_right_collision_03",
)
RIGHT_ARM_ACTUATORS = tuple(f"right_joint{i}_ctrl" for i in range(1, 8))
LEFT_ARM_ACTUATORS = tuple(f"left_joint{i}_ctrl" for i in range(1, 8))
RIGHT_GRIPPER_ACTUATOR = "right_finger1_ctrl"
LEFT_GRIPPER_ACTUATOR = "left_finger1_ctrl"
RIGHT_FINGER_JOINTS = (
    "openarm_right_finger_joint1",
    "openarm_right_finger_joint2",
)
RIGHT_FINGER_MIMIC_EQ = "openarm_right_ee_finger_joint_mimic"
SUPPORT_BODY_NAME = "eval_support"
SUPPORT_GEOM_NAME = "eval_support_top"
BOX_BODY_NAME = "target_box"
BOX_JOINT_NAME = "target_box_freejoint"
BOX_GEOM_NAME = "target_box_geom"


@dataclass(frozen=True)
class SceneConfig:
    """Parameters for the evaluation-only box-on-support scene."""

    table_top_z: float = 0.42
    table_center_xy: tuple[float, float] = (0.53, -0.15)
    table_half_extents: tuple[float, float, float] = (0.20, 0.25, 0.04)
    table_rgba: tuple[float, float, float, float] = (0.78, 0.72, 0.62, 1.0)
    table_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    box_half_extents: tuple[float, float, float] = (0.020, 0.020, 0.032)
    box_mass_kg: float = 0.06
    box_rgba: tuple[float, float, float, float] = (0.88, 0.32, 0.12, 1.0)
    box_friction: tuple[float, float, float] = (1.0, 0.01, 0.001)
    open_finger_rad: float = -0.45
    pinch_z_offset_above_box_center_m: float = 0.018
    # Visual clipping tuning: OpenArm pedestal.xml sets extent=3.5, giving znear=35 mm
    # which clips objects held between the jaws in camera_wrist_right. Setting
    # extent=1.5 and znear=0.016 (24 mm absolute) clears the wrist camera bezel
    # while preventing near-plane clipping on the grasped box.
    visual_extent: float = 1.5
    visual_znear_ratio: float = 0.016
    settle_duration_s: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        """Serialize scene configuration to plain JSON-compatible types."""
        return asdict(self)


@dataclass(frozen=True)
class HeadCameraCalibration:
    """OpenArm Cell head camera parameters relative to the arm origin."""

    source_xml_sha256: str
    position_relative_m: tuple[float, ...]
    quaternion_relative_wxyz: tuple[float, ...]
    fovy_deg: float
    resolution_px: tuple[int, int]


@dataclass(frozen=True)
class SceneContract:
    """Resolved MuJoCo IDs and names for the evaluation scene."""

    pedestal_xml_path: str
    pedestal_xml_sha256: str
    camera_name: str
    camera_id: int
    right_ee_site_id: int
    left_finger_geom_ids: tuple[int, ...]
    right_finger_geom_ids: tuple[int, ...]
    left_finger_geom_names: tuple[str, ...]
    right_finger_geom_names: tuple[str, ...]
    robot_collision_geom_ids: frozenset[int]
    right_arm_actuator_ids: tuple[int, ...]
    left_arm_actuator_ids: tuple[int, ...]
    right_gripper_actuator_id: int
    left_gripper_actuator_id: int
    right_arm_qpos_indices: tuple[int, ...]
    left_arm_qpos_indices: tuple[int, ...]
    right_finger_qpos_indices: tuple[int, int]
    right_finger_mimic_eq_id: int
    support_geom_id: int
    box_body_id: int
    box_geom_id: int
    box_joint_id: int
    box_qpos_adr: int
    box_dof_adr: int
    floor_geom_id: int
    head_camera: HeadCameraCalibration | None = None


def _require_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Required MuJoCo object not found: {obj_type.name} '{name}'")
    return int(obj_id)


def _attach_head_camera(spec: mujoco.MjSpec) -> HeadCameraCalibration:
    """Transfer the right head camera from Cell to the pedestal arm mount."""
    cell_xml = Path(openarm_mujoco.openarm_cell_xml())
    cell_model = mujoco.MjModel.from_xml_path(str(cell_xml))
    cell_data = mujoco.MjData(cell_model)
    mujoco.mj_forward(cell_model, cell_data)
    camera_id = _require_id(cell_model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_HEAD_RIGHT)
    origin_id = _require_id(cell_model, mujoco.mjtObj.mjOBJ_SITE, ARM_ORIGIN_SITE)
    origin_axes = cell_data.site_xmat[origin_id].reshape(3, 3)
    position = origin_axes.T @ (
        cell_data.cam_xpos[camera_id] - cell_data.site_xpos[origin_id]
    )
    rotation = origin_axes.T @ cell_data.cam_xmat[camera_id].reshape(3, 3)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    fovy = float(cell_model.cam_fovy[camera_id])
    width, height = map(int, cell_model.cam_resolution[camera_id])

    origin = spec.site(ARM_ORIGIN_SITE)
    if origin is None:
        raise ValueError(f"Required MuJoCo site not found: {ARM_ORIGIN_SITE!r}")
    local_position = np.empty(3)
    local_quaternion = np.empty(4)
    mujoco.mju_mulPose(
        local_position, local_quaternion, origin.pos, origin.quat, position, quaternion
    )
    camera = origin.parent.add_camera(
        name=CAMERA_HEAD_RIGHT,
        pos=local_position,
        quat=local_quaternion,
        fovy=fovy,
        resolution=[width, height],
    )
    if origin.frame is not None:
        camera.set_frame(origin.frame)
    return HeadCameraCalibration(
        source_xml_sha256=hashlib.sha256(cell_xml.read_bytes()).hexdigest(),
        position_relative_m=tuple(position.tolist()),
        quaternion_relative_wxyz=tuple(quaternion.tolist()),
        fovy_deg=fovy,
        resolution_px=(width, height),
    )


def build_eval_scene(
    config: SceneConfig | None = None,
    *,
    include_head_camera: bool = False,
) -> tuple[mujoco.MjModel, SceneContract]:
    """Build the evaluation-only MuJoCo model from the pinned OpenArm v2 pedestal XML."""
    cfg = config or SceneConfig()
    pedestal_xml = Path(openarm_mujoco.openarm_pedestal_xml()).resolve()
    xml_sha256 = hashlib.sha256(pedestal_xml.read_bytes()).hexdigest()

    spec = mujoco.MjSpec.from_file(str(pedestal_xml))
    spec.stat.extent = float(cfg.visual_extent)
    spec.visual.map.znear = float(cfg.visual_znear_ratio)

    table_pos = [
        float(cfg.table_center_xy[0]),
        float(cfg.table_center_xy[1]),
        float(cfg.table_top_z - cfg.table_half_extents[2]),
    ]
    table_body = spec.worldbody.add_body(name=SUPPORT_BODY_NAME, pos=table_pos)
    table_body.add_geom(
        name=SUPPORT_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(cfg.table_half_extents),
        rgba=list(cfg.table_rgba),
        friction=list(cfg.table_friction),
        contype=1,
        conaffinity=1,
        condim=3,
        solref=[0.005, 1.0],
    )

    default_box_pos = [
        0.41,
        -0.15,
        float(cfg.table_top_z + cfg.box_half_extents[2]),
    ]
    box_body = spec.worldbody.add_body(name=BOX_BODY_NAME, pos=default_box_pos)
    box_body.add_freejoint(name=BOX_JOINT_NAME)
    box_body.add_geom(
        name=BOX_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(cfg.box_half_extents),
        mass=float(cfg.box_mass_kg),
        rgba=list(cfg.box_rgba),
        friction=list(cfg.box_friction),
        contype=1,
        conaffinity=1,
        condim=4,
        solref=[0.005, 1.0],
    )

    head_camera = _attach_head_camera(spec) if include_head_camera else None
    model = spec.compile()

    camera_id = _require_id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_WRIST_RIGHT)
    right_ee_site_id = _require_id(model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_EE_SITE)
    left_finger_ids = tuple(
        _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in LEFT_FINGER_COLLISION_GEOMS
    )
    right_finger_ids = tuple(
        _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in RIGHT_FINGER_COLLISION_GEOMS
    )
    right_arm_act_ids = tuple(
        _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in RIGHT_ARM_ACTUATORS
    )
    left_arm_act_ids = tuple(
        _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in LEFT_ARM_ACTUATORS
    )
    right_grip_act_id = _require_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_GRIPPER_ACTUATOR
    )
    left_grip_act_id = _require_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_GRIPPER_ACTUATOR
    )
    right_arm_qpos = tuple(
        int(
            model.jnt_qposadr[
                _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_right_joint{i}")
            ]
        )
        for i in range(1, 8)
    )
    left_arm_qpos = tuple(
        int(
            model.jnt_qposadr[
                _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_left_joint{i}")
            ]
        )
        for i in range(1, 8)
    )
    rf1_qpos = int(
        model.jnt_qposadr[
            _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_FINGER_JOINTS[0])
        ]
    )
    rf2_qpos = int(
        model.jnt_qposadr[
            _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_FINGER_JOINTS[1])
        ]
    )
    mimic_eq_id = _require_id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, RIGHT_FINGER_MIMIC_EQ
    )
    support_geom_id = _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, SUPPORT_GEOM_NAME)
    box_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY_NAME)
    box_geom_id = _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, BOX_GEOM_NAME)
    box_joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, BOX_JOINT_NAME)
    floor_geom_id = _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    non_robot_geoms = {support_geom_id, box_geom_id, floor_geom_id}
    robot_collision_geoms = frozenset(
        gid
        for gid in range(model.ngeom)
        if gid not in non_robot_geoms and int(model.geom_contype[gid]) > 0
    )

    contract = SceneContract(
        pedestal_xml_path=str(pedestal_xml),
        pedestal_xml_sha256=xml_sha256,
        camera_name=CAMERA_WRIST_RIGHT,
        camera_id=camera_id,
        right_ee_site_id=right_ee_site_id,
        left_finger_geom_ids=left_finger_ids,
        right_finger_geom_ids=right_finger_ids,
        left_finger_geom_names=LEFT_FINGER_COLLISION_GEOMS,
        right_finger_geom_names=RIGHT_FINGER_COLLISION_GEOMS,
        robot_collision_geom_ids=robot_collision_geoms,
        right_arm_actuator_ids=right_arm_act_ids,
        left_arm_actuator_ids=left_arm_act_ids,
        right_gripper_actuator_id=right_grip_act_id,
        left_gripper_actuator_id=left_grip_act_id,
        right_arm_qpos_indices=right_arm_qpos,
        left_arm_qpos_indices=left_arm_qpos,
        right_finger_qpos_indices=(rf1_qpos, rf2_qpos),
        right_finger_mimic_eq_id=mimic_eq_id,
        support_geom_id=support_geom_id,
        box_body_id=box_body_id,
        box_geom_id=box_geom_id,
        box_joint_id=box_joint_id,
        box_qpos_adr=int(model.jnt_qposadr[box_joint_id]),
        box_dof_adr=int(model.jnt_dofadr[box_joint_id]),
        floor_geom_id=floor_geom_id,
        head_camera=head_camera,
    )
    verify_scene_contract(model, contract)
    return model, contract


def verify_scene_contract(
    model: mujoco.MjModel, contract: SceneContract
) -> dict[str, Any]:
    """Verify named finger collision geoms, actuation, camera, and contact options."""
    if model.opt.cone != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        raise ValueError(
            "Evaluation scene requires elliptic friction cone for grasping."
        )
    if float(model.opt.impratio) < 10.0:
        raise ValueError(f"Expected impratio >= 10, got {model.opt.impratio}")
    if not np.isclose(float(model.opt.timestep), 0.001, atol=1e-6):
        raise ValueError(f"Expected 0.001 s timestep, got {model.opt.timestep}")

    for gid in (*contract.left_finger_geom_ids, *contract.right_finger_geom_ids):
        if int(model.geom_contype[gid]) == 0 or int(model.geom_conaffinity[gid]) == 0:
            raise ValueError(
                f"Finger geom {model.geom(gid).name} has collisions disabled."
            )
        if int(model.geom_condim[gid]) < 4:
            raise ValueError(
                f"Finger geom {model.geom(gid).name} requires condim>=4 for torsional grasp friction."
            )

    for gid in (contract.box_geom_id, contract.support_geom_id):
        if int(model.geom_contype[gid]) == 0 or int(model.geom_conaffinity[gid]) == 0:
            raise ValueError(
                f"Scene geom {model.geom(gid).name} has collisions disabled."
            )

    ctrl_range = model.actuator_ctrlrange[contract.right_gripper_actuator_id]
    if not (ctrl_range[0] < -0.5 and np.isclose(ctrl_range[1], 0.0, atol=1e-4)):
        raise ValueError(f"Unexpected right gripper ctrlrange: {ctrl_range.tolist()}")

    return {
        "camera_name": model.cam(contract.camera_id).name,
        "camera_fovy_deg": float(model.cam_fovy[contract.camera_id]),
        "left_finger_collision_geoms": list(contract.left_finger_geom_names),
        "right_finger_collision_geoms": list(contract.right_finger_geom_names),
        "left_finger_note": "openarm_right_ee_inner_finger (+Y, left jaw in camera_wrist_right view)",
        "right_finger_note": "openarm_right_ee_outer_finger (-Y, right jaw in camera_wrist_right view)",
        "right_arm_actuators": list(RIGHT_ARM_ACTUATORS),
        "gripper_actuator": model.actuator(contract.right_gripper_actuator_id).name,
        "gripper_ctrlrange_rad": [float(ctrl_range[0]), float(ctrl_range[1])],
        "mimic_equality": model.eq(contract.right_finger_mimic_eq_id).name,
        "target_box_geom": model.geom(contract.box_geom_id).name,
        "support_geom": model.geom(contract.support_geom_id).name,
        "box_mass_kg": float(model.body_mass[contract.box_body_id]),
        "cone": "elliptic",
        "impratio": float(model.opt.impratio),
        "timestep": float(model.opt.timestep),
    }


def _yaw_to_quat_wxyz(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


class RightArmController:
    """FK/IK solver and gravity-compensated position servo helper for OpenArm v2."""

    def __init__(
        self,
        model: mujoco.MjModel,
        contract: SceneContract,
        open_finger_rad: float = SceneConfig().open_finger_rad,
    ) -> None:
        self.eval_model = model
        self.contract = contract
        self.open_finger_rad = float(open_finger_rad)
        self.setup = ArmSetup.from_args(
            xml=contract.pedestal_xml_path,
            mode="bimanual",
            frame_right=RIGHT_EE_SITE,
            frame_type_right="site",
            frame_left=LEFT_EE_SITE,
            frame_type_left="site",
            keyframe="home",
            origin_frame="world",
            origin_frame_type="world",
        )
        self.fk = Kinematics(self.setup)
        self.ik = Kinematics(self.setup, IKParams(dt=0.01, max_iters=5))
        self.arm_dofs_right = self.setup.joint_resolver.arm_dof_indices("right")
        self.arm_dofs_left = self.setup.joint_resolver.arm_dof_indices("left")
        self.arm_kp_right = np.array(
            [
                self.setup.model.actuator(name).gainprm[0]
                for name in RIGHT_ARM_ACTUATORS
            ],
            dtype=np.float64,
        )
        self.arm_kp_left = np.array(
            [self.setup.model.actuator(name).gainprm[0] for name in LEFT_ARM_ACTUATORS],
            dtype=np.float64,
        )
        self.limits_right = np.array(
            [self.setup.model.actuator(name).ctrlrange for name in RIGHT_ARM_ACTUATORS],
            dtype=np.float64,
        )
        self.limits_left = np.array(
            [self.setup.model.actuator(name).ctrlrange for name in LEFT_ARM_ACTUATORS],
            dtype=np.float64,
        )
        self.home_q = np.array(
            [0.0, 0.0, 0.0, 1.570796, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        home_r, home_l = self.fk.fk_bimanual(self.home_q, self.home_q)
        self.home_pose_right = np.asarray(home_r, dtype=np.float64)
        self.home_pose_left = np.asarray(home_l, dtype=np.float64)

        # Measure nominal pinch-center offset from right_ee_control_point at home pose
        # with the configured open gripper angle.
        probe = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, probe, 0)
        rf1, rf2 = contract.right_finger_qpos_indices
        probe.qpos[rf1] = self.open_finger_rad
        probe.qpos[rf2] = self.open_finger_rad
        mujoco.mj_forward(model, probe)
        p_left = probe.geom_xpos[contract.left_finger_geom_ids[0]]
        p_right = probe.geom_xpos[contract.right_finger_geom_ids[0]]
        self.pinch_offset_world_at_home = (
            0.5 * (p_left + p_right) - probe.site_xpos[contract.right_ee_site_id]
        ).astype(np.float64)

    def nominal_pregrasp_pose(
        self,
        box_pos: np.ndarray,
        hand_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        pinch_z_above_box_center_m: float = 0.018,
    ) -> np.ndarray:
        """Compute a world-frame right-EE target pose [x, y, z, qw, qx, qy, qz]."""
        pose = self.home_pose_right.copy()
        pinch_target = np.asarray(box_pos, dtype=np.float64) + np.array(
            [0.0, 0.0, float(pinch_z_above_box_center_m)], dtype=np.float64
        )
        pose[:3] = (
            pinch_target
            - self.pinch_offset_world_at_home
            + np.asarray(hand_offset_xyz, dtype=np.float64)
        )
        return pose

    def read_right_arm_qpos(self, data: mujoco.MjData) -> np.ndarray:
        """Read current 7-DOF right arm joint positions."""
        return np.array(
            [float(data.qpos[idx]) for idx in self.contract.right_arm_qpos_indices],
            dtype=np.float64,
        )

    def read_left_arm_qpos(self, data: mujoco.MjData) -> np.ndarray:
        """Read current 7-DOF left arm joint positions."""
        return np.array(
            [float(data.qpos[idx]) for idx in self.contract.left_arm_qpos_indices],
            dtype=np.float64,
        )

    def read_right_ee_pose_world(self, data: mujoco.MjData) -> np.ndarray:
        """Read current right_ee_control_point world pose [x, y, z, qw, qx, qy, qz]."""
        pos = data.site_xpos[self.contract.right_ee_site_id].copy()
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, data.site_xmat[self.contract.right_ee_site_id])
        return np.concatenate([pos, quat])

    def gravity_offsets(
        self, q_right: np.ndarray, q_left: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute position-servo gravity compensation offsets (qfrc_bias / kp)."""
        self.fk.fk_bimanual(np.r_[q_right, 0.0], np.r_[q_left, 0.0])
        g_right = self.fk.setup.data.qfrc_bias[self.arm_dofs_right] / self.arm_kp_right
        g_left = self.fk.setup.data.qfrc_bias[self.arm_dofs_left] / self.arm_kp_left
        return g_right.astype(np.float64), g_left.astype(np.float64)

    def apply_position_controls(
        self,
        data: mujoco.MjData,
        q_right_target: np.ndarray,
        right_finger_target: float,
        q_left_target: np.ndarray | None = None,
        left_finger_target: float = 0.0,
    ) -> None:
        """Write gravity-compensated position targets into ``data.ctrl``."""
        q_l_cmd = self.home_q[:7] if q_left_target is None else q_left_target
        g_r, g_l = self.gravity_offsets(
            self.read_right_arm_qpos(data), self.read_left_arm_qpos(data)
        )
        ctrl_r = np.clip(
            q_right_target + g_r, self.limits_right[:, 0], self.limits_right[:, 1]
        )
        ctrl_l = np.clip(q_l_cmd + g_l, self.limits_left[:, 0], self.limits_left[:, 1])
        for act_id, val in zip(
            self.contract.right_arm_actuator_ids, ctrl_r, strict=True
        ):
            data.ctrl[act_id] = float(val)
        for act_id, val in zip(
            self.contract.left_arm_actuator_ids, ctrl_l, strict=True
        ):
            data.ctrl[act_id] = float(val)
        r_grip_range = self.eval_model.actuator_ctrlrange[
            self.contract.right_gripper_actuator_id
        ]
        l_grip_range = self.eval_model.actuator_ctrlrange[
            self.contract.left_gripper_actuator_id
        ]
        data.ctrl[self.contract.right_gripper_actuator_id] = float(
            np.clip(right_finger_target, r_grip_range[0], r_grip_range[1])
        )
        data.ctrl[self.contract.left_gripper_actuator_id] = float(
            np.clip(left_finger_target, l_grip_range[0], l_grip_range[1])
        )

    def solve_right_ik(
        self,
        target_pose_world: np.ndarray,
        start_q_right: np.ndarray | None = None,
        steps: int = 100,
    ) -> np.ndarray | None:
        """Converge IK to a static right-arm target pose in world coordinates."""
        q_start = (
            self.home_q if start_q_right is None else np.r_[start_q_right[:7], 0.0]
        )
        self.ik.sync(np.concatenate([q_start, self.home_q]).astype(np.float32))
        self.ik.set_gripper("right", 0.0)
        self.ik.set_gripper("left", 0.0)
        sol = None
        for _ in range(steps):
            self.ik.set_target("right", np.asarray(target_pose_world, dtype=np.float32))
            self.ik.set_target("left", self.home_pose_left.astype(np.float32))
            sol = self.ik.solve()
            if sol is None or not np.all(np.isfinite(sol)):
                return None
        return None if sol is None else np.asarray(sol[:7], dtype=np.float64)

    def sync_ik_from_data(self, data: mujoco.MjData) -> None:
        """Synchronize the incremental IK solver from current simulator joint state."""
        q_r = np.r_[self.read_right_arm_qpos(data), 0.0]
        q_l = np.r_[self.read_left_arm_qpos(data), 0.0]
        self.ik.sync(np.concatenate([q_r, q_l]).astype(np.float32))
        self.ik.set_gripper("right", 0.0)
        self.ik.set_gripper("left", 0.0)

    def step_right_ik(self, target_pose_world: np.ndarray) -> np.ndarray | None:
        """Advance the incremental IK solver by one 10 ms tick."""
        self.ik.set_target("right", np.asarray(target_pose_world, dtype=np.float32))
        self.ik.set_target("left", self.home_pose_left.astype(np.float32))
        sol = self.ik.solve()
        if sol is None or not np.all(np.isfinite(sol)):
            return None
        return np.asarray(sol[:7], dtype=np.float64)


def prepare_settled_state(
    model: mujoco.MjModel,
    contract: SceneContract,
    controller: RightArmController,
    *,
    box_pos: np.ndarray,
    box_yaw_rad: float,
    target_hand_pose: np.ndarray,
    open_finger_rad: float = -0.45,
    settle_duration_s: float = 0.25,
) -> mujoco.MjData:
    """Initialize box and right arm at a pre-grasp pose and settle under physics."""
    q_init = controller.solve_right_ik(target_hand_pose)
    if q_init is None:
        q_init = controller.home_q[:7].copy()

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for idx, val in zip(contract.right_arm_qpos_indices, q_init, strict=True):
        data.qpos[idx] = float(val)
    rf1, rf2 = contract.right_finger_qpos_indices
    data.qpos[rf1] = float(open_finger_rad)
    data.qpos[rf2] = float(open_finger_rad)

    qadr = contract.box_qpos_adr
    data.qpos[qadr : qadr + 3] = np.asarray(box_pos, dtype=np.float64)
    data.qpos[qadr + 3 : qadr + 7] = _yaw_to_quat_wxyz(box_yaw_rad)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    n_steps = max(1, round(settle_duration_s / float(model.opt.timestep)))
    for _ in range(n_steps):
        controller.apply_position_controls(
            data,
            q_right_target=q_init,
            right_finger_target=open_finger_rad,
        )
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return data
