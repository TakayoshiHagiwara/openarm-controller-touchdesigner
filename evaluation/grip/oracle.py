"""Genuine force/contact grasp-and-lift physics oracle.

Evaluates a captured MuJoCo integration state by restoring it, calling
``mj_forward``, closing the right gripper via ``right_finger1_ctrl`` while
holding arm pose, lifting through right-arm IK + joint position actuation, and
holding. Never uses welds, teleports, or grasp-distance heuristics, and never
classifies invalid scenes or failed robot motion as physics negatives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import mujoco
import numpy as np

from evaluation.grip.scene import RightArmController, SceneContract
from evaluation.grip.snapshot import StateSnapshot, restore_integration_state

OutcomeType = Literal["physics_success", "physics_failure", "invalid"]


@dataclass(frozen=True)
class OracleProfile:
    """Close-lift-hold actuation profile and validity/success thresholds."""

    close_ramp_s: float = 0.35
    close_settle_s: float = 0.15
    close_finger_target_rad: float = 0.0
    lift_height_m: float = 0.040
    lift_duration_s: float = 0.40
    hold_duration_s: float = 0.40
    control_hz: int = 100

    # Initial validity thresholds
    min_initial_support_force_n: float = 0.05
    max_initial_penetration_m: float = 0.005
    max_initial_box_linvel_mps: float = 0.03
    max_initial_box_angvel_radps: float = 0.30
    max_initial_box_tilt_deg: float = 10.0
    max_initial_hand_linvel_mps: float = 0.03
    max_initial_hand_pos_err_m: float = 0.015
    max_initial_hand_rot_err_deg: float = 8.0

    # Robot motion tracking validity thresholds (failure => invalid, NOT negative)
    max_close_hand_pos_err_m: float = 0.015
    max_close_hand_rot_err_deg: float = 10.0
    max_lift_rise_deficit_m: float = 0.010
    max_lift_end_pos_err_m: float = 0.015
    max_lift_end_rot_err_deg: float = 10.0
    max_hold_hand_pos_err_m: float = 0.015
    max_hold_hand_rot_err_deg: float = 10.0
    max_gripper_mimic_err_rad: float = 0.05
    # The unloaded servo is about 0.054 rad short at the end of close settle.
    max_unloaded_close_err_rad: float = 0.075

    # Physics grasp success thresholds (pure contact + rise + retention)
    min_finger_contact_force_n: float = 0.10
    min_support_contact_force_n: float = 0.05
    min_object_rise_m: float = 0.025
    min_support_clearance_m: float = 0.015
    max_object_drift_rel_hand_m: float = 0.025
    min_hold_bilateral_fraction: float = 0.80

    rationale: str = (
        "The 40 mm lift and 0.40 s hold leave clearance for verifying retention "
        "after the object leaves the support. The drive and contact solver are "
        "held fixed across trials."
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize oracle profile to plain JSON-compatible dict."""
        return asdict(self)


@dataclass(frozen=True)
class OracleEvidence:
    """Structured tri-state oracle outcome and full physical evidence."""

    outcome: OutcomeType
    label: int | None
    reasons: list[str]
    initial_validity: dict[str, Any]
    contacts: dict[str, Any]
    object_motion: dict[str, Any]
    hand_tracking: dict[str, Any]
    profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize oracle evidence to JSON-compatible dict."""
        return asdict(self)


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def _box_tilt_deg(data: mujoco.MjData, contract: SceneContract) -> float:
    rot = data.xmat[contract.box_body_id].reshape(3, 3)
    cos_tilt = float(np.clip(rot[2, 2], -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_tilt)))


def _site_linear_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, site_id: int
) -> np.ndarray:
    vel6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, vel6, 0)
    # mj_objectVelocity returns [angular(3), linear(3)]
    return vel6[3:6].copy()


def _box_in_hand_frame(data: mujoco.MjData, contract: SceneContract) -> np.ndarray:
    hand_pos = data.site_xpos[contract.right_ee_site_id]
    hand_rot = data.site_xmat[contract.right_ee_site_id].reshape(3, 3)
    box_pos = data.qpos[contract.box_qpos_adr : contract.box_qpos_adr + 3]
    return (hand_rot.T @ (box_pos - hand_pos)).astype(np.float64)


def _box_clearance(
    model: mujoco.MjModel, data: mujoco.MjData, contract: SceneContract
) -> float:
    """Measure the oriented box's lowest point above the top of the support."""
    box_id = contract.box_geom_id
    support_id = contract.support_geom_id
    box_z_axis = data.geom_xmat[box_id].reshape(3, 3)[2]
    half_height = np.abs(box_z_axis) @ model.geom_size[box_id]
    support_top = data.geom_xpos[support_id, 2] + model.geom_size[support_id, 2]
    return float(data.geom_xpos[box_id, 2] - half_height - support_top)


def inspect_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contract: SceneContract,
    *,
    finger_force_threshold_n: float = 0.10,
    support_force_threshold_n: float = 0.05,
) -> dict[str, Any]:
    """Inspect active MuJoCo contacts and normal forces on the target box."""
    left_ids = set(contract.left_finger_geom_ids)
    right_ids = set(contract.right_finger_geom_ids)
    box_id = contract.box_geom_id
    sup_id = contract.support_geom_id

    left_geoms: set[str] = set()
    right_geoms: set[str] = set()
    other_robot_geoms: set[str] = set()
    left_max_f = 0.0
    right_max_f = 0.0
    support_max_f = 0.0
    wrench = np.zeros(6, dtype=np.float64)

    for i in range(int(data.ncon)):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if box_id not in (g1, g2):
            continue
        other = g2 if g1 == box_id else g1
        mujoco.mj_contactForce(model, data, i, wrench)
        fn = float(wrench[0])
        if other in left_ids:
            left_max_f = max(left_max_f, fn)
            if fn >= finger_force_threshold_n:
                left_geoms.add(model.geom(other).name)
        elif other in right_ids:
            right_max_f = max(right_max_f, fn)
            if fn >= finger_force_threshold_n:
                right_geoms.add(model.geom(other).name)
        elif other == sup_id:
            support_max_f = max(support_max_f, fn)
        elif (
            other in contract.robot_collision_geom_ids
            and fn >= support_force_threshold_n
        ):
            other_robot_geoms.add(model.geom(other).name)

    left_contact = left_max_f >= finger_force_threshold_n
    right_contact = right_max_f >= finger_force_threshold_n
    return {
        "left_finger_contact": bool(left_contact),
        "left_finger_max_normal_force_n": round(left_max_f, 4),
        "left_finger_geoms": sorted(left_geoms),
        "right_finger_contact": bool(right_contact),
        "right_finger_max_normal_force_n": round(right_max_f, 4),
        "right_finger_geoms": sorted(right_geoms),
        "bilateral_contact": bool(left_contact and right_contact),
        "support_contact": bool(support_max_f >= support_force_threshold_n),
        "support_max_normal_force_n": round(support_max_f, 4),
        "other_robot_box_geoms": sorted(other_robot_geoms),
    }


def check_initial_validity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contract: SceneContract,
    controller: RightArmController,
    profile: OracleProfile,
    expected_hand_pose: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Verify the captured scene is physically valid before executing the grasp oracle."""
    box_id = contract.box_geom_id
    sup_id = contract.support_geom_id
    robot_ids = contract.robot_collision_geom_ids

    robot_box_pairs: list[str] = []
    robot_support_pairs: list[str] = []
    support_max_f = 0.0
    max_penetration = 0.0
    wrench = np.zeros(6, dtype=np.float64)

    for i in range(int(data.ncon)):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        pair = {g1, g2}
        if box_id in pair or sup_id in pair:
            max_penetration = max(max_penetration, max(0.0, -float(con.dist)))
        if pair == {box_id, sup_id}:
            mujoco.mj_contactForce(model, data, i, wrench)
            support_max_f = max(support_max_f, float(wrench[0]))
        elif box_id in pair:
            other = g2 if g1 == box_id else g1
            if other in robot_ids:
                robot_box_pairs.append(model.geom(other).name)
        elif sup_id in pair:
            other = g2 if g1 == sup_id else g1
            if other in robot_ids:
                robot_support_pairs.append(model.geom(other).name)

    dadr = contract.box_dof_adr
    box_linvel = float(np.linalg.norm(data.qvel[dadr : dadr + 3]))
    box_angvel = float(np.linalg.norm(data.qvel[dadr + 3 : dadr + 6]))
    box_tilt = _box_tilt_deg(data, contract)
    box_z = float(data.qpos[contract.box_qpos_adr + 2])
    hand_linvel = float(
        np.linalg.norm(_site_linear_velocity(model, data, contract.right_ee_site_id))
    )

    hand_pos_err: float | None = None
    hand_rot_err: float | None = None
    if expected_hand_pose is not None:
        actual_pose = controller.read_right_ee_pose_world(data)
        hand_pos_err = float(np.linalg.norm(actual_pose[:3] - expected_hand_pose[:3]))
        hand_rot_err = _quat_angle_deg(actual_pose[3:7], expected_hand_pose[3:7])

    reasons: list[str] = []
    box_on_support = support_max_f >= profile.min_initial_support_force_n
    if not box_on_support:
        reasons.append("initial_box_not_on_support")
    if robot_box_pairs:
        reasons.append("initial_robot_box_contact")
    if robot_support_pairs:
        reasons.append("initial_robot_support_contact")
    if max_penetration > profile.max_initial_penetration_m:
        reasons.append("initial_excessive_penetration")
    if box_linvel > profile.max_initial_box_linvel_mps:
        reasons.append("initial_box_not_settled_linvel")
    if box_angvel > profile.max_initial_box_angvel_radps:
        reasons.append("initial_box_not_settled_angvel")
    if box_tilt > profile.max_initial_box_tilt_deg:
        reasons.append("initial_box_tilted")
    if hand_linvel > profile.max_initial_hand_linvel_mps:
        reasons.append("initial_hand_not_settled")
    if hand_pos_err is not None and hand_pos_err > profile.max_initial_hand_pos_err_m:
        reasons.append("initial_hand_pose_position_error")
    if hand_rot_err is not None and hand_rot_err > profile.max_initial_hand_rot_err_deg:
        reasons.append("initial_hand_pose_rotation_error")

    validity = {
        "valid": len(reasons) == 0,
        "box_on_support": bool(box_on_support),
        "box_support_normal_force_n": round(support_max_f, 4),
        "robot_box_contact": bool(robot_box_pairs),
        "robot_box_contact_geoms": sorted(set(robot_box_pairs)),
        "robot_support_contact": bool(robot_support_pairs),
        "robot_support_contact_geoms": sorted(set(robot_support_pairs)),
        "max_penetration_m": round(max_penetration, 6),
        "box_z_m": round(box_z, 5),
        "box_tilt_deg": round(box_tilt, 3),
        "box_linvel_norm_mps": round(box_linvel, 5),
        "box_angvel_norm_radps": round(box_angvel, 5),
        "hand_linvel_norm_mps": round(hand_linvel, 5),
        "hand_pose_err_m": None if hand_pos_err is None else round(hand_pos_err, 5),
        "hand_rot_err_deg": None if hand_rot_err is None else round(hand_rot_err, 3),
    }
    return validity, reasons


def evaluate_snapshot(
    model: mujoco.MjModel,
    contract: SceneContract,
    controller: RightArmController,
    snapshot: StateSnapshot,
    profile: OracleProfile | None = None,
    *,
    expected_hand_pose: np.ndarray | None = None,
) -> OracleEvidence:
    """Restore full integration state and run genuine contact close-lift-hold evaluation."""
    prof = profile or OracleProfile()
    data = restore_integration_state(model, snapshot)

    initial_validity, init_reasons = check_initial_validity(
        model, data, contract, controller, prof, expected_hand_pose=expected_hand_pose
    )
    box_pos_init = data.qpos[contract.box_qpos_adr : contract.box_qpos_adr + 3].copy()
    hand_pose_init = controller.read_right_ee_pose_world(data)
    initial_contacts = inspect_contacts(
        model,
        data,
        contract,
        finger_force_threshold_n=prof.min_finger_contact_force_n,
        support_force_threshold_n=prof.min_support_contact_force_n,
    )

    if not initial_validity["valid"]:
        return OracleEvidence(
            outcome="invalid",
            label=None,
            reasons=init_reasons,
            initial_validity=initial_validity,
            contacts={"initial": initial_contacts},
            object_motion={"initial_pos_m": [round(float(v), 5) for v in box_pos_init]},
            hand_tracking={
                "tracking_valid": False,
                "skipped_due_to_invalid_initial": True,
            },
            profile=prof.to_dict(),
        )

    dt = float(model.opt.timestep)
    substeps_per_tick = max(1, round((1.0 / prof.control_hz) / dt))
    q_hold = controller.read_right_arm_qpos(data)
    q_left_hold = controller.read_left_arm_qpos(data)
    rf1, rf2 = contract.right_finger_qpos_indices
    finger_start_rad = float(data.qpos[rf1])

    # Phase 1: Close gripper while holding captured arm pose.
    close_ramp_steps = max(1, round(prof.close_ramp_s / dt))
    close_settle_steps = max(1, round(prof.close_settle_s / dt))
    close_max_pos_err = 0.0
    close_max_rot_err = 0.0
    for step in range(close_ramp_steps + close_settle_steps):
        alpha = min(1.0, (step + 1) / float(close_ramp_steps))
        finger_cmd = finger_start_rad + alpha * (
            prof.close_finger_target_rad - finger_start_rad
        )
        controller.apply_position_controls(
            data,
            q_right_target=q_hold,
            right_finger_target=finger_cmd,
            q_left_target=q_left_hold,
        )
        mujoco.mj_step(model, data)
        if (step + 1) % substeps_per_tick == 0:
            cur_pose = controller.read_right_ee_pose_world(data)
            close_max_pos_err = max(
                close_max_pos_err,
                float(np.linalg.norm(cur_pose[:3] - hand_pose_init[:3])),
            )
            close_max_rot_err = max(
                close_max_rot_err, _quat_angle_deg(cur_pose[3:7], hand_pose_init[3:7])
            )

    mujoco.mj_forward(model, data)
    contacts_after_close = inspect_contacts(
        model,
        data,
        contract,
        finger_force_threshold_n=prof.min_finger_contact_force_n,
        support_force_threshold_n=prof.min_support_contact_force_n,
    )
    finger_after_close_rad = float(data.qpos[rf1])
    finger_close_error = abs(finger_after_close_rad - prof.close_finger_target_rad)
    box_in_jaws_after_close = (
        contacts_after_close["left_finger_contact"]
        or contacts_after_close["right_finger_contact"]
    )
    box_pos_after_close = data.qpos[
        contract.box_qpos_adr : contract.box_qpos_adr + 3
    ].copy()
    box_rel_hand_after_close = _box_in_hand_frame(data, contract)

    # Phase 2: Lift through right-arm IK + joint position actuation.
    lift_ticks = max(1, round(prof.lift_duration_s * prof.control_hz))
    lift_target_pose = hand_pose_init.copy()
    lift_target_pose[2] += float(prof.lift_height_m)
    controller.sync_ik_from_data(data)
    q_cmd = controller.read_right_arm_qpos(data)
    ik_failures = 0

    for tick in range(lift_ticks):
        alpha = (tick + 1) / float(lift_ticks)
        waypoint = hand_pose_init.copy()
        waypoint[2] += float(prof.lift_height_m) * alpha
        sol = controller.step_right_ik(waypoint)
        if sol is None:
            ik_failures += 1
        else:
            q_cmd = sol
        for _ in range(substeps_per_tick):
            controller.apply_position_controls(
                data,
                q_right_target=q_cmd,
                right_finger_target=prof.close_finger_target_rad,
                q_left_target=q_left_hold,
            )
            mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    contacts_after_lift = inspect_contacts(
        model,
        data,
        contract,
        finger_force_threshold_n=prof.min_finger_contact_force_n,
        support_force_threshold_n=prof.min_support_contact_force_n,
    )
    hand_pose_after_lift = controller.read_right_ee_pose_world(data)
    box_pos_after_lift = data.qpos[
        contract.box_qpos_adr : contract.box_qpos_adr + 3
    ].copy()
    lift_end_pos_err = float(
        np.linalg.norm(hand_pose_after_lift[:3] - lift_target_pose[:3])
    )
    lift_end_rot_err = _quat_angle_deg(hand_pose_after_lift[3:7], lift_target_pose[3:7])

    # Phase 3: Hold lifted pose and monitor sustained contact & retention.
    hold_ticks = max(1, round(prof.hold_duration_s * prof.control_hz))
    hold_bilateral_hits = 0
    hold_max_pos_err = 0.0
    hold_max_rot_err = 0.0
    min_box_z_hold = float(box_pos_after_lift[2])
    min_box_clearance_hold = _box_clearance(model, data, contract)

    for _ in range(hold_ticks):
        for _ in range(substeps_per_tick):
            controller.apply_position_controls(
                data,
                q_right_target=q_cmd,
                right_finger_target=prof.close_finger_target_rad,
                q_left_target=q_left_hold,
            )
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        tick_contacts = inspect_contacts(
            model,
            data,
            contract,
            finger_force_threshold_n=prof.min_finger_contact_force_n,
            support_force_threshold_n=prof.min_support_contact_force_n,
        )
        if tick_contacts["bilateral_contact"]:
            hold_bilateral_hits += 1
        min_box_z_hold = min(
            min_box_z_hold, float(data.qpos[contract.box_qpos_adr + 2])
        )
        min_box_clearance_hold = min(
            min_box_clearance_hold, _box_clearance(model, data, contract)
        )
        cur_pose = controller.read_right_ee_pose_world(data)
        hold_max_pos_err = max(
            hold_max_pos_err, float(np.linalg.norm(cur_pose[:3] - lift_target_pose[:3]))
        )
        hold_max_rot_err = max(
            hold_max_rot_err, _quat_angle_deg(cur_pose[3:7], lift_target_pose[3:7])
        )

    contacts_after_hold = inspect_contacts(
        model,
        data,
        contract,
        finger_force_threshold_n=prof.min_finger_contact_force_n,
        support_force_threshold_n=prof.min_support_contact_force_n,
    )
    hand_pose_after_hold = controller.read_right_ee_pose_world(data)
    box_pos_after_hold = data.qpos[
        contract.box_qpos_adr : contract.box_qpos_adr + 3
    ].copy()
    box_rel_hand_after_hold = _box_in_hand_frame(data, contract)

    actual_hand_rise = float(hand_pose_after_hold[2] - hand_pose_init[2])
    gripper_mimic_err = float(abs(data.qpos[rf1] - data.qpos[rf2]))
    hold_bilateral_fraction = hold_bilateral_hits / float(hold_ticks)

    object_rise = float(box_pos_after_hold[2] - box_pos_init[2])
    object_drift_rel_hand = float(
        np.linalg.norm(box_rel_hand_after_hold - box_rel_hand_after_close)
    )
    object_support_clearance = _box_clearance(model, data, contract)

    # Evaluate robot motion tracking validity first (failure => invalid, NOT negative).
    tracking_reasons: list[str] = []
    if ik_failures > 0:
        tracking_reasons.append("lift_ik_failure")
    if close_max_pos_err > prof.max_close_hand_pos_err_m:
        tracking_reasons.append("close_hand_tracking_position_error")
    if close_max_rot_err > prof.max_close_hand_rot_err_deg:
        tracking_reasons.append("close_hand_tracking_rotation_error")
    if (
        finger_close_error > prof.max_unloaded_close_err_rad
        and not box_in_jaws_after_close
    ):
        tracking_reasons.append("gripper_close_tracking_error")
    if actual_hand_rise < prof.lift_height_m - prof.max_lift_rise_deficit_m:
        tracking_reasons.append("lift_hand_tracking_insufficient_rise")
    if lift_end_pos_err > prof.max_lift_end_pos_err_m:
        tracking_reasons.append("lift_end_hand_tracking_position_error")
    if lift_end_rot_err > prof.max_lift_end_rot_err_deg:
        tracking_reasons.append("lift_end_hand_tracking_rotation_error")
    if hold_max_pos_err > prof.max_hold_hand_pos_err_m:
        tracking_reasons.append("hold_hand_tracking_position_error")
    if hold_max_rot_err > prof.max_hold_hand_rot_err_deg:
        tracking_reasons.append("hold_hand_tracking_rotation_error")
    if gripper_mimic_err > prof.max_gripper_mimic_err_rad:
        tracking_reasons.append("gripper_mimic_tracking_error")

    tracking_valid = len(tracking_reasons) == 0
    hand_tracking = {
        "tracking_valid": tracking_valid,
        "ik_failures": int(ik_failures),
        "close_phase_max_pos_err_m": round(close_max_pos_err, 5),
        "close_phase_max_rot_err_deg": round(close_max_rot_err, 3),
        "lift_target_rise_m": round(float(prof.lift_height_m), 5),
        "actual_hand_rise_m": round(actual_hand_rise, 5),
        "lift_end_pos_err_m": round(lift_end_pos_err, 5),
        "lift_end_rot_err_deg": round(lift_end_rot_err, 3),
        "hold_max_pos_err_m": round(hold_max_pos_err, 5),
        "hold_max_rot_err_deg": round(hold_max_rot_err, 3),
        "gripper_start_qpos_rad": round(finger_start_rad, 5),
        "gripper_commanded_close_rad": round(float(prof.close_finger_target_rad), 5),
        "gripper_after_close_qpos_rad": round(finger_after_close_rad, 5),
        "gripper_final_qpos_rad": round(float(data.qpos[rf1]), 5),
        "gripper_mimic_err_rad": round(gripper_mimic_err, 5),
    }
    object_motion = {
        "initial_pos_m": [round(float(v), 5) for v in box_pos_init],
        "after_close_pos_m": [round(float(v), 5) for v in box_pos_after_close],
        "after_lift_pos_m": [round(float(v), 5) for v in box_pos_after_lift],
        "after_hold_pos_m": [round(float(v), 5) for v in box_pos_after_hold],
        "object_rise_m": round(object_rise, 5),
        "object_min_z_during_hold_m": round(min_box_z_hold, 5),
        "object_drift_rel_hand_m": round(object_drift_rel_hand, 5),
        "object_support_clearance_m": round(object_support_clearance, 5),
        "object_min_support_clearance_during_hold_m": round(min_box_clearance_hold, 5),
    }
    contacts = {
        "initial": initial_contacts,
        "after_close": contacts_after_close,
        "after_lift": contacts_after_lift,
        "after_hold": contacts_after_hold,
        "hold_bilateral_fraction": round(hold_bilateral_fraction, 4),
    }

    if not tracking_valid:
        return OracleEvidence(
            outcome="invalid",
            label=None,
            reasons=tracking_reasons,
            initial_validity=initial_validity,
            contacts=contacts,
            object_motion=object_motion,
            hand_tracking=hand_tracking,
            profile=prof.to_dict(),
        )

    # Pure physics grasp verdict: rise + clearance + sustained bilateral finger contact + retention.
    failure_reasons: list[str] = []
    if object_rise < prof.min_object_rise_m:
        failure_reasons.append("insufficient_object_rise")
    if min_box_clearance_hold < prof.min_support_clearance_m:
        failure_reasons.append("insufficient_support_clearance")
    if contacts_after_hold["support_contact"]:
        failure_reasons.append("still_in_support_contact")
    if not contacts_after_hold["left_finger_contact"]:
        failure_reasons.append("missing_left_finger_contact")
    if not contacts_after_hold["right_finger_contact"]:
        failure_reasons.append("missing_right_finger_contact")
    if hold_bilateral_fraction < prof.min_hold_bilateral_fraction:
        failure_reasons.append("unsustained_bilateral_contact_during_hold")
    if object_drift_rel_hand > prof.max_object_drift_rel_hand_m:
        failure_reasons.append("excessive_object_drift_relative_to_hand")

    if not failure_reasons:
        return OracleEvidence(
            outcome="physics_success",
            label=1,
            reasons=[],
            initial_validity=initial_validity,
            contacts=contacts,
            object_motion=object_motion,
            hand_tracking=hand_tracking,
            profile=prof.to_dict(),
        )
    return OracleEvidence(
        outcome="physics_failure",
        label=0,
        reasons=failure_reasons,
        initial_validity=initial_validity,
        contacts=contacts,
        object_motion=object_motion,
        hand_tracking=hand_tracking,
        profile=prof.to_dict(),
    )
