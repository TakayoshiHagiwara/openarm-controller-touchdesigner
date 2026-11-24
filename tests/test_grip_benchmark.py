"""Focused regression tests for the Jev Grip Benchmark local milestone.

Covers:
1. Pinned OpenArm v2 Pedestal evaluation scene contract (named wrist camera,
   named inner/outer finger collision geoms, actuation, contacts, solver options).
2. Lossless MuJoCo integration state snapshot save/restore and deterministic rerun.
3. Genuine force/contact close-lift-hold physics oracle for confirmed positive
   and nearby plausible negative states.
4. Separation of invalid initial scenes and failed robot motion from physics
   negatives (never misclassified as label=0).
5. Offline seed-driven dataset CLI reproducibility, visibility evidence,
   observation/evaluator artifact separation, and anti-leakage checks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mujoco
import numpy as np
from openarm_mujoco.v2 import openarm_cell_xml
import pytest

from evaluation.grip.camera import CameraConfig, WristCameraCapture
from evaluation.grip.cli import main as cli_main
from evaluation.grip.dataset import (
    generate_dataset,
    load_evaluator_records,
    load_observations,
)
from evaluation.grip.oracle import OracleProfile, evaluate_snapshot
from evaluation.grip.scene import (
    ARM_ORIGIN_SITE,
    CAMERA_HEAD_RIGHT,
    RightArmController,
    SceneConfig,
    build_eval_scene,
    prepare_settled_state,
    verify_scene_contract,
)
from evaluation.grip.snapshot import (
    capture_integration_state,
    load_state_snapshot,
    restore_integration_state,
    save_state_snapshot,
)


@pytest.fixture(scope="module")
def scene_bundle():
    """Build the pinned OpenArm v2 evaluation scene and controller once."""
    config = SceneConfig()
    model, contract = build_eval_scene(config)
    controller = RightArmController(model, contract)
    return config, model, contract, controller


def test_scene_contract_verifies_pinned_pedestal_geoms_camera_and_actuation(
    scene_bundle,
):
    """Scene must expose named wrist camera, finger collision geoms, and elliptic contacts."""
    config, model, contract, _ = scene_bundle
    summary = verify_scene_contract(model, contract)

    assert summary["camera_name"] == "camera_wrist_right"
    assert summary["left_finger_collision_geoms"] == [
        "finger_inner_right_collision_00",
        "finger_inner_right_collision_01",
        "finger_inner_right_collision_02",
        "finger_inner_right_collision_03",
    ]
    assert summary["right_finger_collision_geoms"] == [
        "finger_outer_right_collision_00",
        "finger_outer_right_collision_01",
        "finger_outer_right_collision_02",
        "finger_outer_right_collision_03",
    ]
    assert summary["gripper_actuator"] == "right_finger1_ctrl"
    assert summary["mimic_equality"] == "openarm_right_ee_finger_joint_mimic"
    assert summary["target_box_geom"] == "target_box_geom"
    assert summary["support_geom"] == "eval_support_top"
    assert summary["cone"] == "elliptic"
    assert summary["impratio"] == pytest.approx(10.0)
    assert summary["timestep"] == pytest.approx(0.001)
    assert summary["box_mass_kg"] == pytest.approx(config.box_mass_kg)


def test_integration_state_snapshot_roundtrip_and_deterministic_oracle_rerun(
    scene_bundle, tmp_path: Path
):
    """Saved mjSTATE_INTEGRATION must restore bit-exactly and reproduce oracle outputs."""
    config, model, contract, controller = scene_bundle
    box_pos = np.array([0.41, -0.15, config.table_top_z + config.box_half_extents[2]])
    target_pose = controller.nominal_pregrasp_pose(box_pos)
    data = prepare_settled_state(
        model,
        contract,
        controller,
        box_pos=box_pos,
        box_yaw_rad=0.0,
        target_hand_pose=target_pose,
        open_finger_rad=config.open_finger_rad,
    )
    snapshot = capture_integration_state(model, data, config)
    assert snapshot.state_spec == int(mujoco.mjtState.mjSTATE_INTEGRATION)
    assert snapshot.state.ndim == 1
    assert snapshot.state.size == mujoco.mj_stateSize(
        model, mujoco.mjtState.mjSTATE_INTEGRATION
    )

    snap_path = tmp_path / "snap.npz"
    save_state_snapshot(snap_path, snapshot)
    loaded = load_state_snapshot(snap_path)

    np.testing.assert_array_equal(loaded.state, snapshot.state)
    assert loaded.state_spec == snapshot.state_spec

    restored = restore_integration_state(model, loaded)
    np.testing.assert_array_equal(restored.qpos, data.qpos)
    np.testing.assert_array_equal(restored.qvel, data.qvel)
    np.testing.assert_array_equal(restored.ctrl, data.ctrl)

    profile = OracleProfile()
    ev1 = evaluate_snapshot(
        model, contract, controller, snapshot, profile, expected_hand_pose=target_pose
    )
    ev2 = evaluate_snapshot(
        model, contract, controller, loaded, profile, expected_hand_pose=target_pose
    )

    assert ev1.outcome == ev2.outcome == "physics_success"
    assert ev1.label == ev2.label == 1
    assert ev1.object_motion["object_rise_m"] == pytest.approx(
        ev2.object_motion["object_rise_m"], abs=1e-9
    )
    assert ev1.object_motion["object_drift_rel_hand_m"] == pytest.approx(
        ev2.object_motion["object_drift_rel_hand_m"], abs=1e-9
    )


def test_physics_oracle_distinguishes_confirmed_positive_and_nearby_negative(
    scene_bundle,
):
    """Genuine contact close-lift-hold must confirm grasp and reject nearby short-X miss."""
    config, model, contract, controller = scene_bundle
    box_pos = np.array([0.41, -0.15, config.table_top_z + config.box_half_extents[2]])
    profile = OracleProfile()

    # Confirmed positive: pinch center aligned with upper-mid box.
    pos_pose = controller.nominal_pregrasp_pose(box_pos)
    pos_data = prepare_settled_state(
        model,
        contract,
        controller,
        box_pos=box_pos,
        box_yaw_rad=0.0,
        target_hand_pose=pos_pose,
        open_finger_rad=config.open_finger_rad,
    )
    pos_snap = capture_integration_state(model, pos_data, config)
    pos_ev = evaluate_snapshot(
        model, contract, controller, pos_snap, profile, expected_hand_pose=pos_pose
    )

    assert pos_ev.outcome == "physics_success"
    assert pos_ev.label == 1
    assert pos_ev.initial_validity["valid"] is True
    assert pos_ev.hand_tracking["tracking_valid"] is True
    assert pos_ev.contacts["after_hold"]["left_finger_contact"] is True
    assert pos_ev.contacts["after_hold"]["right_finger_contact"] is True
    assert pos_ev.contacts["after_hold"]["support_contact"] is False
    assert pos_ev.object_motion["object_rise_m"] >= 0.025
    assert pos_ev.object_motion["object_drift_rel_hand_m"] <= 0.025

    # Nearby plausible negative: hand stops 35 mm short along X, valid scene & tracking.
    neg_pose = controller.nominal_pregrasp_pose(
        box_pos, hand_offset_xyz=(-0.035, 0.0, 0.0)
    )
    neg_data = prepare_settled_state(
        model,
        contract,
        controller,
        box_pos=box_pos,
        box_yaw_rad=0.0,
        target_hand_pose=neg_pose,
        open_finger_rad=config.open_finger_rad,
    )
    neg_snap = capture_integration_state(model, neg_data, config)
    neg_ev = evaluate_snapshot(
        model, contract, controller, neg_snap, profile, expected_hand_pose=neg_pose
    )

    assert neg_ev.outcome == "physics_failure"
    assert neg_ev.label == 0
    assert neg_ev.initial_validity["valid"] is True
    assert neg_ev.hand_tracking["tracking_valid"] is True
    assert neg_ev.hand_tracking["actual_hand_rise_m"] >= 0.030
    assert neg_ev.object_motion["object_rise_m"] < 0.010
    assert neg_ev.contacts["after_hold"]["left_finger_contact"] is False
    assert neg_ev.contacts["after_hold"]["right_finger_contact"] is False
    assert len(neg_ev.reasons) >= 1

    # Second nearby plausible negative family: hand 50 mm too high above box.
    high_pose = controller.nominal_pregrasp_pose(
        box_pos, hand_offset_xyz=(0.0, 0.0, 0.050)
    )
    high_data = prepare_settled_state(
        model,
        contract,
        controller,
        box_pos=box_pos,
        box_yaw_rad=0.0,
        target_hand_pose=high_pose,
        open_finger_rad=config.open_finger_rad,
    )
    high_snap = capture_integration_state(model, high_data, config)
    high_ev = evaluate_snapshot(
        model, contract, controller, high_snap, profile, expected_hand_pose=high_pose
    )
    assert high_ev.outcome == "physics_failure"
    assert high_ev.label == 0
    assert high_ev.initial_validity["valid"] is True
    assert high_ev.hand_tracking["tracking_valid"] is True
    assert high_ev.object_motion["object_rise_m"] < 0.010


def test_invalid_scene_is_classified_invalid_never_negative(scene_bundle):
    """Pre-capture collision or unsupported box must yield outcome='invalid', label=None."""
    config, model, contract, controller = scene_bundle
    box_pos = np.array([0.41, -0.15, config.table_top_z + config.box_half_extents[2]])
    # Case 1: hand deep enough in -Z to collide with the support table and/or box.
    colliding_pose = controller.nominal_pregrasp_pose(
        box_pos, hand_offset_xyz=(0.0, 0.045, -0.055)
    )
    bad_data = prepare_settled_state(
        model,
        contract,
        controller,
        box_pos=box_pos,
        box_yaw_rad=0.0,
        target_hand_pose=colliding_pose,
        open_finger_rad=config.open_finger_rad,
    )
    bad_snap = capture_integration_state(model, bad_data, config)
    ev = evaluate_snapshot(
        model,
        contract,
        controller,
        bad_snap,
        OracleProfile(),
        expected_hand_pose=colliding_pose,
    )

    assert ev.outcome == "invalid"
    assert ev.label is None
    assert ev.initial_validity["valid"] is False
    assert len(ev.reasons) >= 1

    # Case 2: box placed off the physical support (not resting on support).
    floating_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, floating_data, 0)
    floating_data.qpos[contract.box_qpos_adr : contract.box_qpos_adr + 3] = [
        0.41,
        -0.15,
        0.70,
    ]
    floating_data.qpos[contract.box_qpos_adr + 3 : contract.box_qpos_adr + 7] = [
        1,
        0,
        0,
        0,
    ]
    mujoco.mj_forward(model, floating_data)
    floating_snap = capture_integration_state(model, floating_data, config)
    ev_float = evaluate_snapshot(
        model, contract, controller, floating_snap, OracleProfile()
    )
    assert ev_float.outcome == "invalid"
    assert ev_float.label is None
    assert ev_float.initial_validity["box_on_support"] is False
    assert "initial_box_not_on_support" in ev_float.reasons


def test_failed_robot_lift_motion_is_classified_invalid_never_negative(scene_bundle):
    """If the arm fails to execute the lift trajectory, classify as invalid, not negative."""
    config, model, contract, controller = scene_bundle
    box_pos = np.array([0.41, -0.15, config.table_top_z + config.box_half_extents[2]])
    pose = controller.nominal_pregrasp_pose(box_pos, hand_offset_xyz=(-0.035, 0.0, 0.0))
    data = prepare_settled_state(
        model,
        contract,
        controller,
        box_pos=box_pos,
        box_yaw_rad=0.0,
        target_hand_pose=pose,
        open_finger_rad=config.open_finger_rad,
    )
    snap = capture_integration_state(model, data, config)

    # Simulate an unreachable lift target (e.g., +0.80 m) so arm tracking genuinely fails.
    impossible_profile = OracleProfile(lift_height_m=0.80, lift_duration_s=0.30)
    ev = evaluate_snapshot(
        model,
        contract,
        controller,
        snap,
        impossible_profile,
        expected_hand_pose=pose,
    )

    assert ev.outcome == "invalid"
    assert ev.label is None
    assert ev.initial_validity["valid"] is True
    assert ev.hand_tracking["tracking_valid"] is False
    assert any("tracking" in r or "ik" in r for r in ev.reasons)


def test_finger_stuck_open_is_invalid_without_object_contact(scene_bundle, monkeypatch):
    """An unsuccessful close command must leave the sample without a label."""
    config, model, contract, controller = scene_bundle
    box_pos = np.array([0.41, -0.15, config.table_top_z + config.box_half_extents[2]])
    pose = controller.nominal_pregrasp_pose(box_pos, hand_offset_xyz=(-0.04, 0.0, 0.0))
    data = prepare_settled_state(
        model,
        contract,
        controller,
        box_pos=box_pos,
        box_yaw_rad=0.0,
        target_hand_pose=pose,
    )
    apply_controls = controller.apply_position_controls

    def keep_fingers_open(
        data, q_right_target, right_finger_target, q_left_target=None
    ):
        apply_controls(data, q_right_target, config.open_finger_rad, q_left_target)

    monkeypatch.setattr(controller, "apply_position_controls", keep_fingers_open)
    evidence = evaluate_snapshot(
        model, contract, controller, capture_integration_state(model, data, config)
    )
    assert evidence.outcome == "invalid"
    assert evidence.label is None
    assert "gripper_close_tracking_error" in evidence.reasons


def test_generation_preserves_existing_dataset(tmp_path):
    out = tmp_path / "existing"
    image = out / "observation" / "images" / "smp_existing.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"existing evidence")

    with pytest.raises(FileExistsError, match="already exists"):
        generate_dataset(out, camera_config=CameraConfig(camera_name="invalid_camera"))

    assert image.read_bytes() == b"existing evidence"


def test_failed_generation_leaves_no_partial_dataset(tmp_path):
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="Camera mismatch"):
        generate_dataset(out, camera_config=CameraConfig(camera_name="invalid_camera"))

    assert list(tmp_path.iterdir()) == []


def test_cli_generates_reproducible_separated_dataset_with_visibility_and_no_leakage(
    tmp_path: Path,
):
    """End-to-end CLI check: confirmed pos+neg, visible object, clean separation, rerun parity."""
    # Verify host GL availability explicitly; fail clearly if rendering is unavailable.
    config = SceneConfig()
    model, contract = build_eval_scene(config)
    try:
        with WristCameraCapture(
            model, contract, CameraConfig(width=160, height=100)
        ) as _:
            pass
    except RuntimeError as exc:  # pragma: no cover - host GL diagnostic guard
        pytest.skip(f"Offscreen MuJoCo GL unavailable in this test environment: {exc}")

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    rc_a = cli_main(
        [
            "--seed",
            "101",
            "--groups",
            "1",
            "--width",
            "960",
            "--height",
            "600",
            "--out",
            str(out_a),
        ]
    )
    rc_b = cli_main(
        [
            "--seed",
            "101",
            "--groups",
            "1",
            "--width",
            "960",
            "--height",
            "600",
            "--out",
            str(out_b),
        ]
    )
    assert rc_a == 0
    assert rc_b == 0

    obs_a = load_observations(out_a)
    obs_b = load_observations(out_b)
    eval_a = load_evaluator_records(out_a)
    eval_b = load_evaluator_records(out_b)

    assert len(obs_a) == 2
    assert [o.sample_id for o in obs_a] == [o.sample_id for o in obs_b]
    assert [o.group_id for o in obs_a] == [o.group_id for o in obs_b]
    assert {o.sample_id for o in obs_a} == eval_a.keys()

    # Anti-leakage: observation directory and manifest must never expose labels/offsets/outcomes.
    leak_pattern = re.compile(
        r"(label|outcome|positive|negative|offset|success|failure|oracle|snapshot|proposal|_pos|_neg)",
        re.IGNORECASE,
    )
    manifest_text = (out_a / "observation" / "manifest.jsonl").read_text(
        encoding="utf-8"
    )
    for line in manifest_text.strip().splitlines():
        record = json.loads(line)
        assert set(record.keys()) == {
            "sample_id",
            "group_id",
            "rgb_path",
            "capture_timestamp_utc",
            "sim_time_s",
            "camera",
            "observation_schema_version",
        }
        assert not leak_pattern.search(record["sample_id"])
        assert not leak_pattern.search(record["rgb_path"])
        assert not leak_pattern.search(record["group_id"])
        assert (record["camera"]["width"], record["camera"]["height"]) == (960, 600)
        img_file = out_a / "observation" / record["rgb_path"]
        assert img_file.is_file()
        assert img_file.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    # Capture timestamps are monotonic in shuffled manifest order (no evaluation-order leak).
    timestamps = [o.capture_timestamp_utc for o in obs_a]
    assert timestamps == sorted(timestamps)

    # Confirmed positive + confirmed nearby negative in the same group, object visible in both.
    outcomes = {eval_a[o.sample_id].outcome for o in obs_a}
    labels = {eval_a[o.sample_id].label for o in obs_a}
    assert outcomes == {"physics_success", "physics_failure"}
    assert labels == {0, 1}
    assert len({o.group_id for o in obs_a}) == 1

    # Reproducibility across runs & exact rerun from saved state snapshots.
    controller = RightArmController(model, contract)
    profile = OracleProfile()
    for obs in obs_a:
        rec_a = eval_a[obs.sample_id]
        rec_b = eval_b[obs.sample_id]
        assert rec_a.outcome == rec_b.outcome
        assert rec_a.label == rec_b.label
        assert rec_a.visibility["visible"] is True
        assert rec_a.visibility["box_pixel_count"] > 0

        snap_a = load_state_snapshot(out_a / "evaluator" / rec_a.snapshot_path)
        snap_b = load_state_snapshot(out_b / "evaluator" / rec_b.snapshot_path)
        np.testing.assert_array_equal(snap_a.state, snap_b.state)
        assert (out_a / "observation" / obs.rgb_path).read_bytes() == (
            out_b / "observation" / obs.rgb_path
        ).read_bytes()

        rerun_ev = evaluate_snapshot(
            model,
            contract,
            controller,
            snap_a,
            profile,
            expected_hand_pose=np.asarray(rec_a.proposal["target_hand_pose_world"]),
        )
        assert rerun_ev.to_dict() == rec_a.oracle

    # Provenance metadata exists in evaluator/ only.
    meta = json.loads(
        (out_a / "evaluator" / "dataset_meta.json").read_text(encoding="utf-8")
    )
    assert meta["seed"] == 101
    assert meta["provenance"]["mujoco_version"] == mujoco.__version__
    assert meta["provenance"]["camera_name"] == "camera_wrist_right"
    assert not (out_a / "observation" / "dataset_meta.json").exists()

    assert cli_main(["--out", str(out_a), "--verify-only"]) == 0


def test_head_camera_matches_cell_calibration_at_the_arm_mount():
    pedestal, contract = build_eval_scene(include_head_camera=True)
    cell = mujoco.MjModel.from_xml_path(openarm_cell_xml())
    cell_data, pedestal_data = mujoco.MjData(cell), mujoco.MjData(pedestal)
    mujoco.mj_forward(cell, cell_data)
    mujoco.mj_forward(pedestal, pedestal_data)
    cell_camera = cell.camera(CAMERA_HEAD_RIGHT).id
    pedestal_camera = pedestal.camera(CAMERA_HEAD_RIGHT).id

    np.testing.assert_allclose(
        pedestal_data.cam_xpos[pedestal_camera]
        - pedestal_data.site_xpos[pedestal.site(ARM_ORIGIN_SITE).id],
        cell_data.cam_xpos[cell_camera]
        - cell_data.site_xpos[cell.site(ARM_ORIGIN_SITE).id],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        pedestal_data.cam_xmat[pedestal_camera],
        cell_data.cam_xmat[cell_camera],
        atol=1e-9,
    )
    assert contract.head_camera is not None
    assert contract.head_camera.resolution_px == (1280, 720)
    assert pedestal.cam_fovy[pedestal_camera] == cell.cam_fovy[cell_camera] == 52.0
