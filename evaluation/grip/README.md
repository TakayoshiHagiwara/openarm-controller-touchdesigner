# Jev Grip Benchmark (Local Milestone 1)

Offline, seed-driven MuJoCo grip dataset generator and genuine force/contact grasp oracle built on the pinned `openarm-mujoco` OpenArm v2 Pedestal model (`openarm-mujoco==2.3.0`).

## Verified Model & Scene Contract

- **Base model**: `openarm_mujoco.v2.openarm_pedestal_xml()` (`timestep=0.001 s`, elliptic friction cone, `impratio=10`).
- **Right wrist camera**: `camera_wrist_right` on `openarm_right_ee_base_link` (`fovy=66°`, native `960x600`; default render `480x300` preserving 16:10 aspect ratio). Scene visual near-plane is set to `extent=1.5, znear=0.016` (`24 mm` absolute) to clear the wrist camera bezel while avoiding near-plane clipping on grasped objects.
- **Named right finger collision geoms** (`condim=4`, `contype=1`, `conaffinity=1`):
  - Left jaw in wrist view (`openarm_right_ee_inner_finger`, robot `+Y`): `finger_inner_right_collision_00..03`
  - Right jaw in wrist view (`openarm_right_ee_outer_finger`, robot `-Y`): `finger_outer_right_collision_00..03`
- **Actuation**: `right_joint1_ctrl..right_joint7_ctrl` (gravity-compensated position servos driven via `openarm_control.Kinematics` IK) and `right_finger1_ctrl` (`[-0.7854, 0.0]` rad, coupled to `openarm_right_finger_joint2` by `openarm_right_ee_finger_joint_mimic`).
- **Evaluation-only objects**: static physical support `eval_support` (`eval_support_top`, top at `z=0.42 m`, front edge at `x=0.33 m`) and lightweight free-joint box `target_box` (`target_box_geom`, `40 x 40 x 64 mm`, `0.06 kg`, `condim=4`).

## Physics Oracle Profile & Rationale

Each candidate is evaluated strictly by restoring its captured full MuJoCo integration state (`mjSTATE_INTEGRATION`) into a fresh `MjData`, calling `mj_forward`, and executing:

1. **Initial validity gate**: verifies `target_box_geom` rests on `eval_support_top` (`>= 0.05 N`), zero robot-box and robot-support contact, penetration `<= 5 mm`, settled velocities, upright box, and settled hand pose. Violations yield `outcome="invalid"`, `label=None`.
2. **Close while holding arm pose**: `0.35 s` linear ramp of `right_finger1_ctrl` to `0.0 rad` + `0.15 s` squeeze settle while holding captured arm pose via gravity-compensated position servos.
3. **Lift via arm actuation**: `40 mm` vertical lift (`+Z`) over `0.40 s` (`100 mm/s`) at `100 Hz` IK + gravity-compensated joint position actuation.
4. **Hold**: `0.40 s` hold at the lifted pose while sampling contacts at `100 Hz`.

The lift and hold profile checks that the box clears the support throughout the hold, stays near its grasped position in the hand frame, and retains bilateral fingertip contact. A sample with an IK failure, hand or mimic tracking error, or an unloaded gripper that fails to close is recorded as `outcome="invalid"` with `label=None`. The label describes this simulator, object, and controller profile.

## Visibility Signal

Every captured frame renders both RGB and MuJoCo geom segmentation (`mjOBJ_GEOM`) from `camera_wrist_right`, recording `box_pixel_count`, `box_pixel_fraction`, `box_bbox_px`, and 3D frustum depth (`signal_type="mujoco_geom_segmentation_and_frustum"`). Candidates require `box_pixel_count >= 200`, `box_pixel_fraction >= 0.005`, and positive depth.

## Observation vs. Evaluator Separation

```text
<out_dir>/
  observation/
    manifest.jsonl        # Public observation-only records (opaque sample_id, group_id, rgb_path, timestamp, sim_time_s, camera)
    images/
      smp_<12hex>.png     # Right wrist RGB only
  evaluator/
    dataset_meta.json     # Provenance, versions, XML SHA-256, scene/camera config, oracle profile & rationale
    labels.jsonl          # Evaluator-only outcomes, labels, visibility, oracle evidence, proposal metadata (joined by sample_id)
    states/
      smp_<12hex>.npz     # Lossless mjSTATE_INTEGRATION snapshots sufficient for deterministic rerun
```

Pair order is seeded before rendering. The manifest records the UTC time at the start of rendering and the corresponding simulator time. Each snapshot is an independent settled pose, so this UTC field measures generation time and is unsuitable for trajectory replay.

## Usage

Generate a smoke dataset at a fresh path under ignored `out/`. The dataset appears at its final path after generation succeeds.

```bash
uv run --locked python -m evaluation.grip --seed 42 --groups 2 --out out/grip_smoke --verify-rerun
```

Recheck the existing snapshots and their labels:

```bash
uv run --locked python -m evaluation.grip --out out/grip_smoke --verify-only
```

Run the focused regression suite:

```bash
uv run --locked pytest -v tests/test_grip_benchmark.py
```
