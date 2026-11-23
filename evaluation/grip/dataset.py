"""Seed-driven offline dataset generator and observation/evaluator artifact readers.

Artifact layout strictly separates observation-only data from evaluator-only state
and ground-truth oracle labels, joined exclusively by opaque ``sample_id``:

    <out_dir>/
      observation/
        manifest.jsonl        # Observation-only metadata (no labels, offsets, or outcomes)
        images/
          <sample_id>.png     # Right wrist camera RGB
      evaluator/
        dataset_meta.json     # Runtime/scene provenance, seed, oracle profile & rationale
        labels.jsonl          # Evaluator-only outcomes, labels, oracle/visibility evidence
        states/
          <sample_id>.npz     # Full MuJoCo mjSTATE_INTEGRATION snapshots for rerun
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mujoco
import numpy as np

from evaluation.grip.camera import (
    NATIVE_WRIST_RESOLUTION,
    CameraConfig,
    WristCameraCapture,
    write_png_rgb,
)
from evaluation.grip.oracle import OracleProfile, evaluate_snapshot
from evaluation.grip.scene import (
    RightArmController,
    SceneConfig,
    build_eval_scene,
    prepare_settled_state,
    verify_scene_contract,
)
from evaluation.grip.snapshot import capture_integration_state, save_state_snapshot

OBSERVATION_SCHEMA_VERSION = "1.0"
EVALUATOR_SCHEMA_VERSION = "1.0"

# Fixed states at one box placement. Offsets are in world X, Y, Z, in meters.
# The expected labels describe the design, while the oracle supplies the labels.
DESIGNED_CASES = (
    ("centered", (0.0, 0.0, 0.0), 1),
    ("near_reach", (-0.025, 0.0, 0.0), 1),
    ("raised_reachable", (0.0, 0.0, 0.029), 1),
    ("too_short", (-0.039, 0.0, 0.0), 0),
    ("too_high", (0.0, 0.0, 0.048), 0),
    ("short_lateral", (-0.025, 0.016, 0.0), 0),
)


@dataclass(frozen=True)
class ObservationRecord:
    """Public observation-only contract exposed to downstream evaluators/models."""

    sample_id: str
    group_id: str
    rgb_path: str
    capture_timestamp_utc: str
    sim_time_s: float
    camera: dict[str, Any]
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluatorRecord:
    """Evaluator-only contract containing ground-truth oracle labels and state pointers."""

    sample_id: str
    group_id: str
    outcome: str
    label: int | None
    snapshot_path: str
    capture_timestamp_utc: str
    visibility: dict[str, Any]
    oracle: dict[str, Any]
    proposal: dict[str, Any]
    evaluator_schema_version: str = EVALUATOR_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def opaque_sample_id(seed: int, group_index: int, slot_index: int) -> str:
    """Generate a deterministic opaque sample ID with zero label/offset leakage."""
    payload = (
        f"jev-grip-v1|seed={int(seed)}|group={int(group_index)}|slot={int(slot_index)}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"smp_{digest}"


def load_observations(dataset_dir: str | Path) -> list[ObservationRecord]:
    """Read ONLY ``observation/manifest.jsonl`` without touching ``evaluator/``."""
    manifest_path = Path(dataset_dir) / "observation" / "manifest.jsonl"
    records: list[ObservationRecord] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        records.append(ObservationRecord(**raw))
    return records


def load_evaluator_records(dataset_dir: str | Path) -> dict[str, EvaluatorRecord]:
    """Read ``evaluator/labels.jsonl`` indexed by ``sample_id`` for clean joining."""
    labels_path = Path(dataset_dir) / "evaluator" / "labels.jsonl"
    records: dict[str, EvaluatorRecord] = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rec = EvaluatorRecord(**raw)
        records[rec.sample_id] = rec
    return records


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_head_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        return out or None
    except (OSError, subprocess.SubprocessError):
        return None


def _sample_positive_candidates(
    rng: np.random.Generator,
) -> list[tuple[str, tuple[float, float, float]]]:
    """Generate seeded candidate hand offsets around the nominal pinch pose."""
    candidates: list[tuple[str, tuple[float, float, float]]] = []
    for _ in range(4):
        dx = float(rng.uniform(-0.008, 0.005))
        dy = float(rng.uniform(-0.010, 0.010))
        dz = float(rng.uniform(-0.004, 0.006))
        candidates.append(("centered_jitter", (dx, dy, dz)))
    candidates.append(("centered_nominal", (0.0, 0.0, 0.0)))
    return candidates


def _sample_negative_candidates(
    rng: np.random.Generator,
) -> list[tuple[str, tuple[float, float, float]]]:
    """Generate nearby plausible near-miss hand offsets while keeping the box visible."""
    families = ["short_x", "high_z", "lateral_y_neg"]
    order = list(rng.permutation(families))
    candidates: list[tuple[str, tuple[float, float, float]]] = []
    for family in order:
        if family == "short_x":
            dx = float(rng.uniform(-0.046, -0.034))
            dy = float(rng.uniform(-0.008, 0.008))
            dz = float(rng.uniform(-0.003, 0.006))
            candidates.append(("near_miss_short_x", (dx, dy, dz)))
        elif family == "high_z":
            dx = float(rng.uniform(-0.006, 0.004))
            dy = float(rng.uniform(-0.008, 0.008))
            dz = float(rng.uniform(0.048, 0.056))
            candidates.append(("near_miss_high_z", (dx, dy, dz)))
        else:
            dx = float(rng.uniform(-0.020, -0.014))
            dy = float(rng.uniform(-0.066, -0.058))
            dz = float(rng.uniform(-0.002, 0.006))
            candidates.append(("near_miss_lateral_y", (dx, dy, dz)))
    # Deterministic safe fallback known to remain valid, visible, and non-grasping.
    candidates.append(("near_miss_short_x_fallback", (-0.040, 0.0, 0.0)))
    return candidates


def _pilot_offsets(
    rng: np.random.Generator, group_index: int
) -> tuple[
    list[tuple[str, tuple[float, float, float]]],
    list[tuple[str, tuple[float, float, float]]],
]:
    """Propose close-width pairs and leave the physical label to the oracle."""
    x_jitter, y_jitter, z_jitter = rng.uniform(-0.002, 0.002, 3).tolist()
    if group_index % 4 == 0:
        return (
            [("pilot_high_positive", (x_jitter, y_jitter, 0.029 + z_jitter))],
            [
                ("pilot_high_near_miss", (x_jitter, y_jitter, 0.043 + z_jitter)),
                ("pilot_high_near_miss", (0.0, 0.0, 0.048)),
            ],
        )
    return (
        [("pilot_short_positive", (-0.022 + x_jitter, y_jitter, 0.005 + z_jitter))],
        [
            ("pilot_short_near_miss", (-0.034 + x_jitter, y_jitter, -0.006 + z_jitter)),
            ("pilot_short_near_miss", (-0.036, 0.0, -0.004)),
        ],
    )


def generate_dataset(
    out_dir: str | Path,
    *,
    seed: int = 42,
    num_groups: int = 1,
    scene_config: SceneConfig | None = None,
    camera_config: CameraConfig | None = None,
    oracle_profile: OracleProfile | None = None,
    pilot: bool = False,
    designed: bool = False,
) -> dict[str, Any]:
    """Generate a reproducible offline grip dataset with confirmed positive and negative pairs."""
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if designed:
        if pilot:
            raise ValueError("Choose either designed cases or pilot pairs")
        num_groups = len(DESIGNED_CASES)

    out_path = Path(out_dir)
    if out_path.exists() or out_path.is_symlink():
        raise FileExistsError(f"Dataset destination already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{out_path.name}-", dir=out_path.parent
    ) as temporary:
        staged = Path(temporary)
        meta = _write_dataset(
            staged,
            seed,
            num_groups,
            scene_config or SceneConfig(),
            camera_config or CameraConfig(),
            oracle_profile or OracleProfile(),
            pilot=pilot,
            designed=designed,
        )
        staged.rename(out_path)
    return meta


def _write_dataset(
    out_path: Path,
    seed: int,
    num_groups: int,
    s_cfg: SceneConfig,
    c_cfg: CameraConfig,
    o_prof: OracleProfile,
    *,
    pilot: bool,
    designed: bool,
) -> dict[str, Any]:
    obs_dir = out_path / "observation"
    eval_dir = out_path / "evaluator"
    (obs_dir / "images").mkdir(parents=True)
    (eval_dir / "states").mkdir(parents=True)

    model, contract = build_eval_scene(s_cfg)
    contract_summary = verify_scene_contract(model, contract)
    controller = RightArmController(
        model, contract, open_finger_rad=s_cfg.open_finger_rad
    )
    rng = np.random.default_rng(int(seed))

    obs_records: list[ObservationRecord] = []
    eval_records: list[EvaluatorRecord] = []
    rejected_candidates = 0

    with WristCameraCapture(model, contract, c_cfg) as camera:
        for group_idx in range(num_groups):
            box_z = s_cfg.table_top_z + s_cfg.box_half_extents[2]
            if designed:
                name, offset, expected_label = DESIGNED_CASES[group_idx]
                group_id = "grp_0000"
                box_pos = np.array([0.411, -0.150, box_z])
                box_yaw = 0.0
                candidates = ((None, [(name, offset)]),)
            else:
                group_id = f"grp_{group_idx:04d}"
                expected_label = None
                box_pos = np.array(
                    [rng.uniform(0.400, 0.422), rng.uniform(-0.168, -0.132), box_z]
                )
                box_yaw = float(rng.uniform(-0.14, 0.14))
                positive = _sample_positive_candidates(rng)
                negative = _sample_negative_candidates(rng)
                if pilot:
                    near_positive, near_negative = _pilot_offsets(rng, group_idx)
                    positive = near_positive + positive
                    negative = near_negative + negative
                candidates = (
                    ("physics_success", positive),
                    ("physics_failure", negative),
                )
            # Choose capture order before rendering so timestamps do not encode outcomes.
            for slot_idx, candidate_idx in enumerate(
                rng.permutation(len(candidates)).tolist()
            ):
                desired_outcome, candidate_list = candidates[candidate_idx]
                for proposal_kind, hand_offset in candidate_list:
                    target_pose = controller.nominal_pregrasp_pose(
                        box_pos,
                        hand_offset_xyz=hand_offset,
                        pinch_z_above_box_center_m=s_cfg.pinch_z_offset_above_box_center_m,
                    )
                    data = prepare_settled_state(
                        model,
                        contract,
                        controller,
                        box_pos=box_pos,
                        box_yaw_rad=box_yaw,
                        target_hand_pose=target_pose,
                        open_finger_rad=s_cfg.open_finger_rad,
                        settle_duration_s=s_cfg.settle_duration_s,
                    )
                    snapshot = capture_integration_state(model, data, s_cfg)
                    capture_ts = datetime.now(UTC).isoformat()
                    rgb, visibility = camera.capture(data)
                    oracle_ev = evaluate_snapshot(
                        model,
                        contract,
                        controller,
                        snapshot,
                        o_prof,
                        expected_hand_pose=target_pose,
                    )
                    if desired_outcome is None or (
                        visibility.visible and oracle_ev.outcome == desired_outcome
                    ):
                        break
                    rejected_candidates += 1
                else:
                    raise RuntimeError(
                        f"Failed to confirm a valid, visible {desired_outcome!r} sample "
                        f"for {group_id} (seed={seed})."
                    )

                sample_id = opaque_sample_id(seed, group_idx, slot_idx)
                rel_rgb_path = f"images/{sample_id}.png"
                rel_snap_path = f"states/{sample_id}.npz"
                write_png_rgb(obs_dir / rel_rgb_path, rgb)
                save_state_snapshot(eval_dir / rel_snap_path, snapshot)

                obs_records.append(
                    ObservationRecord(
                        sample_id=sample_id,
                        group_id=group_id,
                        rgb_path=rel_rgb_path,
                        capture_timestamp_utc=capture_ts,
                        sim_time_s=snapshot.sim_time_s,
                        camera={
                            "name": c_cfg.camera_name,
                            "width": c_cfg.width,
                            "height": c_cfg.height,
                            "fovy_deg": contract_summary["camera_fovy_deg"],
                            "native_resolution": list(NATIVE_WRIST_RESOLUTION),
                        },
                    )
                )
                eval_records.append(
                    EvaluatorRecord(
                        sample_id=sample_id,
                        group_id=group_id,
                        outcome=oracle_ev.outcome,
                        label=oracle_ev.label,
                        snapshot_path=rel_snap_path,
                        capture_timestamp_utc=capture_ts,
                        visibility=visibility.to_dict(),
                        oracle=oracle_ev.to_dict(),
                        proposal={
                            "proposal_kind": proposal_kind,
                            **({"expected_label": expected_label} if designed else {}),
                            "box_pos_m": box_pos.tolist(),
                            "box_yaw_rad": box_yaw,
                            "hand_offset_xyz_m": list(hand_offset),
                            "target_hand_pose_world": target_pose.tolist(),
                        },
                    )
                )

    (obs_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(rec.to_dict(), sort_keys=True) for rec in obs_records)
        + "\n",
        encoding="utf-8",
    )
    (eval_dir / "labels.jsonl").write_text(
        "\n".join(json.dumps(rec.to_dict(), sort_keys=True) for rec in eval_records)
        + "\n",
        encoding="utf-8",
    )

    if pilot:
        # Each box placement and both of its observations stay in the same split.
        dev_count = min(num_groups - 1, max(1, num_groups // 5))
        dev_indices = set(rng.permutation(num_groups)[:dev_count].tolist())
        splits = {
            f"grp_{index:04d}": "dev" if index in dev_indices else "eval"
            for index in range(num_groups)
        }
        (eval_dir / "splits.json").write_text(
            json.dumps(splits, indent=2) + "\n", encoding="utf-8"
        )

    dataset_meta = {
        "benchmark": "jev-grip-benchmark",
        "milestone": "designed-cases"
        if designed
        else ("pilot" if pilot else "local-milestone-1"),
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": int(seed),
        "num_groups": len({record.group_id for record in obs_records}),
        "summary": {
            "total_samples": len(eval_records),
            "physics_success_count": sum(
                1 for r in eval_records if r.outcome == "physics_success"
            ),
            "physics_failure_count": sum(
                1 for r in eval_records if r.outcome == "physics_failure"
            ),
            "invalid_count": sum(1 for r in eval_records if r.outcome == "invalid"),
            "rejected_candidates_during_search": int(rejected_candidates),
        },
        "provenance": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "mujoco_version": mujoco.__version__,
            "openarm_mujoco_version": _package_version("openarm-mujoco"),
            "openarm_control_version": _package_version("openarm-control"),
            "numpy_version": np.__version__,
            "git_head": _git_head_sha(),
            "pedestal_xml_path": contract.pedestal_xml_path,
            "pedestal_xml_sha256": contract.pedestal_xml_sha256,
            "camera_name": contract.camera_name,
        },
        "scene_contract": contract_summary,
        "scene_config": s_cfg.to_dict(),
        "camera_config": c_cfg.to_dict(),
        "oracle_profile": o_prof.to_dict(),
    }
    (eval_dir / "dataset_meta.json").write_text(
        json.dumps(dataset_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset_meta
