"""Offline seed-driven CLI for generating and verifying Jev Grip Benchmark datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.grip.camera import CameraConfig
from evaluation.grip.dataset import (
    generate_dataset,
    load_evaluator_records,
    load_observations,
)
from evaluation.grip.oracle import OracleProfile, evaluate_snapshot
from evaluation.grip.scene import RightArmController, SceneConfig, build_eval_scene
from evaluation.grip.snapshot import load_state_snapshot


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for ``python -m evaluation.grip``."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.grip",
        description="Generate a reproducible offline Jev Grip Benchmark dataset in MuJoCo.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("out/grip_smoke"),
        help="Destination directory (default: out/grip_smoke; ignored by git).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed (default: 42).",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="Number of matched positive/negative groups to generate (default: 1).",
    )
    scenario_mode = parser.add_mutually_exclusive_group()
    scenario_mode.add_argument(
        "--pilot",
        action="store_true",
        help="Prefer close boundary pairs and write a placement-disjoint development/evaluation split.",
    )
    scenario_mode.add_argument(
        "--cases",
        action="store_true",
        help="Render the fixed designed scenarios once each and retain every physical outcome.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Wrist camera render width in pixels (default: 480).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=300,
        help="Wrist camera render height in pixels (default: 300).",
    )
    verification = parser.add_mutually_exclusive_group()
    verification.add_argument(
        "--verify-rerun",
        action="store_true",
        help="Reload saved integration state snapshots and verify oracle rerun parity.",
    )
    verification.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify snapshots in an existing dataset without generating it again.",
    )
    return parser


def _verify_saved_snapshots(out_dir: Path) -> dict[str, int]:
    """Re-evaluate every saved state snapshot and confirm exact outcome/label match."""
    obs_list = load_observations(out_dir)
    eval_map = load_evaluator_records(out_dir)
    meta = json.loads(
        (out_dir / "evaluator" / "dataset_meta.json").read_text(encoding="utf-8")
    )
    if (
        not obs_list
        or len(obs_list) != len(eval_map)
        or {obs.sample_id for obs in obs_list} != eval_map.keys()
    ):
        raise ValueError("Observation and evaluator sample IDs do not match")
    scene_config: dict[str, Any] = meta["scene_config"]
    for key, value in scene_config.items():
        if isinstance(value, list):
            scene_config[key] = tuple(value)
    s_cfg = SceneConfig(**scene_config)
    o_prof = OracleProfile(**meta["oracle_profile"])
    model, contract = build_eval_scene(s_cfg)
    controller = RightArmController(
        model, contract, open_finger_rad=s_cfg.open_finger_rad
    )

    verified = 0
    for obs in obs_list:
        rec = eval_map[obs.sample_id]
        if (obs.group_id, obs.capture_timestamp_utc) != (
            rec.group_id,
            rec.capture_timestamp_utc,
        ):
            raise ValueError(
                f"Observation and evaluator metadata mismatch for {obs.sample_id}"
            )
        snap = load_state_snapshot(out_dir / "evaluator" / rec.snapshot_path)
        expected = np.asarray(rec.proposal["target_hand_pose_world"], dtype=np.float64)
        ev = evaluate_snapshot(
            model, contract, controller, snap, o_prof, expected_hand_pose=expected
        )
        if ev.outcome != rec.outcome or ev.label != rec.label:
            raise RuntimeError(
                f"Snapshot rerun mismatch for {obs.sample_id}: "
                f"saved=({rec.outcome}, {rec.label}) rerun=({ev.outcome}, {ev.label})"
            )
        verified += 1
    return {"verified_snapshots": verified}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verify_only:
        print(json.dumps(_verify_saved_snapshots(args.out), indent=2))
        return 0
    try:
        meta = generate_dataset(
            args.out,
            seed=args.seed,
            num_groups=args.groups,
            camera_config=CameraConfig(width=args.width, height=args.height),
            pilot=args.pilot,
            designed=args.cases,
        )
    except FileExistsError as exc:
        parser.error(str(exc))
    report: dict[str, object] = {
        "out": str(args.out),
        "seed": meta["seed"],
        "summary": meta["summary"],
        "observation_manifest": str(args.out / "observation" / "manifest.jsonl"),
        "evaluator_labels": str(args.out / "evaluator" / "labels.jsonl"),
        "evaluator_meta": str(args.out / "evaluator" / "dataset_meta.json"),
    }
    if args.verify_rerun:
        report["rerun_verification"] = _verify_saved_snapshots(args.out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
