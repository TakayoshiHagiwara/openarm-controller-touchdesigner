"""Jev Grip Benchmark local milestone package.

Separates observation-only interfaces (`ObservationRecord`, `load_observations`)
from evaluator-only state, oracle evidence, and labels (`EvaluatorRecord`,
`load_evaluator_records`, `evaluate_snapshot`, `generate_dataset`).
"""

from evaluation.grip.camera import CameraConfig, VisibilityEvidence, WristCameraCapture
from evaluation.grip.dataset import (
    EvaluatorRecord,
    ObservationRecord,
    generate_dataset,
    load_evaluator_records,
    load_observations,
    opaque_sample_id,
)
from evaluation.grip.oracle import OracleEvidence, OracleProfile, evaluate_snapshot
from evaluation.grip.scene import (
    RightArmController,
    SceneConfig,
    SceneContract,
    build_eval_scene,
    prepare_settled_state,
    verify_scene_contract,
)
from evaluation.grip.snapshot import (
    StateSnapshot,
    capture_integration_state,
    load_state_snapshot,
    restore_integration_state,
    save_state_snapshot,
)

__all__ = [
    "CameraConfig",
    "EvaluatorRecord",
    "ObservationRecord",
    "OracleEvidence",
    "OracleProfile",
    "RightArmController",
    "SceneConfig",
    "SceneContract",
    "StateSnapshot",
    "VisibilityEvidence",
    "WristCameraCapture",
    "build_eval_scene",
    "capture_integration_state",
    "evaluate_snapshot",
    "generate_dataset",
    "load_evaluator_records",
    "load_observations",
    "load_state_snapshot",
    "opaque_sample_id",
    "prepare_settled_state",
    "restore_integration_state",
    "save_state_snapshot",
    "verify_scene_contract",
]
