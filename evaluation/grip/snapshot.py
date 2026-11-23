"""Full MuJoCo integration state snapshot capture, persistence, and restore."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from evaluation.grip.scene import SceneConfig

INTEGRATION_STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


@dataclass(frozen=True)
class StateSnapshot:
    """Lossless full MuJoCo integration state sufficient for deterministic rerun."""

    state: np.ndarray
    state_spec: int
    sim_time_s: float
    model_nq: int
    model_nv: int
    model_nu: int
    scene_config: dict[str, Any]


def capture_integration_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene_config: SceneConfig | dict[str, Any] | None = None,
) -> StateSnapshot:
    """Capture the complete MuJoCo ``mjSTATE_INTEGRATION`` vector."""
    size = mujoco.mj_stateSize(model, INTEGRATION_STATE_SPEC)
    state = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(model, data, state, INTEGRATION_STATE_SPEC)
    if isinstance(scene_config, SceneConfig):
        cfg_dict = scene_config.to_dict()
    elif isinstance(scene_config, dict):
        cfg_dict = dict(scene_config)
    else:
        cfg_dict = SceneConfig().to_dict()
    return StateSnapshot(
        state=state,
        state_spec=INTEGRATION_STATE_SPEC,
        sim_time_s=float(data.time),
        model_nq=int(model.nq),
        model_nv=int(model.nv),
        model_nu=int(model.nu),
        scene_config=cfg_dict,
    )


def restore_integration_state(
    model: mujoco.MjModel, snapshot: StateSnapshot
) -> mujoco.MjData:
    """Restore a saved integration state into a fresh ``MjData`` and run ``mj_forward``."""
    if snapshot.state_spec != INTEGRATION_STATE_SPEC:
        raise ValueError(
            f"Expected full MuJoCo integration state, got {snapshot.state_spec:#x}"
        )
    if (snapshot.model_nq, snapshot.model_nv, snapshot.model_nu) != (
        model.nq,
        model.nv,
        model.nu,
    ):
        raise ValueError("Snapshot and MuJoCo model dimensions do not match")
    expected_size = mujoco.mj_stateSize(model, INTEGRATION_STATE_SPEC)
    state = np.asarray(snapshot.state, dtype=np.float64)
    if state.ndim != 1 or state.size != expected_size:
        raise ValueError(
            f"State vector size mismatch: got {state.shape}, expected ({expected_size},) "
            f"for spec {snapshot.state_spec:#x}"
        )
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, INTEGRATION_STATE_SPEC)
    mujoco.mj_forward(model, data)
    return data


def save_state_snapshot(path: str | Path, snapshot: StateSnapshot) -> Path:
    """Write a lossless compressed ``.npz`` snapshot to disk."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "state_spec": int(snapshot.state_spec),
        "sim_time_s": float(snapshot.sim_time_s),
        "model_nq": int(snapshot.model_nq),
        "model_nv": int(snapshot.model_nv),
        "model_nu": int(snapshot.model_nu),
        "scene_config": snapshot.scene_config,
    }
    np.savez_compressed(
        dest,
        state=np.asarray(snapshot.state, dtype=np.float64),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    return dest


def load_state_snapshot(path: str | Path) -> StateSnapshot:
    """Load a previously saved ``.npz`` integration state snapshot."""
    src = Path(path)
    with np.load(src, allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=np.float64).copy()
        metadata = json.loads(str(archive["metadata_json"].item()))
    return StateSnapshot(
        state=state,
        state_spec=int(metadata["state_spec"]),
        sim_time_s=float(metadata["sim_time_s"]),
        model_nq=int(metadata["model_nq"]),
        model_nv=int(metadata["model_nv"]),
        model_nu=int(metadata["model_nu"]),
        scene_config=dict(metadata["scene_config"]),
    )
