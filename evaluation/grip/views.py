"""Derive synchronized head-camera and empty-close overlay observations."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from evaluation.grip.benchmark import load_manifest, resolve_rgb_path
from evaluation.grip.camera import write_png_rgb
from evaluation.grip.oracle import OracleProfile
from evaluation.grip.scene import (
    ARM_ORIGIN_SITE,
    CAMERA_HEAD_RIGHT,
    SceneConfig,
    SceneContract,
    build_eval_scene,
)
from evaluation.grip.snapshot import (
    StateSnapshot,
    load_state_snapshot,
    restore_integration_state,
)

OVERLAY_RGB = (38, 223, 255)
OVERLAY_OPACITY = 0.40


def _finger_visuals(model: mujoco.MjModel, contract: SceneContract) -> list[int]:
    collision_geoms = contract.left_finger_geom_ids + contract.right_finger_geom_ids
    finger_bodies = set(model.geom_bodyid[list(collision_geoms)].tolist())
    visible_groups = mujoco.MjvOption().geomgroup
    visual_geoms = [
        index
        for index in range(model.ngeom)
        if int(model.geom_bodyid[index]) in finger_bodies
        and index not in contract.robot_collision_geom_ids
        and visible_groups[model.geom_group[index]]
    ]
    if set(model.geom_bodyid[visual_geoms].tolist()) != finger_bodies:
        raise ValueError("Could not resolve the right hand's two visible fingers")
    return visual_geoms


def _empty_close_mask(
    model: mujoco.MjModel,
    contract: SceneContract,
    renderer: mujoco.Renderer,
    original: mujoco.MjData,
    visual_fingers: list[int],
) -> np.ndarray:
    """Project the two unloaded, fully closed fingers at the original camera pose."""
    empty_close = mujoco.MjData(model)
    arm_indices = list(contract.right_arm_qpos_indices)
    empty_close.qpos[arm_indices] = original.qpos[arm_indices]
    empty_close.qpos[list(contract.right_finger_qpos_indices)] = (
        OracleProfile().close_finger_target_rad
    )
    mujoco.mj_kinematics(model, empty_close)
    mujoco.mj_camlight(model, empty_close)
    camera_id = contract.camera_id
    if not np.allclose(
        empty_close.cam_xpos[camera_id],
        original.cam_xpos[camera_id],
        rtol=0,
        atol=1e-10,
    ) or not np.allclose(
        empty_close.cam_xmat[camera_id],
        original.cam_xmat[camera_id],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(
            "The right wrist camera pose changed in the empty-close render"
        )

    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    groups = model.geom_group.copy()
    model.geom_group[:] = 5
    model.geom_group[visual_fingers] = 0
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(
            empty_close, camera=contract.camera_name, scene_option=option
        )
        segmentation = renderer.render()
        mask = (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
            segmentation[..., 0], visual_fingers
        )
    finally:
        renderer.disable_segmentation_rendering()
        model.geom_group[:] = groups
    if not mask.any():
        raise ValueError("The empty-close fingers are outside the wrist camera image")
    return mask


def _read_snapshot(dataset: Path, observation: dict[str, Any]) -> StateSnapshot:
    states = (dataset / "evaluator" / "states").resolve()
    path = (states / f"{observation['sample_id']}.npz").resolve()
    if path.parent != states:
        raise ValueError(f"Invalid snapshot path: {path}")
    return load_state_snapshot(path)


def derive_views(dataset_dir: str | Path, out_dir: str | Path) -> Path:
    """Render both observation variants from existing snapshots and keep their labels."""
    source = Path(dataset_dir).resolve()
    output = Path(out_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the source dataset")

    observations = load_manifest(source)
    if not observations:
        raise ValueError("The source dataset has no observations")
    first_state = _read_snapshot(source, observations[0])
    dimensions = observations[0]["camera"]
    width, height = dimensions["width"], dimensions["height"]
    model, contract = build_eval_scene(
        SceneConfig(**first_state.scene_config), include_head_camera=True
    )
    head_camera = contract.head_camera
    if head_camera is None:
        raise ValueError("The evaluation scene has no head camera calibration")
    head_width, head_height = head_camera.resolution_px
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width, head_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height, head_height)
    visual_fingers = _finger_visuals(model, contract)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".grip-views-", dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        obs_dir = stage / "observation"
        (obs_dir / "main").mkdir(parents=True)
        (obs_dir / "overlay").mkdir()
        with (
            mujoco.Renderer(model, height=height, width=width) as renderer,
            mujoco.Renderer(
                model, height=head_height, width=head_width
            ) as head_renderer,
        ):
            for observation in observations:
                sample_id = observation["sample_id"]
                snapshot = _read_snapshot(source, observation)
                if snapshot.scene_config != first_state.scene_config:
                    raise ValueError(f"Scene configuration differs for {sample_id}")
                if observation["camera"] != dimensions:
                    raise ValueError(
                        f"Wrist camera configuration differs for {sample_id}"
                    )

                data = restore_integration_state(model, snapshot)
                renderer.update_scene(data, camera=contract.camera_name)
                wrist_rgb = renderer.render().copy()
                source_rgb = resolve_rgb_path(source, observation["rgb_path"])
                raw_relative = source_rgb.relative_to(source / "observation")
                target_rgb = obs_dir / raw_relative
                target_rgb.parent.mkdir(parents=True, exist_ok=True)
                write_png_rgb(target_rgb, wrist_rgb)
                if target_rgb.read_bytes() != source_rgb.read_bytes():
                    raise ValueError(f"Rerendered wrist pixels differ for {sample_id}")

                main_relative = f"main/{sample_id}.png"
                head_renderer.update_scene(data, camera=CAMERA_HEAD_RIGHT)
                write_png_rgb(obs_dir / main_relative, head_renderer.render())

                mask = _empty_close_mask(
                    model, contract, renderer, data, visual_fingers
                )
                wrist_rgb[mask] = np.rint(
                    (1 - OVERLAY_OPACITY) * wrist_rgb[mask]
                    + OVERLAY_OPACITY * np.asarray(OVERLAY_RGB)
                ).astype(np.uint8)
                overlay_relative = f"overlay/{sample_id}.png"
                write_png_rgb(obs_dir / overlay_relative, wrist_rgb)
                observation.update(
                    main_rgb_path=main_relative, wrist_overlay_rgb_path=overlay_relative
                )

        (obs_dir / "manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in observations),
            encoding="utf-8",
        )
        shutil.copytree(source / "evaluator", stage / "evaluator")
        metadata = {
            "source_dataset": str(source),
            "count": len(observations),
            "main_camera": {
                "name": CAMERA_HEAD_RIGHT,
                "source": f"openarm-mujoco {version('openarm-mujoco')} v2/cell/cell.xml",
                "reference_frame": ARM_ORIGIN_SITE,
                **asdict(head_camera),
            },
            "image_size": [width, height],
            "overlay": {
                "rgb": OVERLAY_RGB,
                "opacity": OVERLAY_OPACITY,
                "finger_target_rad": OracleProfile().close_finger_target_rad,
            },
        }
        (stage / "evaluator" / "view_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stage.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the OpenArm head view and a wrist overlay from Grip snapshots."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(derive_views(args.dataset, args.out))


if __name__ == "__main__":
    main()
