"""Right wrist camera RGB rendering, stdlib PNG encoding, and visibility signal."""

from __future__ import annotations

import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import mujoco
import numpy as np

from evaluation.grip.scene import CAMERA_WRIST_RIGHT, SceneContract

NATIVE_WRIST_RESOLUTION = (960, 600)


@dataclass(frozen=True)
class CameraConfig:
    """Offscreen capture settings for the named right wrist camera."""

    camera_name: str = CAMERA_WRIST_RIGHT
    width: int = 480
    height: int = 300
    min_box_pixels: int = 200
    min_box_pixel_fraction: float = 0.005

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisibilityEvidence:
    """Measurable object visibility signal from MuJoCo geom segmentation + frustum check."""

    signal_type: str
    visible: bool
    box_pixel_count: int
    box_pixel_fraction: float
    image_width: int
    image_height: int
    box_bbox_px: list[int] | None
    min_box_pixels: int
    min_box_pixel_fraction: float
    box_center_in_frustum: bool
    box_center_depth_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_png_rgb(path: str | Path, rgb: np.ndarray) -> Path:
    """Encode an ``(H, W, 3) uint8`` RGB array as a standard PNG using only stdlib."""
    arr = np.ascontiguousarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(
            f"Expected (H, W, 3) uint8 RGB image, got {arr.shape} {arr.dtype}"
        )
    height, width, _ = arr.shape
    raw_scanlines = b"".join(b"\x00" + arr[y].tobytes() for y in range(height))

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw_scanlines, level=6))
        + _chunk(b"IEND", b"")
    )
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(png_bytes)
    return dest


def _box_center_frustum_check(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contract: SceneContract,
    width: int,
    height: int,
) -> tuple[bool, float]:
    cam_pos = data.cam_xpos[contract.camera_id]
    cam_mat = data.cam_xmat[contract.camera_id].reshape(3, 3)
    box_pos = data.geom_xpos[contract.box_geom_id]
    rel = cam_mat.T @ (box_pos - cam_pos)
    depth = float(-rel[2])
    if depth <= 1e-4:
        return False, depth
    fovy_rad = np.deg2rad(float(model.cam_fovy[contract.camera_id]))
    fy = 0.5 * height / np.tan(0.5 * fovy_rad)
    fx = fy
    u = 0.5 * width + fx * (float(rel[0]) / depth)
    v = 0.5 * height - fy * (float(rel[1]) / depth)
    in_bounds = (0.0 <= u < width) and (0.0 <= v < height)
    return bool(in_bounds), depth


class WristCameraCapture:
    """Reusable offscreen MuJoCo renderer for ``camera_wrist_right`` RGB + visibility."""

    def __init__(
        self,
        model: mujoco.MjModel,
        contract: SceneContract,
        config: CameraConfig | None = None,
    ) -> None:
        self.model = model
        self.contract = contract
        self.config = config or CameraConfig()
        if self.config.camera_name != contract.camera_name:
            raise ValueError(
                f"Camera mismatch: config={self.config.camera_name!r}, "
                f"contract={contract.camera_name!r}"
            )
        width, height = self.config.width, self.config.height
        if width < 1 or height < 1:
            raise ValueError("Render width and height must be positive")
        visual = model.vis.global_
        visual.offwidth = max(visual.offwidth, width)
        visual.offheight = max(visual.offheight, height)
        try:
            self._renderer = mujoco.Renderer(model, height=height, width=width)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize MuJoCo offscreen renderer for "
                f"{self.config.camera_name} ({self.config.width}x{self.config.height}): {exc}"
            ) from exc

    def close(self) -> None:
        """Release underlying MuJoCo GL resources."""
        renderer = self._renderer
        if renderer is not None:
            renderer.close()
            self._renderer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, VisibilityEvidence]:
        """Render RGB and compute geom-segmentation + frustum visibility of the box."""
        if self._renderer is None:
            raise RuntimeError("WristCameraCapture renderer is closed.")
        self._renderer.disable_segmentation_rendering()
        self._renderer.update_scene(data, camera=self.config.camera_name)
        rgb = np.asarray(self._renderer.render(), dtype=np.uint8).copy()

        self._renderer.enable_segmentation_rendering()
        self._renderer.update_scene(data, camera=self.config.camera_name)
        seg = np.asarray(self._renderer.render(), dtype=np.int32)
        self._renderer.disable_segmentation_rendering()

        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        mask = (seg[..., 0] == self.contract.box_geom_id) & (seg[..., 1] == geom_type)
        pixel_count = int(mask.sum())
        total_pixels = int(self.config.width * self.config.height)
        pixel_fraction = pixel_count / float(total_pixels)
        if pixel_count > 0:
            ys, xs = np.where(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        else:
            bbox = None

        in_frustum, depth_m = _box_center_frustum_check(
            self.model, data, self.contract, self.config.width, self.config.height
        )
        visible = (
            pixel_count >= self.config.min_box_pixels
            and pixel_fraction >= self.config.min_box_pixel_fraction
            and depth_m > 0.0
        )
        evidence = VisibilityEvidence(
            signal_type="mujoco_geom_segmentation_and_frustum",
            visible=bool(visible),
            box_pixel_count=pixel_count,
            box_pixel_fraction=round(pixel_fraction, 6),
            image_width=int(self.config.width),
            image_height=int(self.config.height),
            box_bbox_px=bbox,
            min_box_pixels=int(self.config.min_box_pixels),
            min_box_pixel_fraction=float(self.config.min_box_pixel_fraction),
            box_center_in_frustum=bool(in_frustum),
            box_center_depth_m=round(depth_m, 5),
        )
        return rgb, evidence
