"""Stdlib-only helpers for encoding input images in OpenArm Grip evaluation."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["ImageInput", "encode_data_url", "image_content_parts"]


@dataclass(frozen=True)
class ImageInput:
    """Immutable encoded image payload and its MIME type."""

    rgb_bytes: bytes
    mime_type: str = "image/png"


def encode_data_url(rgb_bytes: bytes, mime_type: str) -> str:
    """Encode raw image bytes as a standard base64 data URL.

    Args:
        rgb_bytes: Encoded image bytes (for example PNG or JPEG data).
        mime_type: MIME type of the image payload (for example ``'image/png'``).

    Returns:
        A ``data:<mime_type>;base64,<payload>`` URL string.
    """
    if not mime_type or not mime_type.strip():
        raise ValueError("mime_type must be a non-empty string")
    encoded = base64.b64encode(rgb_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_content_parts(
    images: Sequence[ImageInput],
    *,
    kind: Literal["input_image", "image_url"],
) -> list[dict[str, Any]]:
    """Build ordered multimodal content parts for OpenAI-compatible APIs.

    Args:
        images: Ordered sequence of ``ImageInput`` instances to serialize.
        kind: Target wire format:
            - ``'input_image'`` for OpenAI Decisions and Responses APIs
              (``{'type': 'input_image', 'image_url': url}``).
            - ``'image_url'`` for Chat Completions and vLLM
              (``{'type': 'image_url', 'image_url': {'url': url}}``).

    Returns:
        A list of content-part dictionaries preserving the input image order.

    Raises:
        ValueError: If ``kind`` is not ``'input_image'`` or ``'image_url'``.
    """
    if kind not in ("input_image", "image_url"):
        raise ValueError(f"Unsupported image content part kind: {kind!r}")

    parts: list[dict[str, Any]] = []
    for image in images:
        url = encode_data_url(image.rgb_bytes, image.mime_type)
        if kind == "input_image":
            parts.append({"type": "input_image", "image_url": url})
        else:
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts
