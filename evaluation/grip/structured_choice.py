"""Shared structured-output schema and verifier for OpenArm Grip Benchmark choices."""

from __future__ import annotations

import json
from typing import Any, Final, Literal

Choice = Literal["close", "wait", "uncertain"]

CHOICES: Final[tuple[Choice, ...]] = ("close", "wait", "uncertain")

CHOICE_OUTPUT_FORMAT: Final[dict[str, Any]] = {
    "name": "grip_choice",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "choice": {
                "type": "string",
                "enum": list(CHOICES),
            },
        },
        "required": ["choice"],
        "additionalProperties": False,
    },
}

_CHOICE_BY_VALUE: Final[dict[str, Choice]] = {choice: choice for choice in CHOICES}


def _load_unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Decode a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_choice_json(text: object) -> Choice | None:
    """Parse and strictly validate a JSON-encoded grip choice response.

    Accepts a JSON string representing an object with exactly one ``choice``
    key whose value is one of ``CHOICES`` (outer JSON whitespace is allowed).
    Returns the validated choice string, or ``None`` for any invalid input.
    """
    if not isinstance(text, str):
        return None

    try:
        payload = json.loads(text, object_pairs_hook=_load_unique_object)
    except (ValueError, RecursionError):
        return None

    if not isinstance(payload, dict) or len(payload) != 1 or "choice" not in payload:
        return None

    value = payload["choice"]
    if not isinstance(value, str):
        return None

    return _CHOICE_BY_VALUE.get(value)


__all__ = [
    "CHOICE_OUTPUT_FORMAT",
    "CHOICES",
    "Choice",
    "parse_choice_json",
]
