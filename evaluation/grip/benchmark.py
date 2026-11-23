"""Offline Grip Pilot benchmark runner.

Runs a fixed wrist-camera grippability decision task over a Stage-1 dataset
(`observation/manifest.jsonl` + `evaluator/labels.jsonl`, plus optional
`evaluator/splits.json`) while strictly isolating model backends from all
evaluator metadata, paths, IDs, and labels.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

Choice = Literal["close", "wait", "uncertain"]
Status = Literal["ok", "timeout", "http_error", "invalid_response", "refusal"]
SplitFilter = Literal["dev", "eval", "all"]

SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "grip-pilot-v1"
QUESTION_NAME = "grip"
CHOICES: tuple[Choice, ...] = ("close", "wait", "uncertain")
VALID_CHOICES: frozenset[str] = frozenset(CHOICES)
DEADLINE_MS_THRESHOLDS: tuple[int, ...] = (100, 200, 500)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_LOCAL_VLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TIMEOUT_S = 20.0

SHARED_QUESTION = (
    "Look at the robot's wrist-camera image. Can the open gripper pick up the orange "
    "box by closing its jaws at the current position and orientation, then lifting?"
)
CHOICE_DESCRIPTIONS: dict[Choice, str] = {
    "close": "The box is positioned within reach between the jaws. Close now.",
    "wait": "The box is outside the region the jaws can grip. Do not close yet.",
    "uncertain": "The image does not give enough evidence about the box and jaws.",
}
LOCAL_VLM_PROMPT = (
    SHARED_QUESTION
    + "\n"
    + "\n".join(f"{key}: {text}" for key, text in CHOICE_DESCRIPTIONS.items())
    + "\nReply with exactly one choice word."
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class BackendResult:
    """Validated single-image result from a benchmark backend."""

    status: Status
    prediction: Choice | None
    elapsed_ms: float
    response_model: str | None = None
    usage: dict[str, Any] | None = None
    http_status: int | None = None
    error: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None


class DecisionBackend(Protocol):
    """Strict observation-only backend interface: RGB bytes + mime_type only."""

    name: str
    model: str

    @property
    def base_url(self) -> str | None: ...

    def predict(
        self, rgb_bytes: bytes, *, mime_type: str = "image/png"
    ) -> BackendResult: ...


def _endpoint_url(base_url: str, v1_path: str) -> str:
    """Join base URL with an OpenAI-style /v1/<resource> path without duplicating /v1."""
    clean_base = base_url.strip().rstrip("/")
    clean_path = v1_path.lstrip("/")
    if not clean_base:
        raise ValueError("base_url must be non-empty")
    if clean_base.endswith(f"/v1/{clean_path}"):
        return clean_base
    if clean_base.endswith("/v1"):
        return f"{clean_base}/{clean_path}"
    return f"{clean_base}/v1/{clean_path}"


def _encode_data_url(rgb_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(rgb_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def detect_image_mime_type(rgb_bytes: bytes, path: Path) -> str:
    """Infer MIME type strictly from image bytes or extension."""
    if rgb_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if rgb_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if rgb_bytes.startswith(b"RIFF") and rgb_bytes[8:12] == b"WEBP":
        return "image/webp"
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def parse_local_vlm_choice(raw_text: Any) -> Choice | None:
    """Strictly parse a chat completion content string as an enum or simple JSON choice."""
    if isinstance(raw_text, list):
        parts: list[str] = []
        for item in raw_text:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        text = "".join(parts)
    elif isinstance(raw_text, str):
        text = raw_text
    else:
        return None

    stripped = text.strip()
    if not stripped:
        return None

    fenced = _CODE_FENCE_RE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()

    lowered = stripped.lower()
    if lowered in VALID_CHOICES:
        return cast(Choice, lowered)

    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and isinstance(obj.get("choice"), str):
            val = obj["choice"].strip().lower()
            if val in VALID_CHOICES:
                return cast(Choice, val)
    return None


def parse_openai_decisions_payload(
    body: Any,
) -> tuple[Choice | None, float | None, dict[str, float] | None, str | None]:
    """Validate an OpenAI /v1/decisions JSON response body."""
    if not isinstance(body, dict):
        return None, None, None, "response JSON root is not an object"
    answers = body.get("answers")
    if not isinstance(answers, list) or not answers:
        return None, None, None, "missing or empty 'answers' list"

    matched: dict[str, Any] | None = None
    for entry in answers:
        if isinstance(entry, dict) and entry.get("name") == QUESTION_NAME:
            matched = entry
            break
    if matched is None:
        if (
            len(answers) == 1
            and isinstance(answers[0], dict)
            and answers[0].get("name") is None
        ):
            matched = answers[0]
        else:
            return None, None, None, f"no answer matching question '{QUESTION_NAME}'"

    ans_type = matched.get("type")
    if ans_type == "refusal":
        return None, None, None, "refusal"
    if ans_type is not None and ans_type != "choice":
        return None, None, None, f"unexpected answer type: {ans_type!r}"

    choice_raw = matched.get("choice")
    if not isinstance(choice_raw, str):
        return None, None, None, "missing string 'choice' field"
    choice_norm = choice_raw.strip().lower()
    if choice_norm not in VALID_CHOICES:
        return None, None, None, f"invalid choice value: {choice_raw!r}"

    confidence: float | None = None
    raw_conf = matched.get("confidence")
    if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool):
        confidence = float(raw_conf)

    probabilities: dict[str, float] | None = None
    raw_probs = matched.get("probabilities")
    if isinstance(raw_probs, list):
        parsed_probs: dict[str, float] = {}
        for entry in raw_probs:
            if isinstance(entry, dict):
                value, probability = entry.get("value"), entry.get("probability")
                if (
                    isinstance(value, str)
                    and value in CHOICES
                    and type(probability) in (int, float)
                ):
                    parsed_probs[value] = float(probability)
        probabilities = parsed_probs or None

    return cast(Choice, choice_norm), confidence, probabilities, None


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout_s: float,
) -> tuple[int | None, bytes | None, Status | None, str | None]:
    """Execute a single HTTP POST with fixed timeout and no retries."""
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        **headers,
    }
    req = urllib.request.Request(url=url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status_code = int(getattr(resp, "status", 200))
            raw = resp.read()
            return status_code, raw, None, None
    except urllib.error.HTTPError as exc:
        return int(exc.code), None, "http_error", f"HTTP {exc.code}"
    except TimeoutError as exc:
        return None, None, "timeout", f"request timed out: {exc}"
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return None, None, "timeout", f"request timed out: {reason}"
        return None, None, "http_error", f"URL error: {reason}"
    except (http.client.HTTPException, OSError) as exc:
        if isinstance(exc, TimeoutError):
            return None, None, "timeout", f"request timed out: {exc}"
        return None, None, "http_error", f"transport error: {exc}"


def _elapsed_ms(clock: Callable[[], float], start_s: float) -> float:
    return max(0.0, (clock() - start_s) * 1000.0)


class AlwaysWaitBackend:
    """Deterministic zero-network baseline that always returns 'wait'."""

    name = "always-wait"

    def __init__(
        self,
        *,
        model: str = "always-wait",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.model = model
        self.base_url: str | None = None
        self._clock = clock

    def predict(
        self, rgb_bytes: bytes, *, mime_type: str = "image/png"
    ) -> BackendResult:
        t0 = self._clock()
        if not rgb_bytes:
            return BackendResult(
                status="invalid_response",
                prediction=None,
                elapsed_ms=_elapsed_ms(self._clock, t0),
                response_model=self.model,
                error="empty RGB input",
            )
        return BackendResult(
            status="ok",
            prediction="wait",
            elapsed_ms=_elapsed_ms(self._clock, t0),
            response_model=self.model,
        )


class OpenAIDecisionsBackend:
    """OpenAI Decisions API (`POST /v1/decisions`) backend for gpt-6-luna."""

    name = "openai-decisions"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key_env: str = DEFAULT_OPENAI_API_KEY_ENV,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env!r} is required for backend 'openai-decisions'."
            )
        self.base_url = base_url.strip().rstrip("/")
        self.endpoint = _endpoint_url(self.base_url, "decisions")
        self.model = model
        self.timeout_s = float(timeout_s)
        self._api_key = api_key
        self._clock = clock

    def build_payload(
        self, rgb_bytes: bytes, *, mime_type: str = "image/png"
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": _encode_data_url(rgb_bytes, mime_type),
                        },
                    ],
                }
            ],
            "questions": [
                {
                    "type": "choice",
                    "name": QUESTION_NAME,
                    "instructions": SHARED_QUESTION,
                    "choices": [
                        {"value": value, "description": desc}
                        for value, desc in CHOICE_DESCRIPTIONS.items()
                    ],
                }
            ],
        }

    def predict(
        self, rgb_bytes: bytes, *, mime_type: str = "image/png"
    ) -> BackendResult:
        t0 = self._clock()
        payload = self.build_payload(rgb_bytes, mime_type=mime_type)
        http_status, raw, err_status, err_detail = _post_json(
            self.endpoint,
            payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout_s=self.timeout_s,
        )
        if err_status is not None:
            return BackendResult(
                status=err_status,
                prediction=None,
                elapsed_ms=_elapsed_ms(self._clock, t0),
                http_status=http_status,
                error=err_detail,
            )

        raw_text = (raw or b"").decode("utf-8", errors="replace")
        try:
            body = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            return BackendResult(
                status="invalid_response",
                prediction=None,
                elapsed_ms=_elapsed_ms(self._clock, t0),
                http_status=http_status,
                error=f"invalid JSON: {exc}",
            )

        response_model = (
            body.get("model")
            if isinstance(body, dict) and isinstance(body.get("model"), str)
            else None
        )
        usage = (
            body.get("usage")
            if isinstance(body, dict) and isinstance(body.get("usage"), dict)
            else None
        )

        choice, confidence, probabilities, parse_err = parse_openai_decisions_payload(
            body
        )
        elapsed_ms = _elapsed_ms(self._clock, t0)
        if choice is None:
            return BackendResult(
                status="refusal" if parse_err == "refusal" else "invalid_response",
                prediction=None,
                elapsed_ms=elapsed_ms,
                response_model=response_model,
                usage=usage,
                http_status=http_status,
                error=parse_err or "invalid decisions response",
            )

        return BackendResult(
            status="ok",
            prediction=choice,
            elapsed_ms=elapsed_ms,
            response_model=response_model,
            usage=usage,
            http_status=http_status,
            confidence=confidence,
            probabilities=probabilities,
        )


class LocalVLMBackend:
    """OpenAI-compatible `/v1/chat/completions` backend for local VLM servers."""

    name = "local-vlm"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_LOCAL_VLM_BASE_URL,
        model: str,
        api_key_env: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if not model or not model.strip():
            raise ValueError("--model is required for backend 'local-vlm'.")
        api_key: str | None = None
        if api_key_env:
            api_key = os.environ.get(api_key_env, "").strip()
            if not api_key:
                raise ValueError(
                    f"Environment variable {api_key_env!r} was specified via --api-key-env but is unset or empty."
                )
        self.base_url = base_url.strip().rstrip("/")
        self.endpoint = _endpoint_url(self.base_url, "chat/completions")
        self.model = model.strip()
        self.timeout_s = float(timeout_s)
        self._api_key = api_key
        self._clock = clock

    def build_payload(
        self, rgb_bytes: bytes, *, mime_type: str = "image/png"
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 16,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": LOCAL_VLM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _encode_data_url(rgb_bytes, mime_type)
                            },
                        },
                    ],
                }
            ],
        }

    def predict(
        self, rgb_bytes: bytes, *, mime_type: str = "image/png"
    ) -> BackendResult:
        t0 = self._clock()
        payload = self.build_payload(rgb_bytes, mime_type=mime_type)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        http_status, raw, err_status, err_detail = _post_json(
            self.endpoint,
            payload,
            headers=headers,
            timeout_s=self.timeout_s,
        )
        if err_status is not None:
            return BackendResult(
                status=err_status,
                prediction=None,
                elapsed_ms=_elapsed_ms(self._clock, t0),
                http_status=http_status,
                error=err_detail,
            )

        raw_text = (raw or b"").decode("utf-8", errors="replace")
        try:
            body = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            return BackendResult(
                status="invalid_response",
                prediction=None,
                elapsed_ms=_elapsed_ms(self._clock, t0),
                http_status=http_status,
                error=f"invalid JSON: {exc}",
            )

        response_model = (
            body.get("model")
            if isinstance(body, dict) and isinstance(body.get("model"), str)
            else None
        )
        usage = (
            body.get("usage")
            if isinstance(body, dict) and isinstance(body.get("usage"), dict)
            else None
        )

        content: Any = None
        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                msg = choices[0].get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")

        choice = parse_local_vlm_choice(content)
        elapsed_ms = _elapsed_ms(self._clock, t0)
        if choice is None:
            return BackendResult(
                status="invalid_response",
                prediction=None,
                elapsed_ms=elapsed_ms,
                response_model=response_model,
                usage=usage,
                http_status=http_status,
                error="could not parse valid choice ('close'|'wait'|'uncertain') from chat completion",
            )

        return BackendResult(
            status="ok",
            prediction=choice,
            elapsed_ms=elapsed_ms,
            response_model=response_model,
            usage=usage,
            http_status=http_status,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise TypeError(f"{path}:{lineno}: expected JSON object")
        rows.append(obj)
    return rows


def load_manifest(dataset_dir: Path) -> list[dict[str, Any]]:
    """Read `observation/manifest.jsonl` without touching `evaluator/`."""
    manifest_path = dataset_dir / "observation" / "manifest.jsonl"
    rows = _read_jsonl(manifest_path)
    for idx, row in enumerate(rows):
        for key in ("sample_id", "group_id", "rgb_path"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(
                    f"{manifest_path}: row {idx} missing non-empty string {key!r}"
                )
    return rows


def load_splits(dataset_dir: Path) -> dict[str, str] | None:
    """Read optional `evaluator/splits.json` mapping group_id -> 'dev' | 'eval'."""
    splits_path = dataset_dir / "evaluator" / "splits.json"
    if not splits_path.exists():
        return None
    raw = json.loads(splits_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(
            f"{splits_path}: expected JSON object mapping group_id -> split"
        )

    mapping: Any = raw
    for wrapper_key in ("groups", "splits"):
        if wrapper_key in raw and isinstance(raw[wrapper_key], dict):
            mapping = raw[wrapper_key]
            break

    result: dict[str, str] = {}
    for gid, split_val in mapping.items():
        if not isinstance(gid, str) or split_val not in ("dev", "eval"):
            raise ValueError(
                f"{splits_path}: invalid split entry {gid!r}: {split_val!r} (expected 'dev' or 'eval')"
            )
        result[gid] = split_val
    return result


def load_labels(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    """Read `evaluator/labels.jsonl` indexed by `sample_id`."""
    labels_path = dataset_dir / "evaluator" / "labels.jsonl"
    rows = _read_jsonl(labels_path)
    indexed: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        sid = row.get("sample_id")
        label = row.get("label")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"{labels_path}: row {idx} missing 'sample_id'")
        if label not in (0, 1, None) or isinstance(label, bool):
            raise ValueError(f"{labels_path}: row {idx} invalid label {label!r}")
        indexed[sid] = row
    return indexed


def resolve_rgb_path(dataset_dir: Path, rgb_path: str) -> Path:
    """Resolve image paths within the dataset's observation directory."""
    observation_dir = (dataset_dir / "observation").resolve()
    candidate = (observation_dir / rgb_path).resolve()
    if not candidate.is_relative_to(observation_dir):
        raise ValueError(f"Image path escapes the observation directory: {rgb_path}")
    return candidate


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / float(denominator)


def _percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile matching numpy.percentile(..., method='linear')."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def compute_summary(
    records: list[dict[str, Any]],
    *,
    dataset_dir: str | Path,
    backend_name: str,
    requested_model: str,
    base_url: str | None,
    split: SplitFilter,
    limit: int | None,
    timeout_s: float | None,
) -> dict[str, Any]:
    """Compute Grip Pilot metrics with explicit sample counts, denominators, and coverage."""
    total_samples = len(records)
    positives = [r for r in records if r["label"] == 1]
    negatives = [r for r in records if r["label"] == 0]
    null_labels = [r for r in records if r["label"] is None]
    labeled_samples = len(positives) + len(negatives)

    ok_records = [r for r in records if r["status"] == "ok"]
    ok_count = len(ok_records)

    timeout_count = sum(1 for r in records if r["status"] == "timeout")
    http_error_count = sum(1 for r in records if r["status"] == "http_error")
    invalid_response_count = sum(
        1 for r in records if r["status"] == "invalid_response"
    )
    refused_count = sum(1 for r in records if r["status"] == "refusal")
    total_failures = (
        timeout_count + http_error_count + invalid_response_count + refused_count
    )

    pred_close_count = sum(1 for r in ok_records if r["prediction"] == "close")
    pred_wait_count = sum(1 for r in ok_records if r["prediction"] == "wait")
    pred_uncertain_count = sum(1 for r in ok_records if r["prediction"] == "uncertain")
    pred_none_count = sum(1 for r in records if r["prediction"] is None)

    # FCR: false closes on ground-truth negatives (label == 0).
    neg_total = len(negatives)
    neg_ok = [r for r in negatives if r["status"] == "ok"]
    false_closes = sum(1 for r in neg_ok if r["prediction"] == "close")

    # Close recall: true closes on ground-truth positives (label == 1).
    pos_total = len(positives)
    pos_ok = [r for r in positives if r["status"] == "ok"]
    true_closes = sum(1 for r in pos_ok if r["prediction"] == "close")

    # Deadline-qualified close recall at 100 / 200 / 500 ms over all positives (label == 1).
    deadline_close_recall: dict[str, Any] = {}
    for deadline_ms in DEADLINE_MS_THRESHOLDS:
        hits = sum(
            1
            for r in pos_ok
            if r["prediction"] == "close"
            and float(r["elapsed_ms"]) <= float(deadline_ms)
        )
        deadline_close_recall[f"{deadline_ms}ms"] = {
            "deadline_ms": deadline_ms,
            "rate": _safe_rate(hits, pos_total),
            "numerator": hits,
            "denominator": pos_total,
            "ok_count": len(pos_ok),
            "coverage": _safe_rate(len(pos_ok), pos_total),
        }

    # Valid-only latency percentiles (excludes timeout, http_error, invalid_response).
    valid_latencies = [float(r["elapsed_ms"]) for r in ok_records]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_dir": str(dataset_dir),
        "backend": backend_name,
        "requested_model": requested_model,
        "base_url": base_url,
        "split": split,
        "limit": limit,
        "timeout_s": timeout_s,
        "prompt_version": PROMPT_VERSION,
        "choices": list(CHOICES),
        "counts": {
            "total_samples": total_samples,
            "labeled_samples": labeled_samples,
            "positive_samples": pos_total,
            "negative_samples": neg_total,
            "null_label_samples": len(null_labels),
            "ok_count": ok_count,
            "refusals": refused_count,
            "valid_response_coverage": _safe_rate(ok_count, total_samples),
            "predictions": {
                "close": pred_close_count,
                "wait": pred_wait_count,
                "uncertain": pred_uncertain_count,
                "none": pred_none_count,
            },
            "failures": {
                "timeout": timeout_count,
                "http_error": http_error_count,
                "invalid_response": invalid_response_count,
                "total": total_failures,
            },
        },
        "metrics": {
            "fcr": {
                "rate": _safe_rate(false_closes, neg_total),
                "numerator": false_closes,
                "denominator": neg_total,
                "ok_count": len(neg_ok),
                "coverage": _safe_rate(len(neg_ok), neg_total),
                "rate_among_ok": _safe_rate(false_closes, len(neg_ok)),
            },
            "close_recall": {
                "rate": _safe_rate(true_closes, pos_total),
                "numerator": true_closes,
                "denominator": pos_total,
                "ok_count": len(pos_ok),
                "coverage": _safe_rate(len(pos_ok), pos_total),
                "rate_among_ok": _safe_rate(true_closes, len(pos_ok)),
            },
            "abstention": {
                "rate": _safe_rate(pred_uncertain_count, total_samples),
                "numerator": pred_uncertain_count,
                "denominator": total_samples,
                "ok_count": ok_count,
                "coverage": _safe_rate(ok_count, total_samples),
                "rate_among_ok": _safe_rate(pred_uncertain_count, ok_count),
            },
            "deadline_close_recall": deadline_close_recall,
            "valid_latency_ms": {
                "p50": _percentile(valid_latencies, 50.0),
                "p95": _percentile(valid_latencies, 95.0),
                "count": ok_count,
                "total_samples": total_samples,
                "coverage": _safe_rate(ok_count, total_samples),
            },
        },
    }


def create_backend(
    backend_name: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key_env: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> DecisionBackend:
    """Instantiate a configured benchmark backend."""
    if backend_name == "always-wait":
        return AlwaysWaitBackend(model=model or "always-wait")
    if backend_name == "openai-decisions":
        return OpenAIDecisionsBackend(
            base_url=base_url or DEFAULT_OPENAI_BASE_URL,
            model=model or DEFAULT_OPENAI_MODEL,
            api_key_env=api_key_env or DEFAULT_OPENAI_API_KEY_ENV,
            timeout_s=timeout_s,
        )
    if backend_name == "local-vlm":
        return LocalVLMBackend(
            base_url=base_url or DEFAULT_LOCAL_VLM_BASE_URL,
            model=model or "",
            api_key_env=api_key_env,
            timeout_s=timeout_s,
        )
    raise ValueError(f"Unsupported backend: {backend_name!r}")


def run_benchmark(
    *,
    dataset_dir: str | Path,
    out_dir: str | Path,
    backend_name: str | None = None,
    backend: DecisionBackend | None = None,
    split: SplitFilter = "all",
    limit: int | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key_env: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run offline inference on selected observations, then join labels and persist outputs."""
    if split not in ("dev", "eval", "all"):
        raise ValueError(f"split must be 'dev', 'eval', or 'all', got {split!r}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")

    ds_path = Path(dataset_dir)
    out_path = Path(out_dir)
    if out_path.exists() or out_path.is_symlink():
        raise FileExistsError(f"Output directory already exists: {out_path}")

    if backend is None:
        if not backend_name:
            raise ValueError("Either backend or backend_name must be provided.")
        active_backend = create_backend(
            backend_name,
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            timeout_s=timeout_s,
        )
    else:
        active_backend = backend

    # Phase 1: Observation manifest + optional split filtering, then --limit.
    observations = load_manifest(ds_path)
    split_map = load_splits(ds_path)
    if split != "all":
        if split_map is None:
            raise FileNotFoundError(
                f"Requested --split {split!r} but {ds_path / 'evaluator' / 'splits.json'} does not exist."
            )
        observations = [
            obs for obs in observations if split_map.get(obs["group_id"]) == split
        ]
    if limit is not None:
        observations = observations[:limit]

    # Phase 2: Inference loop. Backend sees ONLY raw RGB bytes and MIME type.
    raw_results: list[tuple[dict[str, Any], BackendResult]] = []
    for obs in observations:
        img_path = resolve_rgb_path(ds_path, obs["rgb_path"])
        rgb_bytes = img_path.read_bytes()
        mime_type = detect_image_mime_type(rgb_bytes, img_path)
        result = active_backend.predict(rgb_bytes, mime_type=mime_type)
        raw_results.append((obs, result))

    # Phase 3: Load evaluator labels AFTER inference and join strictly by sample_id.
    labels_by_id = load_labels(ds_path)
    prediction_rows: list[dict[str, Any]] = []
    for obs, result in raw_results:
        sid = obs["sample_id"]
        gid = obs["group_id"]
        if sid not in labels_by_id:
            raise ValueError(
                f"sample_id {sid!r} from manifest not found in evaluator/labels.jsonl"
            )
        label_row = labels_by_id[sid]
        prediction_rows.append(
            {
                "sample_id": sid,
                "group_id": gid,
                "split": split_map.get(gid) if split_map is not None else None,
                "backend": active_backend.name,
                "requested_model": active_backend.model,
                "response_model": result.response_model,
                "status": result.status,
                "prediction": result.prediction,
                "elapsed_ms": round(float(result.elapsed_ms), 4),
                "http_status": result.http_status,
                "error": result.error,
                "usage": result.usage,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "label": label_row["label"],
                "outcome": label_row.get("outcome"),
            }
        )

    summary = compute_summary(
        prediction_rows,
        dataset_dir=ds_path,
        backend_name=active_backend.name,
        requested_model=active_backend.model,
        base_url=active_backend.base_url,
        split=split,
        limit=limit,
        timeout_s=getattr(active_backend, "timeout_s", None),
    )

    out_path.mkdir(parents=True)
    predictions_file = out_path / "predictions.jsonl"
    summary_file = out_path / "summary.json"
    predictions_file.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    summary_file.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.grip.benchmark",
        description="Run the offline Grip Pilot benchmark over a Stage-1 dataset.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Dataset directory containing observation/manifest.jsonl and evaluator/labels.jsonl.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory where predictions.jsonl and summary.json will be written.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("openai-decisions", "local-vlm", "always-wait"),
        help="Inference backend to evaluate.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=("dev", "eval", "all"),
        help="Optional group split from evaluator/splits.json (default: all).",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Optional max number of samples to evaluate, applied after --split.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            f"API base URL (default: {DEFAULT_OPENAI_BASE_URL} for openai-decisions, "
            f"{DEFAULT_LOCAL_VLM_BASE_URL} for local-vlm)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Model identifier (default: {DEFAULT_OPENAI_MODEL} for openai-decisions; "
            "required for local-vlm)."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help=(
            f"Environment variable holding the Bearer API key "
            f"(default: {DEFAULT_OPENAI_API_KEY_ENV} for openai-decisions; optional for local-vlm)."
        ),
    )
    parser.add_argument(
        "--timeout",
        default=DEFAULT_TIMEOUT_S,
        type=float,
        help=f"Fixed per-request HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_benchmark(
        dataset_dir=args.dataset,
        out_dir=args.out,
        backend_name=args.backend,
        split=args.split,
        limit=args.limit,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout_s=args.timeout,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
