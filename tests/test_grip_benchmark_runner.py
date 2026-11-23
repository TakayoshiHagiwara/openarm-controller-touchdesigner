"""Tests for the offline Grip Pilot benchmark runner."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

import pytest

from evaluation.grip import benchmark as bm


EXPECTED_CHOICE_FORMAT = {
    "name": "grip_choice",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": ["close", "wait", "uncertain"]}
        },
        "required": ["choice"],
        "additionalProperties": False,
    },
}

# Minimal valid 1x1 orange PNG bytes (no external image libraries required).
TINY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _write_dataset(
    dataset_dir: Path,
    samples: list[dict[str, Any]],
    *,
    splits: dict[str, str] | None = None,
) -> Path:
    """Create a self-contained observation/evaluator dataset fixture."""
    obs_dir = dataset_dir / "observation"
    eval_dir = dataset_dir / "evaluator"
    (obs_dir / "images").mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    obs_lines: list[str] = []
    label_lines: list[str] = []
    for item in samples:
        sid = item["sample_id"]
        gid = item["group_id"]
        rel_rgb = item.get("rgb_path", f"images/{sid}.png")
        img_bytes = item.get("rgb_bytes", TINY_PNG_BYTES)
        img_path = obs_dir / rel_rgb
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(img_bytes)

        obs_lines.append(
            json.dumps(
                {
                    "sample_id": sid,
                    "group_id": gid,
                    "rgb_path": rel_rgb,
                    "capture_timestamp_utc": "2026-11-23T00:00:00+00:00",
                    "sim_time_s": 0.5,
                    "camera": {"name": "right_wrist_cam", "width": 640, "height": 480},
                    "observation_schema_version": "1.0",
                },
                sort_keys=True,
            )
        )
        label_lines.append(
            json.dumps(
                {
                    "sample_id": sid,
                    "group_id": gid,
                    "outcome": item.get(
                        "outcome",
                        "physics_success"
                        if item["label"] == 1
                        else ("physics_failure" if item["label"] == 0 else "invalid"),
                    ),
                    "label": item["label"],
                    "snapshot_path": f"states/{sid}.npz",
                    "proposal": {
                        "proposal_kind": "secret_offset_hint",
                        "hand_offset_xyz_m": [0.001, -0.002, 0.003],
                    },
                    "evaluator_schema_version": "1.0",
                },
                sort_keys=True,
            )
        )

    (obs_dir / "manifest.jsonl").write_text(
        "\n".join(obs_lines) + "\n", encoding="utf-8"
    )
    (eval_dir / "labels.jsonl").write_text(
        "\n".join(label_lines) + "\n", encoding="utf-8"
    )
    if splits is not None:
        (eval_dir / "splits.json").write_text(
            json.dumps(splits, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return dataset_dir


class _LoopbackServer:
    """Deterministic local HTTP stub recording incoming requests."""

    def __init__(self, responder: Any) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responder = responder

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length)
                parsed_body = json.loads(raw_body.decode("utf-8")) if raw_body else None
                rec = {
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "raw_body": raw_body.decode("utf-8"),
                    "json": parsed_body,
                }
                outer.requests.append(rec)
                status, payload, delay_s = outer._responder(
                    len(outer.requests) - 1, rec
                )
                if delay_s > 0:
                    time.sleep(delay_s)
                if isinstance(payload, (dict, list)):
                    body_bytes = json.dumps(payload).encode("utf-8")
                    content_type = "application/json"
                else:
                    body_bytes = str(payload).encode("utf-8")
                    content_type = "text/plain"
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body_bytes)))
                    self.end_headers()
                    self.wfile.write(body_bytes)
                except BrokenPipeError:
                    pass

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


def test_terra_disables_reasoning_for_vision_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-test-key")
    backend = bm.create_backend("openai-terra")
    assert isinstance(backend, bm.OpenAITerraBackend)
    payload = backend.build_payload(TINY_PNG_BYTES)
    assert backend.endpoint == "https://api.openai.com/v1/chat/completions"
    assert payload["model"] == "gpt-6.1-terra"
    assert payload["reasoning_effort"] == "none"
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": EXPECTED_CHOICE_FORMAT,
    }
    assert payload["messages"][0]["content"][1]["type"] == "image_url"


def test_openai_decisions_payload_parse_and_clean_separation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _write_dataset(
        tmp_path / "ds",
        [
            {"sample_id": "smp_alpha_secret", "group_id": "grp_9999", "label": 1},
            {"sample_id": "smp_beta_secret", "group_id": "grp_9999", "label": 0},
        ],
        splits={"grp_9999": "eval"},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")

    responses = [
        (
            200,
            {
                "model": "gpt-6-luna-2026",
                "answers": [
                    {
                        "type": "choice",
                        "name": bm.QUESTION_NAME,
                        "choice": "close",
                        "confidence": 0.94,
                        "probabilities": [
                            {"value": "close", "probability": 0.94},
                            {"value": "wait", "probability": 0.04},
                            {"value": "uncertain", "probability": 0.02},
                        ],
                    }
                ],
                "usage": {"input_tokens": 120, "output_tokens": 0, "total_tokens": 120},
            },
            0.0,
        ),
        (
            200,
            {
                "model": "gpt-6-luna-2026",
                "answers": [
                    {
                        "type": "choice",
                        "name": bm.QUESTION_NAME,
                        "choice": "wait",
                        "confidence": 0.88,
                        "probabilities": [
                            {"value": "close", "probability": 0.07},
                            {"value": "wait", "probability": 0.88},
                            {"value": "uncertain", "probability": 0.05},
                        ],
                    }
                ],
                "usage": {"input_tokens": 120, "output_tokens": 0, "total_tokens": 120},
            },
            0.0,
        ),
    ]

    with _LoopbackServer(lambda idx, _req: responses[idx]) as srv:
        out_dir = tmp_path / "out"
        summary = bm.run_benchmark(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            backend_name="openai-decisions",
            base_url=srv.base_url,
            split="eval",
            timeout_s=5.0,
        )

    assert len(srv.requests) == 2
    expected_data_url = (
        f"data:image/png;base64,{base64.b64encode(TINY_PNG_BYTES).decode('ascii')}"
    )
    for req in srv.requests:
        assert req["path"] == "/v1/decisions"
        assert req["headers"]["authorization"] == "Bearer sk-test-dummy-key"
        body = req["json"]
        assert body["model"] == "gpt-6-luna"
        content = body["input"][0]["content"]
        img_parts = [p for p in content if p.get("type") == "input_image"]
        assert len(img_parts) == 1
        assert img_parts[0]["image_url"] == expected_data_url
        q = body["questions"][0]
        assert q["type"] == "choice"
        assert q["name"] == bm.QUESTION_NAME
        assert [c["value"] for c in q["choices"]] == ["close", "wait", "uncertain"]
        assert "orange box" in q["instructions"].lower()

        # Clean separation: zero metadata/label/split/path leakage in outbound payload.
        for forbidden in (
            "smp_alpha_secret",
            "smp_beta_secret",
            "grp_9999",
            "images/",
            "physics_success",
            "physics_failure",
            "secret_offset_hint",
            "hand_offset_xyz_m",
            "eval",
        ):
            assert forbidden not in req["raw_body"]

    pred_rows = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [r["prediction"] for r in pred_rows] == ["close", "wait"]
    assert all(r["status"] == "ok" for r in pred_rows)
    assert pred_rows[0]["response_model"] == "gpt-6-luna-2026"
    assert pred_rows[0]["usage"]["total_tokens"] == 120
    assert pred_rows[0]["confidence"] == pytest.approx(0.94)
    assert pred_rows[0]["probabilities"] == {
        "close": 0.94,
        "wait": 0.04,
        "uncertain": 0.02,
    }
    assert "sk-test-dummy-key" not in (out_dir / "predictions.jsonl").read_text(
        encoding="utf-8"
    )
    assert "sk-test-dummy-key" not in (out_dir / "summary.json").read_text(
        encoding="utf-8"
    )
    assert summary["metrics"]["fcr"]["rate"] == pytest.approx(0.0)
    assert summary["metrics"]["close_recall"]["rate"] == pytest.approx(1.0)


def test_local_vlm_payload_strict_parse_and_optional_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [
        {"sample_id": f"smp_{i:02d}", "group_id": "grp_0000", "label": i % 2}
        for i in range(6)
    ]
    dataset_dir = _write_dataset(tmp_path / "ds", samples)
    monkeypatch.setenv("LOCAL_VLM_TOKEN", "local-secret-token")

    reply_contents = [
        '{"choice": "close"}',
        '  {"choice": "wait"} \n',
        '{"choice": "uncertain"}',
        "close",
        '{"choice": "close", "explanation": "aligned"}',
        '{"choice": "grab"}',
    ]

    def responder(idx: int, _req: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
        return (
            200,
            {
                "model": "qwen2.5-vl-7b",
                "choices": [
                    {"message": {"role": "assistant", "content": reply_contents[idx]}}
                ],
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 5,
                    "total_tokens": 85,
                },
            },
            0.0,
        )

    with _LoopbackServer(responder) as srv:
        out_dir = tmp_path / "out"
        summary = bm.run_benchmark(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            backend_name="local-vlm",
            base_url=srv.base_url,
            model="qwen2.5-vl-7b",
            api_key_env="LOCAL_VLM_TOKEN",
            timeout_s=5.0,
        )

    # Exactly 1 request per sample (no retries on invalid_response).
    assert len(srv.requests) == 6
    first = srv.requests[0]
    assert first["path"] == "/v1/chat/completions"
    assert first["headers"]["authorization"] == "Bearer local-secret-token"
    assert first["json"]["model"] == "qwen2.5-vl-7b"
    assert first["json"]["response_format"] == {
        "type": "json_schema",
        "json_schema": EXPECTED_CHOICE_FORMAT,
    }
    msg_content = first["json"]["messages"][0]["content"]
    img_parts = [p for p in msg_content if p.get("type") == "image_url"]
    assert len(img_parts) == 1
    assert img_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "smp_00" not in first["raw_body"]
    assert "grp_0000" not in first["raw_body"]

    pred_rows = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(r["status"], r["prediction"]) for r in pred_rows] == [
        ("ok", "close"),
        ("ok", "wait"),
        ("ok", "uncertain"),
        ("invalid_response", None),
        ("invalid_response", None),
        ("invalid_response", None),
    ]
    assert summary["response_format"] == "json_schema"
    assert summary["counts"]["failures"]["invalid_response"] == 3
    assert summary["counts"]["ok_count"] == 3


def test_openai_responses_uses_the_same_schema_and_validates_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _write_dataset(
        tmp_path / "ds",
        [
            {"sample_id": f"smp_{i}", "group_id": "grp_0000", "label": 0}
            for i in range(3)
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-test-key")

    def responder(
        index: int, _req: dict[str, Any]
    ) -> tuple[int, dict[str, Any], float]:
        content = (
            {"type": "refusal", "refusal": "unavailable"}
            if index == 1
            else {"type": "output_text", "text": '{"choice":"wait"}'}
        )
        return (
            200,
            {
                "status": "incomplete" if index == 2 else "completed",
                "model": "gpt-6-luna",
                "output": [
                    {"type": "reasoning"},
                    {"type": "message", "content": [content]},
                ],
            },
            0.0,
        )

    out_dir = tmp_path / "out"
    with _LoopbackServer(responder) as srv:
        summary = bm.run_benchmark(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            backend_name="openai-responses",
            base_url=srv.base_url,
        )

    assert len(srv.requests) == 3
    assert srv.requests[0]["path"] == "/v1/responses"
    payload = srv.requests[0]["json"]
    assert payload["model"] == "gpt-6-luna"
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["text"]["format"] == {
        "type": "json_schema",
        **EXPECTED_CHOICE_FORMAT,
    }
    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": bm.LOCAL_VLM_PROMPT}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert "smp_0" not in srv.requests[0]["raw_body"]
    assert summary["response_format"] == "json_schema"
    assert summary["reasoning_effort"] == "none"
    pred_rows = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert [(row["status"], row["prediction"]) for row in pred_rows] == [
        ("ok", "wait"),
        ("refusal", None),
        ("invalid_response", None),
    ]


@pytest.mark.parametrize(
    "backend_name",
    ("local-vlm", "openai-terra", "openai-responses", "openai-decisions"),
)
@pytest.mark.parametrize("view_mode", ("main-wrist", "wrist-overlay"))
def test_view_modes_send_ordered_images_and_shared_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    view_mode: bm.ViewMode,
) -> None:
    dataset = _write_dataset(
        tmp_path / "dataset",
        [{"sample_id": "secret-sample", "group_id": "secret-group", "label": 1}],
    )
    obs = dataset / "observation"
    manifest = obs / "manifest.jsonl"
    row = json.loads(manifest.read_text().strip())
    row.update(main_rgb_path="main.png", wrist_overlay_rgb_path="overlay.png")
    manifest.write_text(json.dumps(row) + "\n")
    main = TINY_PNG_BYTES + b"main"
    overlay = TINY_PNG_BYTES + b"overlay"
    (obs / "main.png").write_bytes(main)
    (obs / "overlay.png").write_bytes(overlay)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-test-key")

    bodies = {
        "openai-decisions": {"answers": [{"name": "grip", "choice": "close"}]},
        "openai-responses": {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"choice":"close"}'}],
                }
            ],
        },
    }
    payloads = []

    def post_json(_url, payload, **_kwargs):
        payloads.append(payload)
        body = bodies.get(
            backend_name, {"choices": [{"message": {"content": '{"choice":"close"}'}}]}
        )
        return 200, json.dumps(body).encode(), None, None

    monkeypatch.setattr(bm, "_post_json", post_json)
    summary = bm.run_benchmark(
        dataset_dir=dataset,
        out_dir=tmp_path / "results",
        view_mode=view_mode,
        backend_name=backend_name,
        model="test-model",
    )

    assert summary["view_mode"] == view_mode
    assert summary["counts"]["predictions"]["close"] == 1
    (payload,) = payloads
    if backend_name == "openai-decisions":
        content = payload["input"][0]["content"]
        text = payload["questions"][0]["instructions"]
    elif backend_name == "openai-responses":
        content = payload["input"][0]["content"]
        text = content[0]["text"]
    else:
        content = payload["messages"][0]["content"]
        text = content[0]["text"]

    imgs = [
        base64.b64decode(
            (
                part["image_url"]
                if part["type"] == "input_image"
                else part["image_url"]["url"]
            ).split(",", 1)[1]
        )
        for part in content
        if part["type"] in ("input_image", "image_url")
    ]
    if view_mode == "main-wrist":
        assert imgs == [main, TINY_PNG_BYTES]
        assert "first image" in text.lower()
        assert "second image" in text.lower()
    else:
        assert imgs == [overlay]
        assert "fully closed" in text.lower()
    assert "secret-sample" not in json.dumps(payload)
    assert "secret-group" not in json.dumps(payload)


def test_http_error_timeout_and_invalid_decisions_response_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _write_dataset(
        tmp_path / "ds",
        [
            {"sample_id": "smp_err", "group_id": "grp_0000", "label": 1},
            {"sample_id": "smp_to", "group_id": "grp_0000", "label": 1},
            {"sample_id": "smp_bad", "group_id": "grp_0000", "label": 0},
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")

    def responder(idx: int, _req: dict[str, Any]) -> tuple[int, Any, float]:
        if idx == 0:
            return (503, {"error": "overloaded"}, 0.0)
        if idx == 1:
            return (
                200,
                {
                    "answers": [
                        {"type": "choice", "name": bm.QUESTION_NAME, "choice": "close"}
                    ]
                },
                0.35,
            )
        return (
            200,
            {
                "answers": [
                    {"type": "choice", "name": bm.QUESTION_NAME, "choice": "maybe"}
                ]
            },
            0.0,
        )

    with _LoopbackServer(responder) as srv:
        out_dir = tmp_path / "out"
        summary = bm.run_benchmark(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            backend_name="openai-decisions",
            base_url=srv.base_url,
            timeout_s=0.12,
        )

    # No retries on any failure mode.
    assert len(srv.requests) == 3
    pred_rows = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [r["status"] for r in pred_rows] == [
        "http_error",
        "timeout",
        "invalid_response",
    ]
    assert pred_rows[0]["http_status"] == 503
    assert all(r["prediction"] is None for r in pred_rows)
    assert pred_rows[1]["elapsed_ms"] >= 80.0
    assert summary["counts"]["failures"] == {
        "timeout": 1,
        "http_error": 1,
        "invalid_response": 1,
        "total": 3,
    }
    assert summary["metrics"]["valid_latency_ms"]["p50"] is None
    assert summary["metrics"]["valid_latency_ms"]["p95"] is None
    assert summary["metrics"]["valid_latency_ms"]["count"] == 0
    assert summary["metrics"]["valid_latency_ms"]["coverage"] == pytest.approx(0.0)


def test_split_filtering_and_limit_applied_after_split(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(
        tmp_path / "ds_splits",
        [
            {"sample_id": "smp_d0", "group_id": "grp_0000", "label": 1},
            {"sample_id": "smp_d1", "group_id": "grp_0000", "label": 0},
            {"sample_id": "smp_e0", "group_id": "grp_0001", "label": 1},
            {"sample_id": "smp_e1", "group_id": "grp_0001", "label": 0},
            {"sample_id": "smp_e2", "group_id": "grp_0002", "label": 1},
            {"sample_id": "smp_e3", "group_id": "grp_0002", "label": 0},
        ],
        splits={"grp_0000": "dev", "grp_0001": "eval", "grp_0002": "eval"},
    )

    out_dir = tmp_path / "out_eval"
    summary = bm.run_benchmark(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        backend_name="always-wait",
        split="eval",
        limit=3,
    )
    rows = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [r["sample_id"] for r in rows] == ["smp_e0", "smp_e1", "smp_e2"]
    assert all(r["split"] == "eval" for r in rows)
    assert summary["counts"]["total_samples"] == 3
    assert summary["split"] == "eval"
    assert summary["limit"] == 3

    # Missing splits.json is allowed for split="all", rejected for split="dev".
    dataset_no_splits = _write_dataset(
        tmp_path / "ds_no_splits",
        [
            {"sample_id": "smp_0", "group_id": "grp_0000", "label": 1},
            {"sample_id": "smp_1", "group_id": "grp_0000", "label": 0},
        ],
    )
    summary_all = bm.run_benchmark(
        dataset_dir=dataset_no_splits,
        out_dir=tmp_path / "out_all",
        backend_name="always-wait",
        split="all",
        limit=1,
    )
    assert summary_all["counts"]["total_samples"] == 1

    with pytest.raises(FileNotFoundError):
        bm.run_benchmark(
            dataset_dir=dataset_no_splits,
            out_dir=tmp_path / "out_dev_fail",
            backend_name="always-wait",
            split="dev",
        )


def test_metrics_denominators_coverage_deadlines_and_clean_backend_boundary(
    tmp_path: Path,
) -> None:
    dataset_dir = _write_dataset(
        tmp_path / "ds_metrics",
        [
            {"sample_id": "pos1", "group_id": "g0", "label": 1},
            {"sample_id": "pos2", "group_id": "g0", "label": 1},
            {"sample_id": "pos3", "group_id": "g1", "label": 1},
            {"sample_id": "pos4", "group_id": "g1", "label": 1},
            {"sample_id": "neg1", "group_id": "g2", "label": 0},
            {"sample_id": "neg2", "group_id": "g2", "label": 0},
            {"sample_id": "neg3", "group_id": "g3", "label": 0},
            {"sample_id": "inv1", "group_id": "g3", "label": None},
        ],
    )

    scripted: list[tuple[bm.Status, bm.Choice | None, float]] = [
        ("ok", "close", 80.0),
        ("ok", "close", 180.0),
        ("ok", "wait", 40.0),
        ("timeout", None, 1000.0),
        ("ok", "close", 120.0),
        ("ok", "uncertain", 260.0),
        ("http_error", None, 20.0),
        ("ok", "close", 400.0),
    ]

    class ScriptedFakeBackend:
        name = "fake"
        model = "fake-v1"
        base_url = None

        def __init__(self) -> None:
            self.calls: list[tuple[bytes, str]] = []

        def predict(
            self, rgb_bytes: bytes, *, mime_type: str = "image/png"
        ) -> bm.BackendResult:
            idx = len(self.calls)
            self.calls.append((rgb_bytes, mime_type))
            status, choice, elapsed_ms = scripted[idx]
            return bm.BackendResult(
                status=status,
                prediction=choice,
                elapsed_ms=elapsed_ms,
                response_model="fake-v1" if status == "ok" else None,
                http_status=500
                if status == "http_error"
                else (200 if status == "ok" else None),
                error=None if status == "ok" else f"simulated_{status}",
            )

    fake = ScriptedFakeBackend()
    out_dir = tmp_path / "out_metrics"
    summary = bm.run_benchmark(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        backend=fake,
    )

    # Provider saw ONLY RGB bytes and mime_type.
    assert len(fake.calls) == 8
    assert all(rgb == TINY_PNG_BYTES and mt == "image/png" for rgb, mt in fake.calls)

    counts = summary["counts"]
    assert counts["total_samples"] == 8
    assert counts["labeled_samples"] == 7
    assert counts["positive_samples"] == 4
    assert counts["negative_samples"] == 3
    assert counts["null_label_samples"] == 1
    assert counts["ok_count"] == 6
    assert counts["valid_response_coverage"] == pytest.approx(6 / 8)
    assert counts["failures"] == {
        "timeout": 1,
        "http_error": 1,
        "invalid_response": 0,
        "total": 2,
    }

    metrics = summary["metrics"]
    # FCR: denominator is negative_samples (3), NOT including label=None (inv1).
    assert metrics["fcr"]["numerator"] == 1
    assert metrics["fcr"]["denominator"] == 3
    assert metrics["fcr"]["rate"] == pytest.approx(1 / 3)
    assert metrics["fcr"]["ok_count"] == 2
    assert metrics["fcr"]["coverage"] == pytest.approx(2 / 3)
    assert metrics["fcr"]["rate_among_ok"] == pytest.approx(0.5)

    # Close recall: denominator is positive_samples (4), including timeout pos4.
    assert metrics["close_recall"]["numerator"] == 2
    assert metrics["close_recall"]["denominator"] == 4
    assert metrics["close_recall"]["rate"] == pytest.approx(0.5)
    assert metrics["close_recall"]["ok_count"] == 3
    assert metrics["close_recall"]["coverage"] == pytest.approx(0.75)
    assert metrics["close_recall"]["rate_among_ok"] == pytest.approx(2 / 3)

    # Abstention: uncertain across total_samples (8), with coverage 6/8.
    assert metrics["abstention"]["numerator"] == 1
    assert metrics["abstention"]["denominator"] == 8
    assert metrics["abstention"]["rate"] == pytest.approx(1 / 8)
    assert metrics["abstention"]["ok_count"] == 6
    assert metrics["abstention"]["coverage"] == pytest.approx(6 / 8)
    assert metrics["abstention"]["rate_among_ok"] == pytest.approx(1 / 6)

    # Deadline-qualified close recall at 100/200/500 ms over positive_samples (4).
    dcr = metrics["deadline_close_recall"]
    assert dcr["100ms"]["numerator"] == 1
    assert dcr["100ms"]["denominator"] == 4
    assert dcr["100ms"]["rate"] == pytest.approx(0.25)
    assert dcr["100ms"]["coverage"] == pytest.approx(0.75)

    assert dcr["200ms"]["numerator"] == 2
    assert dcr["200ms"]["denominator"] == 4
    assert dcr["200ms"]["rate"] == pytest.approx(0.5)
    assert dcr["200ms"]["coverage"] == pytest.approx(0.75)

    assert dcr["500ms"]["numerator"] == 2
    assert dcr["500ms"]["denominator"] == 4
    assert dcr["500ms"]["rate"] == pytest.approx(0.5)
    assert dcr["500ms"]["coverage"] == pytest.approx(0.75)

    # Valid latency p50/p95 excludes both 20ms http_error and 1000ms timeout.
    # Sorted ok latencies: [40, 80, 120, 180, 260, 400]
    lat = metrics["valid_latency_ms"]
    assert lat["count"] == 6
    assert lat["total_samples"] == 8
    assert lat["coverage"] == pytest.approx(0.75)
    assert lat["p50"] == pytest.approx(150.0)
    assert lat["p95"] == pytest.approx(365.0)


def test_cli_always_wait_and_loopback_smoke(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(
        tmp_path / "ds_cli",
        [
            {"sample_id": "smp_1", "group_id": "grp_0000", "label": 1},
            {"sample_id": "smp_2", "group_id": "grp_0000", "label": 0},
        ],
    )

    out_wait = tmp_path / "out_wait"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.grip.benchmark",
            "--dataset",
            str(dataset_dir),
            "--out",
            str(out_wait),
            "--backend",
            "always-wait",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    summary_wait = json.loads((out_wait / "summary.json").read_text(encoding="utf-8"))
    assert summary_wait["backend"] == "always-wait"
    assert summary_wait["counts"]["total_samples"] == 2
    assert summary_wait["counts"]["ok_count"] == 2
    assert summary_wait["metrics"]["fcr"]["rate"] == pytest.approx(0.0)
    assert summary_wait["metrics"]["close_recall"]["rate"] == pytest.approx(0.0)
    assert summary_wait["metrics"]["abstention"]["rate"] == pytest.approx(0.0)

    def local_vlm_responder(
        _idx: int, _req: dict[str, Any]
    ) -> tuple[int, dict[str, Any], float]:
        return (
            200,
            {
                "model": "fake-local-vlm",
                "choices": [{"message": {"content": '{"choice": "close"}'}}],
            },
            0.0,
        )

    out_vlm = tmp_path / "out_vlm"
    with _LoopbackServer(local_vlm_responder) as srv:
        env = os.environ.copy()
        proc_vlm = subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.grip.benchmark",
                "--dataset",
                str(dataset_dir),
                "--out",
                str(out_vlm),
                "--backend",
                "local-vlm",
                "--base-url",
                srv.base_url,
                "--model",
                "fake-local-vlm",
                "--timeout",
                "5",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    assert proc_vlm.returncode == 0, proc_vlm.stderr
    summary_vlm = json.loads((out_vlm / "summary.json").read_text(encoding="utf-8"))
    assert summary_vlm["metrics"]["close_recall"]["rate"] == pytest.approx(1.0)
    assert summary_vlm["metrics"]["fcr"]["rate"] == pytest.approx(1.0)


def test_elapsed_through_validation_zero_denominators_and_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        bm.OpenAIDecisionsBackend()

    # Local VLM without api_key_env sends no Authorization header, and elapsed spans validation.
    captured_headers: list[dict[str, str]] = []
    ticks = iter([10.0, 10.045, 10.060])

    def fake_clock() -> float:
        return next(ticks)

    class _FakeResp:
        status = 200

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            # Advance clock inside HTTP read; final clock read happens after choice validation.
            fake_clock()
            return json.dumps(
                {
                    "model": "stub-vlm",
                    "choices": [{"message": {"content": '{"choice": "wait"}'}}],
                }
            ).encode("utf-8")

    def fake_urlopen(req: Any, timeout: float = 10.0) -> _FakeResp:
        assert timeout == 3.0
        captured_headers.append({k.lower(): v for k, v in req.header_items()})
        return _FakeResp()

    monkeypatch.setattr(bm.urllib.request, "urlopen", fake_urlopen)
    backend = bm.LocalVLMBackend(
        base_url="http://127.0.0.1:9999",
        model="stub-vlm",
        timeout_s=3.0,
        clock=fake_clock,
    )
    assert backend.endpoint == "http://127.0.0.1:9999/v1/chat/completions"
    res = backend.predict(TINY_PNG_BYTES)
    assert res.status == "ok"
    assert res.prediction == "wait"
    assert res.elapsed_ms == pytest.approx(60.0)
    assert "authorization" not in captured_headers[0]

    # Zero-denominator safety: positives-only dataset has fcr.rate=None; limit=0 has all rates None.
    ds_pos = _write_dataset(
        tmp_path / "ds_pos",
        [{"sample_id": "only_pos", "group_id": "g0", "label": 1}],
    )
    summary_pos = bm.run_benchmark(
        dataset_dir=ds_pos,
        out_dir=tmp_path / "out_pos",
        backend_name="always-wait",
    )
    assert summary_pos["metrics"]["fcr"]["denominator"] == 0
    assert summary_pos["metrics"]["fcr"]["rate"] is None
    assert summary_pos["metrics"]["fcr"]["coverage"] is None

    summary_empty = bm.run_benchmark(
        dataset_dir=ds_pos,
        out_dir=tmp_path / "out_empty",
        backend_name="always-wait",
        limit=0,
    )
    assert summary_empty["counts"]["total_samples"] == 0
    assert summary_empty["metrics"]["close_recall"]["rate"] is None
    assert summary_empty["metrics"]["abstention"]["rate"] is None
    assert summary_empty["metrics"]["valid_latency_ms"]["p50"] is None
