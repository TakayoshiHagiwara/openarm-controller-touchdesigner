"""Opt-in smoke tests for the actual Dora processes; no real hardware is used."""

import os
import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time

from pythonosc import udp_client
from pythonosc import dispatcher, osc_server
import pytest
import yaml

from status_monitor import StatusMonitor


@pytest.mark.skipif(os.getenv("OPENARM_RUN_DATAFLOW_TESTS") != "1",
                    reason="Set OPENARM_RUN_DATAFLOW_TESTS=1 to launch the real Dora processes")
@pytest.mark.parametrize("preset", ["forward", "down"])
def test_pedestal_dora_home_follow(preset):
    """Verify each Home, Follow and stop through three real, auto-stopping nodes."""
    root = Path(__file__).resolve().parents[1]
    work = root / "out" / f"pedestal-smoke-{preset}-{time.time_ns()}"
    work.mkdir(parents=True)
    config = yaml.safe_load((root / "config/control.yml").read_text(encoding="utf-8"))
    config["home_preset"] = preset
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        config["osc_port"] = reservation.getsockname()[1]
    config_path = work / "control.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    graph = yaml.safe_load((root / "dataflow-mujoco.yml").read_text(encoding="utf-8"))
    for node in graph["nodes"]:
        if node["id"] == "td-bridge":
            node["args"] = f'--backend sim --listen 127.0.0.1 --config "{config_path.as_posix()}"'
        if node["id"] == "ik":
            args = shlex.split(node["args"])
            args[args.index("--xml")+1] = (root / args[args.index("--xml")+1]).as_posix()
            node["args"] = shlex.join(args)
    graph_path = work / "dataflow.yml"
    graph_path.write_text(yaml.safe_dump(graph), encoding="utf-8")
    environment = os.environ.copy()
    environment.update(DORA_OPENARM_MUJOCO_VIEWER="false", DORA_OPENARM_MUJOCO_CTRL="false",
                       DORA_OPENARM_MUJOCO_ENABLE_COLLISION="false")
    log_path = work / "dora.log"
    client = udp_client.SimpleUDPClient("127.0.0.1", config["osc_port"])

    def read_log():
        """Read the current UTF-8 process log for startup and mode acknowledgements."""
        return log_path.read_text(encoding="utf-8", errors="replace")

    def send_controls(home=0, follow=0):
        """Send the unchanged TD protocol to this test's isolated simulation port."""
        offset = [0.01, 0, 0.01 if preset == "down" else 0] if follow else [0, 0, 0]
        for side in ("left", "right"):
            client.send_message(f"/openarm/{side}", [*offset, 1, 0, 0, 0, 0])
            client.send_message(f"/openarm/enable_{side}", 0)
        client.send_message("/openarm/home", home)
        client.send_message("/openarm/follow", follow)
        time.sleep(0.01)

    with log_path.open("wb") as log:
        process = subprocess.Popen([shutil.which("dora"), "run", str(graph_path), "--stop-after", "20s"],
                                   cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+15
            started = ("[OSC] 127.0.0.1:", f"[MODEL] pedestal; Home preset={preset}")
            while not all(message in read_log() for message in started):
                assert process.poll() is None, read_log()[-6000:]
                assert time.monotonic() < deadline, read_log()[-6000:]
                time.sleep(0.05)
            deadline = time.monotonic()+0.2
            while time.monotonic() < deadline: send_controls()
            deadline = time.monotonic()+0.1
            while time.monotonic() < deadline: send_controls(home=1)
            deadline = time.monotonic()+config["home_timeout_s"]
            ready = ("[right] HOME complete -> READY", "[left] HOME complete -> READY")
            while not all(message in read_log() for message in ready):
                assert process.poll() is None, read_log()[-6000:]
                assert time.monotonic() < deadline, read_log()[-6000:]
                send_controls()
            deadline = time.monotonic()+2
            while time.monotonic() < deadline: send_controls(follow=1)
            assert all(f"[{side}] FOLLOW start" in read_log() for side in ("left", "right"))
            deadline = time.monotonic()+0.3
            while time.monotonic() < deadline: send_controls()
            assert all(f"[{side}] IDLE: Follow OFF" in read_log() for side in ("left", "right"))
        finally:
            client.close()
            # Dora's own timer sends STOP and waits for the nodes; no Ctrl+C is needed.
            process.wait(timeout=35)
    assert process.returncode == 0, read_log()[-6000:]
    assert "Traceback" not in read_log(), read_log()[-6000:]


@pytest.mark.skipif(os.getenv("OPENARM_RUN_DATAFLOW_TESTS") != "1",
                    reason="Set OPENARM_RUN_DATAFLOW_TESTS=1 to launch real Dora with fake drivers")
def test_hardware_protocol_with_real_dora_and_status_udp():
    """Route real OSC, Dora, IK and status UDP through test-only motor workers."""
    root = Path(__file__).resolve().parents[1]
    work = root / "out" / f"hardware-protocol-{time.time_ns()}"
    work.mkdir(parents=True)
    config = yaml.safe_load((root / "config/control.yml").read_text(encoding="utf-8"))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        config["osc_port"] = reservation.getsockname()[1]
    monitor = StatusMonitor()
    routes = dispatcher.Dispatcher()
    routes.set_default_handler(monitor.receive)
    server = osc_server.OSCUDPServer(("127.0.0.1", 0), routes)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config["status"] = {"enabled": True, "host": "127.0.0.1", "port": server.server_address[1]}
    config_path = work / "control.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    profile = root / "config/openarm_pedestal.yaml"
    graph = yaml.safe_load((root / "dataflow-hardware.yml").read_text(encoding="utf-8"))
    for node in graph["nodes"]:
        if node["id"] == "td-bridge":
            node["path"] = sys.executable
            node["args"] = f'"{(root / "bridge.py").as_posix()}" --backend hardware --listen 127.0.0.1 --config "{config_path.as_posix()}" --hardware-config "{profile.as_posix()}"'
        elif node["id"].startswith("follower-"):
            side = node["id"].removeprefix("follower-")
            node["path"] = sys.executable
            node["args"] = f'"{(root / "tests/mock_hardware_node.py").as_posix()}" --side {side} --config "{profile.as_posix()}"'
        elif node["id"] == "ik":
            args = shlex.split(node["args"])
            args[args.index("--xml")+1] = (root / args[args.index("--xml")+1]).as_posix()
            node["args"] = shlex.join(args)
    graph_path = work / "dataflow.yml"
    graph_path.write_text(yaml.safe_dump(graph), encoding="utf-8")
    client = udp_client.SimpleUDPClient("127.0.0.1", config["osc_port"])
    log_path = work / "dora.log"
    stages = []

    def send(enable=0, home=0, follow=0, x=0.0):
        """Send the operator's TD fields only to the isolated loopback test graph."""
        client.send_message("/openarm/left", [x, 0, 0, 1, 0, 0, 0, -0.4])
        client.send_message("/openarm/enable_left", enable)
        client.send_message("/openarm/right", [0, 0, 0, 1, 0, 0, 0, -0.4])
        client.send_message("/openarm/enable_right", 0)
        client.send_message("/openarm/home", home)
        client.send_message("/openarm/follow", follow)
        time.sleep(0.01)

    def wait_phase(phase, timeout, **controls):
        """Wait for the phase actually received by the Windows-compatible monitor."""
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            send(**controls)
            if "left" in monitor.arms and monitor.arms["left"].phase == phase:
                stages.append({"phase": phase, "elapsed_s": time.monotonic()-run_started,
                               "disabled": monitor.arms["left"].disabled})
                return
            assert process.poll() is None, log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise AssertionError(f"phase {phase} not received: {monitor.arms}")

    with log_path.open("wb") as log:
        run_started = time.monotonic()
        process = subprocess.Popen([shutil.which("dora"), "run", str(graph_path), "--stop-after", "60s"],
                                   cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_phase("STOPPED", 10)
            wait_phase("STARTING", 2, enable=1)
            deadline = time.monotonic()+0.5
            while time.monotonic() < deadline: send(enable=1)
            # STOPPING is a worker acknowledgement, distinct from bridge STOP_REQUESTED.
            cancel_at = time.monotonic()
            wait_phase("STOPPING", 1, enable=0)
            cancel_latency = time.monotonic()-cancel_at
            assert cancel_latency < 1.0
            wait_phase("STOPPED", 17)
            assert monitor.arms["left"].disabled == 1
            wait_phase("STARTING", 2, enable=1)
            # Late timer delivery slows the trajectory; never change robot timing to fit a test.
            wait_phase("WAIT_HOME", 16, enable=1)
            wait_phase("READY", 8, enable=1, home=1)
            wait_phase("FOLLOW", 2, enable=1, home=1, follow=1)
            deadline = time.monotonic()+2
            while time.monotonic() < deadline: send(enable=1, home=1, follow=1, x=0.01)
            wait_phase("WAIT_HOME", 2, enable=1, home=1)
            wait_phase("STOPPING", 2, home=1)
            wait_phase("STOPPED", 17, home=1)
            assert monitor.arms["left"].disabled == 1
            assert monitor.arms["right"].phase == "STOPPED"
        finally:
            client.close()
            process.wait(timeout=60)
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
    assert process.returncode == 0
    assert "Traceback" not in log_path.read_text(encoding="utf-8", errors="replace")
    (work / "result.json").write_text(json.dumps({"real_hardware_used": False,
        "transport": "real Dora and loopback OSC/UDP; fake motor drivers", "cancel_ack_s": cancel_latency,
        "stages": stages}, indent=2)+"\n", encoding="utf-8")
