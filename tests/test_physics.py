"""Measure physical tracking through isolated, automatically stopped Dora nodes."""

import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

import dora
import numpy as np
from openarm_control import ArmSetup, Kinematics
import openarm_mujoco.v2 as openarm_mujoco
from pythonosc import udp_client
import pytest
import yaml


def observe(path):
    """Record observed hand positions and source timestamps until Dora stops.

    Parameters
    ----------
    path : str
        CSV destination belonging to this test run.
    """
    setup = ArmSetup.from_args(openarm_mujoco.openarm_pedestal_xml(), "bimanual",
                               "right_ee_control_point", "site", "left_ee_control_point", "site")
    fk = Kinematics(setup)
    node = dora.Node()
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["side", "timestamp_ns", "x", "y", "z"])
        print("[PROBE] ready", flush=True)
        for event in node:
            if event["type"] == "STOP": break
            if event["type"] != "INPUT": continue
            side = event["id"]
            joints = np.array(event["value"].field("qpos")[0].values)
            xyz = fk.fk(side, joints)[:3]
            writer.writerow([side, event["metadata"]["observation_timestamp"], *xyz])
            stream.flush()


@pytest.mark.skipif(os.getenv("OPENARM_RUN_DATAFLOW_TESTS") != "1",
                    reason="Set OPENARM_RUN_DATAFLOW_TESTS=1 to launch real Dora processes")
def test_physical_tracking():
    """Measure a forward step and sinusoid with gravity and real process transport."""
    root = Path(__file__).resolve().parents[1]
    work = root / "out" / f"tracking-{time.time_ns()}"
    work.mkdir(parents=True)
    config = yaml.safe_load((root / "config/control.yml").read_text(encoding="utf-8"))
    config["home_preset"] = "forward"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        config["osc_port"] = reservation.getsockname()[1]
    config_path = work / "control.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    graph = yaml.safe_load((root / "dataflow-mujoco.yml").read_text(encoding="utf-8"))
    for node in graph["nodes"]:
        if node["id"] == "td-bridge":
            node["path"] = sys.executable
            node["args"] = f'"{(root / "bridge.py").as_posix()}" --backend sim --listen 127.0.0.1 --config "{config_path.as_posix()}"'
        if node["id"] == "ik":
            args = shlex.split(node["args"])
            args[args.index("--xml")+1] = (root / args[args.index("--xml")+1]).as_posix()
            node["args"] = shlex.join(args)
    observations = work / "observations.csv"
    graph["nodes"].append({"id": "probe", "path": sys.executable,
                           "args": f'"{Path(__file__).as_posix()}" --observe "{observations.as_posix()}"',
                           "env": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                           "inputs": {"right": "mujoco/position_right", "left": "mujoco/position_left"}})
    graph_path = work / "dataflow.yml"
    graph_path.write_text(yaml.safe_dump(graph), encoding="utf-8")
    environment = os.environ.copy()
    environment.update(DORA_OPENARM_MUJOCO_VIEWER="false", DORA_OPENARM_MUJOCO_CTRL="true",
                       DORA_OPENARM_MUJOCO_ENABLE_COLLISION="false")
    log_path = work / "dora.log"
    client = udp_client.SimpleUDPClient("127.0.0.1", config["osc_port"])
    references = []

    def read_log():
        """Read mode acknowledgements without interacting with the user's system."""
        return log_path.read_text(encoding="utf-8", errors="replace")

    def send(x=0.0, home=0, follow=0):
        """Send the TD protocol at approximately 100 Hz and record reference time."""
        references.append((time.time_ns(), x, follow))
        for side in ("right", "left"):
            client.send_message(f"/openarm/{side}", [x, 0, 0, 1, 0, 0, 0, 0])
        client.send_message("/openarm/home", home)
        client.send_message("/openarm/follow", follow)
        time.sleep(0.01)

    with log_path.open("wb") as log:
        process = subprocess.Popen([shutil.which("dora"), "run", str(graph_path), "--stop-after", "30s"],
                                   cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+15
            while "[PROBE] ready" not in read_log() or "[OSC] 127.0.0.1:" not in read_log():
                assert process.poll() is None and time.monotonic() < deadline, read_log()[-6000:]
                time.sleep(0.05)
            for _ in range(25): send()
            for _ in range(10): send(home=1)
            deadline = time.monotonic()+config["home_timeout_s"]
            while not all(f"[{side}] HOME complete -> READY" in read_log() for side in ("right", "left")):
                assert process.poll() is None and time.monotonic() < deadline, read_log()[-6000:]
                send()
            step_start = time.time_ns()
            until = time.monotonic()+3
            while time.monotonic() < until: send(0.10, follow=1)
            assert all(f"[{side}] FOLLOW start" in read_log() for side in ("right", "left")), read_log()[-6000:]
            until = time.monotonic()+2
            while time.monotonic() < until: send(follow=1)
            sine_start = time.time_ns()
            until = time.monotonic()+8
            while time.monotonic() < until:
                x = 0.03*np.sin(np.pi*(time.time_ns()-sine_start)/1e9)
                send(x, follow=1)
            sine_end = time.time_ns()
            for _ in range(30): send()
            assert all(f"[{side}] IDLE: Follow OFF" in read_log() for side in ("right", "left"))
        finally:
            client.close()
            process.wait(timeout=40)
    assert process.returncode == 0 and "Traceback" not in read_log(), read_log()[-6000:]
    np.savetxt(work / "references.csv", references, delimiter=",", header="timestamp_ns,x,follow")
    setup = ArmSetup.from_args(openarm_mujoco.openarm_pedestal_xml(), "bimanual",
                               "right_ee_control_point", "site", "left_ee_control_point", "site")
    fk = Kinematics(setup)
    metrics = {}
    with observations.open(encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
    for side in ("right", "left"):
        origin = fk.fk(side, np.array(config["home_presets"]["forward"][side]))[:3]
        samples = np.array([[int(row["timestamp_ns"]), float(row["x"]), float(row["y"]), float(row["z"])]
                            for row in rows if row["side"] == side])
        step = samples[(samples[:, 0] > step_start+2e9) & (samples[:, 0] < step_start+2.9e9)]
        sine = samples[(samples[:, 0] > sine_start+2e9) & (samples[:, 0] < sine_end)]
        assert len(step) >= 20 and len(sine) >= 100
        times = (sine[:, 0]-sine_start)/1e9
        actual = sine[:, 1]-origin[0]
        lags = np.arange(0, 0.301, 0.001)
        rms_by_lag = [np.mean((actual-0.03*np.sin(np.pi*(times-lag)))**2) for lag in lags]
        metrics[side] = {
            "step_x_m": float(np.mean(step[:, 1])-origin[0]),
            "step_z_error_m": float(np.mean(step[:, 3])-origin[2]),
            "sine_rms_error_m": float(np.sqrt(rms_by_lag[0])),
            "sine_lag_ms": int(lags[np.argmin(rms_by_lag)]*1000),
            "sine_max_z_error_m": float(np.max(np.abs(sine[:, 3]-origin[2]))) }
    (work / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n{json.dumps(metrics)}; results={work}")
    for values in metrics.values():
        assert abs(values["step_x_m"]-0.10) < 0.005
        assert abs(values["step_z_error_m"]) < 0.005
        assert values["sine_rms_error_m"] < 0.006
        assert values["sine_lag_ms"] < 100


if __name__ == "__main__": observe(sys.argv[2])
