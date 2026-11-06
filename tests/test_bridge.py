"""Check OSC validation, independent arm control, and the real model/IK path."""

import csv
import io
import json
from pathlib import Path
import shlex
import threading
import time

import mujoco
import numpy as np
from openarm_control import ArmSetup, IKParams, Kinematics
import openarm_mujoco.v2 as openarm_mujoco
from openarm_mujoco.v2 import JointResolver
import pyarrow as pa
from pythonosc import dispatcher, osc_server, udp_client
import pytest
import yaml

from bridge import (Bridge, FollowRecorder, HARDWARE_ALIGN_THRESHOLD_RAD,
                    RECORD_HEADER, SIDES, extract_qpos, limit_position_step,
                    scale_joint_step)


class MemoryFiles:
    """Open in-memory text files and keep their contents after closing."""

    def __init__(self):
        """Start with no files."""
        self.files = {}

    def __call__(self, path):
        """Return a new text file for ``path``.

        Parameters
        ----------
        path : Path
            Name the recorder would write on disk.

        Returns
        -------
        io.StringIO
            File whose text stays readable after close.
        """
        text = io.StringIO()
        text.close = lambda: None
        self.files[Path(path).name] = text
        return text

    def rows(self, name):
        """Return the CSV rows written to one file.

        Parameters
        ----------
        name : str
            File name, for example ``follow_001_right.csv``.

        Returns
        -------
        list of list of str
            Header and data rows.
        """
        return list(csv.reader(io.StringIO(self.files[name].getvalue())))


class Sink:
    """Collect Dora outputs without starting a dataflow."""

    def __init__(self):
        """Initialize an empty output list."""
        self.outputs = []

    def send_output(self, name, value, metadata=None):
        """Record one output and its metadata."""
        self.outputs.append((name, value, metadata or {}))


@pytest.fixture
def control_factory():
    """Return a controller factory using the checked-in configuration."""
    config = yaml.safe_load(Path("config/control.yml").read_text(encoding="utf-8"))
    config["home_move_s"] = 0.01

    def create(backend="sim", preset="forward", sim_ctrl=False, hardware_gravity=None,
               hardware_config=Path("config/openarm_pedestal.yaml"),
               recorder=None):
        """Create an isolated controller with a recording Dora sink."""
        settings = config.copy()
        settings["home_preset"] = preset
        if hardware_gravity is not None: settings["hardware_gravity_compensation"] = hardware_gravity
        return Bridge(Sink(), settings, backend, sim_ctrl=sim_ctrl,
                      hardware_config=hardware_config, recorder=recorder)

    return create


@pytest.mark.parametrize("preset, elbow", [("forward", np.pi/2), ("down", 0)])
def test_home_presets(control_factory, preset, elbow):
    """Resolve each selected Home against the official pedestal model."""
    control = control_factory(preset=preset)
    assert control.fk.setup.model.nq == 18  # Two arms and fingers, without the Cell lifter.
    for side in SIDES:
        assert control.home[side][3] == pytest.approx(elbow, abs=1e-6)
        np.testing.assert_allclose(control.home_pose[side], control.fk.fk(side, control.home[side]))


def test_unknown_home_preset_is_rejected(control_factory):
    """Reject a misspelled preset instead of selecting a different posture."""
    with pytest.raises(ValueError, match="home_preset"):
        control_factory(preset="unknown")


def receive_pose(control, side, x=0.0):
    """Send a valid relative pose using the existing eight-value OSC layout."""
    control.receive(f"/openarm/{side}", x, 0, 0, 1, 0, 0, 0, 0)


def start_hardware_arm(control, side, aligned=True):
    """Mock driver readiness; use aligned=False to test the first-command handshake."""
    receive_pose(control, side)
    control.receive(f"/openarm/enable_{side}", 1)
    control.tick()
    metadata = {"start_epoch": 1, "observation_timestamp": time.time_ns()}
    control.update_status(side, pa.array(["started"]), metadata)
    control.update_state(side, pa.array(control.home[side]), metadata)
    if aligned: control.update_status(side, pa.array(["aligned"]), metadata)


@pytest.mark.parametrize("values", [(0,)*7, (0, 0, 0, 0, 0, 0, 0, 0),
                                    (np.nan, 0, 0, 1, 0, 0, 0, 0),
                                    (0.31, 0, 0, 1, 0, 0, 0, 0)])
def test_reject_invalid_pose(control_factory, values):
    """Reject malformed targets without making their timestamps fresh."""
    control = control_factory()
    control.receive("/openarm/right", *values)
    assert control.packets["right"][0] is None


def test_quaternion_and_gripper_conversion(control_factory):
    """Normalize the quaternion and preserve the sender's negative gripper scale."""
    control = control_factory()
    control.receive("/openarm/left", 0, 0, 0, 2, 0, 0, 0, -2)
    pose = control.packets["left"][0]
    np.testing.assert_allclose(pose[3:7], [1, 0, 0, 0])
    assert pose[7] == pytest.approx(-np.pi/4)


def test_enable_is_edge_triggered_per_arm(control_factory):
    """Repeated right Enable packets start only the right driver once."""
    control = control_factory("hardware")
    for _ in range(20):
        receive_pose(control, "right")
        control.receive("/openarm/enable_right", 1)
        control.tick()
    commands = [(name, value.to_pylist()) for name, value, _ in control.node.outputs]
    assert commands.count(("arm_command_right", ["start"])) == 1
    assert not any(name == "arm_command_left" for name, _ in commands)


def test_sim_ignores_real_enable(control_factory):
    """Keep both simulated arms available when the real Enable switches are OFF."""
    control = control_factory()
    control.receive("/openarm/enable_left", 0)
    control.receive("/openarm/enable_right", 1)
    control.tick()
    assert control.active("right") and control.active("left")
    assert not any(name.startswith("arm_command") for name, _, _ in control.node.outputs)


@pytest.mark.parametrize("side", SIDES)
def test_single_arm_home_follow_without_other_state(control_factory, side):
    """Run Home/Follow and forward commands only to the enabled arm.

    Parameters
    ----------
    control_factory : callable
        Controller factory with simulated driver acknowledgements.
    side : str
        The sole enabled arm; the other arm has no observations.
    """
    control = control_factory("hardware")
    other = "left" if side == "right" else "right"
    start_hardware_arm(control, side)
    started_at = time.monotonic()
    control.receive("/openarm/home", 1)
    control.tick(started_at)
    assert control.arms[side].mode == "home"
    assert control.arms[other].mode == "idle"
    control.tick(started_at+0.02)
    assert control.arms[side].mode == "ready"
    control.receive("/openarm/follow", 1)
    control.tick(started_at+0.03)
    assert control.arms[side].mode == "follow"
    assert control.arms[other].state is None
    output_names = [name for name, _, _ in control.node.outputs]
    assert "target_right" in output_names and "target_left" in output_names
    control.node.outputs.clear()
    for candidate in SIDES:
        control.forward_solution(candidate, pa.array(control.home[candidate]), {"timestamp": time.time_ns()})
    assert [name for name, _, _ in control.node.outputs] == [f"position_{side}"]


def test_left_enable_waits_for_ten_second_driver_start(control_factory, monkeypatch):
    """Keep a fresh left Enable pending throughout the slower driver startup.

    Parameters
    ----------
    control_factory : callable
        Controller factory using the hardware protocol without real CAN.
    monkeypatch : pytest.MonkeyPatch
        Overrides the monotonic clock without waiting or moving motors.
    """
    now = 1000.0
    monkeypatch.setattr("bridge.time.monotonic", lambda: now)
    control = control_factory("hardware")
    for now in (1000.0, 1005.0, 1010.5):
        receive_pose(control, "left")
        control.receive("/openarm/enable_left", 1)
        control.tick()
        assert control.arms["left"].enabled and not control.active("left")
    assert [(name, value.to_pylist()) for name, value, _ in control.node.outputs] == [("arm_command_left", ["start"])]
    metadata = {"start_epoch": 1, "observation_timestamp": time.time_ns()}
    control.update_status("left", pa.array(["started"]), metadata)
    control.node.outputs.clear()
    control.tick()
    assert [name for name, _, _ in control.node.outputs] == ["request_state_left"]
    control.update_state("left", pa.array(control.home["left"]), metadata)
    control.receive("/openarm/home", 1)
    control.tick()
    assert control.arms["left"].mode == "home"
    assert not control.active("right")


@pytest.mark.parametrize("cause", ["disable", "osc_timeout", "state_timeout", "driver_stopped", "home_timeout"])
def test_hardware_polling_stops_with_arm(control_factory, monkeypatch, capsys, cause):
    """Poll only an active driver and stop polling after each stop path.

    Parameters
    ----------
    control_factory : callable
        Hardware controller factory without CAN access.
    monkeypatch : pytest.MonkeyPatch
        Controls the simulated driver clock.
    capsys : pytest.CaptureFixture
        Captures final diagnostic values before the state is invalidated.
    cause : str
        Stop condition under test.
    """
    now = 1000.0
    monkeypatch.setattr("bridge.time.monotonic", lambda: now)
    control = control_factory("hardware")
    control.home_diagnostics = True
    start_hardware_arm(control, "left")
    control.arms["left"].mode = "follow"
    if cause == "home_timeout":
        measured = control.home["left"].copy()
        measured[0] += 0.2
        control.update_state("left", pa.array(measured), {"start_epoch": 1, "observation_timestamp": time.time_ns()})
        control.receive("/openarm/home", 1)
    control.node.outputs.clear()
    control.tick()
    assert any(name == "request_state_left" for name, _, _ in control.node.outputs)
    assert not any(name == "request_state_right" for name, _, _ in control.node.outputs)
    control.node.outputs.clear()
    if cause == "disable": control.receive("/openarm/enable_left", 0)
    elif cause == "driver_stopped": control.update_status("left", pa.array(["stopped"]), {})
    else:
        now += 15.1 if cause == "home_timeout" else 1.0
        if cause != "osc_timeout":
            receive_pose(control, "left")
            control.receive("/openarm/enable_left", 1)
        if cause != "state_timeout":
            control.update_state("left", pa.array(control.arms["left"].state),
                                 {"start_epoch": 1, "observation_timestamp": time.time_ns()})
    control.tick()
    assert not control.active("left")
    requested_stop = any(name == "arm_command_left" and value.to_pylist() == ["stop"] for name, value, _ in control.node.outputs)
    assert requested_stop == (cause != "driver_stopped")  # Completed stop acknowledgements must not echo stop.
    if cause == "home_timeout":
        reports = [json.loads(line.split(" HOME diag ", 1)[1]) for line in capsys.readouterr().out.splitlines() if " HOME diag " in line]
        assert reports[-1]["event"] == "stop: Home timed out"
        assert reports[-1]["angle_err_deg"] > control.config["home_angle_tol_deg"]
        assert reports[-1]["measured_q"] is not None
    control.node.outputs.clear()
    receive_pose(control, "left")
    control.receive("/openarm/enable_left", 0 if cause == "disable" else 1)
    control.tick()
    assert not any(name.startswith("request_state_") or name.startswith("position_") for name, _, _ in control.node.outputs)


def test_home_diagnostics_values_and_rate(control_factory, monkeypatch, capsys):
    """Report measured errors and actual sent commands with source health data.

    Parameters
    ----------
    control_factory : callable
        Hardware controller factory with an output recorder.
    monkeypatch : pytest.MonkeyPatch
        Supplies deterministic monotonic and Unix clocks.
    capsys : pytest.CaptureFixture
        Captures the single-line JSON diagnostic reports.
    """
    clock = {"mono": 1000.0, "wall": 2_000_000_000}
    monkeypatch.setattr("bridge.time.monotonic", lambda: clock["mono"])
    monkeypatch.setattr("bridge.time.time_ns", lambda: clock["wall"])
    control = control_factory("hardware")
    control.home_diagnostics, control.gravity_compensation = True, False
    start_hardware_arm(control, "left")
    initial_ref = control.arms["left"].command.copy()
    clock["mono"] += 0.1
    clock["wall"] += 100_000_000
    measured = control.home["left"].copy()
    measured[0] += 0.2
    health = {"carrier": True, "bus_off": 0}
    value = pa.array([{"qpos": measured.tolist(), "motor_status": ["ENABLED"]*8, "bus": health}])
    control.update_state("left", value, {"start_epoch": 1, "observation_timestamp": clock["wall"]-50_000_000})
    control.receive("/openarm/home", 1)
    control.tick()
    reports = [json.loads(line.split(" HOME diag ", 1)[1]) for line in capsys.readouterr().out.splitlines() if " HOME diag " in line]
    assert len(reports) == 1 and reports[0]["event"] == "start"
    actual_pose = control.fk.fk("left", measured)
    assert reports[0]["pos_err_m"] == pytest.approx(np.linalg.norm(actual_pose[:3]-control.home_pose["left"][:3]), abs=1e-6)
    assert reports[0]["angle_err_deg"] == pytest.approx(np.rad2deg(0.2), abs=0.001)
    assert reports[0]["path_err_m"] == pytest.approx(np.linalg.norm(actual_pose[:3]-control.fk.fk("left", initial_ref)[:3]), abs=1e-6)
    assert reports[0]["state_age_s"] == 0
    assert reports[0]["snapshot_age_s"] == 0.05
    assert reports[0]["sent_q"] is None and reports[0]["ik_q"] is None
    assert reports[0]["rate_ref_q"] == np.round(initial_ref, 5).tolist()
    assert reports[0]["motor_status"] == ["ENABLED"]*8 and reports[0]["bus"] == health
    solution = measured.copy()
    solution[3] += 0.1
    control.forward_solution("left", pa.array(solution), {"timestamp": clock["wall"]-1})
    assert " HOME diag " not in capsys.readouterr().out
    assert control.arms["left"].solution is None and control.arms["left"].last_sent is None
    control.forward_solution("left", pa.array(solution), {"timestamp": clock["wall"]})
    reports = [json.loads(line.split(" HOME diag ", 1)[1]) for line in capsys.readouterr().out.splitlines() if " HOME diag " in line]
    assert len(reports) == 1 and reports[0]["event"] == "first_command"
    assert reports[0]["ik_q"] == np.round(solution, 5).tolist()
    assert reports[0]["sent_q"] == np.round(control.arms["left"].command, 5).tolist()
    for elapsed in (0.5, 1.2):
        clock["mono"], clock["wall"] = 1000.0+elapsed, 2_000_000_000+int(elapsed*1e9)
        receive_pose(control, "left")
        control.receive("/openarm/enable_left", 1)
        control.update_state("left", value, {"start_epoch": 1, "observation_timestamp": clock["wall"]})
        control.tick()
        reports = [json.loads(line.split(" HOME diag ", 1)[1]) for line in capsys.readouterr().out.splitlines() if " HOME diag " in line]
        assert len(reports) == (0 if elapsed == 0.5 else 1)
        if reports: assert reports[0]["event"] == "periodic"


def test_home_completion_logs_unknown_health_as_null(control_factory, monkeypatch, capsys):
    """Preserve a final report when a legacy observation has no health fields.

    Parameters
    ----------
    control_factory : callable
        Factory with an accelerated Home interpolation for unit tests.
    monkeypatch : pytest.MonkeyPatch
        Controls the elapsed Home time.
    capsys : pytest.CaptureFixture
        Reads the completion report.
    """
    now = 1000.0
    monkeypatch.setattr("bridge.time.monotonic", lambda: now)
    control = control_factory("hardware")
    control.home_diagnostics = True
    start_hardware_arm(control, "left")
    control.receive("/openarm/home", 1)
    control.tick()
    capsys.readouterr()
    now += 0.02
    control.tick()
    reports = [json.loads(line.split(" HOME diag ", 1)[1]) for line in capsys.readouterr().out.splitlines() if " HOME diag " in line]
    assert len(reports) == 1 and reports[0]["event"] == "complete"
    assert reports[0]["pos_err_m"] == 0 and reports[0]["angle_err_deg"] == 0
    assert reports[0]["motor_status"] is None and reports[0]["bus"] is None


def test_left_home_and_follow_preserve_start_gripper(control_factory):
    """Hold the initial opening when the TD slider matches the driver start.

    Parameters
    ----------
    control_factory : callable
        Controller factory with a recording output sink.
    """
    control = control_factory("hardware")
    start_hardware_arm(control, "left")
    measured = control.home["left"].copy()
    measured[7] = 0.4
    control.update_state("left", pa.array(measured), {"start_epoch": 1, "observation_timestamp": time.time_ns()})
    control.receive("/openarm/left", 0, 0, 0, 1, 0, 0, 0, -0.4)
    control.receive("/openarm/home", 1)
    control.tick()
    control.tick(time.monotonic()+0.02)
    assert control.arms["left"].mode == "ready"
    control.receive("/openarm/follow", 1)
    control.tick()
    targets = [value for name, value, _ in control.node.outputs if name == "target_left"]
    assert targets and all(value.field("pose")[0].values[-1].as_py() == pytest.approx(0.4) for value in targets)


def test_follow_off_keeps_polling_for_next_home(control_factory):
    """Continue observations while an enabled arm holds its last command.

    Parameters
    ----------
    control_factory : callable
        Factory using a simulated left-driver handshake.
    """
    control = control_factory("hardware")
    start_hardware_arm(control, "left")
    control.arms["left"].mode = "ready"
    control.receive("/openarm/follow", 1)
    control.tick()
    assert control.arms["left"].mode == "follow"
    control.node.outputs.clear()
    control.receive("/openarm/follow", 0)
    control.tick()
    assert control.active("left") and control.arms["left"].mode == "idle"
    assert [name for name, _, _ in control.node.outputs] == ["request_state_left"]
    control.receive("/openarm/home", 1)
    control.tick()
    assert control.arms["left"].mode == "home"


def test_disable_left_does_not_stop_right(control_factory):
    """Stop only the disabled arm, including late IK results addressed to it."""
    control = control_factory("hardware")
    for side in SIDES:
        start_hardware_arm(control, side)
        control.arms[side].mode = "follow"
    control.receive("/openarm/enable_left", 0)
    control.tick()
    assert control.arms["right"].mode == "follow"
    assert control.arms["left"].mode == "idle"
    control.node.outputs.clear()
    control.forward_solution("left", pa.array(control.home["left"]), {"timestamp": time.time_ns()})
    assert control.node.outputs == []


@pytest.mark.parametrize("backend,mode,speed", [("sim", "home", 0.5), ("sim", "follow", 2.0),
                                               ("hardware", "home", 0.5), ("hardware", "follow", 0.75)])
def test_ik_command_is_rate_limited(control_factory, backend, mode, speed):
    """Bound outgoing motion even when a new mode receives a distant IK solution."""
    control = control_factory(backend)
    if backend == "hardware": start_hardware_arm(control, "right")
    arm = control.arms["right"]
    control.update_state("right", pa.array(control.home["right"]), {})
    arm.mode, arm.command_at = mode, time.monotonic()-0.1
    target = control.home["right"].copy()
    target[3] += 0.5
    before = arm.command.copy()
    control.forward_solution("right", pa.array(target), {"timestamp": time.time_ns()})
    assert arm.command[3] > before[3]
    assert arm.command[3]-before[3] == pytest.approx(speed / control.config["tick_hz"])


def test_small_position_step_is_unchanged():
    """Pass a hand target through when it is within the allowed distance."""
    previous, desired = np.zeros(3), np.array([0.001, -0.001, 0.0])
    np.testing.assert_array_equal(limit_position_step(previous, desired, 0.002), desired)


def test_large_position_step_keeps_direction():
    """Shorten a distant hand target to the allowed distance, same direction."""
    previous, desired = np.array([0.1, 0.0, -0.2]), np.array([0.1, 0.3, 0.2])
    limited = limit_position_step(previous, desired, 0.002)
    assert np.linalg.norm(limited-previous) == pytest.approx(0.002)
    np.testing.assert_allclose((limited-previous)/0.002, [0, 0.6, 0.8])


def test_follow_segments_are_recorded(control_factory):
    """Write one file per Follow segment and nothing outside Follow."""
    files = MemoryFiles()
    recorder = FollowRecorder(Path("unused"), files)
    control = control_factory("hardware", recorder=recorder)
    start_hardware_arm(control, "right")
    started_at = time.monotonic()
    control.receive("/openarm/home", 1)
    control.tick(started_at)
    control.tick(started_at+0.02)
    assert set(files.files) == {"events.csv"}
    receive_pose(control, "right", x=0.01)
    control.receive("/openarm/follow", 1)
    for index in range(3):
        control.tick(started_at+0.03+index/control.config["tick_hz"])
    control.receive("/openarm/follow", 0)
    control.tick(started_at+0.07)
    control.tick(started_at+0.08)
    rows = files.rows("follow_001_right.csv")
    assert rows[0] == RECORD_HEADER
    assert len(rows) == 4
    assert all(len(row) == len(RECORD_HEADER) for row in rows)
    column = {name: index for index, name in enumerate(RECORD_HEADER)}
    # Row times come from the real perf_counter, not the tick override.
    times = [float(row[column["t_s"]]) for row in rows[1:]]
    assert times[0] == 0 and times[0] < times[1] < times[2]
    assert float(rows[1][column["osc_px"]]) == pytest.approx(0.01)
    assert rows[1][column["ik_q1"]] == ""  # No IK answer arrived yet.
    assert "follow_001_left.csv" not in files.files  # The left arm never ran.
    events = files.rows("events.csv")
    assert [row[2:5] for row in events[1:]] == [["right", "follow_start", "1"],
                                               ["right", "follow_end", "1"]]
    assert events[2][5] == "Follow OFF; Home required"


def test_next_follow_opens_a_new_segment(control_factory):
    """Number each Follow start and close open files when recording stops."""
    files = MemoryFiles()
    recorder = FollowRecorder(Path("unused"), files)
    control = control_factory("hardware", recorder=recorder)
    start_hardware_arm(control, "right")
    for segment in (1, 2):
        # Keep Enable, OSC and joint state fresh for the 0.2 s timeouts.
        receive_pose(control, "right")
        control.receive("/openarm/enable_right", 1)
        metadata = {"start_epoch": 1, "observation_timestamp": time.time_ns()}
        state = pa.array(control.home["right"])
        control.update_state("right", state, metadata)
        started_at = time.monotonic()
        control.receive("/openarm/home", 1)
        control.tick(started_at)
        control.tick(started_at+0.02)
        receive_pose(control, "right")
        control.receive("/openarm/follow", 1)
        control.tick(started_at+0.03)
        assert f"follow_{segment:03d}_right.csv" in files.files
        control.receive("/openarm/home", 0)
        if segment == 1:
            control.receive("/openarm/follow", 0)
            control.tick(started_at+0.04)
    recorder.close()
    last_event = files.rows("events.csv")[-1]
    assert last_event[3:] == ["follow_end", "2", "bridge exit"]


def test_follow_hand_speed_bounds_each_tick(control_factory):
    """Move the Follow target toward a far pose at the configured hand speed."""
    control = control_factory("hardware")
    start_hardware_arm(control, "right")
    started_at = time.monotonic()
    control.receive("/openarm/home", 1)
    control.tick(started_at)
    control.tick(started_at+0.02)
    control.receive("/openarm/follow", 1)
    control.tick(started_at+0.03)
    arm = control.arms["right"]
    assert arm.mode == "follow"
    np.testing.assert_allclose(arm.target, control.home_pose["right"], atol=1e-12)
    receive_pose(control, "right", x=0.2)
    step = control.config["follow_hand_speed_m_s"]["hardware"] / control.config["tick_hz"]
    for index in range(1, 4):
        before = arm.target.copy()
        control.tick(started_at+0.03+index/control.config["tick_hz"])
        # The FK pose is float32, so compare at float32 precision.
        assert np.linalg.norm(arm.target[:3]-before[:3]) == pytest.approx(step, rel=1e-5)
        assert arm.target[0]-before[0] == pytest.approx(step, rel=1e-5)


def test_joint_steps_are_scaled_together(control_factory):
    """Slow every arm joint by one ratio so the commanded direction is kept."""
    np.testing.assert_array_equal(scale_joint_step(np.array([0.001, -0.002]), 0.003),
                                  [0.001, -0.002])
    control = control_factory()  # Kinematic sim: no gravity offset in the target.
    control.update_state("right", pa.array(control.home["right"]), {})
    arm = control.arms["right"]
    arm.mode, arm.command_at = "follow", time.monotonic()-0.1
    before = arm.command.copy()
    request = np.array([0.04, -0.01, 0.0, 0.02, 0.0, 0.005, 0.0])
    target = before.copy()
    target[:7] += request
    control.forward_solution("right", pa.array(target), {"timestamp": time.time_ns()})
    step = control.config["follow_joint_speed_rad_s"]["sim"] / control.config["tick_hz"]
    moved = arm.command[:7]-before[:7]
    np.testing.assert_allclose(moved, request*(step/0.04))
    assert np.max(np.abs(moved)) == pytest.approx(step)


def test_hardware_follow_speed_is_configurable(control_factory):
    """Allow a slower first hardware test without changing Home or sim speed."""
    control = control_factory("hardware")
    control.config["follow_joint_speed_rad_s"] = {"sim": 2.0, "hardware": 0.05}
    start_hardware_arm(control, "right")
    arm = control.arms["right"]
    arm.mode, arm.command_at = "follow", time.monotonic()-0.1
    before = arm.command.copy()
    control.forward_solution("right", pa.array(before+0.5), {"timestamp": time.time_ns()})
    # Gravity offsets make the joint changes unequal, so only the largest reaches the limit.
    moved = arm.command[:7]-before[:7]
    assert np.max(np.abs(moved)) == pytest.approx(0.0005)
    assert np.all(moved > 0)


@pytest.mark.parametrize("backend,physics,expected", [("sim", True, True), ("sim", False, False),
                                                     ("hardware", True, True), ("hardware", False, True)])
def test_gravity_offset_only_affects_physical_sim(control_factory, backend, physics, expected):
    """Keep sim physics and the explicitly enabled hardware compensation independent."""
    control = control_factory(backend, sim_ctrl=physics)
    control.arms["right"].state = control.home["right"].copy()
    offset = control.gravity_offset("right")
    assert bool(np.linalg.norm(offset[:7]) > 0.001) == expected
    assert offset[7] == 0
    control.gravity_compensation = False
    np.testing.assert_array_equal(control.gravity_offset("right"), np.zeros(8))


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("scale", [0.0, 0.25, 0.5, 1.0])
def test_hardware_gravity_uses_real_gains_and_configured_scale(control_factory, side, scale):
    """Scale model gravity with hardware gains while excluding the gripper."""
    settings = {"enabled": True, "scale": scale, "max_offset_rad": 0.15}
    control = control_factory("hardware", hardware_gravity=settings)
    state = control.home[side].copy()
    state[0], state[3], state[5] = 0.04, 1.48, -0.06
    control.arms[side].state = state
    control.fk.fk(side, state)
    torque = control.fk.setup.data.qfrc_bias[control.arm_dofs[side]].copy()
    profile = yaml.safe_load(Path("config/openarm_pedestal.yaml").read_text(encoding="utf-8"))
    kp = np.array(profile["control_gains"]["kps"][:7])
    np.testing.assert_array_equal(control.arm_kp[side], kp)
    expected = np.r_[np.clip(scale*torque/kp, -0.15, 0.15), 0.0]
    np.testing.assert_allclose(control.gravity_offset(side), expected)
    assert control.gravity_offset(side)[7] == 0


@pytest.mark.parametrize("settings", [
    {"enabled": "true", "scale": 0.5, "max_offset_rad": 0.15},
    {"enabled": True, "scale": -0.1, "max_offset_rad": 0.15},
    {"enabled": True, "scale": 1.1, "max_offset_rad": 0.15},
    {"enabled": True, "scale": float("nan"), "max_offset_rad": 0.15},
    {"enabled": True, "scale": True, "max_offset_rad": 0.15},
    {"enabled": True, "scale": 0.5, "max_offset_rad": 0.0},
    {"enabled": True, "scale": 0.5, "max_offset_rad": float("inf")},
])
def test_invalid_hardware_gravity_settings_are_rejected(control_factory, settings):
    """Reject unsafe correction settings before a controller can send commands."""
    with pytest.raises(ValueError, match="hardware gravity"):
        control_factory("hardware", hardware_gravity=settings)


def test_hardware_gravity_cap_and_slew_limits(control_factory):
    """Limit correction magnitude and the final servo position and velocity."""
    control = control_factory("hardware", hardware_gravity={"enabled": True, "scale": 1.0, "max_offset_rad": 0.01})
    start_hardware_arm(control, "left")
    arm = control.arms["left"]
    arm.mode, arm.command_at = "follow", time.monotonic()-0.1
    before = arm.command.copy()
    correction = control.gravity_offset("left")
    assert np.max(np.abs(correction)) == pytest.approx(0.01)
    control.forward_solution("left", pa.array(control.limits["left"][:, 1]+1), {"timestamp": time.time_ns()})
    step = control.config["follow_joint_speed_rad_s"]["hardware"] / control.config["tick_hz"]
    assert np.all(np.abs(arm.command[:7]-before[:7]) <= step+1e-8)
    assert np.all(arm.command >= control.limits["left"][:, 0])
    assert np.all(arm.command <= control.limits["left"][:, 1])
    assert arm.gravity_q[7] == 0


@pytest.mark.parametrize("physics", [False, True])
def test_hardware_gravity_settings_do_not_change_sim(control_factory, physics):
    """Ignore hardware gains, caps and toggles in both simulation modes."""
    normal = control_factory(sim_ctrl=physics)
    hardware_enabled = control_factory(sim_ctrl=physics, hardware_gravity={"enabled": True, "scale": 0.25, "max_offset_rad": 0.001},
                                       hardware_config=Path("missing-hardware-profile.yaml"))
    for control in (normal, hardware_enabled): control.arms["left"].state = control.home["left"].copy()
    np.testing.assert_array_equal(normal.gravity_offset("left"), hardware_enabled.gravity_offset("left"))
    assert not hardware_enabled.driver_start


@pytest.mark.parametrize("preset", ["forward", "down"])
@pytest.mark.parametrize("enabled", [False, True])
def test_first_home_holds_start_command_until_aligned(control_factory, monkeypatch, preset, enabled):
    """Prevent the logged sagged-feedback handover and wait for driver alignment."""
    now = 1000.0
    monkeypatch.setattr("bridge.time.monotonic", lambda: now)
    control = control_factory("hardware", preset=preset, hardware_gravity={"enabled": enabled, "scale": 0.5, "max_offset_rad": 0.15})
    control.config["home_move_s"] = 4.0
    start_hardware_arm(control, "left", aligned=False)
    held = control.driver_start["left"].copy()
    measured = np.array([0.04978, -0.00515, -0.00668, 1.4765, 0.01431, -0.09174, 0.00973, 0.3996])
    control.update_state("left", pa.array(measured), {"start_epoch": 1, "observation_timestamp": time.time_ns()})
    np.testing.assert_array_equal(control.arms["left"].command, held)
    control.receive("/openarm/home", 1)
    control.tick()
    value = next(value for name, value, _ in reversed(control.node.outputs) if name == "position")
    seed = np.array(value.field("qpos")[0].values)[8:]
    np.testing.assert_allclose(seed+control.gravity_offset("left"), held, atol=1e-6)
    np.testing.assert_allclose(control.arms["left"].home_start_pose, control.fk.fk("left", seed), atol=1e-6)
    now += 0.008
    control.forward_solution("left", pa.array(seed), {"timestamp": time.time_ns()})
    np.testing.assert_array_equal(control.arms["left"].last_sent, held)
    control.node.outputs.clear()
    now += 0.01
    control.forward_solution("left", pa.array(control.home["left"]), {"timestamp": time.time_ns()})
    assert not control.node.outputs
    control.update_status("left", pa.array(["aligned"]), {"start_epoch": 1})
    now += 0.01
    control.forward_solution("left", pa.array(seed), {"timestamp": time.time_ns()})
    np.testing.assert_allclose(control.arms["left"].command, held, atol=1e-6)


def test_home_waits_for_alignment_and_rejects_large_start_tracking_error(control_factory, monkeypatch):
    """Keep Home pending until aligned and avoid the aligner's measured-q fallback."""
    now = 1000.0
    monkeypatch.setattr("bridge.time.monotonic", lambda: now)
    control = control_factory("hardware")
    start_hardware_arm(control, "left", aligned=False)
    control.receive("/openarm/home", 1)
    control.tick()
    now += 0.02
    control.tick()
    assert control.arms["left"].mode == "home"
    measured = control.home["left"].copy()
    measured[3] -= HARDWARE_ALIGN_THRESHOLD_RAD+0.01
    control.update_state("left", pa.array(measured), {"start_epoch": 1, "observation_timestamp": time.time_ns()})
    control.node.outputs.clear()
    control.forward_solution("left", pa.array(measured), {"timestamp": time.time_ns()})
    assert not control.active("left")
    assert not any(name.startswith("position_") for name, _, _ in control.node.outputs)
    assert any(name == "arm_command_left" and value.to_pylist() == ["stop"] for name, value, _ in control.node.outputs)


def test_rehome_preserves_last_compensated_command(control_factory, monkeypatch):
    """Use the held servo command as the reference when Home follows Follow OFF."""
    now = 1000.0
    monkeypatch.setattr("bridge.time.monotonic", lambda: now)
    control = control_factory("hardware", hardware_gravity={"enabled": True, "scale": 0.5, "max_offset_rad": 0.15})
    start_hardware_arm(control, "left")
    arm = control.arms["left"]
    arm.mode = "ready"
    control.receive("/openarm/follow", 1)
    control.tick()
    arm.command_at = now-0.01
    control.forward_solution("left", pa.array(control.home["left"]), {"timestamp": time.time_ns()})
    held = arm.command.copy()
    control.receive("/openarm/follow", 0)
    control.tick()
    control.receive("/openarm/home", 1)
    control.tick()
    value = next(value for name, value, _ in reversed(control.node.outputs) if name == "position")
    seed = np.array(value.field("qpos")[0].values)[8:]
    now += 0.008
    control.forward_solution("left", pa.array(seed), {"timestamp": time.time_ns()})
    np.testing.assert_allclose(arm.command, held, atol=1e-6)


@pytest.mark.parametrize("side", SIDES)
def test_hardware_gain_physics_home_and_follow(control_factory, monkeypatch, side):
    """Compare sag and compensated tracking using physical servos with real PD gains.

    Parameters
    ----------
    control_factory : callable
        Hardware protocol connected only to an in-process MuJoCo plant.
    monkeypatch : pytest.MonkeyPatch
        Advances controller time with simulated physics rather than wall time.
    side : str
        Sole enabled arm; no real driver, CAN, or network is started.
    """
    profile = yaml.safe_load(Path("config/openarm_pedestal.yaml").read_text(encoding="utf-8"))
    clock = [1000.0]
    monkeypatch.setattr("bridge.time.monotonic", lambda: clock[0])
    for scale in (0.0, 1.0):
        control = control_factory("hardware", hardware_gravity={"enabled": True, "scale": scale, "max_offset_rad": 0.15})
        control.config["home_move_s"] = 4.0
        model = mujoco.MjModel.from_xml_path(openarm_mujoco.openarm_pedestal_xml())
        data, resolver = mujoco.MjData(model), JointResolver(model)
        model.geom_contype[:], model.geom_conaffinity[:] = 0, 0
        for candidate in SIDES:
            resolver.set_qpos(data.qpos, control.driver_start[candidate], candidate)
            for index in range(7):
                actuator = model.actuator(f"{candidate}_joint{index+1}_ctrl")
                actuator.gainprm[0] = profile["control_gains"]["kps"][index]
                actuator.biasprm[1] = -profile["control_gains"]["kps"][index]
                actuator.biasprm[2] = -profile["control_gains"]["kds"][index]
                data.ctrl[actuator.id] = control.driver_start[candidate][index]
            data.ctrl[model.actuator(f"{candidate}_finger1_ctrl").id] = control.driver_start[candidate][7]
        mujoco.mj_forward(model, data)
        for _ in range(round(2/model.opt.timestep)): mujoco.mj_step(model, data)
        setup = ArmSetup.from_args(openarm_mujoco.openarm_pedestal_xml(), "bimanual",
                                   "right_ee_control_point", "site", "left_ee_control_point", "site")
        kin = Kinematics(setup, IKParams(dt=0.01, max_iters=5))
        receive_pose(control, side)
        control.receive(f"/openarm/enable_{side}", 1)
        control.tick()
        control.update_status(side, pa.array(["started"]), {"start_epoch": 1})
        control.node.outputs.clear()
        previous = control.driver_start[side].copy()

        def cycle(home, follow=0, offset=0.0):
            """Route one 100 Hz observation/IK/servo cycle into physical simulation."""
            nonlocal previous
            clock[0] += 0.01
            control.receive(f"/openarm/{side}", offset, 0, 0, 1, 0, 0, 0, -0.4)
            control.receive(f"/openarm/enable_{side}", 1)
            joints, opening = resolver.get_driver(data.qpos, side)
            control.update_state(side, pa.array(np.r_[joints, opening]),
                                 {"start_epoch": 1, "observation_timestamp": time.time_ns()})
            control.receive("/openarm/home", home)
            control.receive("/openarm/follow", follow)
            control.tick()
            pending, control.node.outputs = control.node.outputs, []
            for name, value, _ in pending:
                if name == "position": kin.sync(np.array(value.field("qpos")[0].values))
                elif name.startswith("target_"):
                    target = np.array(value.field("pose")[0].values)
                    kin.set_target(name[7:], target[:7])
                    kin.set_gripper(name[7:], target[7])
            if kin.ready():
                result = kin.solve()
                assert result is not None
                for candidate, target in zip(SIDES, (result[:8], result[8:])):
                    control.forward_solution(candidate, pa.array(target), {"timestamp": time.time_ns()})
            for name, value, _ in control.node.outputs:
                if name != f"position_{side}": continue
                command = extract_qpos(value)
                speed = control.config["home_joint_speed_rad_s"] if not follow else control.config["follow_joint_speed_rad_s"]["hardware"]
                assert np.max(np.abs(command[:7]-previous[:7])) <= speed*0.01+1e-6
                if not control.arms[side].aligned:
                    np.testing.assert_allclose(command, control.driver_start[side], atol=1e-7)
                    control.update_status(side, pa.array(["aligned"]), {"start_epoch": 1})
                previous = command.copy()
                for index in range(7): data.ctrl[model.actuator(f"{side}_joint{index+1}_ctrl").id] = command[index]
                data.ctrl[model.actuator(f"{side}_finger1_ctrl").id] = command[7]
            control.node.outputs.clear()
            for _ in range(round(0.01/model.opt.timestep)): mujoco.mj_step(model, data)

        for _ in range(1600):
            cycle(home=1)
            if control.arms[side].mode == "ready" or not control.active(side): break
        joints, grip = resolver.get_driver(data.qpos, side)
        actual = control.fk.fk(side, np.r_[joints, grip])
        error = np.linalg.norm(actual[:3]-control.home_pose[side][:3])
        if scale == 0:
            assert control.arms[side].mode == "idle" and not control.active(side)
            assert error > control.config["home_position_tol_m"]
            continue
        assert control.arms[side].mode == "ready" and error < 0.003
        for _ in range(200): cycle(home=1, follow=1, offset=0.01)
        joints, grip = resolver.get_driver(data.qpos, side)
        actual = control.fk.fk(side, np.r_[joints, grip])
        expected = control.home_pose[side][:3]+[0.01, 0, 0]
        assert control.arms[side].mode == "follow" and np.linalg.norm(actual[:3]-expected) < 0.004


def test_compensated_commands_still_obey_limits(control_factory):
    """Apply gravity correction inside joint position and speed restrictions."""
    control = control_factory(sim_ctrl=True)
    control.update_state("right", pa.array(control.home["right"]), {})
    arm = control.arms["right"]
    arm.mode, arm.command_at = "home", time.monotonic()-0.1
    before = arm.command.copy()
    control.forward_solution("right", pa.array(control.limits["right"][:, 1]+1), {"timestamp": time.time_ns()})
    assert np.all(np.abs(arm.command[:7]-before[:7]) <= 0.00500001)
    assert np.all(arm.command <= control.limits["right"][:, 1])
    assert np.all(arm.command >= control.limits["right"][:, 0])


def test_feedback_does_not_reset_ik_each_tick(control_factory):
    """Seed IK on mode transitions while continuing to receive live observations."""
    control = control_factory()
    for side in SIDES:
        receive_pose(control, side)
        control.update_state(side, pa.array(control.home[side]), {})
    control.receive("/openarm/home", 1)
    control.tick()
    assert sum(name == "position" for name, _, _ in control.node.outputs) == 1
    control.node.outputs.clear()
    newer = control.home["right"].copy()
    newer[0] += 0.001
    control.update_state("right", pa.array(newer), {})
    control.tick(time.monotonic()+0.02)
    np.testing.assert_array_equal(control.arms["right"].state, newer)
    assert not any(name == "position" for name, _, _ in control.node.outputs)
    assert all(arm.mode == "ready" for arm in control.arms.values())
    control.receive("/openarm/follow", 1)
    control.tick()
    assert sum(name == "position" for name, _, _ in control.node.outputs) == 1


def test_stale_observation_blocks_solution(control_factory):
    """Prevent a valid IK result from moving an arm whose feedback has expired."""
    control = control_factory()
    arm = control.arms["right"]
    arm.mode, arm.command, arm.state_at = "follow", control.home["right"].copy(), time.monotonic()-1
    control.forward_solution("right", pa.array(control.home["right"]), {"timestamp": time.time_ns()})
    assert control.node.outputs == []


@pytest.mark.parametrize("side", SIDES)
def test_timeout_requires_a_new_enable_edge(control_factory, side):
    """Stop on a heartbeat timeout and keep the driver stopped when packets return."""
    control = control_factory("hardware")
    start_hardware_arm(control, side)
    control.tick(time.monotonic()+1)
    assert not control.active(side)
    receive_pose(control, side)
    control.receive(f"/openarm/enable_{side}", 1)
    control.tick()
    starts = [value.to_pylist() for name, value, _ in control.node.outputs if name == f"arm_command_{side}"]
    assert starts == [["start"], ["stop"]]
    assert not any(name == f"arm_command_{'left' if side == 'right' else 'right'}" for name, _, _ in control.node.outputs)


def test_old_session_observation_is_ignored(control_factory):
    """Ignore measurements that belong to an earlier real-driver start epoch."""
    control = control_factory("hardware")
    start_hardware_arm(control, "right")
    before = control.arms["right"].state.copy()
    control.update_state("right", pa.array(before+0.1),
                         {"start_epoch": 0, "observation_timestamp": time.time_ns()})
    np.testing.assert_array_equal(control.arms["right"].state, before)


def test_current_yaml_observations_are_independent_of_ik():
    """Keep sim polling on a timer and hardware polling on the bridge tick."""
    for path in ("dataflow-mujoco.yml", "dataflow-hardware.yml"):
        graph = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        outputs = {node["id"]: set(node.get("outputs", [])) for node in graph["nodes"]}
        for node in graph["nodes"]:
            for port, source in node.get("inputs", {}).items():
                if isinstance(source, dict): source = source["source"]
                if source.startswith("dora/"): continue
                owner, output = source.split("/", 1)
                assert output in outputs[owner]
                if port == "request_position":
                    assert source.startswith("dora/timer/")
                if port == "request_state":
                    side = node["id"].removeprefix("follower-")
                    assert source == f"td-bridge/request_state_{side}"
        if path == "dataflow-hardware.yml":
            bridge = next(node for node in graph["nodes"] if node["id"] == "td-bridge")
            assert bridge["inputs"]["tick"]["source"] == "dora/timer/millis/10"


def test_hardware_data_queues_keep_latest_without_dropping_control_edges():
    """Bound motion and observation queues while retaining ordered control events."""
    graph = yaml.safe_load(Path("dataflow-hardware.yml").read_text(encoding="utf-8"))
    for node in graph["nodes"]:
        for port, source in node.get("inputs", {}).items():
            if port in ("command", "right_status", "left_status", "position"): continue
            assert source["queue_size"] == 1
    bridge = next(node for node in graph["nodes"] if node["id"] == "td-bridge")
    assert isinstance(bridge["inputs"]["left_status"], str)
    for node in graph["nodes"]:
        if node["id"].startswith("follower-"): assert isinstance(node["inputs"]["command"], str)


def test_dataflows_select_the_same_pedestal_model():
    """Keep display, FK, and both IK profiles on the same bundled scene."""
    expected = Path(openarm_mujoco.openarm_pedestal_xml()).resolve()
    for path in ("dataflow-mujoco.yml", "dataflow-hardware.yml"):
        nodes = {node["id"]: node for node in yaml.safe_load(Path(path).read_text(encoding="utf-8"))["nodes"]}
        args = shlex.split(nodes["ik"]["args"])
        assert Path(args[args.index("--xml")+1]).resolve() == expected
        if "mujoco" in nodes:
            args = shlex.split(nodes["mujoco"]["args"])
            assert args[args.index("--scene")+1] == "pedestal"


def test_hardware_dataflow_uses_local_profile_without_automatic_start():
    """Resolve the reviewed driver profile for both arms without auto-start."""
    graph = yaml.safe_load(Path("dataflow-hardware.yml").read_text(encoding="utf-8"))
    profile = Path("config/openarm_pedestal.yaml")
    assert profile.is_file()
    settings = yaml.safe_load(profile.read_text(encoding="utf-8"))
    bridge = next(node for node in graph["nodes"] if node["id"] == "td-bridge")
    args = shlex.split(bridge["args"])
    assert Path(args[args.index("--hardware-config")+1]) == profile
    assert settings["can_interface"] == {"right_arm": "can0", "left_arm": "can1"}
    for side in SIDES:
        node = next(node for node in graph["nodes"] if node["id"] == f"follower-{side}")
        args = shlex.split(node["args"])
        assert args[args.index("--side")+1] == side
        assert Path(args[args.index("--config")+1]) == profile
        assert node["path"] == "openarm-td-hardware"
        assert "--start-on-startup" not in args
        assert float(args[args.index("--timeout")+1]) == 0.2
        assert node["inputs"]["motion_tick"] == {"source": "dora/timer/millis/20", "queue_size": 1}


def test_driver_start_stop_and_home_fit_pedestal_limits():
    """Keep both Home presets and slow driver trajectories inside joint limits."""
    profile = yaml.safe_load(Path("config/openarm_pedestal.yaml").read_text(encoding="utf-8"))
    control = yaml.safe_load(Path("config/control.yml").read_text(encoding="utf-8"))
    for side in SIDES:
        limits = np.array(profile["joint_limits"][f"{side}_arm"])
        for home in control["home_presets"].values():
            target = np.array(home[side])
            assert np.all((target >= limits[:, 0]) & (target <= limits[:, 1]))
        for mode in ("start", "stop"):
            for move in profile[mode]["moves"]:
                target = np.array(move["position"][f"{side}_arm"])
                assert np.all((target >= limits[:, 0]) & (target <= limits[:, 1]))
                np.testing.assert_allclose(target[:7], control["home_presets"]["forward"][side][:7], atol=1e-6)
                assert move["duration"] == 10 and move["hz"] == 50


@pytest.mark.parametrize("preset", ["forward", "down"])
def test_udp_to_official_ik_and_mujoco(control_factory, preset):
    """Move between Home presets and follow both arms through real UDP and IK."""
    control = control_factory(preset=preset)
    control.config["home_move_s"] = 4.0
    model = mujoco.MjModel.from_xml_path(openarm_mujoco.openarm_pedestal_xml())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(model, data)
    mapper = JointResolver(model)
    opposite = "down" if preset == "forward" else "forward"
    for side in SIDES:
        mapper.set_qpos(data.qpos, np.array(control.config["home_presets"][opposite][side]), side)
    mujoco.mj_forward(model, data)
    setup = ArmSetup.from_args(openarm_mujoco.openarm_pedestal_xml(), "bimanual",
                               "right_ee_control_point", "site", "left_ee_control_point", "site")
    kin = Kinematics(setup, IKParams(dt=0.01, max_iters=5))
    routes = dispatcher.Dispatcher()
    routes.set_default_handler(control.receive)
    server = osc_server.OSCUDPServer(("127.0.0.1", 0), routes)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = udp_client.SimpleUDPClient("127.0.0.1", server.server_address[1])

    def cycle(right_offset=(0, 0, 0), left_offset=(0, 0, 0)):
        """Advance one complete feedback/OSC/IK/position-forwarding cycle."""
        for side, offset in zip(SIDES, (right_offset, left_offset)):
            joints, grip = mapper.get_driver(data.qpos, side)
            control.update_state(side, pa.array(np.r_[joints, grip]), {})
            client.send_message(f"/openarm/{side}", [*offset, 1, 0, 0, 0, 0])
        time.sleep(0.01)
        control.tick()
        pending, control.node.outputs = control.node.outputs, []
        for name, value, _ in pending:
            if name == "position": kin.sync(np.array(value.field("qpos")[0].values))
            elif name.startswith("target_"):
                values = np.array(value.field("pose")[0].values)
                kin.set_target(name[7:], values[:7])
                kin.set_gripper(name[7:], values[7])
        if kin.ready():
            result = kin.solve()
            assert result is not None and np.all(np.isfinite(result))
            for side, values in zip(SIDES, (result[:8], result[8:])):
                control.forward_solution(side, pa.array(values), {"timestamp": time.time_ns()})
        commanded = set()
        for name, value, _ in control.node.outputs:
            if name.startswith("position_"): mapper.set_qpos(data.qpos, extract_qpos(value), name[9:])
            if name.startswith("position_"): commanded.add(name[9:])
        control.node.outputs.clear()
        mujoco.mj_forward(model, data)
        return commanded

    try:
        cycle()
        client.send_message("/openarm/home", 1)
        deadline = time.monotonic()+control.config["home_timeout_s"]+1
        while time.monotonic() < deadline:
            cycle()
            if all(arm.mode == "ready" for arm in control.arms.values()): break
        assert all(arm.mode == "ready" for arm in control.arms.values())
        for side in SIDES:
            joints, grip = mapper.get_driver(data.qpos, side)
            pose = control.fk.fk(side, np.r_[joints, grip])
            assert np.linalg.norm(pose[:3]-control.home_pose[side][:3]) <= control.config["home_position_tol_m"]
        client.send_message("/openarm/follow", 1)
        # The straight-down pose needs an upward component to stay inside the workspace.
        offset = np.array([0.01, 0, 0.01 if preset == "down" else 0])
        for _ in range(160): cycle(offset)
        if preset == "down":
            deadline = time.monotonic()+6
            while time.monotonic() < deadline:
                cycle(offset)
                joints, grip = mapper.get_driver(data.qpos, "left")
                left_pose = control.fk.fk("left", np.r_[joints, grip])
                if np.linalg.norm(left_pose[:3]-control.home_pose["left"][:3]) < 0.003: break
        assert all(arm.mode == "follow" for arm in control.arms.values())
        joints, grip = mapper.get_driver(data.qpos, "right")
        pose = control.fk.fk("right", np.r_[joints, grip])
        target = control.home_pose["right"].copy()
        target[:3] += offset
        assert np.linalg.norm(pose[:3]-target[:3]) < 0.003
        joints, grip = mapper.get_driver(data.qpos, "left")
        left_pose = control.fk.fk("left", np.r_[joints, grip])
        assert np.linalg.norm(left_pose[:3]-control.home_pose["left"][:3]) < 0.003
        for _ in range(160): cycle(offset, offset)
        joints, grip = mapper.get_driver(data.qpos, "left")
        left_pose = control.fk.fk("left", np.r_[joints, grip])
        assert np.linalg.norm(left_pose[:3]-(control.home_pose["left"][:3]+offset)) < 0.003
        client.send_message("/openarm/follow", 0)
        for _ in range(3): cycle(offset, offset)
        assert all(arm.mode == "idle" for arm in control.arms.values())
        assert not cycle(offset, offset)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
