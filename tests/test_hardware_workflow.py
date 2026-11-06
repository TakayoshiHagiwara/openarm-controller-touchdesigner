"""Exercise the shipped hardware workflow without constructing a CAN driver."""

from pathlib import Path
import sys
import types

import mujoco
import numpy as np
from openarm_control import ArmSetup, IKParams, Kinematics
import openarm_mujoco.v2 as om
import pyarrow as pa
import pytest
import yaml

import bridge
from bridge import Bridge, SIDES, extract_qpos
from hardware_node import HardwareController, create_driver
from status_monitor import StatusMonitor
from fake_hardware import Clock, FakeDriver, Plant, Sink


class NotificationSink:
    """Route status payloads into the same parser used by the Windows monitor."""

    def __init__(self, monitor):
        """Initialize an in-memory packet history without any socket."""
        self.monitor, self.packets, self.fail = monitor, [], False

    def send_message(self, address, values):
        """Deliver one OSC payload or inject a display-only network failure."""
        if self.fail: raise OSError("mock status network failure")
        self.packets.append((address, values.copy()))
        self.monitor.receive(address, *values)


class Workflow:
    """Connect the real bridge, IK and cooperative hardware controller in memory."""

    def __init__(self, monkeypatch, side="left", physics=False, compensation=None):
        """Use the checked-in config; the ideal fake has no sag, so it runs uncompensated."""
        self.clock, self.side = Clock(), side
        monkeypatch.setattr(bridge, "time", self.clock)
        self.config = yaml.safe_load(Path("config/control.yml").read_text(encoding="utf-8"))
        # The ideal fake returns commands as measurements, so it has no gravity sag to compensate.
        if not physics and compensation is None: compensation = False
        if compensation is not None: self.config["hardware_gravity_compensation"]["enabled"] = compensation
        self.profile = yaml.safe_load(Path("config/openarm_pedestal.yaml").read_text(encoding="utf-8"))
        self.monitor = StatusMonitor(self.clock.monotonic)
        self.notifications = NotificationSink(self.monitor)
        self.bridge_sink, self.worker_sink = Sink(self.clock), Sink(self.clock)
        self.bridge = Bridge(self.bridge_sink, self.config, "hardware", status_client=self.notifications)
        self.plant = Plant(side, self.profile) if physics else None
        self.driver = FakeDriver(side, self.profile, self.clock, self.plant)
        self.factory_calls = 0
        self.worker = HardwareController(self.worker_sink, side, self.profile, self.factory, self.config["timeout_s"], self.clock)
        setup = ArmSetup.from_args(om.openarm_pedestal_xml(), "bimanual",
                                   "right_ee_control_point", "site", "left_ee_control_point", "site")
        self.ik = Kinematics(setup, IKParams(dt=0.01, max_iters=5))
        self.enable, self.home, self.follow = 0, 0, 0
        self.xyz, self.quat, self.grip = np.zeros(3), np.array([1., 0, 0, 0]), -0.4
        self.send_osc, self.send_observations = True, True
        self.steps, self.trace = 0, []
        self.worker.status("stopped")
        self.pump()

    def factory(self):
        """Count actual Enable attempts and return only the in-memory fake."""
        if self.factory_calls:
            previous = self.driver.fetch_position()
            self.driver = FakeDriver(self.side, self.profile, self.clock, self.plant)
            if self.plant is None: self.driver.last_command = previous
        self.factory_calls += 1
        return self.driver

    def pump(self):
        """Route all current messages while retaining command/status event order."""
        for _ in range(20):
            if not self.bridge_sink.pending and not self.worker_sink.pending: return
            pending, self.bridge_sink.pending = self.bridge_sink.pending, []
            for name, value, metadata in pending:
                if name.startswith("arm_command_"):
                    assert name == f"arm_command_{self.side}"
                    self.worker.handle({"id": "command", "value": value, "metadata": metadata})
                elif name.startswith("request_state_"):
                    assert name == f"request_state_{self.side}"
                    if self.send_observations: self.worker.observation()
                elif name.startswith("position_"):
                    assert name == f"position_{self.side}"
                    self.worker.position(value, metadata)
                elif name == "position": self.ik.sync(np.array(value.field("qpos")[0].values))
                elif name.startswith("target_"):
                    values = np.array(value.field("pose")[0].values)
                    self.ik.set_target(name[7:], values[:7])
                    self.ik.set_gripper(name[7:], values[7])
            pending, self.worker_sink.pending = self.worker_sink.pending, []
            for name, value, metadata in pending:
                if name == "status":
                    self.trace.append((self.clock.now, value[0].as_py(), metadata.get("reason", "")))
                    self.bridge.update_status(self.side, value, metadata)
                elif name == "state": self.bridge.update_state(self.side, value, metadata)
        raise AssertionError("message routing did not become idle")

    def cycle(self):
        """Advance 10 ms of fresh TD input, driver ticks, IK and feedback."""
        if self.send_osc:
            self.bridge.receive(f"/openarm/{self.side}", *self.xyz, *self.quat, self.grip)
            self.bridge.receive(f"/openarm/enable_{self.side}", self.enable)
            self.bridge.receive("/openarm/home", self.home)
            self.bridge.receive("/openarm/follow", self.follow)
        self.bridge.tick()
        self.pump()
        if self.steps % 2 == 0:
            self.worker.tick()
            self.pump()
        if self.ik.ready() and self.bridge.active(self.side):
            result = self.ik.solve()
            assert result is not None and np.all(np.isfinite(result))
            for side, values in zip(SIDES, (result[:8], result[8:])):
                self.bridge.forward_solution(side, pa.array(values), {"timestamp": self.clock.time_ns()})
            self.pump()
        if self.plant is not None: self.plant.advance(0.01)
        self.clock.now += 0.01
        self.steps += 1

    def advance(self, seconds):
        """Run enough deterministic cycles to cover an operator wait."""
        for _ in range(round(seconds*100)): self.cycle()

    def until(self, condition, timeout):
        """Wait for a state transition and include the trace on failure."""
        for _ in range(round(timeout*100)):
            self.cycle()
            if condition(): return
        raise AssertionError(f"condition timed out: {self.trace}")

    def start_home_follow(self):
        """Execute explicit Enable, completed start, completed Home and Follow."""
        assert self.factory_calls == 0
        self.enable = 1
        self.until(lambda: self.worker.phase == "started", 12)
        self.advance(0.05)
        self.home = 1
        self.until(lambda: self.bridge.arms[self.side].mode == "ready", 16)
        self.follow = 1
        self.cycle()
        assert self.bridge.arms[self.side].mode == "follow"


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("physics", [False, True])
def test_shipped_enable_home_follow_slider_stop(monkeypatch, side, physics):
    """Check the shipped ON/1.0 config, then run startup, tracking and confirmed stop."""
    shipped = yaml.safe_load(Path("config/control.yml").read_text(encoding="utf-8"))
    assert shipped["hardware_gravity_compensation"] == {"enabled": True, "scale": 1.0, "max_offset_rad": 0.15}
    w = Workflow(monkeypatch, side, physics)
    w.start_home_follow()
    before = w.bridge.fk.fk(side, w.driver.fetch_position())[:3].copy()
    w.xyz[0] = 0.01
    w.advance(3)
    after = w.bridge.fk.fk(side, w.driver.fetch_position())[:3].copy()
    assert after[0]-before[0] == pytest.approx(0.01, abs=0.002)
    assert abs(after[2]-before[2]) < 0.002
    w.grip = -0.2
    w.advance(0.5)
    assert w.driver.last_command[7] == pytest.approx(0.2 if side == "left" else -0.2)
    w.follow = 0
    w.cycle()
    count = len(w.driver.commands)
    w.advance(0.1)
    assert len(w.driver.commands) == count and w.driver.enabled
    w.enable = 0
    w.cycle()
    assert w.worker.phase == "stopping" and w.driver.enabled
    assert w.monitor.arms[side].phase == "STOPPING"
    w.until(lambda: w.worker.phase == "stopped", 13)
    assert not w.driver.enabled and w.driver.last_command[7] == 0
    assert w.monitor.arms[side].phase == "STOPPED" and w.monitor.arms[side].disabled == 1
    assert w.factory_calls == 1
    assert "openarm_can" not in sys.modules


@pytest.mark.parametrize("cause", ["disable", "osc_timeout"])
def test_stop_interrupts_start_without_waiting_for_trajectory(monkeypatch, cause):
    """Process stop during the start trajectory on the next bridge/worker event."""
    w = Workflow(monkeypatch)
    w.enable = 1
    w.advance(0.5)
    assert w.worker.phase == "starting"
    assert 0 < w.driver.last_command[3] < 0.2
    before = w.clock.now
    held = w.driver.last_command.copy()
    if cause == "disable": w.enable = 0
    else: w.send_osc = False
    w.until(lambda: w.worker.phase == "stopping", 0.25)
    assert w.clock.now-before <= (0.02 if cause == "disable" else 0.22)
    np.testing.assert_allclose(w.worker.origin, held if cause == "disable" else w.driver.last_command, atol=0.05)
    assert not any(phase == "started" for _, phase, _ in w.trace)
    assert w.driver.enabled  # It is still a normal return, not an emergency stop.
    w.until(lambda: w.worker.phase == "stopped", 13)
    assert not w.driver.enabled
    assert "OSC timeout" in w.monitor.arms["left"].reason if cause == "osc_timeout" else "Enable OFF" in w.monitor.arms["left"].reason


def test_stop_queued_during_initialization_cancels_enable(monkeypatch):
    """Return to command processing after setup before sending any motor Enable."""
    w = Workflow(monkeypatch)

    def delayed_factory():
        """Model a slow constructor without touching the real CAN implementation."""
        w.clock.now += 0.3
        return w.driver

    w.worker.driver_factory = delayed_factory
    w.enable = 1
    w.cycle()
    assert w.worker.enable_pending and not w.driver.enabled
    w.enable = 0
    w.cycle()
    assert w.worker.phase == "disabling" and not w.driver.enabled
    assert not w.driver.commands
    w.until(lambda: w.worker.phase == "stopped", 2)
    assert w.monitor.arms["left"].disabled == 1


def test_early_buttons_reject_without_losing_ready_or_auto_follow(monkeypatch):
    """Early Home/Follow never arm later motion; a fresh valid Follow edge works."""
    w = Workflow(monkeypatch)
    w.enable, w.home = 1, 1
    w.advance(0.1)
    assert "Home rejected" in w.monitor.arms["left"].reason
    w.until(lambda: w.worker.phase == "started", 12)
    w.advance(0.05)
    assert w.bridge.arms["left"].mode == "idle"
    w.home = 0
    w.cycle()
    w.home = 1
    w.cycle()
    w.follow = 1
    w.cycle()
    assert "Follow rejected" in w.monitor.arms["left"].reason
    w.until(lambda: w.bridge.arms["left"].mode == "ready", 16)
    w.advance(0.1)
    assert w.bridge.arms["left"].mode == "ready"
    w.follow = 0
    w.cycle()
    assert w.bridge.arms["left"].mode == "ready"
    w.follow = 1
    w.cycle()
    assert w.bridge.arms["left"].mode == "follow"


@pytest.mark.parametrize("cause", ["osc_timeout", "state_timeout", "read_error", "position_rejected", "carrier_lost"])
def test_follow_faults_stop_and_require_explicit_reenable(monkeypatch, cause):
    """Invalidate tracking and complete shutdown for independent fault paths."""
    w = Workflow(monkeypatch)
    w.start_home_follow()
    if cause == "osc_timeout": w.send_osc = False
    elif cause == "state_timeout": w.send_observations = False
    elif cause == "read_error": w.driver.raise_state = True
    elif cause == "position_rejected": w.driver.reject_commands = True
    else: w.driver.carrier = False
    w.until(lambda: w.worker.phase in ("stopping", "disabling"), 0.3)
    assert not w.bridge.active("left")
    reason = w.bridge.arms["left"].reason
    w.driver.raise_state = False
    w.driver.reject_commands = False
    w.driver.carrier = True
    w.until(lambda: w.worker.phase == "stopped", 13)
    assert w.monitor.arms["left"].reason == reason
    w.send_osc, w.send_observations = True, True
    w.advance(0.1)
    assert not w.driver.enabled and w.factory_calls == 1


def test_disable_without_acknowledgement_remains_fault(monkeypatch):
    """Never display confirmed motor-off merely because disable was transmitted."""
    w = Workflow(monkeypatch)
    w.start_home_follow()
    w.driver.disable_ack = False
    w.enable = 0
    w.until(lambda: w.worker.phase == "fault", 13)
    assert w.monitor.arms["left"].phase == "FAULT"
    assert w.monitor.arms["left"].disabled == 0
    assert "disable unconfirmed" in w.monitor.arms["left"].reason


def test_compensation_off_still_reports_home_timeout_without_relaxing_it(monkeypatch):
    """Keep the diagnosed OFF case as an expected, explicit Home failure."""
    w = Workflow(monkeypatch, physics=True, compensation=False)
    w.enable = 1
    w.until(lambda: w.worker.phase == "started", 12)
    w.advance(0.05)
    w.home = 1
    w.until(lambda: w.worker.phase == "stopping", 16)
    assert "Home timed out" in w.monitor.arms["left"].reason
    assert w.config["home_position_tol_m"] == 0.02 and w.config["home_angle_tol_deg"] == 8


def test_trajectory_jitter_and_stall_do_not_catch_up_with_fast_points():
    """Retain configured start speed under late and slightly early motion ticks."""
    clock = Clock()
    profile = yaml.safe_load(Path("config/openarm_pedestal.yaml").read_text(encoding="utf-8"))
    driver = FakeDriver("left", profile, clock)
    controller = HardwareController(Sink(clock), "left", profile, lambda: driver, clock=clock)
    controller.start()
    for dt in (0.02, 0.0198, 0.0202, 0.001, 0.5, 0.02):
        before, at = driver.last_command.copy(), clock.now
        clock.now += dt
        controller.tick()
        maximum_step = (np.pi/2)/499
        assert np.max(np.abs(driver.last_command[:7]-before[:7])) <= maximum_step+1e-9
        assert np.max(np.abs(driver.last_command[:7]-before[:7]))/(clock.now-at) <= maximum_step/0.02+1e-7


def test_stale_and_inflight_positions_are_ignored_after_stop():
    """Drop prior-session and in-flight motion commands during normal stopping."""
    clock = Clock()
    profile = yaml.safe_load(Path("config/openarm_pedestal.yaml").read_text(encoding="utf-8"))
    driver = FakeDriver("left", profile, clock)
    controller = HardwareController(Sink(clock), "left", profile, lambda: driver, clock=clock)
    controller.start()
    for _ in range(505):
        clock.now += 0.02
        controller.tick()
    assert controller.phase == "started"
    count = len(driver.commands)
    controller.position(pa.array(np.zeros(8)), {"start_epoch": 0})
    assert len(driver.commands) == count
    controller.stop()
    controller.position(pa.array(np.zeros(8)), {"start_epoch": controller.epoch})
    assert len(driver.commands) == count


def test_status_failure_never_changes_control_and_stale_ui_is_unknown(monkeypatch):
    """Isolate display failures and erase stale READY from the Windows view."""
    w = Workflow(monkeypatch)
    w.notifications.fail = True
    w.start_home_follow()
    assert w.bridge.arms["left"].mode == "follow"
    w.notifications.fail = False
    w.advance(0.6)
    assert w.monitor.arms["left"].phase == "FOLLOW"
    w.clock.now += 2.1
    assert "状態不明" in "\n".join(w.monitor.lines())


def test_started_notification_before_first_observation_is_not_home_ready(monkeypatch):
    """Expose WAIT_STATE rather than racing the operator's first Home edge."""
    w = Workflow(monkeypatch)
    w.enable = 1
    w.send_observations = False
    w.until(lambda: w.worker.phase == "started", 12)
    assert w.monitor.arms["left"].phase == "WAIT_STATE"
    assert w.monitor.arms["left"].can_home == 0
    w.home = 1
    w.cycle()
    assert w.bridge.arms["left"].mode == "idle"
    w.send_observations = True
    w.cycle()
    w.cycle()
    assert w.monitor.arms["left"].phase == "WAIT_HOME" and w.monitor.arms["left"].can_home == 1
    assert w.bridge.arms["left"].mode == "idle"  # No delayed, automatic Home.
    w.home = 0
    w.cycle()
    w.home = 1
    w.cycle()
    assert w.bridge.arms["left"].mode == "home"


def test_lost_worker_heartbeat_is_not_an_endless_stopping_display(monkeypatch):
    """Keep unconfirmed motor state visible even if the driver process disappears."""
    w = Workflow(monkeypatch)
    w.enable = 1
    w.advance(0.5)
    assert w.worker.phase == "starting"
    # Stop delivering any worker events while the TD and bridge remain alive.
    for _ in range(205):
        w.bridge.receive("/openarm/left", 0, 0, 0, 1, 0, 0, 0, -0.4)
        w.bridge.receive("/openarm/enable_left", 1)
        w.bridge.tick()
        w.clock.now += 0.01
    state = w.monitor.arms["left"]
    assert state.phase == "FAULT" and state.disabled == 0
    assert "driver heartbeat timeout" in state.reason
    assert not w.bridge.active("left")


@pytest.mark.parametrize("side", SIDES)
def test_all_pose_slider_fields_and_gripper_in_ideal_isolation(monkeypatch, side):
    """Verify all six pose fields independently of gravity load and compensation."""
    w = Workflow(monkeypatch, side, compensation=False)
    w.start_home_follow()
    for axis in range(6):
        for sign in (1, -1):
            w.xyz[:], w.quat[:] = 0, [1., 0, 0, 0]
            if axis < 3: w.xyz[axis] = sign*0.01
            else:
                angle = np.deg2rad(sign*5)
                w.quat[0], w.quat[axis-2] = np.cos(angle/2), np.sin(angle/2)
            w.advance(2)
            actual = w.bridge.fk.fk(side, w.driver.fetch_position())
            orientation = np.empty(4)
            mujoco.mju_mulQuat(orientation, w.bridge.home_pose[side][3:7].astype(float), w.quat)
            angle_error = np.rad2deg(2*np.arccos(np.clip(abs(actual[3:7]@orientation), 0, 1)))
            assert np.linalg.norm(actual[:3]-(w.bridge.home_pose[side][:3]+w.xyz)) < 0.001
            assert angle_error < 0.2
            w.xyz[:], w.quat[:] = 0, [1., 0, 0, 0]
            w.advance(2)
    for raw in (-0.2, -0.4):
        w.grip = raw
        w.advance(0.5)
        assert w.driver.last_command[7] == pytest.approx(-raw if side == "left" else raw)


def test_enable_ack_failure_does_not_publish_started(monkeypatch):
    """Keep a missing enable acknowledgement out of the Home-ready protocol."""
    w = Workflow(monkeypatch)
    w.driver.enable_ack = False
    w.enable = 1
    w.until(lambda: w.worker.phase == "stopping", 0.3)
    assert "motor enable acknowledgement timed out" in w.monitor.arms["left"].reason
    assert not any(phase == "started" for _, phase, _ in w.trace)


def test_duplicate_stop_and_enable_during_return_do_not_restart_motion(monkeypatch):
    """Make repeated stop idempotent and reject Enable while returning."""
    w = Workflow(monkeypatch)
    w.enable = 1
    w.advance(0.5)
    w.enable = 0
    w.cycle()
    origin = w.worker.origin.copy()
    w.worker.stop("duplicate stop")
    np.testing.assert_array_equal(w.worker.origin, origin)
    w.enable = 1
    w.cycle()
    assert w.factory_calls == 1 and w.worker.phase == "stopping"
    w.until(lambda: w.worker.phase == "stopped", 13)
    w.advance(0.1)
    assert w.factory_calls == 1 and not w.driver.enabled


def test_disable_send_failure_is_visible_and_never_confirmed(monkeypatch):
    """Retain FAULT and an unconfirmed state when the CAN disable call fails."""
    w = Workflow(monkeypatch)
    w.start_home_follow()
    w.driver.raise_disable = True
    w.enable = 0
    w.until(lambda: w.worker.phase == "fault", 13)
    assert w.monitor.arms["left"].phase == "FAULT" and w.monitor.arms["left"].disabled == 0
    assert "disable failed" in w.monitor.arms["left"].reason


def test_protected_robot_parameters_are_unchanged():
    """Pin the shipped gains, velocities, durations and Home tolerances."""
    config = yaml.safe_load(Path("config/control.yml").read_text(encoding="utf-8"))
    profile = yaml.safe_load(Path("config/openarm_pedestal.yaml").read_text(encoding="utf-8"))
    assert profile["control_gains"] == {"kps": [70., 70., 70., 60., 10., 10., 10., 10.],
                                        "kds": [2.75, 2.5, 2., 2., 0.7, 0.6, 0.5, 0.2]}
    assert profile["joint_velocity_limits"] == [2., 2., 3.3, 3.3, 6.3, 6.3, 6.3, 20.]
    assert profile["gripper_posforce_limits"] == [50., 1.]
    assert config["home_joint_speed_rad_s"] == 0.5
    assert config["follow_joint_speed_rad_s"] == {"sim": 2., "hardware": 0.75}
    assert config["follow_hand_speed_m_s"] == {"sim": 2., "hardware": 0.2}
    assert config["gripper_speed_rad_s"] == 2.
    assert config["home_move_s"] == 4 and config["home_timeout_s"] == 15
    assert config["home_position_tol_m"] == 0.02 and config["home_angle_tol_deg"] == 8
    for mode in ("start", "stop"):
        assert profile[mode]["moves"][0]["duration"] == 10
        assert profile[mode]["moves"][0]["hz"] == 50


@pytest.mark.parametrize("responses,status,accepted", [(0, "DISABLED", False), (1, "ENABLED", False), (1, "DISABLED", True)])
def test_factory_preflight_does_not_enable_motors(monkeypatch, responses, status, accepted):
    """Exercise the production factory with a stub package and no CAN import."""
    instances = []

    class Collection:
        """Provide reply counters for the production preflight check."""

        def get_motors(self):
            """Return eight mock MIT motor slots."""
            return [None]*8

        def get_link_stats(self, index):
            """Return a controlled response count."""
            return types.SimpleNamespace(responses=responses)

    class RawDriver:
        """Replace only the external package; factory code itself remains real."""

        def __init__(self, side, config):
            """Record construction without enabling any device."""
            self.gripper_posforce, self.enabled = False, False
            self.openarm = types.SimpleNamespace(get_arm=lambda: Collection())
            instances.append(self)

        def get_health(self):
            """Return the preflight status under test."""
            return [status]*8, {"carrier": True}

    monkeypatch.setitem(sys.modules, "openarm_driver", types.SimpleNamespace(Config=lambda path: path, SingleArmDriver=RawDriver))
    if accepted: assert create_driver("left", Path("config/openarm_pedestal.yaml")) is instances[0]
    else:
        with pytest.raises(RuntimeError, match="preflight"):
            create_driver("left", Path("config/openarm_pedestal.yaml"))
    assert all(not instance.enabled for instance in instances)


def test_critical_status_failure_never_prevents_disable():
    """Refuse unobservable startup and still finish a previously active shutdown."""
    clock = Clock()
    profile = yaml.safe_load(Path("config/openarm_pedestal.yaml").read_text(encoding="utf-8"))

    class FailingSink(Sink):
        """Inject a lost Dora control-status channel."""

        fail = True

        def send_output(self, name, value, metadata=None):
            """Fail only when the test enables channel loss."""
            if self.fail: raise RuntimeError("mock lost Dora status channel")
            super().send_output(name, value, metadata)

    sink = FailingSink(clock)
    driver = FakeDriver("left", profile, clock)
    calls = []
    controller = HardwareController(sink, "left", profile, lambda: calls.append(True) or driver, clock=clock)
    controller.start()
    assert not calls and not driver.enabled and controller.phase == "fault"
    sink.fail = False
    controller.start()
    for _ in range(505):
        clock.now += 0.02
        controller.tick()
    assert controller.phase == "started"
    sink.fail = True
    controller.stop()
    for _ in range(560):
        clock.now += 0.02
        controller.tick()
    assert controller.phase == "stopped" and not driver.enabled
