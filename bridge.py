"""Receive the existing TouchDesigner OSC protocol and gate OpenArm commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import ipaddress
import os
from pathlib import Path
import threading
import time

import dora
import numpy as np
import openarm_mujoco.v2 as openarm_mujoco
from openarm_control import ArmSetup, Kinematics
import pyarrow as pa
from pythonosc import dispatcher, osc_server, udp_client
import yaml

SIDES = ("right", "left")
# Match the explicit --align-threshold in dataflow-hardware.yml.
HARDWARE_ALIGN_THRESHOLD_RAD = 0.1
DRIVER_HEARTBEAT_TIMEOUT_S = 2.0
# Set by run-hardware.sh / run-mujoco.cmd --record to enable recording.
RECORD_FLAG_ENV = "OPENARM_BRIDGE_RECORD"
JOINT_NAMES = [f"q{i}" for i in range(1, 8)] + ["gripper"]
POSE_NAMES = ["px", "py", "pz", "qw", "qx", "qy", "qz", "gripper"]
RECORD_HEADER = (
    ["t_s", "wall_ns", "osc_age_s"]
    + [f"osc_{name}" for name in POSE_NAMES]
    + [f"target_{name}" for name in POSE_NAMES]
    + ["ik_age_s"] + [f"ik_{name}" for name in JOINT_NAMES]
    + ["command_age_s"] + [f"command_{name}" for name in JOINT_NAMES]
    + [f"gravity_{name}" for name in JOINT_NAMES]
    + ["state_age_s", "observation_ns"]
    + [f"state_{name}" for name in JOINT_NAMES]
    + [f"driver_command_{name}" for name in JOINT_NAMES])
EVENT_HEADER = ["wall_ns", "t_s", "side", "event", "segment", "reason"]


class FollowRecorder:
    """Write one CSV per arm and Follow segment, plus an events log.

    Rows go to buffered files and are flushed about once per second, so
    writing one row per tick does not wait for the disk. Row and event
    times use ``time.perf_counter()``, which is fine-grained on Windows
    too (``time.monotonic()`` has 15.6 ms steps there).

    Parameters
    ----------
    folder : Path
        Folder of this bridge run; created when the first file opens.
    open_file : callable, optional
        ``open_file(path)`` returning a writable text file. Tests pass an
        in-memory factory instead of writing to disk.
    """

    def __init__(self, folder: Path, open_file=None) -> None:
        """Open the events log and start with no Follow segment."""
        self.folder = Path(folder)
        self.open_file = open_file or self._open_on_disk
        self.segment = 0
        self.files = {}
        self.started_at = {}
        self.created_at = time.perf_counter()
        self.flushed_at = self.created_at
        self.events = self.open_file(self.folder / "events.csv")
        self.events.write(",".join(EVENT_HEADER) + "\n")

    @staticmethod
    def _open_on_disk(path: Path):
        """Create the parent folder and open a new text file.

        Parameters
        ----------
        path : Path
            File to create.

        Returns
        -------
        io.TextIOWrapper
            Buffered text file opened for writing.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, "w", encoding="utf-8", newline="")

    def event(self, side: str, name: str, reason: str) -> None:
        """Append one Follow start or end to the events log.

        Parameters
        ----------
        side : {'right', 'left'}
            Arm the event belongs to.
        name : str
            ``follow_start`` or ``follow_end``.
        reason : str
            Stop reason, empty for a start.
        """
        t_s = time.perf_counter() - self.created_at
        text = reason.replace('"', "'")
        self.events.write(f'{time.time_ns()},{t_s:.6f},{side},{name},'
                          f'{self.segment},"{text}"\n')

    def start(self, sides: list, now: float) -> None:
        """Open the files of a new Follow segment.

        Parameters
        ----------
        sides : list of str
            Arms that entered Follow on this tick.
        now : float
            ``time.perf_counter()`` of the tick; row times count from here.
        """
        self.segment += 1
        for side in sides:
            name = f"follow_{self.segment:03d}_{side}.csv"
            self.files[side] = self.open_file(self.folder / name)
            self.files[side].write(",".join(RECORD_HEADER) + "\n")
            self.started_at[side] = now
            self.event(side, "follow_start", "")

    def end(self, side: str, reason: str) -> None:
        """Close one arm's segment file if it is open.

        Parameters
        ----------
        side : {'right', 'left'}
            Arm leaving Follow.
        reason : str
            Why Follow ended.
        """
        recording = self.files.pop(side, None)
        if recording is None:
            return
        recording.close()
        self.event(side, "follow_end", reason)
        self.events.flush()

    def write(self, side: str, now: float, values: list) -> None:
        """Append one tick of one arm and flush about once per second.

        Parameters
        ----------
        side : {'right', 'left'}
            Arm in Follow.
        now : float
            ``time.perf_counter()`` of the tick.
        values : list
            Values in the order of ``RECORD_HEADER`` after ``t_s``;
            ``None`` is written as an empty field.
        """
        fields = [now - self.started_at[side]] + values
        self.files[side].write(",".join(
            "" if value is None else
            str(value) if isinstance(value, int) else f"{value:.7g}"
            for value in fields) + "\n")
        
        if now - self.flushed_at >= 1.0:
            for recording in self.files.values():
                recording.flush()
            self.events.flush()
            self.flushed_at = now

    def close(self) -> None:
        """End every open segment and close the events log."""
        for side in list(self.files):
            self.end(side, "bridge exit")
        self.events.close()


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two unit quaternions.

    Parameters
    ----------
    a, b : ndarray, shape (4,)
        Quaternions in wxyz order.

    Returns
    -------
    ndarray, shape (4,)
        Normalized quaternion representing a * b.
    """
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    q = np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                  aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
    return q / np.linalg.norm(q)


def quaternion_slerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate wxyz quaternions along the shortest rotation.

    Parameters
    ----------
    a, b : ndarray, shape (4,)
        Unit quaternions.
    alpha : float
        Interpolation fraction between zero and one.

    Returns
    -------
    ndarray, shape (4,)
        Interpolated unit quaternion.
    """
    dot = float(np.dot(a, b))
    if dot < 0: b, dot = -b, -dot
    if dot > 0.9995:
        q = a + alpha * (b - a)
        return q / np.linalg.norm(q)
    
    theta = np.arccos(np.clip(dot, -1, 1))

    return (np.sin((1-alpha)*theta)*a + np.sin(alpha*theta)*b) / np.sin(theta)


def limit_position_step(previous: np.ndarray, desired: np.ndarray,
                        max_step_m: float) -> np.ndarray:
    """Move a hand position toward its target by at most a fixed distance.

    Parameters
    ----------
    previous : ndarray, shape (3,)
        Position sent in the previous cycle.
    desired : ndarray, shape (3,)
        Position requested in this cycle.
    max_step_m : float
        Largest allowed distance for this cycle.

    Returns
    -------
    ndarray, shape (3,)
        ``desired`` when it is close enough, otherwise the point at
        ``max_step_m`` from ``previous`` in the same direction.
    """
    step = desired - previous
    distance = float(np.linalg.norm(step))

    if distance <= max_step_m:
        return desired.copy()
    
    return previous + step * (max_step_m / distance)


def scale_joint_step(step: np.ndarray, max_step_rad: float) -> np.ndarray:
    """Shrink all joint changes by one ratio so none exceeds the limit.

    Parameters
    ----------
    step : ndarray, shape (n,)
        Requested change of each joint in this cycle.
    max_step_rad : float
        Largest allowed change of any single joint in this cycle.

    Returns
    -------
    ndarray, shape (n,)
        ``step`` when every joint is within the limit, otherwise ``step``
        scaled so that the largest change equals ``max_step_rad``.
    """
    largest = float(np.max(np.abs(step)))

    if largest <= max_step_rad:
        return step.copy()
    
    return step * (max_step_rad / largest)


def extract_qpos(value: pa.Array) -> np.ndarray:
    """Read current and legacy Dora joint payloads.

    Parameters
    ----------
    value : pyarrow.Array
        A length-one qpos struct or a flat eight-element array.

    Returns
    -------
    ndarray, shape (8,)
        Finite joint positions, including the gripper.
    """
    if pa.types.is_struct(value.type): value = value.field("qpos")[0].values
    qpos = np.asarray(value, dtype=np.float64)

    if qpos.shape != (8,) or not np.all(np.isfinite(qpos)):
        raise ValueError("Expected eight finite joint positions")
    return qpos


@dataclass
class ArmRuntime:
    """Store one arm's measured state and independent control mode."""

    enabled: bool = False
    started: bool = False
    aligned: bool = False
    epoch: int | None = None
    mode: str = "idle"
    state: np.ndarray | None = None
    state_at: float = -np.inf
    command: np.ndarray | None = None
    command_at: float = 0.0
    target: np.ndarray | None = None
    target_at: float = 0.0
    home_start_pose: np.ndarray | None = None
    home_started_at: float = 0.0
    accept_after_ns: int = 0
    start_requested_ns: int = 0
    motor_status: list[str] | None = None
    bus: dict | None = None
    observation_ns: int | None = None
    solution: np.ndarray | None = None
    solution_at: float = -np.inf
    driver_command: np.ndarray | None = None
    last_sent: np.ndarray | None = None
    gravity_q: np.ndarray | None = None
    home_logged_at: float = -np.inf
    driver_phase: str = "stopped"
    reason: str = ""
    disabled_confirmed: int = -1
    heartbeat_supported: bool = False
    driver_seen_at: float = -np.inf


class Bridge:
    """Keep the OSC protocol common to simulation and real arms.

    Parameters
    ----------
    node : dora.Node
        Dora output connection.
    config : dict
        Control parameters loaded from config/control.yml.
    backend : {'sim', 'hardware'}
        Simulation ignores the two real-motor Enable switches.
    sim_ctrl : bool
        Whether the simulation uses position actuators and physics stepping.
    hardware_config : Path
        Same driver profile used by both real-arm Dora nodes.
    status_client : object, optional
        Receive-only display destination; tests inject an in-memory sender.
    recorder : FollowRecorder, optional
        Writes each Follow tick when recording is enabled.
    """

    def __init__(self, node, config: dict, backend: str, *,
                 sim_ctrl: bool = False,
                 hardware_config: Path = Path("config/openarm_pedestal.yaml"),
                 status_client=None,
                 recorder: FollowRecorder | None = None) -> None:
        """Initialize protocol state, the official FK model, and command limits."""
        self.node, self.config, self.backend = node, config, backend
        self.recorder = recorder
        self.gravity_compensation = backend == "sim" and sim_ctrl and config["sim_gravity_compensation"]
        self.gravity_scale, self.gravity_max_offset = 1.0, np.inf
        self.driver_start = {}
        self.status_client, self.status_session = status_client, str(time.time_ns())
        self.status_error = False
        self.status_seq = {side: 0 for side in SIDES}
        self.status_sent = {side: (None, -np.inf) for side in SIDES}
        self.home_diagnostics = config.get("home_diagnostics", False)
        self.needs_sync = False
        self.home_preset = config["home_preset"]

        if self.home_preset not in ("forward", "down"):
            raise ValueError("home_preset must be 'forward' or 'down'")
        setup = ArmSetup.from_args(
            xml=openarm_mujoco.openarm_pedestal_xml(), mode="bimanual",
            frame_right="right_ee_control_point", frame_type_right="site",
            frame_left="left_ee_control_point", frame_type_left="site",
            keyframe="home", origin_frame="arm_origin", origin_frame_type="site")
        self.fk = Kinematics(setup)
        self.arm_dofs = {side: setup.joint_resolver.arm_dof_indices(side) for side in SIDES}
        self.arm_kp = {side: np.array([setup.model.actuator(f"{side}_joint{i}_ctrl").gainprm[0]
                                      for i in range(1, 8)]) for side in SIDES}
        self.home = {side: np.asarray(config["home_presets"][self.home_preset][side], dtype=np.float64) for side in SIDES}
        self.limits = {side: np.array([setup.model.actuator(f"{side}_joint{i}_ctrl").ctrlrange
                                      for i in range(1, 8)] +
                                     [setup.model.actuator(f"{side}_finger1_ctrl").ctrlrange])
                       for side in SIDES}
        
        if backend == "hardware":
            compensation = config.get("hardware_gravity_compensation", {"enabled": False, "scale": 0.5, "max_offset_rad": 0.15})
            enabled, scale, maximum = compensation["enabled"], compensation["scale"], compensation["max_offset_rad"]
            if not isinstance(enabled, bool): raise ValueError("hardware gravity enabled must be true or false")
            if isinstance(scale, bool) or not np.isfinite(scale) or not 0 <= scale <= 1:
                raise ValueError("hardware gravity scale must be finite and between 0 and 1")
            if isinstance(maximum, bool) or not np.isfinite(maximum) or maximum <= 0:
                raise ValueError("hardware gravity max_offset_rad must be positive and finite")
            
            self.gravity_compensation = enabled and scale > 0
            self.gravity_scale, self.gravity_max_offset = float(scale), float(maximum)
            profile = yaml.safe_load(Path(hardware_config).read_text(encoding="utf-8"))
            gains = np.asarray(profile["control_gains"]["kps"], dtype=float)

            if gains.shape != (8,) or not np.all(np.isfinite(gains)) or np.any(gains[:7] <= 0):
                raise ValueError("hardware control_gains.kps must contain eight finite values and positive arm gains")
            if not profile["start"]["moves"]: raise ValueError("hardware start.moves must define a final holding command")

            for side in SIDES:
                bounds = np.asarray(profile["joint_limits"][f"{side}_arm"], dtype=float)
                if bounds.shape != (8, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
                    raise ValueError(f"Invalid hardware joint limits for {side}")
                self.limits[side][:, 0] = np.maximum(self.limits[side][:, 0], bounds[:, 0])
                self.limits[side][:, 1] = np.minimum(self.limits[side][:, 1], bounds[:, 1])
                if np.any(self.limits[side][:, 0] >= self.limits[side][:, 1]):
                    raise ValueError(f"Hardware and model joint limits do not overlap for {side}")
                self.arm_kp[side] = gains[:7].copy()
                initial = np.asarray(profile["start"]["moves"][-1]["position"][f"{side}_arm"], dtype=float)
                if initial.shape != (8,) or not np.all(np.isfinite(initial)) or np.any(initial < self.limits[side][:, 0]) or np.any(initial > self.limits[side][:, 1]):
                    raise ValueError(f"Invalid hardware final start command for {side}")
                self.driver_start[side] = initial.copy()

        for side in SIDES:
            q = self.home[side]
            if q.shape != (8,) or not np.all(np.isfinite(q)) or np.any(q < self.limits[side][:, 0]) or np.any(q > self.limits[side][:, 1]):
                raise ValueError(f"Invalid {self.home_preset} Home joints for {side}")
            
        poses = self.fk.fk_bimanual(self.home["right"], self.home["left"])
        self.home_pose = dict(zip(SIDES, poses))
        self.arms = {side: ArmRuntime(target=self.home_pose[side].copy()) for side in SIDES}
        self.lock = threading.Lock()
        self.packets = {side: (None, -np.inf) for side in SIDES}
        self.levels = {"home": False, "follow": False, "enable_right": False, "enable_left": False}
        self.enable_at = {side: -np.inf for side in SIDES}
        self.events = []

    def send(self, name: str, values, field: str | None = None, metadata=None) -> None:
        """Publish a Dora array, optionally wrapping floats in a canonical struct.

        Parameters
        ----------
        name : str
            Declared output name.
        values : array_like
            Float values or command strings.
        field : str, optional
            Struct field, either qpos or pose.
        metadata : dict, optional
            Driver session metadata.
        """
        if field:
            value = pa.array([{field: np.asarray(values).tolist()}],
                             type=pa.struct({field: pa.list_(pa.float32())}))
        else: value = pa.array(values)

        self.node.send_output(name, value, metadata or {})

    def receive(self, address: str, *values) -> None:
        """Validate OSC values and cache packets without touching the control loop.

        Parameters
        ----------
        address : str
            An address used by transform_sender.toe.
        *values : float
            OSC message arguments.
        """
        key = address.removeprefix("/openarm/")
        if not address.startswith("/openarm/") or key not in (*SIDES, *self.levels): return

        try:
            data = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(data)): raise ValueError("Non-finite OSC value")
            if key in SIDES:
                if data.shape != (8,): raise ValueError("Pose requires eight values")
                if np.linalg.norm(data[:3]) > self.config["max_move_m"]: raise ValueError("Position offset exceeds max_move_m")
                norm = np.linalg.norm(data[3:7])
                if norm < 1e-6: raise ValueError("Zero quaternion")
                data[3:7] /= norm
                data[7] = np.clip(data[7], -np.pi/4, 0)
            elif data.shape != (1,) or data[0] not in (0, 1):
                raise ValueError("Control switches require one value, zero or one")
        except (ValueError, TypeError) as error:
            print(f"[OSC] Rejected {address}: {error}", flush=True)
            return
        
        now = time.monotonic()
        with self.lock:
            if key in SIDES: self.packets[key] = (data, now)
            else:
                level = bool(data[0])
                if key.startswith("enable_"): self.enable_at[key[7:]] = now
                if level != self.levels[key]: self.events.append((key, level))
                self.levels[key] = level

    def active(self, side: str) -> bool:
        """Return whether this backend permits commands for the selected arm."""
        arm = self.arms[side]
        return self.backend == "sim" or (arm.enabled and arm.started and arm.driver_phase in ("started", "aligned"))

    def phase(self, side: str) -> str:
        """Return the operator phase without equating stop requests with completion."""
        arm = self.arms[side]
        if self.backend == "hardware":
            if arm.driver_phase in ("starting", "stop_requested", "stopping", "disabling", "fault"): return arm.driver_phase.upper()
            if not arm.enabled: return "STOPPED"
            if not arm.started: return "STARTING"
            if arm.state is None or time.monotonic()-arm.state_at > self.config["timeout_s"]: return "WAIT_STATE"

        return "WAIT_HOME" if arm.mode == "idle" else arm.mode.upper()

    def publish_status(self) -> None:
        """Send phase/reason changes immediately and a 2 Hz receive-only heartbeat."""
        if self.status_client is None: return
        now = time.monotonic()

        with self.lock: pose_times = {side: self.packets[side][1] for side in SIDES}
        for side, arm in self.arms.items():
            fresh = arm.state is not None and now-arm.state_at <= self.config["timeout_s"]
            can_home = int(self.active(side) and fresh)
            can_follow = int(can_home and arm.mode == "ready" and now-pose_times[side] <= self.config["timeout_s"])
            signature = (self.phase(side), arm.reason, can_home, can_follow, arm.disabled_confirmed)
            previous, sent_at = self.status_sent[side]

            if signature == previous and now-sent_at < 0.5: continue
            position_error, angle_error = -1.0, -1.0

            if fresh:
                measured = self.fk.fk(side, arm.state)
                position_error = float(np.linalg.norm(measured[:3]-self.home_pose[side][:3]))
                angle_error = float(np.rad2deg(2*np.arccos(np.clip(abs(measured[3:7]@self.home_pose[side][3:7]), 0, 1))))
            self.status_seq[side] = (self.status_seq[side]+1) % (2**31)
            payload = [self.status_session, self.status_seq[side], signature[0], arm.reason,
                       position_error, angle_error, can_home, can_follow, arm.disabled_confirmed,
                       int(self.gravity_compensation), float(self.gravity_scale)]
            try:
                self.status_client.send_message(f"/openarm/status/{side}", payload)
                self.status_error = False
            except OSError:
                # Display traffic must never break or actuate the control path.
                if not self.status_error: print("[STATUS] UDP notification failed", flush=True)
                self.status_error = True
            self.status_sent[side] = (signature, now)

    def gravity_offset(self, side: str) -> np.ndarray:
        """Return a backend-specific position-servo offset for gravity load.

        Parameters
        ----------
        side : {'right', 'left'}
            Arm with fresh measured joint positions.

        Returns
        -------
        ndarray, shape (8,)
            Scaled gravity torque divided by servo gain, with no gripper offset.
            Hardware offsets are capped before joint and slew limits apply.
        """
        if not self.gravity_compensation: return np.zeros(8)
        # FK data has zero velocity, so qfrc_bias here contains gravity only.
        self.fk.fk(side, self.arms[side].state)
        offset = self.gravity_scale * self.fk.setup.data.qfrc_bias[self.arm_dofs[side]] / self.arm_kp[side]

        return np.r_[np.clip(offset, -self.gravity_max_offset, self.gravity_max_offset), 0.0]

    def stop(self, side: str, reason: str, motors: bool = False, *, driver_ack: bool = False) -> None:
        """Invalidate Home/Follow and optionally stop one real arm.

        Parameters
        ----------
        side : {'right', 'left'}
            Arm to stop; the other arm is unaffected.
        reason : str
            Operator-facing reason.
        motors : bool
            Also send the real driver's stop command.
        driver_ack : bool
            Clear local control after a driver notification without sending stop again.
        """
        arm = self.arms[side]
        was_following = arm.mode == "follow"
        if arm.driver_phase not in ("stop_requested", "stopping", "disabling") or not arm.reason: arm.reason = reason
        if arm.mode == "home": self.log_home(side, time.monotonic(), f"stop: {reason}")

        if motors and self.backend == "hardware":
            if not driver_ack and (arm.enabled or arm.started or (reason == "Enable OFF" and arm.driver_phase == "fault")):
                self.send(f"arm_command_{side}", ["stop"], metadata={"reason": arm.reason})
                arm.driver_phase, arm.disabled_confirmed = "stop_requested", 0
            arm.enabled, arm.started, arm.state, arm.command = False, False, None, None
            arm.aligned = False
            arm.state_at, arm.epoch = -np.inf, None
            arm.motor_status, arm.bus, arm.observation_ns = None, None, None
            arm.solution, arm.last_sent, arm.gravity_q = None, None, None
            arm.solution_at, arm.driver_command = -np.inf, None

        arm.mode = "idle"
        arm.accept_after_ns = time.time_ns()
        print(f"[{side}] {self.phase(side) if motors and self.backend == 'hardware' else 'IDLE'}: {arm.reason}", flush=True)
        self.publish_status()

        # Close the recording last so a disk error cannot block the stop.
        if was_following and self.recorder is not None:
            self.recorder.end(side, arm.reason)

    def update_state(self, side: str, value: pa.Array, metadata: dict) -> None:
        """Cache measured joints only for the current real-arm session.

        Parameters
        ----------
        side : {'right', 'left'}
            Observation source.
        value : pyarrow.Array
            Canonical state/position struct or legacy joint array.
        metadata : dict
            Optional start_epoch and observation_timestamp from the driver.
        """
        arm = self.arms[side]
        if not self.active(side): return
        if self.backend == "hardware":
            if arm.epoch is not None and metadata.get("start_epoch") != arm.epoch: return
            if metadata.get("observation_timestamp", 0) < arm.start_requested_ns: return

        try: arm.state = extract_qpos(value)
        except ValueError as error:
            print(f"[{side}] Invalid observation: {error}", flush=True)
            return
        
        arm.state_at = time.monotonic()
        arm.observation_ns = metadata.get("observation_timestamp")
        names = value.type.names if pa.types.is_struct(value.type) else []
        arm.motor_status = value.field("motor_status")[0].as_py() if "motor_status" in names else None
        arm.bus = value.field("bus")[0].as_py() if "bus" in names else None
        arm.driver_command = (
            np.asarray(value.field("command_qpos")[0].as_py(), dtype=float)
            if "command_qpos" in names else None)
        
        if arm.command is None:
            # A loaded arm sags below its servo target; do not replace that holding target with feedback.
            holding = value.field("command_qpos")[0].as_py() if "command_qpos" in names else self.driver_start.get(side)
            if self.backend == "hardware" and holding is not None:
                try: holding = extract_qpos(pa.array(holding))
                except (ValueError, TypeError) as error:
                    self.stop(side, f"invalid driver holding command: {error}", motors=True)
                    return
                if np.any(holding < self.limits[side][:, 0]) or np.any(holding > self.limits[side][:, 1]):
                    self.stop(side, "driver holding command is outside model limits", motors=True)
                    return
            arm.command = (np.array(holding, dtype=float) if self.backend == "hardware" else
                           np.clip(arm.state + self.gravity_offset(side), self.limits[side][:, 0], self.limits[side][:, 1]))
            arm.command_at = arm.state_at

        if arm.mode == "idle" and arm.reason == "Waiting for fresh joint state": arm.reason = "Home required"

    def log_home(self, side: str, now: float, event: str = "periodic") -> None:
        """Report hardware Home errors without changing control or polling CAN.

        Parameters
        ----------
        side : {'right', 'left'}
            Arm with accepted observations in the current driver session.
        now : float
            Monotonic time for elapsed time and observation receipt age.
        event : str
            Periodic reports are limited to one per second; transition reports
            include the start, first command, completion and stop reason.
        """
        arm = self.arms[side]

        if self.backend != "hardware" or arm.state is None or not self.home_diagnostics: return
        if event == "periodic" and now-arm.home_logged_at < 1.0: return

        measured = self.fk.fk(side, arm.state)
        angle = 2*np.arccos(np.clip(abs(np.dot(measured[3:7], self.home_pose[side][3:7])), 0, 1))
        report = {
            "event": event, "elapsed_s": round(now-arm.home_started_at, 3),
            "pos_err_m": round(float(np.linalg.norm(measured[:3]-self.home_pose[side][:3])), 6),
            "angle_err_deg": round(float(np.rad2deg(angle)), 3),
            "path_err_m": round(float(np.linalg.norm(measured[:3]-arm.target[:3])), 6),
            "state_age_s": round(now-arm.state_at, 4),
            "snapshot_age_s": None if arm.observation_ns is None else round((time.time_ns()-arm.observation_ns)/1e9, 4),
            "measured_q": np.round(arm.state, 5).tolist(),
            "ik_q": None if arm.solution is None else np.round(arm.solution, 5).tolist(),
            "rate_ref_q": None if arm.command is None else np.round(arm.command, 5).tolist(),
            "sent_q": None if arm.last_sent is None else np.round(arm.last_sent, 5).tolist(),
            "gravity_scale": self.gravity_scale if self.gravity_compensation else 0.0,
            "gravity_offset_q": None if arm.gravity_q is None else np.round(arm.gravity_q, 5).tolist(),
            "motor_status": arm.motor_status, "bus": arm.bus,
        }

        print(f"[{side}] HOME diag " + json.dumps(report, separators=(",", ":")), flush=True)
        arm.home_logged_at = now

    def update_status(self, side: str, value: pa.Array, metadata: dict) -> None:
        """Track per-arm start/stop acknowledgements from the real driver.

        Parameters
        ----------
        side : {'right', 'left'}
            Driver source.
        value : pyarrow.Array
            Driver status string.
        metadata : dict
            Session epoch supplied by the official driver.
        """
        arm, status = self.arms[side], value[0].as_py()
        if metadata.get("status_timestamp", arm.start_requested_ns) < arm.start_requested_ns: return

        if status == "heartbeat":
            if arm.epoch is not None and metadata.get("start_epoch") != arm.epoch: return
            if time.time_ns()-metadata.get("status_timestamp", 0) > DRIVER_HEARTBEAT_TIMEOUT_S*1e9: return

        arm.heartbeat_supported = metadata.get("heartbeat_supported", arm.heartbeat_supported)
        arm.driver_seen_at = time.monotonic()

        if status == "heartbeat": return  # Liveness cannot arm, complete, or overwrite a rejection.
        if status == "starting":
            if not arm.enabled:
                self.send(f"arm_command_{side}", ["stop"], metadata={"reason": arm.reason})
                return
            arm.driver_phase, arm.epoch = "starting", metadata.get("start_epoch")
        elif status in ("stopping", "disabling", "stopped", "fault"):
            arm.driver_phase = status
            arm.disabled_confirmed = int(metadata.get("disabled_confirmed", False))
            reason = metadata.get("reason") or arm.reason or "driver stopped; toggle Enable OFF then ON"
            arm.reason = reason
            self.stop(side, reason, motors=True, driver_ack=True)
        elif status in ("started", "aligned"):
            if arm.epoch is not None and metadata.get("start_epoch") != arm.epoch: return
            if not arm.enabled:
                self.send(f"arm_command_{side}", ["stop"], metadata={"reason": arm.reason})
                return
            arm.started, arm.epoch = True, metadata.get("start_epoch")
            arm.driver_phase, arm.disabled_confirmed = status, 0
            arm.reason = "Waiting for fresh joint state" if arm.mode == "idle" and arm.state is None else ("Home required" if arm.mode == "idle" else "")
            if status == "aligned": arm.aligned = True
            print(f"[{side}] DRIVER {status}", flush=True)
        elif status == "rejected":
            arm.reason = metadata.get("reason", "driver rejected command")
            phase = metadata.get("driver_phase", arm.driver_phase)
            if phase in ("stopping", "disabling", "fault"):
                arm.driver_phase = phase
                self.stop(side, arm.reason, motors=True, driver_ack=True)
            else: self.stop(side, arm.reason, motors=True)

        self.publish_status()

    def forward_solution(self, side: str, value: pa.Array, metadata: dict) -> None:
        """Gate and slew-limit an IK result before it reaches either backend.

        Parameters
        ----------
        side : {'right', 'left'}
            Target arm.
        value : pyarrow.Array
            Solved joint positions.
        metadata : dict
            Solver timestamp, used to discard results from earlier modes.
        """
        arm, now = self.arms[side], time.monotonic()

        if not self.active(side) or arm.mode not in ("home", "follow") or arm.command is None: return
        if now - arm.state_at > self.config["timeout_s"]: return
        if metadata.get("timestamp", 0) < arm.accept_after_ns: return

        try: target = extract_qpos(value)
        except ValueError as error:
            self.stop(side, f"invalid IK output: {error}", motors=True)
            return
        
        first_solution = arm.solution is None
        arm.solution, arm.solution_at = target.copy(), now
        if self.backend == "hardware" and not arm.aligned:
            if arm.last_sent is not None: return
            # The official aligner otherwise starts from measured q, losing the existing holding torque.
            if np.any(np.abs(arm.command[:7]-arm.state[:7]) >= HARDWARE_ALIGN_THRESHOLD_RAD):
                self.stop(side, "startup tracking error exceeds driver alignment threshold", motors=True)
                return
            target, arm.gravity_q = arm.command.copy(), np.zeros(8)
        else:
            arm.gravity_q = self.gravity_offset(side)
            target = np.clip(target + arm.gravity_q, self.limits[side][:, 0], self.limits[side][:, 1])

        dt = np.clip(now-arm.command_at, 0, 1/self.config["tick_hz"])
        speed = (self.config["home_joint_speed_rad_s"] if arm.mode == "home"
                 else self.config["follow_joint_speed_rad_s"][self.backend])
        # Slow all arm joints together so the hand keeps its path; the gripper is separate.
        step = target - arm.command
        step[:7] = scale_joint_step(step[:7], speed*dt)
        gripper_step = self.config["gripper_speed_rad_s"] * dt
        step[7] = np.clip(step[7], -gripper_step, gripper_step)
        arm.command += step
        arm.command_at = now

        out_meta = {"start_epoch": arm.epoch} if self.backend == "hardware" and arm.epoch is not None else {}
        self.send(f"position_{side}", arm.command, "qpos", out_meta)
        arm.last_sent = arm.command.copy()

        if arm.mode == "home" and first_solution: self.log_home(side, now, "first_command")

    def tick(self, now: float | None = None) -> None:
        """Process OSC edges and publish paired IK targets at the YAML timer rate.

        Parameters
        ----------
        now : float, optional
            Monotonic clock override used by deterministic tests.
        """
        now = time.monotonic() if now is None else now
        # Recording clocks, read only when recording; perf_counter is
        # fine-grained on Windows too.
        if self.recorder is not None:
            record_at, wall_ns = time.perf_counter(), time.time_ns()
        with self.lock:
            packets, enable_at = self.packets.copy(), self.enable_at.copy()
            events, self.events = self.events, []

        timeout = self.config["timeout_s"]
        follow_started = []
        for key, level in events:
            if key.startswith("enable_"):
                if self.backend == "sim": continue
                side = key[7:]
                arm = self.arms[side]

                if not level: self.stop(side, "Enable OFF", motors=True)
                elif now-packets[side][1] <= timeout and now-enable_at[side] <= timeout:
                    if arm.driver_phase in ("starting", "stop_requested", "stopping", "disabling"):
                        arm.reason = "Enable rejected: wait for driver stop completion"
                        continue
                    self.stop(side, "Enable ON; waiting for driver and Home")
                    arm.enabled, arm.started, arm.state, arm.command = True, False, None, None
                    arm.aligned = False
                    arm.driver_phase, arm.disabled_confirmed = "starting", 0
                    arm.state_at, arm.epoch = -np.inf, None
                    arm.start_requested_ns = time.time_ns()
                    arm.driver_seen_at = now
                    self.send(f"arm_command_{side}", ["start"])
                else: print(f"[{side}] Enable rejected: no fresh OSC; toggle OFF then ON", flush=True)
                continue
            if key == "follow" and not level:
                for side, arm in self.arms.items():
                    if arm.mode == "follow": self.stop(side, "Follow OFF; Home required")
            elif key == "home" and level:
                for side, arm in self.arms.items():
                    if not self.active(side):
                        if arm.enabled: arm.reason = "Home rejected: wait for DRIVER started, then toggle Home OFF then ON"
                        continue
                    
                    if now-arm.state_at > timeout:
                        arm.reason = "Home rejected: no fresh joint state"
                        print(f"[{side}] Home rejected: no fresh joint state", flush=True)
                        continue
                    initial = (np.clip(arm.command-self.gravity_offset(side), self.limits[side][:, 0], self.limits[side][:, 1])
                               if self.backend == "hardware" else arm.state)
                    arm.home_start_pose = self.fk.fk(side, initial).copy()
                    arm.target = arm.home_start_pose.copy()
                    arm.home_started_at, arm.mode, arm.accept_after_ns = now, "home", time.time_ns()
                    arm.command_at, self.needs_sync = now, True
                    arm.solution, arm.home_logged_at = None, -np.inf
                    arm.reason = ""

                    print(f"[{side}] HOME start ({self.home_preset}, {self.config['home_move_s']:.1f}s)", flush=True)
                    self.log_home(side, now, "start")
            elif key == "follow" and level:
                for side, arm in self.arms.items():
                    if not self.active(side):
                        if arm.enabled: arm.reason = "Follow rejected: complete Home and toggle Follow OFF then ON"
                        continue
                    
                    if arm.mode == "ready" and now-packets[side][1] <= timeout and now-arm.state_at <= timeout:
                        arm.mode, arm.accept_after_ns = "follow", time.time_ns()
                        # Start the hand-speed limit from the zero-offset Home pose.
                        arm.target, arm.target_at = self.home_pose[side].copy(), now
                        arm.reason = ""
                        self.needs_sync = True
                        follow_started.append(side)
                        print(f"[{side}] FOLLOW start", flush=True)
                    else:
                        arm.reason = "Follow rejected: complete Home and toggle Follow OFF then ON"
                        print(f"[{side}] {arm.reason}", flush=True)

        if follow_started and self.recorder is not None:
            self.recorder.start(follow_started, record_at)

        for side, arm in self.arms.items():
            relative, pose_at = packets[side]
            if self.backend == "hardware" and arm.heartbeat_supported and arm.driver_phase in ("starting", "started", "aligned", "stop_requested", "stopping", "disabling") and now-arm.driver_seen_at > DRIVER_HEARTBEAT_TIMEOUT_S:
                reason = "driver heartbeat timeout; disable unconfirmed"
                self.stop(side, reason, motors=True)
                arm.driver_phase, arm.disabled_confirmed, arm.reason = "fault", 0, reason
                continue
            
            if self.backend == "hardware" and arm.enabled and (now-pose_at > timeout or now-enable_at[side] > timeout):
                self.stop(side, "OSC timeout; toggle Enable OFF then ON", motors=True)
                continue
            
            if not self.active(side): continue
            if arm.mode in ("home", "follow") and now-arm.state_at > timeout:
                self.stop(side, "joint-state timeout", motors=True)
                continue
            
            if self.backend == "hardware": self.send(f"request_state_{side}", [True])
            if arm.mode == "home":
                elapsed = now-arm.home_started_at
                t = np.clip(elapsed/self.config["home_move_s"], 0, 1)
                alpha = t*t*(3-2*t)
                arm.target = arm.home_start_pose.copy()
                arm.target[:3] += alpha*(self.home_pose[side][:3]-arm.home_start_pose[:3])
                arm.target[3:7] = quaternion_slerp(arm.home_start_pose[3:7], self.home_pose[side][3:7], alpha)
                if elapsed >= self.config["home_move_s"]:
                    measured = self.fk.fk(side, arm.state)
                    position_error = np.linalg.norm(measured[:3]-self.home_pose[side][:3])
                    angle_error = 2*np.arccos(np.clip(abs(np.dot(measured[3:7], self.home_pose[side][3:7])), 0, 1))
                    if position_error <= self.config["home_position_tol_m"] and angle_error <= np.deg2rad(self.config["home_angle_tol_deg"]) and (self.backend != "hardware" or arm.aligned):
                        self.log_home(side, now, "complete")
                        arm.mode = "ready"
                        arm.reason = "Home complete; toggle Follow OFF then ON"
                        print(f"[{side}] HOME complete -> READY", flush=True)
                    elif elapsed > self.config["home_timeout_s"]:
                        self.stop(side, "Home timed out", motors=True)
                if arm.mode == "home": self.log_home(side, now)
            elif arm.mode == "follow":
                if now-pose_at > timeout:
                    self.stop(side, "OSC timeout; Home required", motors=True)
                    continue
                # Limit the hand speed before IK; joint limits still apply after IK.
                dt = np.clip(now-arm.target_at, 0, 1/self.config["tick_hz"])
                max_step = self.config["follow_hand_speed_m_s"][self.backend] * dt
                target = arm.target.copy()
                target[:3] = limit_position_step(
                    arm.target[:3], self.home_pose[side][:3]+relative[:3], max_step)
                target[3:7] = quaternion_multiply(self.home_pose[side][3:7], relative[3:7])
                arm.target, arm.target_at = target, now
            elif now-arm.state_at <= timeout: arm.target = self.fk.fk(side, arm.state).copy()
        
        self.publish_status()
        if not any(arm.mode in ("home", "follow") for arm in self.arms.values()): return
        # Missing disabled-arm measurements are only an IK seed, never observations or commands.
        if self.needs_sync:
            seed = np.concatenate([
                (np.clip(arm.command-self.gravity_offset(side), self.limits[side][:, 0], self.limits[side][:, 1])
                 if self.backend == "hardware" else arm.state)
                if now-arm.state_at <= timeout else self.home[side] for side, arm in self.arms.items()])
            self.send("position", seed, "qpos")
            self.needs_sync = False
        for side, arm in self.arms.items():
            gripper = arm.state[7] if now-arm.state_at <= timeout else self.home[side][7]
            if arm.mode == "follow": gripper = packets[side][0][7] * (1 if side == "right" else -1)
            self.send(f"target_{side}", np.r_[arm.target, gripper], "pose")
            if arm.mode == "follow" and self.recorder is not None:
                self.recorder.write(side, record_at, self.record_values(
                    side, now, wall_ns, packets[side], gripper))

    def record_values(self, side: str, now: float, wall_ns: int,
                      packet: tuple, gripper: float) -> list:
        """Collect one Follow tick of one arm for the recorder.

        Parameters
        ----------
        side : {'right', 'left'}
            Arm in Follow.
        now : float
            Monotonic time of the tick.
        wall_ns : int
            PC clock time of the tick, for later alignment with TD.
        packet : tuple
            Latest OSC relative target (eight values) and its receive time.
        gripper : float
            Gripper target sent to IK on this tick.

        Returns
        -------
        list
            Values in the order of ``RECORD_HEADER`` after ``t_s``.
            Missing values are ``None``.
        """
        arm = self.arms[side]
        relative, pose_at = packet

        def joints(values):
            """Return eight joint values, or eight blanks when missing.

            Parameters
            ----------
            values : ndarray or None
                Joint vector of the arm and gripper.

            Returns
            -------
            list
                Eight floats or eight ``None``.
            """
            return [None]*8 if values is None else [float(v) for v in values]

        def age(at):
            """Return seconds since ``at``, or ``None`` if never set.

            Parameters
            ----------
            at : float
                Monotonic time of the value.

            Returns
            -------
            float or None
                Age in seconds.
            """
            return now - at if np.isfinite(at) else None

        observation = arm.observation_ns
        return ([wall_ns, age(pose_at)] + joints(relative)
                + [float(v) for v in arm.target] + [float(gripper)]
                + [age(arm.solution_at)] + joints(arm.solution)
                + [age(arm.command_at) if arm.last_sent is not None
                   else None] + joints(arm.last_sent)
                + joints(arm.gravity_q)
                + [age(arm.state_at),
                   None if observation is None else int(observation)]
                + joints(arm.state) + joints(arm.driver_command))


def main() -> None:
    """Run the OSC receiver and Dora event loop with the selected backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sim", "hardware"], default="sim")
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--config", type=Path, default=Path("config/control.yml"))
    parser.add_argument("--hardware-config", type=Path, default=Path("config/openarm_pedestal.yaml"))
    args = parser.parse_args()

    # Only hardware_node constructs CAN devices and enforces the Linux boundary.
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("tick_hz", "timeout_s", "max_move_m", "home_move_s", "home_timeout_s", "home_position_tol_m", "home_angle_tol_deg", "home_joint_speed_rad_s", "gripper_speed_rad_s"):
        if not np.isfinite(config[key]) or config[key] <= 0: parser.error(f"{key} must be positive and finite")
    for backend in ("sim", "hardware"):
        speed = config["follow_joint_speed_rad_s"][backend]
        if not np.isfinite(speed) or speed <= 0: parser.error(f"Follow speed for {backend} must be positive and finite")
        hand_speed = config["follow_hand_speed_m_s"][backend]
        if not np.isfinite(hand_speed) or hand_speed <= 0:
            parser.error(f"Follow hand speed for {backend} must be positive and finite")
    
    if not isinstance(config["sim_gravity_compensation"], bool): parser.error("sim_gravity_compensation must be true or false")
    
    if config["tick_hz"] != 100: parser.error("tick_hz must match the 10 ms YAML timer and IK settings (100)")
    sim_ctrl = os.getenv("DORA_OPENARM_MUJOCO_CTRL", "false").lower() in ("1", "true", "yes", "on")
    recorder = None
    
    if os.getenv(RECORD_FLAG_ENV, "").lower() in ("1", "true", "yes", "on"):
        record_dir = config.get("record_dir")
        if not isinstance(record_dir, str) or not record_dir:
            parser.error("record_dir must be a folder path to record")
        run_name = time.strftime("%Y%m%d_%H%M%S")
        recorder = FollowRecorder(Path(record_dir) / run_name)
    # Validate the hardware profile before connecting this bridge to Dora.
    bridge = Bridge(None, config, args.backend, sim_ctrl=sim_ctrl,
                    hardware_config=args.hardware_config, recorder=recorder)
    status_client = None
    
    if args.backend == "hardware" and config.get("status", {}).get("enabled", False):
        status = config["status"]
        try: ipaddress.IPv4Address(status["host"])
        except ipaddress.AddressValueError: parser.error("status.host must be an explicit IPv4 address")
        if not isinstance(status["port"], int) or not 1 <= status["port"] <= 65535: parser.error("Invalid status.port")
        status_client = udp_client.SimpleUDPClient(status["host"], status["port"])
        bridge.status_client = status_client
    
    node = dora.Node()
    bridge.node = node
    routes = dispatcher.Dispatcher()
    routes.set_default_handler(bridge.receive)
    server = osc_server.OSCUDPServer((args.listen, config["osc_port"]), routes)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    print(f"[OSC] {args.listen}:{config['osc_port']} backend={args.backend}", flush=True)
    print(f"[MODEL] pedestal; Home preset={bridge.home_preset}", flush=True)
    print(f"[CONTROL] Home={config['home_joint_speed_rad_s']} Follow={config['follow_joint_speed_rad_s'][args.backend]} rad/s, hand={config['follow_hand_speed_m_s'][args.backend]} m/s; gravity compensation={bridge.gravity_compensation}", flush=True)
    
    if args.backend == "hardware":
        print(f"[GRAVITY] hardware enabled={bridge.gravity_compensation} scale={bridge.gravity_scale:g} max_offset={bridge.gravity_max_offset:g} rad; profile={args.hardware_config}", flush=True)
    
    print("[OSC] /openarm/left, /right, /home, /follow, /enable_left, /enable_right", flush=True)
    record_text = "off" if recorder is None else recorder.folder
    print(f"[RECORD] {record_text}", flush=True)
    
    try:
        for event in node:
            if event["type"] == "STOP": break
            if event["type"] != "INPUT": continue
            name, value, metadata = event["id"], event["value"], event.get("metadata", {})
            if name == "tick": bridge.tick()
            elif name.endswith("_state"): bridge.update_state(name[:-6], value, metadata)
            elif name.endswith("_status"): bridge.update_status(name[:-7], value, metadata)
            elif name.startswith("solution_"): bridge.forward_solution(name[9:], value, metadata)
    finally:
        # Detach the recorder first so a disk error cannot skip an arm stop.
        bridge.recorder = None
        for side in SIDES: bridge.stop(side, "bridge exit", motors=True)
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
        if status_client is not None: status_client.close()
        if recorder is not None: recorder.close()


if __name__ == "__main__": main()
