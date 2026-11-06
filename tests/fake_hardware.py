"""Test-only motor interfaces; no imports, sockets, or calls to openarm_can."""

import time

import mujoco
import numpy as np
import openarm_mujoco.v2 as om
from openarm_mujoco.v2 import JointResolver


class Clock:
    """Advance controller time without sleeping or depending on OS scheduling."""

    def __init__(self):
        """Start at a nonzero deterministic monotonic time."""
        self.now = 1000.0

    def monotonic(self):
        """Return virtual monotonic seconds."""
        return self.now

    def time_ns(self):
        """Return session-comparable virtual Unix nanoseconds."""
        return 1_800_000_000_000_000_000+round((self.now-1000)*1e9)


class Sink:
    """Record Dora outputs and their emission time."""

    def __init__(self, clock=time):
        """Initialize empty pending and historical output lists."""
        self.clock, self.pending, self.history = clock, [], []

    def send_output(self, name, value, metadata=None):
        """Keep one output without sending it to any process or device."""
        item = (name, value, metadata or {})
        self.pending.append(item)
        self.history.append((self.clock.monotonic(), *item))


class Plant:
    """Use the official Pedestal model with the unchanged real-arm PD gains."""

    def __init__(self, side, profile):
        """Initialize a contact-free physical plant with arms down."""
        self.side = side
        self.model = mujoco.MjModel.from_xml_path(om.openarm_pedestal_xml())
        self.data, self.resolver = mujoco.MjData(self.model), JointResolver(self.model)
        self.model.geom_contype[:], self.model.geom_conaffinity[:] = 0, 0
        self.ids = np.array([self.model.actuator(f"{side}_joint{i}_ctrl").id for i in range(1, 8)])
        self.grip_id = self.model.actuator(f"{side}_finger1_ctrl").id
        self.profile = profile
        for candidate in ("right", "left"):
            self.resolver.set_qpos(self.data.qpos, np.zeros(8), candidate)
            for index in range(7):
                actuator = self.model.actuator(f"{candidate}_joint{index+1}_ctrl")
                actuator.gainprm[0] = profile["control_gains"]["kps"][index]
                actuator.biasprm[1] = -profile["control_gains"]["kps"][index]
                actuator.biasprm[2] = -profile["control_gains"]["kds"][index]
        mujoco.mj_forward(self.model, self.data)

    def qpos(self):
        """Read measured joints in the same eight-axis driver convention."""
        joints, grip = self.resolver.get_driver(self.data.qpos, self.side)
        return np.r_[joints, grip].copy()

    def enable(self, enabled):
        """Enable or remove the seven simulated arm PD servos."""
        for index, actuator in enumerate(self.ids):
            self.model.actuator_gainprm[actuator, 0] = self.profile["control_gains"]["kps"][index] if enabled else 0
            self.model.actuator_biasprm[actuator, 1] = -self.profile["control_gains"]["kps"][index] if enabled else 0
            self.model.actuator_biasprm[actuator, 2] = -self.profile["control_gains"]["kds"][index] if enabled else 0

    def command(self, qpos):
        """Drive physical PD targets; the gripper is an approximate position servo."""
        self.data.ctrl[self.ids] = qpos[:7]
        self.data.ctrl[self.grip_id] = qpos[7]

    def advance(self, dt):
        """Step physics independently of the virtual controller clock."""
        for _ in range(round(dt/self.model.opt.timestep)): mujoco.mj_step(self.model, self.data)


class FakeDriver:
    """Expose the upstream driver's small API without CAN or network access."""

    def __init__(self, side, profile, clock=time, plant=None):
        """Create an ideal or physical fake; all fault injection starts disabled."""
        self.side, self.profile, self.clock, self.plant = side, profile, clock, plant
        self.last_command = np.zeros(8) if plant is None else plant.qpos()
        self.enabled, self.started = False, False
        self.safety_stop_reason = None
        self.disable_ack, self.enable_ack, self.carrier = True, True, True
        self.reject_commands, self.raise_commands, self.raise_state, self.raise_disable = False, False, False, False
        self.commands, self.disable_times = [], []

    def start(self):
        """Enable only; the production controller owns the stepped trajectory."""
        self.enabled, self.started = True, True
        if self.plant is not None: self.plant.enable(True)
        return self.send_position(self.last_command)

    def send_position(self, command):
        """Record accepted commands or inject an upstream safety rejection."""
        if self.raise_commands: raise OSError("mock CAN write failure")
        if self.reject_commands:
            self.safety_stop_reason = "mock driver safety rejection"
            return False
        command = np.asarray(command, dtype=float)
        assert command.shape == (8,) and np.all(np.isfinite(command))
        bounds = np.array(self.profile["joint_limits"][f"{self.side}_arm"])
        assert np.all(command >= bounds[:, 0]-1e-6) and np.all(command <= bounds[:, 1]+1e-6)
        self.last_command = command.copy()
        self.commands.append((self.clock.monotonic(), command.copy()))
        if self.plant is not None: self.plant.command(command)
        return True

    def fetch_position(self):
        """Return ideal or physical measured joint positions."""
        return self.last_command.copy() if self.plant is None else self.plant.qpos()

    def fetch_state(self):
        """Return canonical feedback or inject a read failure."""
        if self.raise_state: raise OSError("mock CAN read failure")
        return {"qpos": self.fetch_position(), "qvel": np.zeros(8), "qtorque": np.zeros(8),
                "tmos": np.full(8, 25), "trotor": np.full(8, 25)}

    def get_health(self):
        """Expose explicit enable/disable acknowledgements and bus carrier."""
        status = "ENABLED" if ((self.enabled and self.enable_ack) or (not self.enabled and not self.disable_ack)) else "DISABLED"
        return [status]*8, {"carrier": self.carrier, "bus_off": 0}

    def disable(self):
        """Record disable without sleeping, or inject a transmission failure."""
        if self.raise_disable: raise OSError("mock disable failure")
        self.disable_times.append(self.clock.monotonic())
        self.enabled, self.started = False, False
        if self.plant is not None: self.plant.enable(False)
