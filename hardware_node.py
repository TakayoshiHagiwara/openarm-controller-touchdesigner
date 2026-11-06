"""Run one OpenArm with interruptible, single-threaded start/stop trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import dora
import numpy as np
import pyarrow as pa
import yaml

from bridge import HARDWARE_ALIGN_THRESHOLD_RAD, extract_qpos


def create_driver(side: str, path: Path):
    """Create a real driver only after an explicit Enable command.

    Parameters
    ----------
    side : {'right', 'left'}
        Arm selected by the hardware dataflow.
    path : Path
        Reviewed driver profile; gains and safety checkers remain upstream.
    """
    from openarm_driver import Config, SingleArmDriver

    class SteppedDriver(SingleArmDriver):
        """Leave trajectory timing to the event loop, keeping CAN on this thread."""

        def _on_start(self):
            """Hold the measured enable position before the stepped start move."""
            return self.send_position(self.last_command)

        def disable(self):
            """Send disable without the upstream one-second blocking sleep."""
            self.openarm.disable_all()
            self.set_latest_state(timeout_us=1000)
            self.started = False

    driver = SteppedDriver(f"{side}_arm", Config(path))
    collections = [driver.openarm.get_arm()]
    if driver.gripper_posforce: collections.append(driver.openarm.get_gripper())
    replies = [collection.get_link_stats(i).responses for collection in collections for i in range(len(collection.get_motors()))]
    
    if len(replies) != 8 or any(count == 0 for count in replies):
        missing = [i+1 for i, count in enumerate(replies) if count == 0]
        raise RuntimeError(f"preflight: expected 8 responding motors; no reply at axes {missing}; Enable was not sent")
    statuses, _ = driver.get_health()
    
    if len(statuses) != 8 or any(status != "DISABLED" for status in statuses):
        raise RuntimeError("preflight: all motors must report DISABLED before Enable")
    
    return driver


class HardwareController:
    """Advance one trajectory point per tick and accept stop at every event.

    Parameters
    ----------
    node : dora.Node
        State/status output connection, or a test sink.
    side : {'right', 'left'}
        The sole arm controlled by this process.
    profile : dict
        Unmodified driver gains, limits and start/stop moves.
    driver_factory : callable
        Deferred real driver construction, replaced by a fake in tests.
    timeout_s : float
        Existing communication timeout, also used for enable acknowledgement.
    clock : module or object
        Monotonic and Unix clocks; tests use virtual time.
    """

    def __init__(self, node, side: str, profile: dict, driver_factory, timeout_s: float = 0.2, clock=time):
        """Initialize a stopped process without constructing or enabling a driver."""
        self.node, self.side, self.profile = node, side, profile
        self.driver_factory, self.timeout_s, self.clock = driver_factory, timeout_s, clock
        self.driver, self.epoch, self.phase, self.reason = None, 0, "stopped", "motor state not yet verified"
        self.disabled_confirmed = False
        self.status_failed = False
        self.status_at = -np.inf
        self.moves, self.move_index, self.step_index = [], 0, 0
        self.origin, self.due, self.enable_started, self.disable_due = None, 0.0, 0.0, 0.0
        self.last_step_at = 0.0
        self.enable_pending = False

        for mode in ("start", "stop"):
            for move in profile[mode]["moves"]:
                target = np.asarray(move["position"][f"{side}_arm"], dtype=float)
                if target.shape != (8,) or not np.all(np.isfinite(target)):
                    raise ValueError(f"Invalid {mode} target for {side}")
                if move["duration"] <= 0 or move["hz"] <= 0 or int(move["duration"]*move["hz"]) < 2:
                    raise ValueError(f"Invalid {mode} move duration or rate")

    def status(self, phase: str, reason: str | None = None, *, heartbeat: bool = False) -> None:
        """Publish a phase transition without claiming unverified motor disable."""
        self.phase = phase
        if reason is not None: self.reason = reason
        metadata = {"start_epoch": self.epoch, "reason": self.reason,
                    "disabled_confirmed": self.disabled_confirmed, "status_timestamp": self.clock.time_ns(),
                    "heartbeat_supported": True, "driver_phase": self.phase}
        self.status_at = self.clock.monotonic()
        
        try:
            self.node.send_output("status", pa.array(["heartbeat" if heartbeat else phase]), metadata)
            self.status_failed = False
        except Exception as error:
            if not self.status_failed: print(f"[{self.side}] status channel unavailable: {error}", file=sys.stderr, flush=True)
            self.status_failed = True

    def begin_motion(self, mode: str) -> None:
        """Start configured moves from the last accepted servo command."""
        self.moves = self.profile[mode]["moves"]
        self.move_index, self.step_index = 0, 0
        self.origin = self.driver.last_command.copy()
        self.due = self.last_step_at = self.clock.monotonic()

    def start(self) -> None:
        """Enable once, then schedule the start trajectory without blocking for it."""
        if self.driver is not None or self.phase not in ("stopped", "fault"):
            self.node.send_output("status", pa.array(["rejected"]),
                                  {"start_epoch": self.epoch, "reason": "Enable rejected: driver is busy",
                                   "driver_phase": self.phase, "disabled_confirmed": self.disabled_confirmed,
                                   "status_timestamp": self.clock.time_ns()})
            return
        self.epoch += 1
        self.disabled_confirmed = False
        self.status("starting", "")
        
        if self.status_failed:
            self.status("fault", "status channel unavailable; Enable was not attempted")
            return
        try:
            self.driver = self.driver_factory()
            # Return to event processing before enabling: stop queued during setup can cancel it.
            self.enable_pending = True
            self.due = self.clock.monotonic()+0.02
        except Exception as error:
            if self.driver is None: self.status("fault", str(error))
            else: self.stop(f"startup error: {error}")

    def stop(self, reason: str = "stop requested") -> None:
        """Replace a start/holding trajectory with the existing normal return move."""
        if self.phase in ("stopping", "disabling"): return
        
        if self.driver is None:
            self.status(self.phase if self.phase == "fault" else "stopped", self.reason if self.phase == "fault" else reason)
            return
        
        if self.phase == "fault":
            self.begin_disable()
            return
        self.status("stopping", reason)
        
        if self.enable_pending:
            self.enable_pending = False
            self.begin_disable()  # No start/return motion when Enable has not been sent.
        elif self.driver.safety_stop_reason is not None: self.begin_disable()
        else: self.begin_motion("stop")

    def begin_disable(self) -> None:
        """Send disable and schedule acknowledgement checks without a blocking sleep."""
        try:
            self.driver.disable()
            self.disable_due = self.clock.monotonic()+1.0
            self.status("disabling")
        except Exception as error:
            self.status("fault", f"disable failed: {error}")

    def tick(self) -> None:
        """Send at most one 50 Hz point, allowing commands between every step."""
        now = self.clock.monotonic()
        
        if now-self.status_at >= 0.5: self.status(self.phase, heartbeat=True)
        if self.status_failed and self.driver is not None and self.phase not in ("stopping", "disabling", "fault"):
            self.stop("Dora status channel unavailable")
        
        if self.phase == "disabling":
            try:
                self.driver.fetch_state()
                if now < self.disable_due: return
                statuses, bus = self.driver.get_health()
                self.disabled_confirmed = len(statuses) == len(self.profile["motor_config"]["types"]) and all(s == "DISABLED" for s in statuses) and bus.get("carrier", False)
                if self.disabled_confirmed:
                    self.driver = None
                    self.status("stopped")
                else: self.status("fault", f"disable unconfirmed; original stop reason: {self.reason}")
            except Exception as error:
                self.status("fault", f"disable unconfirmed: {error}")
            return
        
        if self.phase not in ("starting", "stopping") or now+1e-9 < self.due: return
        try:
            if self.phase == "starting" and self.enable_pending:
                self.enable_pending = False
                if self.driver.start() is False:
                    self.stop(self.driver.safety_stop_reason or "driver rejected startup")
                    return
                self.enable_started = self.clock.monotonic()
                self.begin_motion("start")
                return
            
            if self.phase == "starting":
                self.driver.fetch_state()
                statuses, _ = self.driver.get_health()
                if len(statuses) != len(self.profile["motor_config"]["types"]) or any(s != "ENABLED" for s in statuses):
                    if now-self.enable_started >= self.timeout_s: self.stop("motor enable acknowledgement timed out")
                    self.due = now+0.02
                    return
            
            if self.move_index >= len(self.moves):
                if self.phase == "starting": self.status("started", "")
                else: self.begin_disable()
                return
            move = self.moves[self.move_index]
            count = int(move["duration"]*move["hz"])

            if self.step_index >= count-1:
                target = np.asarray(move["position"][f"{self.side}_arm"], dtype=float)
                if not np.allclose(self.driver.last_command, target, atol=1e-5, rtol=0):
                    if self.phase == "stopping":
                        self.reason = "stop trajectory final command was limited"
                        self.begin_disable()
                    else: self.stop("start trajectory final command was limited")
                    return
                self.move_index += 1
                self.step_index = 0
                self.origin = self.driver.last_command.copy()
                self.last_step_at = now
                self.tick()
                return
            
            target = np.asarray(move["position"][f"{self.side}_arm"], dtype=float)
            # Fractional progress handles timer jitter; a stalled loop may advance only one point.
            self.step_index = min(count-1, self.step_index+min(max(now-self.last_step_at, 0)*move["hz"], 1.0))
            command = self.origin+(target-self.origin)*(self.step_index/(count-1))

            if self.driver.send_position(command) is False:
                if self.phase == "stopping":
                    self.reason = "driver safety rejected a stop trajectory command"
                    self.begin_disable()
                else: self.stop("driver safety rejected a start trajectory command")
                return
            
            self.last_step_at = now
            self.due = now+1.0/move["hz"] if self.step_index >= count-1 else now
        except Exception as error:
            if self.phase == "stopping":
                self.reason = f"stop trajectory error: {error}"
                self.begin_disable()
            else: self.stop(f"start trajectory error: {error}")

    def position(self, value: pa.Array, metadata: dict) -> None:
        """Accept only current-session commands while the driver is ready."""
        if self.phase not in ("started", "aligned") or metadata.get("start_epoch") != self.epoch: return

        if self.status_failed:
            self.stop("Dora status channel unavailable")
            return
        
        try:
            command = extract_qpos(value)
            if self.phase == "started":
                measured = self.driver.fetch_position()
                if np.any(np.abs(command[:7]-measured[:7]) >= HARDWARE_ALIGN_THRESHOLD_RAD):
                    self.stop("startup tracking error exceeds driver alignment threshold")
                    return
                
            if self.driver.send_position(command) is False:
                self.stop("driver safety rejected a position command")
                return
            
            if self.phase == "started": self.status("aligned", "")
        except Exception as error: self.stop(f"position command error: {error}")

    def observation(self) -> None:
        """Publish measured state and the actual accepted holding target."""
        if self.phase not in ("started", "aligned"): return

        try:
            state = {name: np.asarray(values).tolist() for name, values in self.driver.fetch_state().items()}
            statuses, bus = self.driver.get_health()
            extract_qpos(pa.array(state["qpos"]))
            if not bus.get("carrier", False):
                self.stop("CAN carrier lost")
                return
            
            if len(statuses) != len(self.profile["motor_config"]["types"]) or any(s != "ENABLED" for s in statuses):
                self.stop("motor feedback is not ENABLED: "+", ".join(statuses))
                return
            state.update(command_qpos=self.driver.last_command.tolist(), motor_status=statuses, bus=bus)
            self.node.send_output("state", pa.array([state]),
                                  {"start_epoch": self.epoch, "observation_timestamp": self.clock.time_ns()})
        except Exception as error: self.stop(f"state read error: {error}")

    def handle(self, event: dict) -> None:
        """Dispatch one Dora input; stop never waits for the rest of a trajectory."""
        name = event["id"]

        if name == "command":
            command = event["value"][0].as_py()
            if command == "stop": self.stop(event.get("metadata", {}).get("reason", "stop requested"))
            elif command == "start": self.start()
        elif name == "motion_tick": self.tick()
        elif name == "request_state": self.observation()
        elif name == "move_position": self.position(event["value"], event.get("metadata", {}))


def run(node, side: str, profile: dict, driver_factory, timeout_s: float = 0.2) -> None:
    """Run the same event loop with a real driver or a test-only factory."""
    controller = HardwareController(node, side, profile, driver_factory, timeout_s)
    controller.status("stopped")

    try:
        for event in node:
            if event["type"] == "STOP": break
            if event["type"] == "INPUT": controller.handle(event)
            if controller.status_failed and controller.phase in ("stopped", "fault"):
                raise RuntimeError("Dora status channel unavailable; motors were not enabled or shutdown has completed")
    finally:
        controller.stop("dataflow shutdown")
        while controller.phase in ("stopping", "disabling"):
            controller.tick()
            time.sleep(0.02)
        if controller.phase == "fault": raise RuntimeError(f"shutdown left disable unconfirmed: {controller.reason}")


def main() -> None:
    """Load configuration without enabling motors; only an OSC Enable starts them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("right", "left"), required=True)
    parser.add_argument("--config", type=Path, default=Path("config/openarm_pedestal.yaml"))
    parser.add_argument("--timeout", type=float, default=0.2)
    args = parser.parse_args()
    
    if not np.isfinite(args.timeout) or args.timeout <= 0: parser.error("timeout must be positive and finite")
    if sys.platform != "linux": parser.error("Real hardware requires Ubuntu/Linux; this command has no simulation mode")
    profile = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(dora.Node(), args.side, profile, lambda: create_driver(args.side, args.config), args.timeout)


if __name__ == "__main__": main()
