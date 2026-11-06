"""Test-only Dora worker; never imports or constructs a real OpenArm driver."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dora
import yaml

from hardware_node import run
from fake_hardware import FakeDriver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=0.2)
    args = parser.parse_args()
    profile = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    instances = []

    def factory():
        """Retain the simulated physical position across explicit enable sessions."""
        driver = FakeDriver(args.side, profile)
        if instances: driver.last_command = instances[-1].fetch_position()
        instances.append(driver)
        return driver

    run(dora.Node(), args.side, profile, factory, args.timeout)
