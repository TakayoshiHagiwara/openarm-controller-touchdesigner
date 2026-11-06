#!/usr/bin/env bash
# Start the MuJoCo dataflow with physics, no collision, and a 60 FPS viewer.
# Add --record to write Follow data to record_dir in config/control.yml.
# The settings apply only to this run and are not kept in the shell.
set -euo pipefail
cd "$(dirname "$0")"
unset OPENARM_BRIDGE_RECORD
case "${1:-}" in
    --record) export OPENARM_BRIDGE_RECORD=1 ;;
    "") ;;
    *) echo "Usage: $0 [--record]" >&2; exit 2 ;;
esac
export DORA_OPENARM_MUJOCO_CTRL=true
export DORA_OPENARM_MUJOCO_ENABLE_COLLISION=false
export DORA_OPENARM_MUJOCO_VIEWER=60
exec uv run --extra sim dora run dataflow-mujoco.yml
