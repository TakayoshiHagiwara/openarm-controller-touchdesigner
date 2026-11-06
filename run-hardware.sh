#!/usr/bin/env bash
# Start the hardware dataflow on Ubuntu.
# Add --record to write Follow data to record_dir in config/control.yml.
# The setting applies only to this run and is not kept in the shell.
set -euo pipefail
cd "$(dirname "$0")"
unset OPENARM_BRIDGE_RECORD
case "${1:-}" in
    --record) export OPENARM_BRIDGE_RECORD=1 ;;
    "") ;;
    *) echo "Usage: $0 [--record]" >&2; exit 2 ;;
esac
exec uv run --locked --extra sim --extra hardware \
    dora run dataflow-hardware.yml
