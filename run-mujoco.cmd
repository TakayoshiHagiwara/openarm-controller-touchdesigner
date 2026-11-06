@echo off
rem Start the MuJoCo dataflow with physics, no collision, and a 60 FPS viewer.
rem Add --record to write Follow data to record_dir in config/control.yml.
rem The settings apply only to this run and are not kept in the shell.
setlocal
cd /d "%~dp0"
set DORA_OPENARM_MUJOCO_CTRL=true
set DORA_OPENARM_MUJOCO_ENABLE_COLLISION=false
set DORA_OPENARM_MUJOCO_VIEWER=60
set OPENARM_BRIDGE_RECORD=
if "%~1"=="--record" (
    set OPENARM_BRIDGE_RECORD=1
) else if not "%~1"=="" (
    echo Unknown option: %~1
    echo Usage: run-mujoco.cmd [--record]
    exit /b 2
)
uv run --extra sim dora run dataflow-mujoco.yml
endlocal
