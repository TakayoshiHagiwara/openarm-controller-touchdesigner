# MuJoCo setup

Move the MuJoCo model of OpenArm v2 Pedestal from the TouchDesigner sender project (`td/transform_sender.toe`). The real robot's left and right Enable are not used with MuJoCo.

## Create the environment

Run this at the repository root. The virtual environment is created in `.venv` at the root and is independent of TouchDesigner's Python. Dependency versions are pinned in `uv.lock`.

```bash
uv sync --extra sim
```

## Start

The startup scripts start MuJoCo with physics on, collision off, and the viewer at 60 FPS. These settings apply only while it runs and do not remain in the shell.

| OS | Command |
|---|---|
| Windows (PowerShell) | `.\run-mujoco.cmd` |
| Linux, macOS | `bash run-mujoco.sh` |

When the following lines appear in the log, you can operate it from TouchDesigner.

```text
[OSC] 127.0.0.1:7000 backend=sim
[MODEL] pedestal; Home preset=forward
```

To quit, press `Ctrl+C` in the terminal where you started it.

With `--record`, the data during Follow is written to `record_dir` in `config/control.yml` (for the format, see "Recording" in [hardware_en.md](hardware_en.md)).

### Fine-tune with environment variables

To start without the scripts, set the following environment variables and then run `uv run --extra sim dora run dataflow-mujoco.yml`.

| Environment variable | Value | Meaning |
|---|---|---|
| `DORA_OPENARM_MUJOCO_VIEWER` | `60` etc. / `false` | Viewer FPS. `false` shows no viewer |
| `DORA_OPENARM_MUJOCO_CTRL` | `true` / `false` | `true`: run the joint position servos with physics. `false`: apply the joint angles of the IK solution directly (for checking kinematics) |
| `DORA_OPENARM_MUJOCO_ENABLE_COLLISION` | `true` / `false` | Whether collisions are computed |

With `CTRL=true` and `sim_gravity_compensation: true` in `config/control.yml`, the bridge adds a correction for gravity to the joint targets (a correction for simulation only). With physics, Home may not complete because of contact or tracking error. In that case, do not widen the tolerance; check the log and the pose.

## TouchDesigner operation

1. Open `td/transform_sender.toe` and set the destination to **MuJoCo** (127.0.0.1, Port 7000).
2. Set the input switch (Input source) to **Slider** and set the position and rotation sliders to 0. Keep Enable sending data OFF.
3. Press **Move HOME**. Wait until `[right] HOME complete -> READY` and `[left] HOME complete -> READY` appear in the log.
4. Turn **Enable sending data** ON. `FOLLOW start` appears in the log for both arms.
5. Move the right arm's forward/backward slider just a little and back, and check that the hand moves forward and backward. Then check the other axes, rotation, gripper, and the left arm in turn.
6. When you turn Enable sending data OFF, the arm stops at the position of the last command. To resume, start again from Move HOME.

For details, see [td-sender_en.md](td-sender_en.md).

## Choose the Home pose

Choose it with `home_preset` in `config/control.yml`. It is read only once at startup, so restart Dora after changing it.

| Value | Pose | Arm joint angles (J1 to J7, rad) |
|---|---|---|
| `forward` (default) | Elbow bent, arm forward | `[0, 0, 0, 1.570796, 0, 0, 0]` |
| `down` | Arm lowered | `[0, 0, 0, 0, 0, 0, 0]` |

- The pose shown right after startup and the Home destination are different. Even with `down`, the arm is shown in the forward pose right after startup and is lowered by Move HOME.
- During Home, the gripper opening at that moment is kept.
- The reference for Follow position and rotation is the hand pose of the chosen Home. With `down`, the arm is nearly fully extended, so targets that extend it further downward cannot be reached.

## Home and Follow behavior

- Home interpolates the hand pose over 4 seconds and completes when the measured hand enters the tolerance (position 2 cm, orientation 8°). If it does not get there in 15 seconds, it stops.
- The Home command is accepted only when it changes from 0 to 1. Holding it does not repeat it.
- Follow starts only when it changes from 0 to 1 after Home completes. Even if it is already ON before Home completes, it does not start automatically.
- Follow speed is limited by the hand position speed `follow_hand_speed_m_s` before IK and by the speed of each joint `follow_joint_speed_rad_s` after IK (separate values for `sim` and `hardware`). The joints are all slowed by the same ratio, so the hand path does not bend.

## Tests

```bash
uv run --extra sim pytest -q
```

Without the real robot, these check the bridge state transitions, speed limits, communication loss, start and stop of the real robot node (with a simulated driver), and the flow from OSC through IK to the MuJoCo model.

Tests that actually start Dora processes run only when the environment variable is set. Results are written to `out/`.

```powershell
$env:OPENARM_RUN_DATAFLOW_TESTS = '1'
uv run --extra sim pytest -q tests/test_dataflow.py
uv run --extra sim pytest -q -s tests/test_physics.py
$env:OPENARM_RUN_DATAFLOW_TESTS = '0'
```

On Linux and macOS, run them like `OPENARM_RUN_DATAFLOW_TESTS=1 uv run --extra sim pytest -q tests/test_dataflow.py`.
