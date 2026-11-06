# Hardware setup

Send OSC over the LAN from the PC running TouchDesigner (the "TD PC" below) to the Ubuntu PC, and move OpenArm v2 Pedestal with the bridge, IK, and real robot nodes on the Ubuntu PC.

**In an emergency, use the real robot's emergency stop.** A software stop is not a substitute for the emergency stop. When running it for the first time, be ready to support the arm, and make sure there are no people or objects in the path of the start and return motions.

## 1. Overview of the motions

| Operation | Real robot motion |
|---|---|
| Enable ON | Starts the driver and moves to the start pose over 10 seconds |
| Move HOME | Moves to the Home pose over 4 seconds (interpolating the hand pose) |
| Enable sending data ON | After Home completes, follows the target (Follow) |
| Enable sending data OFF | Stops at the position of the last command. Move HOME is needed to resume |
| Enable OFF | Normal stop: returns to the end pose over 10 seconds, closes the gripper, and disables the motors |

A normal stop includes the return motion and closing the gripper (it does not cut the motors immediately).

## 2. Ubuntu PC preparation

### Create the environment

```bash
git clone <this repository>
cd openarm-controller-touchdesigner
uv sync --extra sim --extra hardware
```

The `hardware` dependencies (the real robot driver) are installed only on Linux.

### CAN

Connect the right arm to `can0` and the left arm to `can1` (this can be changed in `config/openarm_pedestal.yaml`).

```bash
ip -details link show can0
ip -details link show can1
```

Check that both are `UP` with `bitrate 1000000` and `dbitrate 5000000`. If not (for example after restarting the PC), set them with the OpenArm CAN CLI tool `openarm-can-cli`. For how to install and use `openarm-can-cli`, see the [official OpenArm documentation](https://docs.openarm.dev/api-reference/can/cli/).

```bash
openarm-can-cli -i can0 can_configure -b 1000000 -d 5000000 --fd --sp 0.75 --dsp 0.75 --dsjw 2 --rm 0
openarm-can-cli -i can1 can_configure -b 1000000 -d 5000000 --fd --sp 0.75 --dsp 0.75 --dsjw 2 --rm 0
```

Check that all motors respond (read only).

```bash
openarm-can-cli -i can0 discover
openarm-can-cli -i can1 discover
```

It is ready when **8** motors (ID 1 to 8) are found on both sides. If fewer than 8 are found, do not proceed; check the cable connections and the motor power.

- `change_id`, `change_baud`, `write_param`, and `set_zero` rewrite settings inside the motors

## 3. Network settings

Connect the TD PC and the Ubuntu PC to the same LAN, and decide the IPv4 address of each.

| Where to set | Value |
|---|---|
| Destination in `td/transform_sender.toe` (the `Real robot` row of `/project1/table_addr`) | Address of the Ubuntu PC |
| `status.host` in `config/control.yml` on the Ubuntu PC | Address of the TD PC (destination of status notifications) |

- The bridge receives on UDP 7000 and sends status to UDP 7001. If a firewall blocks them, allow receiving on UDP 7000 on the Ubuntu PC and on UDP 7001 on the TD PC
- Example of allowing UDP 7001 on Windows (PowerShell as administrator; replace `<Ubuntu PC address>`):

```powershell
New-NetFirewallRule -DisplayName "OpenArm status UDP 7001" -Direction Inbound -Action Allow -Protocol UDP -LocalPort 7001 -RemoteAddress <Ubuntu PC address>
```

## 4. Settings for the real robot

Set the following in `config/control.yml`. It is read at startup, so restart Dora after changing it.

```yaml
hardware_gravity_compensation:
  enabled: true
  scale: 1.0
  max_offset_rad: 0.15
home_diagnostics: true
```

- Gravity compensation adds `scale` × model gravity torque ÷ real robot Kp to the joint targets, limited to ±0.15 rad per joint. It is not added to the gripper. `scale` is 0 to 1.
- The real robot can sag more under gravity than the model. Without compensation, Home may not complete because it does not enter the Home tolerance (position 2 cm, orientation 8°).
- With `home_diagnostics: true`, a detailed JSON of the Home error is written to the log.

Main values (`config/control.yml`, `config/openarm_pedestal.yaml`):

| Item | Value |
|---|---|
| Kp | 70, 70, 70, 60, 10, 10, 10, 10 |
| Kd | 2.75, 2.5, 2, 2, 0.7, 0.6, 0.5, 0.2 |
| Home joint speed limit | 0.5 rad/s |
| Follow joint speed limit | 0.75 rad/s (all joints slowed by the same ratio) |
| Follow hand position speed limit | 0.2 m/s (limited before IK) |
| Gripper speed limit | 2.0 rad/s (grip force limit is 1 Nm in the driver settings) |
| Home interpolation / timeout | 4 s / 15 s |
| Home completion condition | Measured hand within position 2 cm and orientation 8° |
| OSC and joint observation timeout | 0.2 s |
| Start and normal stop motion | 10 s, 50 Hz |

These are the values used for operation checks and do not guarantee safety.

## 5. Start

On the Ubuntu PC, run this from the repository root.

```bash
bash run-hardware.sh
```

Check that the following lines appear in the log.

```text
[GRAVITY] hardware enabled=True scale=1 ...
[CONTROL] Home=0.5 Follow=0.75 rad/s, hand=0.2 m/s; ...
[RECORD] off
```

To record during Follow, start with `bash run-hardware.sh --record` (section 7).

## 6. Operation

On the TD PC, open `td/transform_sender.toe` and set the destination to **Real robot**. Keep the left and right Enable, Move HOME, and Enable sending data all OFF. Check the status in the sender's status table ("Status display" in [td-sender_en.md](td-sender_en.md)).

| Step | Operation | What to wait for and check |
|---|---|---|
| 1 | Turn ON Enable for the arm you use | `Starting` → `Waiting for Home` |
| 2 | Move HOME | `Homing` → `Home done (Follow allowed)` |
| 3 | Turn Enable sending data ON | `Following`. Move the sliders little by little at first |
| 4 | Turn Enable sending data OFF | It stops. To continue, go back to 2 |
| 5 | Turn Enable OFF | `Returning (normal stop)` → `Confirming disable` → `Stopped` (disable response confirmed) |
| 6 | `Ctrl+C` in the terminal on the Ubuntu PC | Dora exits |

- Follow before Home completes, or Enable during a stop, is rejected with a reason. It does not start on its own later. If it is rejected, turn that switch OFF→ON again.
- In the following cases, that arm is stopped automatically (normal stop). It does not resume automatically even if communication returns. To resume, turn Enable OFF→ON and start from Home.
  - No value arrives from TouchDesigner for 0.2 s
  - During Home or Follow, no joint observation arrives for 0.2 s
  - Home does not complete in 15 seconds
  - The difference between the joint command and the measurement when Home starts exceeds 0.1 rad (`startup tracking error exceeds driver alignment threshold`)
  - The driver rejected a command for safety
- If sending the disable fails, or the disable response cannot be confirmed, it becomes `FAULT (check robot)`. The motors are not treated as cut off. Check the real robot.

## 7. Recording

When started with `--record`, the bridge records every 10 ms (every control cycle), only during Follow. Without it, nothing is recorded. The destination is `record_dir` in `config/control.yml` (relative paths are from the repository root).

- A `<record_dir>/<startup date and time>/` folder is created for each startup.
- Each time Follow starts, the segment number increases by 1, and `follow_<number>_<right|left>.csv` is written for each arm. Nothing is written outside Follow.
- `events.csv`: the start and end of Follow (`wall_ns, t_s, side, event, segment, reason`). The reason for the end (Follow OFF, OSC timeout, etc.) is kept.
- Writes are buffered and written together about every second and at the end of a segment. If writing fails, the bridge stops with an error (the arms are stopped by the exit handling).

Columns of `follow_*.csv` (1 row = 1 control cycle):

| Column | Content |
|---|---|
| `t_s` | Seconds since Follow started (`time.perf_counter()`) |
| `wall_ns` | Time on that PC (ns). Used to align with records from other PCs |
| `osc_age_s`, `osc_px` to `osc_gripper` | The received OSC target (8 values) and seconds since it was received |
| `target_px` to `target_qz`, `target_gripper` | The target sent to IK after limiting the hand speed (`arm_origin` coordinates, m and wxyz) and the gripper |
| `ik_age_s`, `ik_q1` to `ik_gripper` | The latest IK solution and its age |
| `command_age_s`, `command_q1` to `command_gripper` | The command sent after gravity compensation and speed limits, and its age |
| `gravity_q1` to `gravity_gripper` | The gravity compensation added to the command |
| `state_age_s`, `observation_ns`, `state_q1` to `state_gripper` | The measured joints, their age, and the observation time on the real robot (ns) |
| `driver_command_q1` to `driver_command_gripper` | The last command accepted by the driver (real robot only; empty with MuJoCo) |

- Values that have not arrived yet are empty.
- The measured hand pose is not recorded. In analysis, compute it from `state_*` with FK of the official model.
- `*_age_s` is measured with the clock used by the control (`time.monotonic()`). On Windows it has 15.6 ms steps.

## 8. Troubleshooting

| Symptom | What to check |
|---|---|
| Fewer than 8 found with `discover` | Cable connections, motor power |
| The status table is not updated (`NO STATUS`) | Whether Dora is running, `status.host` in `config/control.yml`, the firewall on the TD PC (UDP 7001), whether `status_monitor.py` is using the same 7001 |
| `Home timed out` | The Home error in the status table and the Home JSON in the log. The gravity compensation settings |
| `startup tracking error exceeds driver alignment threshold` | The joint offset when Home started is large. Turn Enable OFF once and try again |
| Enable or Move HOME is rejected | Read the reason in the status table and turn that switch OFF→ON again |
| Nothing moves after turning Enable sending data ON | Whether you turned it ON after `Home done (Follow allowed)`. The sender's Input source and Enable state ([td-sender_en.md](td-sender_en.md)) |
| `[OSC] Rejected ...` keeps appearing in the log | The format of the values you are sending ([osc-protocol_en.md](osc-protocol_en.md)). Whether the position magnitude exceeds 0.30 m |
