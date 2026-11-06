# TouchDesigner sender project (sender)

`td/transform_sender.toe` receives the targets for the left and right arms, applies range limits and safety checks, and then sends them to the bridge over OSC. The targets come from the sliders in the project or from another TouchDesigner project.

It is made with TouchDesigner 2025.33230.

## 1. Switches on the screen

| Switch | Location | Function |
|---|---|---|
| Destination (MuJoCo / Real robot) | `/project1/container_target` | Chooses the OSC destination. MuJoCo is 127.0.0.1. Real robot is the address in the `Real robot` row of `/project1/table_addr` (rewrite it to the Ubuntu PC's address) |
| Input source (Slider / TouchIn) | `/project1/container_input/container_input_source` | Chooses whether the targets come from the sliders or from another project (section 2) |
| Right arm enable, Left arm enable | `/project1/container_real_robot_enable` | Enable for each arm of the real robot (not used with MuJoCo) |
| Move HOME | `/project1/container_robot_control` | Moves to Home when pressed |
| Enable sending data | `/project1/container_robot_control` | Main switch for Follow. Turning it OFF always stops the arms |
| Open logger, Clear logger | `/project1/container_logger_control` | Opens and clears the status table from the bridge (section 5) |
| Open final state monitor | `/project1/container_logger_control` | Shows the final left and right targets to be sent and the violation state (section 5) |


It sends every frame (240 fps). Even when Follow is OFF, the targets and each switch value keep being sent (Follow switches the bridge's behavior; it does not turn sending itself ON/OFF).

## 2. Target input

Input source switches both arms together (the Index of `/project1/container_input/switch_source` changes).

| Input source | Input |
|---|---|
| Slider | Sliders (`/project1/container_input/container_gui`) |
| TouchIn | Input from another project (Touch In CHOP `/project1/container_input/touchin_openarm`, Port 9310. For the format, see "Format sent to the sender" in [osc-protocol_en.md](osc-protocol_en.md)) |

### Sliders

Each arm (`container_transform_right_gui`, `container_transform_left_gui`) has the following sliders.

| Slider | Value sent |
|---|---|
| px, py, pz | Position from the Home hand (±0.2 m each) |
| rx, ry, rz | Rotation from the Home hand (±18° each; sent as a quaternion) |
| gripper | 0 to 1 is sent as 0 to −1 (the bridge clips it to −0.785 to 0) |

Position and rotation are then limited to the ranges in section 3.

The coordinates are `arm_origin` (x forward, y left, z up), relative to the Home hand position. Rotation is multiplied on the right of the Home hand rotation (rotation in the hand frame).

## 3. Range limits and safety checks

Location: `/project1/container_input/safety`

| Process | Content |
|---|---|
| Position range | Limited per arm and per direction (table below). When exceeded, the arm stops at that position and follows again when you come back |
| Rotation range | The rotation is converted to Euler angles (degrees) about the hand axes, limited to −25 to 25° on each axis, and converted back (Constant `rot_min`, `rot_max`) |
| Sudden change of value | A violation if the limited target changes by 0.10 m or more, or 45° or more, in 0.1 s. Checked only when the input `r_active` / `l_active` and its value delayed by 0.2 s are both 1 |
| Input loss | If the input `heartbeat` does not change for 0.1 s, both arms are in violation |

Position range (relative to Home, m. Constant `range_min`, `range_max`):

| Direction | Right arm (value location) | Left arm (value location) |
|---|---|---|
| Forward | 0.17 (`range_max` `r_px`) | 0.17 (`range_max` `l_px`) |
| Backward | 0.025 (`range_min` `r_px` = −0.025) | 0.025 (`range_min` `l_px` = −0.025) |
| Inward | 0.14 (`range_max` `r_py`) | 0.14 (`range_min` `l_py` = −0.14) |
| Outward | 0.17 (`range_min` `r_py` = −0.17) | 0.17 (`range_max` `l_py`) |
| Up | 0.17 (`range_max` `r_pz`) | 0.17 (`range_max` `l_pz`) |
| Down | 0.15 (`range_min` `r_pz` = −0.15) | 0.15 (`range_min` `l_pz` = −0.15) |

- y is positive to the left, so inward is + for the right arm and − for the left arm
- The bridge discards values whose position magnitude exceeds 0.30 m (`max_move_m` in `config/control.yml`) and stops that arm if no value arrives for 0.2 s. When deciding the ranges, keep the corner magnitude √(max(forward, backward)² + max(outward, inward)² + max(up, down)²) below 0.30 m (0.294 m with the values above). When widening one, narrow the others.
- Forward, inward, and down are matched to the distances the arm can reach while keeping the Home hand orientation (about 0.17, 0.14 to 0.15, and 0.15 m). Widening them further does not move the arm further.
- Backward is narrow because moving backward, inward, and up at the same time brings the arm close to the pillar.
- **Collisions with the pillar or the other arm are not prevented anywhere in the sender, the bridge, or IK.** Bringing both hands close together at the center can make the arms hit each other. The operator must watch and avoid this.
- Setting `/project1/container_input/safety/switch_safety_limit` to 1 switches to a process that limits the sum of the left and right inward amounts (`collision_safety`). When the sum exceeds the limit, both hands are pushed back outward.

## 4. Follow permission

Location: `/project1/follow_gate`

Follow sent to the bridge = Enable sending data × (1 − any of the following)

- A violation of an arm whose Enable is ON (latched; stays stopped until cleared)
- A current violation
- `active` of an arm whose Enable is ON is 0

The latch is cleared when Enable sending data is turned OFF, or when `active` of all arms whose Enable is ON becomes 0.

- Violations of arms whose Enable is OFF are not checked. With MuJoCo, both arms move without Enable, so to stop on violations, turn ON Enable for the arms you use.
- With the sliders, `active` is 1 and `heartbeat` is a value made inside the sender.
- The bridge starts following only when Follow changes from 0 to 1 after Home completes. When Follow becomes 0, Home is needed to resume.

## 5. Status display

Open logger opens the status table from the bridge (`/project1/container_logger`, received on UDP 7001). Clear logger clears the table.

- `status_now`: the current state of left and right (1 row per arm, rewritten on each notification)
- `status_log`: a history that adds 1 row only when the state or reason changes

Main states:

| Display | Meaning |
|---|---|
| `Stopped` | Stopped (Enable OFF) |
| `Starting` | Real robot start motion in progress |
| `Waiting for joint state` | Waiting for the joint state after starting |
| `Waiting for Home` | Accepts Move HOME |
| `Homing` | Moving to Home |
| `Home done (Follow allowed)` | Accepts Enable sending data OFF→ON |
| `Following` | Following |
| `Stop requested`, `Returning (normal stop)`, `Confirming disable` | During a normal stop |
| `FAULT (check robot)` | Fault. It has not been confirmed that the motors were cut off |

Because it uses the same UDP 7001, it cannot be used at the same time as `status_monitor.py` (a monitor that shows the status in a terminal).

The display opened with Open final state monitor (`/project1/container_final_state`) shows the left and right targets sent to the bridge after the range limits (`/project1/transform_left`, `transform_right`) and the violation state (`/project1/violation`).

## 6. Sending

Every frame, `/project1/execute3` sends the last sample (8 values) of `/project1/transform_left` and `transform_right`, Home, Follow, and the left and right Enable from `/project1/oscout_dat` (the addresses are in `/project1/table_osc_address`). For the format, see [osc-protocol_en.md](osc-protocol_en.md).
