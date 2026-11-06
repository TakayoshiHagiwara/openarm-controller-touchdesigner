# Communication formats

## 1. OSC received by the bridge (UDP 7000)

The port can be changed with `osc_port` in `config/control.yml`. The bridge listens on 127.0.0.1 with MuJoCo and on 0.0.0.0 with the real robot (`--listen` in `dataflow-*.yml`).

| Address | Value | Meaning |
|---|---|---|
| `/openarm/right` | `[px, py, pz, qw, qx, qy, qz, gripper]` (8 floats) | Right arm target |
| `/openarm/left` | Same as above | Left arm target |
| `/openarm/home` | `0` or `1` | 0→1 moves to Home |
| `/openarm/follow` | `0` or `1` | 0→1 after Home completes starts following; 1→0 stops it |
| `/openarm/enable_right` | `0` or `1` | Start (1) and normal stop (0) of the real robot's right arm. Not used with MuJoCo |
| `/openarm/enable_left` | `0` or `1` | Start and normal stop of the real robot's left arm |

Target values:

| Value | Content |
|---|---|
| `px, py, pz` | Position relative to the Home hand position (m). The coordinates are the model's `arm_origin` (x forward, y left, z up) |
| `qw, qx, qy, qz` | Rotation relative to the Home hand rotation (unit quaternion). Multiplied on the right of the Home rotation (rotation in the hand frame) |
| `gripper` | −π/4 (about −0.785) to 0. Sent in the same direction for both arms; the bridge flips the sign for the left |

Values the bridge does not accept (`[OSC] Rejected ...` appears in the log and the value is not used):

- Not 8 values, or values that are not finite
- Values whose position magnitude $\sqrt{\mathrm{px}^{2} + \mathrm{py}^{2} + \mathrm{pz}^{2}}$ exceeds `max_move_m` (0.30 m)
- Values whose quaternion length is 0 (otherwise the length is normalized to 1)
- Switch values other than 0 or 1

Values that are not accepted are treated the same as "not arrived". If no new target arrives for 0.2 s (`timeout_s`), that arm is stopped.

- Keep sending the targets and each switch (the sender sends at 240 fps). The bridge uses the latest value at 100 Hz.
- Home and Follow are accepted only when they change from 0 to 1. Holding them does not repeat them.

## 2. Format sent to the sender (Touch In CHOP, Port 9310)

To send targets to `td/transform_sender.toe` from another TouchDesigner project, send the following 19 channels with a Touch Out CHOP and set the sender's Input source to TouchIn ([td-sender_en.md](td-sender_en.md)).

| Channel | Content |
|---|---|
| `r_px r_py r_pz` | Right arm position (same as `px, py, pz` in section 1) |
| `r_qw r_qx r_qy r_qz` | Right arm rotation (same as `qw, qx, qy, qz` in section 1) |
| `r_gripper` | Right gripper (−0.785 to 0) |
| `l_px` to `l_gripper` | The same 8 values for the left arm |
| `r_active l_active` | The arm may move only while this is 1. Setting it to 0 makes the sender cut Follow (Home is needed to resume) |
| `heartbeat` | A value that keeps changing while the sender side is running (for example, the Ramp of an LFO CHOP). If it does not change for 0.1 s, the sender stops both arms |

- For an arm that you do not move, send position 0 and rotation (1, 0, 0, 0).
- When you reset the origin (the hand position corresponding to the Home hand) and the target returns to 0, it does not stop if this happens right after changing `r_active` / `l_active` from 0 to 1 (the sender does not check sudden changes for 0.2 s after active becomes 1).

## 3. Status notification (UDP 7001)

The bridge sends the state of each arm to `status.host` and `status.port` in `config/control.yml` immediately when the state or reason changes, and every 0.5 seconds even when it does not change. This notification is for display; the robot is never operated from it.

The addresses are `/openarm/status/right` and `/openarm/status/left`. The values are the following 11.

| Index | Content |
|---|---|
| 0 | Session string for each startup |
| 1 | Sequence number |
| 2 | State (`STOPPED`, `STARTING`, `WAIT_STATE`, `WAIT_HOME`, `HOME`, `READY`, `FOLLOW`, `STOP_REQUESTED`, `STOPPING`, `DISABLING`, `FAULT`) |
| 3 | Reason string |
| 4, 5 | Difference between the measured hand and Home in position (m) and orientation (degrees). −1 when unknown |
| 6, 7 | Whether Move HOME is accepted, whether Follow is accepted (0 or 1) |
| 8 | Motor disable response (1 = confirmed, 0 = not confirmed, −1 = none yet) |
| 9, 10 | Whether gravity compensation is on (0 or 1), correction scale |

- The receiver should discard values from old sessions or with old sequence numbers
- If no new notification arrives for 2 seconds, treat the state as unknown (do not trust the display)
- "Disable response: confirmed" is a software confirmation of the response; it does not guarantee that the arm has mechanically stopped and is supported
