"""Display OpenArm status on Windows; this program never sends robot commands."""

import argparse
from dataclasses import dataclass
import math
import sys
import time

from pythonosc import dispatcher, osc_server

LABELS = {"STOPPED": "停止処理済み", "STARTING": "開始動作中", "WAIT_HOME": "Home待ち",
          "WAIT_STATE": "新しい関節状態の受信待ち",
          "STOP_REQUESTED": "通常停止を要求済み・実機ノードの受付待ち",
          "HOME": "Home動作中", "READY": "Home完了・Follow操作可", "FOLLOW": "追従中",
          "STOPPING": "通常停止の帰還中", "DISABLING": "無効化応答の確認中", "FAULT": "異常・要確認"}
REASONS = {"Home timed out": "Home未達で通常停止", "OSC timeout": "TDからの操作データが途絶",
           "joint-state timeout": "関節状態の受信が途絶", "startup tracking error": "開始姿勢の実測誤差が整列許容値を超過",
           "Follow rejected": "Follow拒否：Home完了後にOFF→ONしてください",
           "Home rejected": "Home拒否：開始完了後にOFF→ONしてください",
           "Home required": "Home操作をしてください", "Home complete": "Home完了：FollowをOFF→ONしてください",
           "Waiting for fresh joint state": "関節状態を受信してからHome操作を受け付けます",
           "Enable rejected": "Enable拒否：停止処理の完了を待ってください", "Enable OFF": "操作者がEnableをOFF",
           "Follow OFF": "追従解除：再開にはHomeが必要", "preflight": "有効化前の確認で中止（未応答または既に有効なモーター）",
           "motor enable acknowledgement": "モーター有効化の応答が未確認", "disable unconfirmed": "モーター無効化の応答が未確認",
           "disable failed": "モーター無効化の送信で異常", "driver safety": "ドライバの安全制限で指令を中止",
           "CAN carrier lost": "CAN接続の状態を喪失", "motor feedback": "モーターの状態異常・応答不明",
           "state read error": "関節状態の取得で異常", "Joint ": "関節指令がドライバの安全制限を超過",
           "driver heartbeat timeout": "実機ノードの応答途絶・モーターOFF未確認",
           "dataflow shutdown": "プログラム終了に伴う通常停止", "motor state not yet verified": "モーター状態はまだ未確認"}


@dataclass
class Status:
    """Keep one arm's latest ordered status and local reception time."""

    session: int
    sequence: int
    phase: str
    reason: str
    position_error: float
    angle_error: float
    can_home: int
    can_follow: int
    disabled: int
    gravity: int
    scale: float
    received_at: float


class StatusMonitor:
    """Validate receive-only status messages and hide stale READY indications."""

    def __init__(self, clock=time.monotonic):
        """Initialize without opening any socket or connecting to the robot."""
        self.clock, self.arms = clock, {}

    def receive(self, address, *values):
        """Accept the documented eleven-field OSC status payload."""
        side = address.removeprefix("/openarm/status/")
        if side not in ("left", "right") or len(values) != 11: return

        session, sequence, phase, reason, pos, angle, home, follow, disabled, gravity, scale = values
        if not isinstance(session, str) or not session.isdecimal() or len(session) > 30: return
        if not isinstance(sequence, int) or sequence < 0 or not isinstance(phase, str) or phase not in LABELS or not isinstance(reason, str): return
        if any(value not in (0, 1) for value in (home, follow, gravity)) or disabled not in (-1, 0, 1): return
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (pos, angle, scale)): return
        if not 0 <= scale <= 1: return
        
        previous = self.arms.get(side)
        if previous and (int(session) < previous.session or (int(session) == previous.session and sequence <= previous.sequence)): return
        reason = "".join(char if ord(char) >= 32 and char != "\x7f" else " " for char in reason)[:500]
        self.arms[side] = Status(int(session), sequence, phase, reason, pos, angle, home, follow, disabled, gravity, scale, self.clock())

    def lines(self):
        """Render current state; no packet for two seconds means state is unknown."""
        result = ["OpenArm 状態モニター（受信専用）", ""]
        for side, name in (("left", "左腕"), ("right", "右腕")):
            state = self.arms.get(side)
            if state is None or self.clock()-state.received_at > 2.0:
                result.extend([f"{name}: 通知なし・状態不明（Home/Followの可否も未確認）", "", "", "", ""])
                continue
            
            reason = next((translated for prefix, translated in REASONS.items() if state.reason.startswith(prefix)), state.reason)
            confirmation = "確認済み" if state.disabled == 1 else "未確認"
            error = "未取得" if state.position_error < 0 else f"{state.position_error*1000:.1f} mm / {state.angle_error:.1f} deg"
            result.extend([f"{name}: {LABELS[state.phase]} | Home可={state.can_home} Follow可={state.can_follow}",
                           f"  理由: {reason or '―'}", f"  詳細: {state.reason if state.reason != reason else '―'}", f"  Home誤差: {error}",
                           f"  補償: {'ON' if state.gravity else 'OFF'} / {state.scale:g} | 無効化応答: {confirmation}"])
        return result


def main():
    """Listen on Windows UDP 7001 and keep a compact, non-scrolling display."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7001)
    args = parser.parse_args()

    monitor = StatusMonitor()
    routes = dispatcher.Dispatcher()
    routes.set_default_handler(monitor.receive)
    server = osc_server.OSCUDPServer((args.listen, args.port), routes)
    server.timeout = 0.1
    terminal = sys.stdout.isatty()

    if terminal and sys.platform == "win32":
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel.GetConsoleMode(handle, ctypes.byref(mode)): kernel.SetConsoleMode(handle, mode.value | 4)
    
    if terminal: print("\x1b[2J", end="")
    displayed, last_render = None, -math.inf
    try:
        while True:
            server.handle_request()
            if time.monotonic()-last_render < 0.25: continue
            lines = monitor.lines()
            if terminal: print("\x1b[H"+"\n".join(line+"\x1b[K" for line in lines), flush=True)
            elif lines != displayed: print("\n".join(lines), flush=True)
            displayed, last_render = lines, time.monotonic()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        if terminal: print()


if __name__ == "__main__": main()
