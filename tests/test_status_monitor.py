"""Validate Windows status reception, ordering and absence of stale READY."""

import threading
import time

from pythonosc import dispatcher, osc_server, udp_client
import pytest

from status_monitor import StatusMonitor


def packet(session="1800000000000000000", sequence=1, phase="READY"):
    """Return a valid protocol message with visible correction and disable state."""
    return [session, sequence, phase, "Home complete; toggle Follow OFF then ON", 0.015, 5.5, 1, 1, 0, 1, 0.5]


def test_ordered_status_and_stale_ready():
    """Reject delayed old packets and replace READY with unknown after two seconds."""
    now = [100.0]
    monitor = StatusMonitor(lambda: now[0])
    monitor.receive("/openarm/status/left", *packet(sequence=2))
    monitor.receive("/openarm/status/left", *packet(sequence=1, phase="FOLLOW"))
    assert monitor.arms["left"].phase == "READY"
    monitor.receive("/openarm/status/left", *packet(session="1800000000000000001", phase="STARTING"))
    monitor.receive("/openarm/status/left", *packet(sequence=999, phase="READY"))
    assert monitor.arms["left"].phase == "STARTING"
    now[0] += 2.01
    assert "Home完了" not in "\n".join(monitor.lines())
    assert "状態不明" in "\n".join(monitor.lines())


@pytest.mark.parametrize("index,value", [(0, "invalid"), (1, -1), (2, []), (2, "unknown"),
                                          (4, float("nan")), (6, 9), (8, 2), (10, 1.2)])
def test_malformed_status_is_ignored(index, value):
    """Keep invalid UDP data from becoming an operator READY indication."""
    monitor = StatusMonitor()
    values = packet()
    values[index] = value
    monitor.receive("/openarm/status/left", *values)
    assert not monitor.arms


def test_loopback_osc_status_receiver():
    """Receive a real OSC datagram on loopback without any robot command path."""
    monitor = StatusMonitor()
    routes = dispatcher.Dispatcher()
    routes.set_default_handler(monitor.receive)
    server = osc_server.OSCUDPServer(("127.0.0.1", 0), routes)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    client = udp_client.SimpleUDPClient("127.0.0.1", server.server_address[1])
    thread.start()
    try:
        deadline = time.monotonic()+2
        while "left" not in monitor.arms and time.monotonic() < deadline:
            client.send_message("/openarm/status/left", packet())
            time.sleep(0.01)
        assert monitor.arms["left"].phase == "READY"
        assert monitor.arms["left"].scale == pytest.approx(0.5)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_phase_and_disable_ack_are_separate():
    """Show unconfirmed disable instead of equating STOPPED with motor-off proof."""
    monitor = StatusMonitor()
    values = packet(phase="STOPPED")
    values[8] = 0
    monitor.receive("/openarm/status/left", *values)
    assert "無効化応答: 未確認" in "\n".join(monitor.lines())
