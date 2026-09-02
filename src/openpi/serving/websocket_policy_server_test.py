import asyncio
import json
import time
from typing import ClassVar

from openpi_client import msgpack_numpy
import pytest
import websockets

from openpi.serving.websocket_policy_server import QC_HOLD_SECONDS
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.serving.websocket_policy_server import resolve_model_path

IDS = {"exp_v14/70000", "exp_v13/70000", "exp_v12/70000"}
DEFAULT = "exp_v14/70000"


class _StubPolicy:
    metadata: ClassVar[dict] = {"k": "v"}

    def infer(self, obs):  # pragma: no cover - never called in these tests
        raise NotImplementedError


def _multi():
    policies = {mid: _StubPolicy() for mid in ("exp_v14/70000", "exp_v13/70000")}
    return WebsocketPolicyServer(policies=policies, default="exp_v14/70000", labels={"exp_v14/70000": "v14"})


def test_bare_path_serves_default():
    assert resolve_model_path("/", IDS, DEFAULT) == DEFAULT
    assert resolve_model_path("", IDS, DEFAULT) == DEFAULT


def test_model_path_selects_id_with_slash():
    assert resolve_model_path("/m/exp_v13/70000", IDS, DEFAULT) == "exp_v13/70000"


def test_unknown_id_and_other_paths_reject():
    assert resolve_model_path("/m/nope/1", IDS, DEFAULT) is None
    assert resolve_model_path("/exp_v13/70000", IDS, DEFAULT) is None
    assert resolve_model_path("/models", IDS, DEFAULT) is None


def test_constructor_requires_exactly_one_form():
    with pytest.raises(ValueError, match="exactly one"):
        WebsocketPolicyServer()
    with pytest.raises(ValueError, match="exactly one"):
        WebsocketPolicyServer(policy=_StubPolicy(), policies={"a": _StubPolicy()}, default="a")


def test_constructor_requires_default_in_policies():
    with pytest.raises(ValueError, match="not in policies"):
        WebsocketPolicyServer(policies={"a": _StubPolicy()}, default="b")


def test_catalog_lists_ids_with_labels_and_default():
    assert _multi()._catalog() == {  # noqa: SLF001
        "models": [
            {"id": "exp_v14/70000", "label": "v14"},
            {"id": "exp_v13/70000", "label": "exp_v13/70000"},
        ],
        "default": "exp_v14/70000",
    }


def test_connection_policy_routes_and_rejects():
    server = _multi()
    mid, policy = server._connection_policy("/m/exp_v13/70000")  # noqa: SLF001
    assert mid == "exp_v13/70000"
    assert policy is server._policies[mid]  # noqa: SLF001
    assert server._connection_policy("/") == ("exp_v14/70000", server._policies["exp_v14/70000"])  # noqa: SLF001
    assert server._connection_policy("/m/gone/1") == (None, None)  # noqa: SLF001


def test_single_policy_accepts_any_path():
    policy = _StubPolicy()
    server = WebsocketPolicyServer(policy=policy)
    assert server._connection_policy("/whatever") == (None, policy)  # noqa: SLF001


# ── QC probe hold ────────────────────────────────────────────────────────────


class _CountingPolicy:
    def __init__(self):
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        return {"actions": [0.0]}


class _Socket:
    """Feeds the handler a fixed list of requests, then closes the connection."""

    remote_address = ("test", 0)
    request = type("_Req", (), {"path": "/"})()

    def __init__(self, requests):
        self._requests = list(requests)
        self.sent = []

    async def recv(self):
        if not self._requests:
            raise websockets.ConnectionClosed(None, None)
        return msgpack_numpy.Packer().pack(self._requests.pop(0))

    async def send(self, data):
        self.sent.append(data)


class _Connection:
    def respond(self, status, body):
        return status, body


class _Request:
    def __init__(self, path):
        self.path = path


def _clocked(*, last_infer_at=None):
    policy = _CountingPolicy()
    server = WebsocketPolicyServer(policy=policy, port=0)
    server._last_infer_at = last_infer_at  # noqa: SLF001
    return server, policy


def _replies(server, requests):
    socket = _Socket(requests)
    asyncio.run(server._handler(socket))  # noqa: SLF001
    # The handler opens with a metadata frame; the rest are replies.
    return [msgpack_numpy.unpackb(frame) for frame in socket.sent[1:]]


def test_probe_inside_the_hold_window_is_refused_without_inference():
    server, policy = _clocked(last_infer_at=time.time())

    (reply,) = _replies(server, [{"prompt": "p", "_qc_probe": True}])

    assert reply["_qc_refused"] is True
    assert 0 < reply["retry_after"] <= QC_HOLD_SECONDS
    assert policy.calls == 0


def test_probe_is_served_once_the_hold_expires():
    when = time.time() - QC_HOLD_SECONDS - 1
    server, policy = _clocked(last_infer_at=when)

    (reply,) = _replies(server, [{"prompt": "p", "_qc_probe": True}])

    assert "_qc_refused" not in reply
    assert policy.calls == 1
    assert server._last_infer_at == when  # noqa: SLF001 - a probe is not robot activity


def test_a_robot_request_is_never_refused():
    server, policy = _clocked(last_infer_at=time.time())

    (reply,) = _replies(server, [{"prompt": "p"}])

    assert "_qc_refused" not in reply
    assert policy.calls == 1


def test_healthz_advertises_the_hold():
    """The gate reads this to confirm it is talking to a server that enforces."""
    server, _ = _clocked()

    _, body = server._process_request(_Connection(), _Request("/healthz"))  # noqa: SLF001

    assert json.loads(body)["qc_hold_seconds"] == QC_HOLD_SECONDS


def test_the_hold_is_server_wide_across_a_model_set():
    """One GPU serves the set, so a robot on any model holds probes off all."""
    server = _multi()
    server._last_infer_at = time.time()  # noqa: SLF001

    assert server.probe_hold_left() > 0
