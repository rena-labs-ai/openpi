import contextlib
import threading

import pytest
from websockets.sync.server import serve

from openpi_client import msgpack_numpy, websocket_client_policy


@contextlib.contextmanager
def _server():
    seen = []

    def handler(ws):
        seen.append(ws.request.headers.get("Authorization"))
        ws.send(msgpack_numpy.packb({"ok": True}))
        ws.recv()

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}", seen
        server.shutdown()


@pytest.mark.parametrize(
    ("kwargs", "sent"),
    [
        ({}, None),
        ({"api_key": "k"}, "Api-Key k"),
        ({"headers": {"Authorization": "Bearer t"}}, "Bearer t"),
        ({"api_key": "k", "headers": {"Authorization": "Bearer t"}}, "Bearer t"),
    ],
)
def test_handshake_headers(kwargs, sent):
    with _server() as (uri, seen):
        policy = websocket_client_policy.WebsocketClientPolicy(uri, **kwargs)
        assert policy.get_server_metadata() == {"ok": True}
        policy._ws.close()
    assert seen == [sent]
