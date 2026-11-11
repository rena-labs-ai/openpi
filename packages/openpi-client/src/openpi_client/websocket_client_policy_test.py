import contextlib
import threading

import pytest
import websockets.exceptions
from websockets.sync.server import ServerConnection, serve

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
        ({"api_key": "k", "headers": {"authorization": "Bearer t"}}, "Bearer t"),
    ],
)
def test_handshake_headers(kwargs, sent):
    with _server() as (uri, seen):
        policy = websocket_client_policy.WebsocketClientPolicy(uri, **kwargs)
        assert policy.get_server_metadata() == {"ok": True}
        policy.close()
    assert seen == [sent]


needs_respond = pytest.mark.skipif(
    not hasattr(ServerConnection, "respond"), reason="faking an HTTP status needs websockets>=13"
)


@needs_respond
def test_a_gateway_502_is_retried_until_the_server_answers(monkeypatch):
    monkeypatch.setattr(websocket_client_policy.time, "sleep", lambda s: None)
    answers = []

    def process_request(connection, request):
        answers.append(len(answers))
        if len(answers) == 1:
            return connection.respond(502, "starting\n")
        return None

    def handler(ws):
        ws.send(msgpack_numpy.packb({"ok": True}))
        ws.recv()

    with serve(handler, "127.0.0.1", 0, process_request=process_request) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        uri = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        policy = websocket_client_policy.WebsocketClientPolicy(uri)
        assert policy.get_server_metadata() == {"ok": True}
        policy.close()
        server.shutdown()
    assert answers == [0, 1]


@needs_respond
def test_a_refusal_is_not_retried():
    with serve(lambda ws: None, "127.0.0.1", 0, process_request=lambda c, r: c.respond(404, "no\n")) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        uri = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        with pytest.raises(websockets.exceptions.InvalidStatus) as refused:
            websocket_client_policy.WebsocketClientPolicy(uri)
        assert refused.value.response.status_code == 404
        server.shutdown()


def test_close_ends_the_connection():
    closed = threading.Event()

    def handler(ws):
        ws.send(msgpack_numpy.packb({"ok": True}))
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            ws.recv()
        closed.set()

    with serve(handler, "127.0.0.1", 0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        uri = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        websocket_client_policy.WebsocketClientPolicy(uri).close()
        assert closed.wait(5)
        server.shutdown()
