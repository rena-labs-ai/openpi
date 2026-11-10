import json
import socket
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import websocket_client_policy as _wcp
import pytest

from openpi.serving import websocket_policy_server


class _EchoPolicy(_base_policy.BasePolicy):
    """Answers with its name and the first pixel of the image it was sent."""

    def __init__(self, name: str):
        self.name = name
        self.metadata = {"name": name}

    def infer(self, obs: dict) -> dict:
        return {"name": self.name, "pixel": obs["observation.images.image_0"][0, 0], "prompt": obs["prompt"]}


@pytest.fixture(scope="module")
def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = websocket_policy_server.WebsocketPolicyServer(
        {"a/1": _EchoPolicy("a"), "b/2": _EchoPolicy("b")}, "a/1", host="127.0.0.1", port=port
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
            break
        except OSError:
            time.sleep(0.05)
    return port


def test_models_lists_the_roster_and_its_default(port):
    doc = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/models").read())
    assert doc == {"models": [{"id": "a/1", "label": "a/1"}, {"id": "b/2", "label": "b/2"}], "default": "a/1"}


def test_model_path_binds_the_connection_and_jpeg_arrives_as_rgb(port):
    client = _wcp.WebsocketClientPolicy(host=f"ws://127.0.0.1:{port}/m/b/2", port=None)
    assert client.get_server_metadata() == {"name": "b", "model": "b/2", "default": "a/1"}

    red_rgb = np.zeros((48, 64, 3), np.uint8)
    red_rgb[..., 0] = 255
    _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(red_rgb, cv2.COLOR_RGB2BGR))
    out = client.infer({"observation.images.image_0": jpeg.tobytes(), "prompt": "pick toy fruit"})
    assert out["name"] == "b"
    assert out["prompt"] == "pick toy fruit"
    assert np.abs(out["pixel"].astype(int) - [255, 0, 0]).max() <= 2


def test_bare_path_gets_the_default(port):
    client = _wcp.WebsocketClientPolicy(host=f"ws://127.0.0.1:{port}", port=None)
    assert client.get_server_metadata()["model"] == "a/1"


def test_unknown_model_is_refused_with_404(port):
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/m/c/3")
    assert err.value.code == 404
