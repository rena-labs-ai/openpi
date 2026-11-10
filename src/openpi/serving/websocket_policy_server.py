import asyncio
import http
import json
import logging
import time
import traceback

import cv2
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves one or more policies using the websocket protocol. See websocket_client_policy.py for a client
    implementation.

    As rena-training's serving does, so rena_vla_client can choose among them: a connection picks its policy by request
    path, "/m/<id>" for that id (ids may contain slashes, "<exp>/<step>") and "/" for the default, and an unknown id is
    refused with 404 before the upgrade; GET /models lists the ids, /healthz reports them, and each connection's
    metadata frame names its model. Image values sent as JPEG bytes (BGR, as cv2 encodes them) are decoded to RGB.
    """

    def __init__(
        self,
        policies: dict[str, _base_policy.BasePolicy],
        default: str,
        host: str = "0.0.0.0",
        port: int | None = None,
    ) -> None:
        if default not in policies:
            raise ValueError(f"default {default!r} not in policies")
        self._policies = policies
        self._default = default
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
        ) as server:
            await server.serve_forever()

    def _model_id(self, path: str) -> str | None:
        if path in ("", "/"):
            return self._default
        if path.startswith("/m/") and path[3:] in self._policies:
            return path[3:]
        return None

    def _process_request(
        self, connection: _server.ServerConnection, request: _server.Request
    ) -> _server.Response | None:
        catalog = {"models": [{"id": mid, "label": mid} for mid in self._policies], "default": self._default}
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, json.dumps({"status": "ok", **catalog}) + "\n")
        if request.path == "/models":
            return connection.respond(http.HTTPStatus.OK, json.dumps(catalog) + "\n")
        if self._model_id(request.path) is None:
            return connection.respond(http.HTTPStatus.NOT_FOUND, f"unknown model path: {request.path}\n")
        # Continue with the normal request handling.
        return None

    async def _handler(self, websocket: _server.ServerConnection):
        model_id = self._model_id(websocket.request.path)
        policy = self._policies[model_id]
        logger.info(f"Connection from {websocket.remote_address} opened, model {model_id}")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack({**policy.metadata, "model": model_id, "default": self._default}))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = _decode_jpeg_images(msgpack_numpy.unpackb(await websocket.recv()))

                infer_time = time.monotonic()
                action = policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _decode_jpeg_images(obs: dict) -> dict:
    """The observation with every bytes value, at any depth, decoded as a JPEG to an RGB array."""
    for key, val in obs.items():
        if isinstance(val, dict):
            _decode_jpeg_images(val)
        elif isinstance(val, bytes | bytearray):
            img = cv2.imdecode(np.frombuffer(val, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"{key!r} is bytes but not a JPEG")
            obs[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return obs
