import asyncio
import http
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
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
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
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        req_count = 0
        while True:
            try:
                t0 = time.monotonic()
                raw = await websocket.recv()
                t1 = time.monotonic()

                obs = msgpack_numpy.unpackb(raw)
                t2 = time.monotonic()

                obs = _decode_jpeg_images(obs)
                t2b = time.monotonic()

                action = self._policy.infer(obs)
                t3 = time.monotonic()

                # Extract per-stage policy timing before packing
                policy_timing = action.pop("policy_timing", {})

                packed = packer.pack(action)
                t4 = time.monotonic()

                action["server_timing"] = {
                    "recv_ms": (t1 - t0) * 1000,
                    "unpack_ms": (t2 - t1) * 1000,
                    "jpeg_decode_ms": (t2b - t2) * 1000,
                    "infer_ms": (t3 - t2b) * 1000,
                    "pack_ms": (t4 - t3) * 1000,
                    **policy_timing,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packed)
                t5 = time.monotonic()

                prev_total_time = t5 - t0
                req_count += 1

                pt = policy_timing
                logger.info(
                    f"[req {req_count}] recv={(t1-t0)*1000:.0f}ms | unpack={(t2-t1)*1000:.0f}ms | "
                    f"jpeg_dec={(t2b-t2)*1000:.0f}ms | "
                    f"in_xform={pt.get('input_transform_ms',0):.0f}ms | to_dev={pt.get('to_device_ms',0):.0f}ms | "
                    f"build_obs={pt.get('build_obs_ms',0):.0f}ms | sample={pt.get('sample_actions_ms',0):.0f}ms | "
                    f"to_np={pt.get('to_numpy_ms',0):.0f}ms | out_xform={pt.get('output_transform_ms',0):.0f}ms | "
                    f"pack={(t4-t3)*1000:.0f}ms | send={(t5-t4)*1000:.0f}ms | "
                    f"TOTAL={(t5-t0)*1000:.0f}ms | payload={len(raw)/1e6:.2f}MB"
                )

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
    """Decode JPEG-encoded images in-place at any level of the obs dict.

    Handles both flat keys (e.g. "observation.images.base_rgb") and nested
    dicts (e.g. obs["image"]["base_0_rgb"]). If a value is already a numpy
    image array, it passes through unchanged.
    """
    for key, val in obs.items():
        # Recurse into nested dicts (e.g. obs["image"] = {"base_0_rgb": ...})
        if isinstance(val, dict):
            _decode_jpeg_images(val)
            continue

        raw = None
        if isinstance(val, (bytes, bytearray)):
            raw = val
        elif isinstance(val, np.ndarray) and val.dtype.kind in ("S", "U", "V", "O"):
            # msgpack_numpy wraps bytes as a numpy array with bytes/void dtype (e.g. |S30570)
            raw = bytes(val.flat[0]) if val.dtype.kind == "V" else val.flat[0]
        else:
            continue

        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            obs[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            logger.warning(f"Failed to decode JPEG for key '{key}'")
    return obs


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
