import asyncio
import concurrent.futures
import functools
import http
import itertools
import json
import logging
import os
import queue
import threading
import time
import traceback

import cv2
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

# How long rena-training's quality-gate probes wait past a robot's last inference.
QC_HOLD_SECONDS = float(os.environ.get("RENA_QC_HOLD_SECONDS", "1800"))

_ROBOT = 0
_PROBE = 1


class _ProbeRefusedError(Exception):
    def __init__(self, retry_after: float):
        super().__init__(retry_after)
        self.retry_after = retry_after


class _InferenceQueue:
    """Runs inferences one at a time on a single worker thread, robot requests ahead of queued probes.

    One worker keeps inferences serialized on the single GPU, while the event loop stays
    free to answer keepalive pings: a policy's first request compiles for ~80s, and on
    the loop thread that silence is what the client closes the connection over.
    """

    def __init__(self) -> None:
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._order = itertools.count()
        threading.Thread(target=self._work, name="infer", daemon=True).start()

    def submit(self, priority: int, fn) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((priority, next(self._order), fn, future))
        return future

    def _work(self) -> None:
        while True:
            _, _, fn, future = self._queue.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except BaseException as e:
                future.set_exception(e)


def resolve_model_path(path: str, ids, default: str) -> str | None:
    """Model id selected by a websocket request path, or None to reject.

    A bare "/" serves the default; "/m/<id>" selects (ids contain slashes:
    "<exp_name>/<step>"). Anything else — including an unknown id — is a
    rejection, never a silent fallback.
    """
    if path in ("", "/"):
        return default
    if path.startswith("/m/"):
        rid = path[3:]
        if rid in ids:
            return rid
    return None


class WebsocketPolicyServer:
    """Serves one or more policies using the websocket protocol. See
    websocket_client_policy.py for a client implementation.

    Single-policy form: pass `policy` (+ optional `metadata`) — every
    connection gets that policy, whatever its path; `/models` does not exist.

    Multi-policy form: pass `policies` (id -> policy, insertion-ordered),
    `default` (an id in `policies`), optional `labels` (id -> display label)
    and optional `delivered_at` (id -> UTC ISO 8601 of when that model was
    delivered, for a client to render in its own zone). The model is chosen
    per connection by request path (see
    `resolve_model_path`); an unknown id is rejected with HTTP 404 before the
    upgrade. `GET /models` lists the catalog, `/healthz` reports the loaded
    ids, and each connection's metadata frame carries the resolved `model`.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy | None = None,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        *,
        policies: dict[str, _base_policy.BasePolicy] | None = None,
        default: str | None = None,
        labels: dict[str, str] | None = None,
        delivered_at: dict[str, str | None] | None = None,
    ) -> None:
        if (policy is None) == (policies is None):
            raise ValueError("pass exactly one of `policy` or `policies`")
        if policies is not None and default not in policies:
            raise ValueError(f"default {default!r} not in policies")
        self._policy = policy
        self._policies = policies
        self._default = default
        self._labels = labels or {}
        self._delivered_at = delivered_at or {}
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._last_infer_at: float | None = None
        self._in_flight = 0
        self._inference = _InferenceQueue()
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

    def _catalog(self) -> dict:
        # delivered_at travels as the instant the roster recorded, never a
        # formatted day: which day it falls on depends on the reader's zone,
        # and this host's zone is nobody's. null when the roster had none.
        return {
            "models": [
                {
                    "id": mid,
                    "label": self._labels.get(mid, mid),
                    "delivered_at": self._delivered_at.get(mid),
                }
                for mid in self._policies
            ],
            "default": self._default,
        }

    def _connection_policy(self, path: str):
        """(model id, policy) for a connection path; (None, None) = reject."""
        if self._policies is None:
            return None, self._policy
        mid = resolve_model_path(path, self._policies.keys(), self._default)
        if mid is None:
            return None, None
        return mid, self._policies[mid]

    def _process_request(
        self, connection: _server.ServerConnection, request: _server.Request
    ) -> _server.Response | None:
        if request.path == "/healthz":
            body = {
                "status": "ok",
                "last_infer_at": self._last_infer_at,
                "in_flight": self._in_flight,
                "qc_hold_seconds": QC_HOLD_SECONDS,
                # A client keeps several probes outstanding only where robots jump them.
                "robot_first": True,
            }
            if self._policies is not None:
                body["models"] = list(self._policies)
                body["default"] = self._default
            return connection.respond(http.HTTPStatus.OK, json.dumps(body) + "\n")
        if request.path == "/models" and self._policies is not None:
            return connection.respond(http.HTTPStatus.OK, json.dumps(self._catalog()) + "\n")
        if self._connection_policy(request.path)[1] is None:
            return connection.respond(
                http.HTTPStatus.NOT_FOUND,
                json.dumps({"error": f"unknown model path: {request.path}"}) + "\n",
            )
        # Continue with the normal websocket handshake.
        return None

    def probe_hold_left(self) -> float:
        """Seconds until QC probes are allowed again."""
        if self._last_infer_at is None:
            return 0.0
        return max(0.0, QC_HOLD_SECONDS - (time.time() - self._last_infer_at))

    def _run_robot(self, infer, obs: dict) -> dict:
        try:
            return infer(obs)
        finally:
            # On the worker, before it takes the next request: a probe queued behind
            # this inference must already see the hold.
            self._last_infer_at = time.time()

    def _run_probe(self, infer, obs: dict) -> dict:
        # Checked when the probe reaches the worker, not when it arrived: a robot
        # may have inferred while it waited.
        hold_left = self.probe_hold_left()
        if hold_left > 0:
            raise _ProbeRefusedError(hold_left)
        return infer(obs)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        model_id, policy = self._connection_policy(websocket.request.path)
        if policy is None:  # raced a set change since process_request
            await websocket.close(code=websockets.frames.CloseCode.POLICY_VIOLATION)
            return
        if self._policies is None:
            metadata = self._metadata
        else:
            metadata = {
                **(policy.metadata or {}),
                "model": model_id,
                "default": self._default,
            }
            logger.info(f"Connection bound to model {model_id}")
        await websocket.send(packer.pack(metadata))

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

                # QC probes don't count as robot activity for /healthz.
                if obs.pop("_qc_probe", False):
                    priority = _PROBE
                    job = functools.partial(self._run_probe, policy.infer, obs)
                else:
                    priority = _ROBOT
                    job = functools.partial(self._run_robot, policy.infer, obs)
                self._in_flight += 1
                try:
                    action = await asyncio.wrap_future(self._inference.submit(priority, job))
                except _ProbeRefusedError as refused:
                    logger.info(f"QC probe refused: {refused.retry_after:.0f}s of robot hold left")
                    await websocket.send(packer.pack({"_qc_refused": True, "retry_after": refused.retry_after}))
                    continue
                finally:
                    self._in_flight -= 1
                t3 = time.monotonic()

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
        if isinstance(val, bytes | bytearray):
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
