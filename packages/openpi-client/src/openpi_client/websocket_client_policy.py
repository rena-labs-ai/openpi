import logging
import socket
import time
from typing import Dict, Optional, Tuple

import websockets.exceptions
import websockets.sync.client
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        headers: Optional[Dict[str, str]] = None,
        open_timeout: Optional[float] = 10,
    ) -> None:
        """headers: sent with the handshake; an Authorization in any case replaces the api_key one.
        open_timeout: seconds to wait for the handshake; a server that scales from zero needs minutes."""
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._headers = dict(headers or {})
        if api_key and "authorization" not in {h.lower() for h in self._headers}:
            self._headers["Authorization"] = f"Api-Key {api_key}"
        self._open_timeout = open_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=self._headers or None,
                    open_timeout=self._open_timeout,
                )
                try:
                    return conn, msgpack_numpy.unpackb(conn.recv())
                except BaseException:
                    conn.close()
                    raise
            # socket.timeout is TimeoutError only from Python 3.10.
            except (ConnectionRefusedError, TimeoutError, socket.timeout):
                logging.info("Still waiting for server...")
                time.sleep(5)
            except websockets.exceptions.InvalidStatus as e:
                # A gateway in front of a server still starting answers 502 or 504.
                if e.response.status_code not in (502, 504):
                    raise
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass

    def close(self) -> None:
        """An open connection's reader thread keeps the interpreter from exiting."""
        self._ws.close()
