from typing import ClassVar

import pytest

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
