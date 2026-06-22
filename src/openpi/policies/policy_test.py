import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.policy import Policy
from openpi.training import config as _config


class _FakeStageModel(nnx.Module):
    """Minimal nnx.Module exposing both sampling methods for Policy wiring tests."""

    action_horizon = 2
    action_dim = 7

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        b = observation.state.shape[0]
        return jnp.zeros((b, self.action_horizon, self.action_dim))

    def sample_actions_and_stage(self, rng, observation, *, num_steps=10, noise=None):
        b = observation.state.shape[0]
        actions = jnp.zeros((b, self.action_horizon, self.action_dim))
        stage_logits = jnp.broadcast_to(jnp.array([0.0, 5.0, 0.0]), (b, 3))
        return actions, stage_logits


class _FakeActionOnlyModel(nnx.Module):
    """Model without a stage head — Policy must not emit stage_logits."""

    action_horizon = 2
    action_dim = 7

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        b = observation.state.shape[0]
        return jnp.zeros((b, self.action_horizon, self.action_dim))


def _example():
    # unbatched obs in openpi internal format; Policy.infer adds the batch dim.
    return {
        "image": {"base_0_rgb": np.zeros((224, 224, 3), np.float32)},
        "image_mask": {"base_0_rgb": np.ones((), bool)},
        "state": np.zeros((7,), np.float32),
    }


def test_infer_emits_stage_logits_when_model_supports_it():
    policy = Policy(_FakeStageModel())
    assert policy._returns_stage is True  # noqa: SLF001
    out = policy.infer(_example())
    assert out["actions"].shape == (2, 7)
    assert out["stage_logits"].shape == (3,)
    assert int(np.argmax(out["stage_logits"])) == 1


def test_infer_omits_stage_logits_without_stage_head():
    policy = Policy(_FakeActionOnlyModel())
    assert policy._returns_stage is False  # noqa: SLF001
    out = policy.infer(_example())
    assert out["actions"].shape == (2, 7)
    assert "stage_logits" not in out


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
