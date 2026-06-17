import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.pi0 import masked_mean_pool
from openpi.models.pi0 import stage_ce_and_acc
import openpi.models.pi0_config as _pi0_config


def test_masked_mean_pool_ignores_masked():
    tokens = jnp.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])  # (1, 3, 2)
    mask = jnp.array([[True, True, False]])
    out = masked_mean_pool(tokens, mask)  # -> (1, 2)
    assert jnp.allclose(out, jnp.array([[2.0, 3.0]]))  # mean of first two only


def test_masked_mean_pool_all_masked_is_safe():
    tokens = jnp.ones((1, 2, 2))
    mask = jnp.array([[False, False]])
    out = masked_mean_pool(tokens, mask)
    assert jnp.all(jnp.isfinite(out))  # no NaN from divide-by-zero


def test_stage_ce_perfect_prediction():
    logits = jnp.array([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])  # argmax 0,1
    labels = jnp.array([0, 1], dtype=jnp.int32)
    ce, acc = stage_ce_and_acc(logits, labels, num_classes=3)
    assert float(acc) == 1.0
    assert float(ce) < 1e-3


def test_stage_ce_wrong_prediction():
    logits = jnp.array([[-10.0, 10.0, -10.0]])  # argmax 1
    labels = jnp.array([0], dtype=jnp.int32)
    ce, acc = stage_ce_and_acc(logits, labels, num_classes=3)
    assert float(acc) == 0.0
    assert float(ce) > 1.0


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
