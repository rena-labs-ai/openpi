import dataclasses
import re

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.pi0 import ki_masks
from openpi.models.pi0 import masked_mean_pool
from openpi.models.pi0 import stage_ce_and_acc
import openpi.models.pi0_config as _pi0_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer


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
    logits = jnp.array([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
    labels = jnp.array([0, 1], dtype=jnp.int32)
    ce, acc = stage_ce_and_acc(logits, labels)
    assert float(acc) == 1.0
    assert float(ce) < 1e-3


def test_stage_ce_wrong_prediction():
    logits = jnp.array([[-10.0, 10.0, -10.0]])
    labels = jnp.array([0], dtype=jnp.int32)
    ce, acc = stage_ce_and_acc(logits, labels)
    assert float(acc) == 0.0
    assert float(ce) > 1.0


def test_stage_ce_takes_num_stage_classes_from_logits():
    logits = jnp.full((1, 7), -10.0).at[0, 5].set(10.0)
    ce, acc = stage_ce_and_acc(logits, jnp.array([5], dtype=jnp.int32))
    assert float(acc) == 1.0
    assert float(ce) < 1e-3


def test_stage_head_follows_num_stage_classes():
    model = nnx.eval_shape(_pi0_config.Pi0Config(num_stage_classes=7).create, jax.random.key(0))
    assert model.stage_head_out.kernel.value.shape[-1] == 7


def test_load_takes_num_stage_classes_from_checkpoint(monkeypatch):
    monkeypatch.setattr(_model.BaseModelConfig, "load", lambda self, params, **_: self)
    params = {"stage_head_out": {"kernel": np.zeros((4, 7)), "bias": np.zeros(7)}}
    assert _pi0_config.Pi0Config().load(params).num_stage_classes == 7


def test_load_keeps_config_num_stage_classes_without_stage_head(monkeypatch):
    monkeypatch.setattr(_model.BaseModelConfig, "load", lambda self, params, **_: self)
    assert _pi0_config.Pi0Config(num_stage_classes=7).load({}).num_stage_classes == 7


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


def test_compute_loss_and_stage_gives_flow_loss_and_stage_logits():
    config = _pi0_config.Pi0Config(
        pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", num_stage_classes=7
    )
    model = config.create(jax.random.key(0))
    obs, act = config.fake_obs(2), config.fake_act(2)
    flow, logits = model.compute_loss_and_stage(jax.random.key(1), obs, act, train=True)
    assert flow.shape == (2, config.action_horizon)
    assert logits.shape == (2, 7)


def test_ki_masks_keep_the_prompt_bidirectional_and_the_targets_causal_after_it():
    prefix_mask = jnp.array([[True, True, True, True, False]])
    input_mask, attn = ki_masks(prefix_mask, jnp.zeros(5, bool), jnp.array([[True, True, False]]))
    assert input_mask[0].tolist() == [True, True, True, True, False, True, True, False]
    assert attn[0, :4, :4].all()
    assert not attn[0, :5, 5:].any()
    assert attn[0, 5].tolist() == [True, True, True, True, False, True, False, False]
    assert attn[0, 6].tolist() == [True, True, True, True, False, True, True, False]


@pytest.fixture(scope="module")
def ki():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        knowledge_insulation=True,
        max_target_len=6,
        dtype="float32",
    )
    model = config.create(jax.random.key(0))
    rng = np.random.default_rng(0)
    obs = config.fake_obs(2)
    prompt_mask = np.ones((2, config.max_token_len), bool)
    prompt_mask[1, 5:] = False
    obs = dataclasses.replace(
        obs,
        tokenized_prompt=jnp.asarray(rng.integers(0, 1000, (2, config.max_token_len)), jnp.int32),
        tokenized_prompt_mask=jnp.asarray(prompt_mask),
        target_tokens=jnp.asarray(rng.integers(0, 1000, (2, 6)), jnp.int32),
        target_mask=jnp.array([[True] * 6, [True] * 4 + [False] * 2]),
    )
    return config, model, obs


def test_ki_flow_loss_ignores_the_targets_and_trains_only_the_expert(ki):
    config, model, obs = ki
    act = jnp.asarray(np.random.default_rng(1).normal(size=(2, config.action_horizon, config.action_dim)), jnp.float32)
    flow, ce, stage_logits = model.compute_ki_loss(jax.random.key(1), obs, act)
    other = dataclasses.replace(obs, target_tokens=obs.target_tokens + 7)
    other_flow, other_ce, _ = model.compute_ki_loss(jax.random.key(1), other, act)
    assert flow.shape == (2, config.action_horizon)
    assert stage_logits.shape == (2, config.num_stage_classes)
    assert jnp.array_equal(flow, other_flow)
    assert ce != other_ce
    np.testing.assert_allclose(flow, model.compute_loss(jax.random.key(1), obs, act), rtol=1e-5, atol=1e-6)

    grads = nnx.grad(lambda m: jnp.mean(m.compute_ki_loss(jax.random.key(1), obs, act)[0]))(model)
    flat = {"/".join(map(str, path)): np.abs(np.asarray(g.value)).max() for path, g in grads.flat_state().items()}
    backbone = {p: g for p, g in flat.items() if p.startswith("PaliGemma") and not re.search(r"_1(/|$)", p)}
    expert = {p: g for p, g in flat.items() if re.search(r"_1(/|$)", p)}
    assert backbone
    assert all(g == 0 for g in backbone.values())
    assert any(g > 0 for g in expert.values())
    assert flat["action_in_proj/kernel"] > 0


def test_ki_optimizer_splits_modules_by_the_loss_that_trains_them(ki):
    config, model, obs = ki
    act = jnp.asarray(np.random.default_rng(1).normal(size=(2, config.action_horizon, config.action_dim)), jnp.float32)
    # without the image encoder, whose full-size gradients and optimizer state would not fit a small GPU
    params = nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex("PaliGemma/img/.*")))
    flow_grads = nnx.grad(
        lambda m: jnp.mean(m.compute_ki_loss(jax.random.key(1), obs, act)[0]), argnums=nnx.DiffState(0, params)
    )(model)
    ce_grads = nnx.grad(lambda m: m.compute_ki_loss(jax.random.key(1), obs, act)[1], argnums=nnx.DiffState(0, params))(
        model
    )
    labels = _optimizer.module_labels(nnx.state(model, params))
    for grads, module in ((flow_grads, "expert"), (ce_grads, "backbone")):
        trained = jax.tree.map(lambda g: bool(jnp.any(g != 0)), grads)
        for t, label in zip(jax.tree.leaves(trained), jax.tree.leaves(labels), strict=True):
            assert label == module or not t

    params = nnx.state(model, params)
    tx = _optimizer.AdamW(clip_per_module=True).create(1e-3)
    both = jax.tree.map(lambda f, c: f + 1e3 * c, flow_grads, ce_grads)
    expert_only, _ = tx.update(flow_grads, tx.init(params), params)
    mixed, _ = tx.update(both, tx.init(params), params)
    for a, b, label in zip(jax.tree.leaves(expert_only), jax.tree.leaves(mixed), jax.tree.leaves(labels), strict=True):
        if label == "expert":
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-12)
