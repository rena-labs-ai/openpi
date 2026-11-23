import dataclasses
import re

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.pi0 import ki_masks
from openpi.models.pi0 import make_attn_mask
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


def test_ki_masks_keep_the_prompt_bidirectional_and_the_targets_from_the_expert():
    # 2 image tokens, 2 prompt tokens and a pad; 2 target tokens and a pad; 2 action tokens.
    prefix_mask = jnp.array([[True, True, True, True, False]])
    target_mask = jnp.array([[True, True, False]])
    input_mask, attn, expert = ki_masks(
        prefix_mask, jnp.zeros(5, bool), target_mask, jnp.ones((1, 2), bool), jnp.array([True, False])
    )
    assert input_mask[0].tolist() == [True, True, True, True, False, True, True, False]
    assert attn[0, :4, :4].all()  # images and prompt see each other both ways
    assert not attn[0, :5, 5:].any()  # and never the targets
    assert attn[0, 5].tolist() == [True, True, True, True, False, True, False, False]  # targets are causal
    assert attn[0, 6].tolist() == [True, True, True, True, False, True, True, False]
    assert expert[0, :, :4].all()
    assert not expert[0, :, 4:8].any()  # the expert never sees a target
    assert expert[0, :, 8:].all()


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
        target_is_fast=jnp.array([[False, False, True, True, True, True]] * 2),
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

    grads = nnx.grad(lambda m: jnp.mean(m.compute_ki_loss(jax.random.key(1), obs, act)[0]))(model)
    flat = {"/".join(map(str, path)): np.abs(np.asarray(g.value)).max() for path, g in grads.flat_state().items()}
    backbone = {p: g for p, g in flat.items() if p.startswith("PaliGemma") and not re.search(r"_1(/|$)", p)}
    expert = {p: g for p, g in flat.items() if re.search(r"_1(/|$)", p)}
    assert backbone
    assert all(g == 0 for g in backbone.values())
    assert any(g > 0 for g in expert.values())
    assert flat["action_in_proj/kernel"] > 0


def test_ki_action_loss_mask_drops_the_flow_and_fast_tokens_of_a_sample(ki):
    config, model, obs = ki
    act = jnp.ones((2, config.action_horizon, config.action_dim))
    flow, ce, _ = model.compute_ki_loss(jax.random.key(1), obs, act)
    masked = dataclasses.replace(obs, action_loss_mask=jnp.array([True, False]))
    masked_flow, masked_ce, _ = model.compute_ki_loss(jax.random.key(1), masked, act)
    assert jnp.array_equal(masked_flow[0], flow[0])
    assert not masked_flow[1].any()
    # without FAST tokens sample 1's cross-entropy is its two annotation tokens alone
    no_fast = dataclasses.replace(
        obs, target_is_fast=obs.target_is_fast.at[1].set(False), target_mask=obs.target_mask.at[1, 2:].set(False)
    )
    _, want_ce, _ = model.compute_ki_loss(jax.random.key(1), no_fast, act)
    assert jnp.allclose(masked_ce, want_ce)
    assert ce != masked_ce


def _greedy_without_cache(model, obs, n):
    """Greedy decoding that reruns the whole sequence for every token, as fast-eval-scripts/detect.py does."""
    obs = _model.preprocess_observation(None, obs, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask, n_img = model.embed_prefix(obs)
    batch = jnp.arange(prefix_mask.shape[0])
    last_prompt = n_img + jnp.sum(obs.tokenized_prompt_mask, axis=1) - 1
    tokens = []
    for _ in range(n):
        emb, mask, ar = prefix_tokens, prefix_mask, prefix_ar_mask
        if tokens:
            decoded = jnp.stack(tokens, axis=1)
            emb = jnp.concatenate([emb, model.PaliGemma.llm(decoded, method="embed")], axis=1)
            mask = jnp.concatenate([mask, jnp.ones(decoded.shape, bool)], axis=1)
            ar = jnp.concatenate([ar, jnp.ones(decoded.shape[1], bool)])
        (out, _), _ = model.PaliGemma.llm([emb, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, 1) - 1)
        last = out[:, -1] if tokens else out[batch, last_prompt]
        tokens.append(jnp.argmax(model.PaliGemma.llm(last[:, None], method="deembed")[:, 0], axis=-1))
    return jnp.stack(tokens, axis=1)


def test_decode_annotation_matches_decoding_without_a_cache_and_stops_at_the_end_token(ki):
    _, model, obs = ki
    want = _greedy_without_cache(model, obs, 5)
    assert jnp.array_equal(model.decode_annotation(obs, end_token=-1, max_tokens=5), want)
    end = int(want[0, 2])
    got = model.decode_annotation(obs, end_token=end, max_tokens=5)
    assert got[0].tolist() == [*want[0, :3].tolist(), 0, 0]
    stop = next((i for i, t in enumerate(want[1].tolist()) if t == end), 4)
    assert got[1, : stop + 1].tolist() == want[1, : stop + 1].tolist()
