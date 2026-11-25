import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


def test_tokenize_drop_state_keeps_the_pi05_format_without_the_state():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    state = np.zeros(7)
    with_state, _ = tokenizer.tokenize("pick toy", state)
    without, mask = tokenizer.tokenize("pick toy", state, drop_state=True)
    sentencepiece = tokenizer._tokenizer  # noqa: SLF001
    assert (
        sentencepiece.decode([int(t) for t, m in zip(without, mask, strict=True) if m]) == "Task: pick toy;\nAction: "
    )
    assert "State: 128 128" in sentencepiece.decode([int(t) for t in with_state])


def test_ki_fast_ids_round_trip_below_the_loc_tokens():
    tokenizer = _tokenizer.KITargetTokenizer(max_len=128)
    sentencepiece = tokenizer._tokenizer  # noqa: SLF001
    every = np.arange(2048)
    assert _tokenizer.ki_fast_ids(every).min() == 253_952
    assert _tokenizer.ki_fast_ids(every).max() == 255_999
    assert (_tokenizer.ki_fast_ids(_tokenizer.ki_fast_ids(every)) == every).all()
    assert sentencepiece.id_to_piece(256_000) == "<loc0000>"
    actions = np.cumsum(np.random.default_rng(0).normal(0, 0.05, (50, 7)), axis=0).clip(-1, 1)
    ids = tokenizer.fast_ids(actions)
    assert ((ids >= 253_952) & (ids < 256_000)).all()
    fast = tokenizer._fast_tokenizer  # noqa: SLF001
    error = np.abs(fast.decode([_tokenizer.ki_fast_ids(ids).tolist()], time_horizon=50, action_dim=7)[0] - actions)
    assert error.mean() < 0.05
    assert error.max() < 0.15


def test_ki_targets_are_the_annotation_then_fast_tokens():
    tokenizer = _tokenizer.KITargetTokenizer(max_len=128)
    sentencepiece = tokenizer._tokenizer  # noqa: SLF001
    actions = np.zeros((50, 7))
    tokens, mask = tokenizer.tokenize("pick toy at Right: <loc0001><loc0002><loc0003><loc0004>", actions)
    n_text = int(np.argmax(tokens == tokenizer.end_of_annotation)) + 1
    n = int(mask.sum())
    assert tokens[n_text - 5 : n_text - 1].tolist() == [256_001, 256_002, 256_003, 256_004]
    assert sentencepiece.decode(tokens[: n_text - 5].tolist()) == "pick toy at Right: "
    assert (tokens[n_text : n - 1] == tokenizer.fast_ids(actions)).all()
    assert tokens[n - 1] == sentencepiece.eos_id()
    only_text, text_mask = tokenizer.tokenize("pick toy", None)
    assert sentencepiece.decode(only_text[: text_mask.sum()].tolist()) == "pick toy\n"


def test_an_empty_annotation_is_a_target_only_before_fast_tokens():
    tokenizer = _tokenizer.KITargetTokenizer(max_len=128)
    _, mask = tokenizer.tokenize("", None)
    assert not mask.any()
    actions = np.zeros((50, 7))
    tokens, mask = tokenizer.tokenize("", actions)
    assert tokens[0] == tokenizer.end_of_annotation
    assert (tokens[1 : int(mask.sum()) - 1] == tokenizer.fast_ids(actions)).all()
