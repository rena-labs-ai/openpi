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
    decode = lambda t, m: tokenizer._tokenizer.decode([int(x) for x, k in zip(t, m) if k])
    assert decode(without, mask) == "Task: pick toy;\nAction: "
    assert "State: 128 128" in tokenizer._tokenizer.decode([int(x) for x in with_state])


def test_ki_fast_ids_round_trip_below_the_loc_tokens():
    tokenizer = _tokenizer.KITargetTokenizer(max_len=128)
    every = np.arange(2048)
    assert _tokenizer.ki_fast_ids(every).min() == 253_952
    assert _tokenizer.ki_fast_ids(every).max() == 255_999
    assert (_tokenizer.ki_fast_ids(_tokenizer.ki_fast_ids(every)) == every).all()
    assert tokenizer._tokenizer.id_to_piece(256_000) == "<loc0000>"
    actions = np.cumsum(np.random.default_rng(0).normal(0, 0.05, (50, 7)), axis=0).clip(-1, 1)
    ids = tokenizer.fast_ids(actions)
    assert ((ids >= 253_952) & (ids < 256_000)).all()
    error = np.abs(tokenizer.fast_actions(ids, 50, 7) - actions)
    assert error.mean() < 0.05
    assert error.max() < 0.15


def test_ki_targets_are_the_annotation_then_fast_tokens():
    tokenizer = _tokenizer.KITargetTokenizer(max_len=128)
    actions = np.zeros((50, 7))
    tokens, mask, is_fast = tokenizer.tokenize("pick toy at Right: <loc0001><loc0002><loc0003><loc0004>", actions)
    n_text = int(np.argmax(tokens == tokenizer.end_of_annotation)) + 1
    n = int(mask.sum())
    assert tokens[n_text - 5 : n_text - 1].tolist() == [256_001, 256_002, 256_003, 256_004]
    assert tokenizer._tokenizer.decode(tokens[: n_text - 5].tolist()) == "pick toy at Right: "
    assert (tokens[n_text : n - 1] == tokenizer.fast_ids(actions)).all()
    assert tokens[n - 1] == tokenizer._tokenizer.eos_id()
    assert is_fast.tolist() == [False] * n_text + [True] * (n - n_text) + [False] * (128 - n)
    only_text, text_mask, no_fast = tokenizer.tokenize("pick toy", None)
    assert tokenizer._tokenizer.decode(only_text[: text_mask.sum()].tolist()) == "pick toy\n"
    assert not no_fast.any()
