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
