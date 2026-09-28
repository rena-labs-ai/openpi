"""Naming activation-rematerialization policies so they can come from config.

Remat trades activation memory for recompute in the backward pass. The towers
take the policy as a string rather than a `jax.checkpoint_policies` value so it
can be set per training run; this module is the one place a name becomes a
policy.
"""

import jax
from flax import linen as nn

NO_REMAT = "none"


def resolve_policy(name: str):
    """The ``jax.checkpoint_policies`` attribute ``name`` refers to.

    Unknown names raise rather than falling back. A fallback here would silently
    reinstate full recompute, which surfaces only as unexplained slowness.
    """
    try:
        return getattr(jax.checkpoint_policies, name)
    except AttributeError as e:
        available = sorted(n for n in dir(jax.checkpoint_policies) if not n.startswith("_"))
        raise ValueError(
            f"unknown remat policy {name!r}; expected {NO_REMAT!r} or one of: {', '.join(available)}"
        ) from e


def maybe_remat(block_cls, policy_name: str, **remat_kwargs):
    """``block_cls`` wrapped in ``nn.remat`` under ``policy_name``, or bare for ``"none"``."""
    if policy_name == NO_REMAT:
        return block_cls
    return nn.remat(block_cls, policy=resolve_policy(policy_name), **remat_kwargs)
