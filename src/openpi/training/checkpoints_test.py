import jax
import numpy as np

import openpi.training.checkpoints as _checkpoints


def _repl_sharding() -> jax.sharding.NamedSharding:
    mesh = jax.make_mesh((1,), ("x",))
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def test_with_sharding_attaches_sharding_to_abstract_leaves():
    """The FSDP resume fix: abstract restore targets from jax.eval_shape carry no
    sharding, so Orbax materializes them fully on host and reshards on first use --
    which stalls for FSDP-sized models. `_with_sharding` attaches the per-leaf target
    sharding so Orbax restores each shard directly onto its device."""
    repl = _repl_sharding()
    target = {
        "weight": jax.ShapeDtypeStruct((4, 8), np.float32),
        "nested": {"bias": jax.ShapeDtypeStruct((8,), np.float32)},
        "step": jax.ShapeDtypeStruct((), np.int32),
    }
    shardings = {
        "weight": repl,
        "nested": {"bias": repl},
        "step": repl,
    }

    out = _checkpoints._with_sharding(target, shardings)  # noqa: SLF001

    # Shape and dtype are preserved.
    assert out["weight"].shape == (4, 8)
    assert out["weight"].dtype == np.float32
    assert out["nested"]["bias"].shape == (8,)
    assert out["step"].shape == ()
    # Sharding is now attached to every leaf (was None before).
    assert target["weight"].sharding is None
    assert out["weight"].sharding == repl
    assert out["nested"]["bias"].sharding == repl
    assert out["step"].sharding == repl
