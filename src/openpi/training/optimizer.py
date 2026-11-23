import dataclasses
import re
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


# pi0's action expert: its own Gemma weights (named with a _1 suffix beside the backbone's) and the layers around it.
_EXPERT = re.compile(r"PaliGemma/llm/.*_1(/|$)|^(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|state_proj|"
                     r"action_time_mlp_in|action_time_mlp_out)/")


def module_labels(params: at.PyTree) -> at.PyTree:
    """Each parameter's module: "expert" for the action expert, "backbone" for the rest (the VLM and stage head)."""

    def label(path, _):
        name = "/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in path)
        return "expert" if _EXPERT.search(name) else "backbone"

    return jax.tree_util.tree_map_with_path(label, params)


def module_norms(grads: at.PyTree) -> dict[str, jax.Array]:
    """The global norm of each module's gradients."""
    labels = jax.tree.leaves(module_labels(grads))
    leaves = jax.tree.leaves(grads)
    return {
        f"grad_norm_{m}": optax.global_norm([g for g, l in zip(leaves, labels, strict=True) if l == m])
        for m in ("backbone", "expert")
    }


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0
    # Set, the backbone and the action expert are clipped each on its own norm, and the expert's learning rate is the
    # schedule's times this. With knowledge insulation the two learn from different losses, so one clip on the sum
    # would scale the expert's steps by the backbone's gradient.
    expert_lr_scale: float | None = None

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        def clipped_adamw(lr):
            tx = optax.adamw(
                lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
            )
            return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)

        if self.expert_lr_scale is None:
            return clipped_adamw(lr)
        assert weight_decay_mask is None, "Weight decay masks are not split by module"
        scale = self.expert_lr_scale
        expert_lr = (lambda step: scale * lr(step)) if callable(lr) else scale * lr
        return optax.multi_transform(
            {"backbone": clipped_adamw(lr), "expert": clipped_adamw(expert_lr)}, module_labels
        )


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
