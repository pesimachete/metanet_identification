import typing
import jax
import jax.numpy as jnp
from .parameter_handling import Theta


class AdaGradState(typing.NamedTuple):
    r: Theta


class LearningState(typing.NamedTuple):
    theta: Theta
    z: typing.Any
    adagrad: AdaGradState
    adagrad_eps: float
    mcmc_variance: jax.Array
    k: jax.Array


class StepDiagnostic(typing.NamedTuple):
    k: int
    gamma: float
    grad_norm: float


class RunResult(typing.NamedTuple):
    state: LearningState
    theta_history: list[Theta]
    lr_history: list[float]
    grad_norm_history: list[float]
    diagnostics: list[StepDiagnostic]


def init_adagrad(theta: Theta) -> AdaGradState:
    return AdaGradState(r=jax.tree.map(jnp.zeros_like, theta))


def adagrad_update(state: AdaGradState, g: Theta) -> AdaGradState:
    return AdaGradState(r=jax.tree.map(lambda r, gi: r + gi**2, state.r, g))


def adagrad_apply(state: AdaGradState, g: Theta, eps: float) -> Theta:
    return jax.tree.map(lambda r, gi: gi / (jnp.sqrt(r) + eps), state.r, g)


def compute_lr_jax(
    k: jax.Array,
    k_pre: int,
    k_end_heat: jax.Array,
    heat_ended: jax.Array,
    gamma_0: float,
    alpha_decay: float,
) -> jax.Array:
    pre_heat = gamma_0 ** (1.0 - k / k_pre)
    cool = (k - k_end_heat) ** (-alpha_decay)
    return jnp.where(k < k_pre, pre_heat, jnp.where(heat_ended, cool, 1.0))


def compute_lr_jax_constant(
    k: jax.Array,
    k_pre: int,
    k_end_heat: jax.Array,
    heat_ended: jax.Array,
    gamma_0: float,
    alpha_decay: float,
) -> jax.Array:
    # Bypass the pre-heat and cooling schedule
    # Return gamma_0 as a constant learning rate throughout the entire optimization
    return jnp.asarray(gamma_0, dtype=jnp.float32)
