import typing
import jax
import jax.numpy as jnp
from .parameter_handling import Theta
from jax import tree_util


class AdamState(typing.NamedTuple):
    m: Theta  # first moment
    v: Theta  # second moment


class LearningState(typing.NamedTuple):
    theta: Theta
    z: typing.Any
    adam: AdamState
    adam_eps: float
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


def init_adam(theta: Theta) -> AdamState:
    return AdamState(
        m=jax.tree.map(jnp.zeros_like, theta), v=jax.tree.map(jnp.zeros_like, theta)
    )


def adam_update(
    state: AdamState, g: Theta, omb1: float = 1e-1, omb2: float = 1e-4
) -> AdamState:

    g_clipped = jax.tree.map(lambda gi: jnp.clip(gi, -1e10, 1e10), g)

    m_new = jax.tree.map(lambda m, gi: m + (omb1) * (gi - m), state.m, g_clipped)
    v_new = jax.tree.map(lambda v, gi: v + omb2 * (gi**2 - v), state.v, g_clipped)

    return AdamState(m=m_new, v=v_new)


def adam_apply(
    state: AdamState, eps: float, k: jax.Array, omb1: float = 1e-3, omb2: float = 1e-5
) -> Theta:
    # k is 1-indexed for bias correction
    def compute_update(m, v):
        # Stably compute 1 - (1 - omb)**k without computing 1 - omb
        bias_correction1 = -jnp.expm1(k * jnp.log1p(-omb1))
        bias_correction2 = -jnp.expm1(k * jnp.log1p(-omb2))

        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        return m_hat / (jnp.sqrt(v_hat) + eps)

    # Note: Passed positionally to avoid Python syntax errors
    # with kwargs preceding positional args
    return jax.tree.map(compute_update, state.m, state.v)


def compute_lr_jax_constant(
    k: jax.Array,
    k_pre: int,
    k_end_heat: jax.Array,
    heat_ended: jax.Array,
    gamma_0: float,
    alpha_decay: float,
) -> jax.Array:
    # Bypass the pre-heat and cooling schedule
    return jnp.asarray(gamma_0, dtype=jnp.float64)


def exponential_warmup_scheduler(
    k: jax.Array, gamma_0: float, warmup_steps: int = 2000
) -> jax.Array:

    # Ensure k is treated as a float for division
    k_float = k.astype(jnp.float64)
    warmup_steps_float = jnp.float64(warmup_steps)

    # Calculate the exponential warmup factor
    # At k=0, factor = 10^-6. At k=warmup_steps, factor = 10^0 = 1.0
    factor = jnp.power(10.0, -6.0 * (1.0 - k_float / warmup_steps_float))

    warmup_lr = gamma_0 * factor

    # Apply the warmup condition: if k < warmup_steps, use warmup_lr, else use gamma_0
    return jnp.where(k < warmup_steps, warmup_lr, gamma_0)


def three_phase_scheduler(
    k: jax.Array,
    k_pre: int = 1000,
    k_end_heat: jax.Array = jnp.array(5000),
    heat_ended: jax.Array = jnp.array(False),
    gamma_0: float = 1e-8,
    gamma_max: float = 1e-2,
    alpha: float = 2.0 / 3.0,
) -> jax.Array:
    """
    Three-phase learning rate scheduler adapted from Baey et al. (2023),
    modified to support a configurable maximum learning rate.
    """
    k_float = k.astype(jnp.float64)
    k_pre_float = jnp.float64(k_pre)

    # 1. Pre-heating phase (exponential growth from gamma_0 to gamma_max)
    # Using log-space interpolation for numerical stability
    pre_heat_lr = jnp.exp(
        jnp.log(gamma_max)
        + (1.0 - k_float / k_pre_float) * jnp.log(gamma_0 / gamma_max)
    )

    # 2. Decreasing steps phase
    # Scaled by gamma_max so it decays smoothly from the top
    steps_since_heat_ended = jnp.maximum(k_float - k_end_heat.astype(jnp.float64), 1.0)
    decreasing_lr = gamma_max * jnp.power(steps_since_heat_ended, -alpha)

    # 3. Apply conditions via JAX control flow
    return jnp.where(
        k < k_pre,
        pre_heat_lr,
        jnp.where(heat_ended, decreasing_lr, jnp.float64(gamma_max)),
    )
