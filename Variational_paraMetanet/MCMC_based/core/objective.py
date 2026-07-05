import jax
import jax.numpy as jnp
from Variational_paraMetanet.MCMC_based.simulator import parametanet
from .parameter_handling import Theta, map_to_physical_params
from .adagrad import (
    LearningState,
    adagrad_update,
    adagrad_apply,
    adagrad_apply_stabilised,
    compute_lr_jax_constant as compute_lr_jax,
)


def log_joint(
    theta: Theta,
    z: parametanet.ParaNetworkLatentParameters,
    params_static: parametanet.ParaNetworkStaticParameters,
    scales_scalar: parametanet.ParaNetworkScalarParameters,
    init_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
) -> jax.Array:

    # 1. Centralized Physical Mapping
    candidate_params = map_to_physical_params(
        unconstrained_latent=z,
        unconstrained_scalar=theta.scalar,
        params_static=params_static,
        scales_scalar=scales_scalar,
    )

    # 2. Physics Rollout
    sim = parametanet.rollout_simulation(init_state, boundaries, candidate_params)

    # 3. Likelihood
    r2_v = jnp.exp(theta.obs_var.log_var_speed) + 1e-8
    r2_f = jnp.exp(theta.obs_var.log_var_flow) + 1e-8
    ll = (
        -0.5
        * jnp.sum((jnp.log(sim.speed + 1e-8) - jnp.log(traj_true.speed + 1e-8)) ** 2)
        / r2_v
        - 0.5
        * jnp.sum((jnp.log(sim.flow + 1e-8) - jnp.log(traj_true.flow + 1e-8)) ** 2)
        / r2_f
        - 0.5 * sim.speed.size * jnp.log(r2_v)
        - 0.5 * sim.flow.size * jnp.log(r2_f)
    )

    # 4. Spatial Prior on z
    var_prior = jnp.exp(theta.prior.log_var)
    corr = jnp.tanh(theta.prior.corr)  # bijector: keeps |corr| < 1 under SGD
    coeff = (var_prior * (1 - corr**2) + 1e-8) ** (-1)
    e = jnp.stack(
        [
            z.alpha - theta.prior.mean[0],
            z.critical_density - theta.prior.mean[1],
            z.free_flow_speed - theta.prior.mean[2],
        ]
    )
    quad = coeff * (
        e[:, 0] ** 2
        + e[:, -1] ** 2
        + (1 + corr**2) * jnp.sum(e[:, 1:-1] ** 2, axis=1)
        - 2 * corr * jnp.sum(e[:, :-1] * e[:, 1:], axis=1)
    )
    lp = -0.5 * jnp.sum(quad)

    return ll + lp


log_joint_jit = jax.jit(log_joint)


@jax.jit
def pure_update_step(
    state: LearningState,
    z_new: parametanet.ParaNetworkLatentParameters,
    mcmc_var_new: jax.Array,
    k_pre: int,
    gamma_0: float,
    alpha_decay: float,
    c_heat: float,
    params_static: parametanet.ParaNetworkStaticParameters,
    scales_scalar: parametanet.ParaNetworkScalarParameters,
    init_net_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
) -> tuple[LearningState, jax.Array, jax.Array, jax.Array]:

    k = state.k
    eps = state.adagrad_eps
    gamma = compute_lr_jax(
        k, k_pre, state.k_end_heat, state.heat_ended, gamma_0, alpha_decay
    )

    score_fn = jax.grad(log_joint)
    g = score_fn(
        state.theta,
        z_new,
        params_static,
        scales_scalar,
        init_net_state,
        traj_true,
        boundaries,
    )

    adagrad_new = adagrad_update(state.adagrad, g)
    direction = jax.lax.cond(
        k < k_pre,
        lambda _: adagrad_apply_stabilised(adagrad_new, g, gamma, eps),
        lambda _: adagrad_apply(adagrad_new, g, eps),
        operand=None,
    )
    theta_new = jax.tree.map(lambda t, d: t + gamma * d, state.theta, direction)

    leaves = jax.tree.leaves(g)
    norm = jnp.sqrt(sum(jnp.sum(l**2) for l in leaves))
    ema_new = state.ema_norm + c_heat * (norm - state.ema_norm)

    next_state = LearningState(
        theta=theta_new,
        z=z_new,
        adagrad=adagrad_new,
        adagrad_eps=eps,
        mcmc_variance=mcmc_var_new,
        k=k + 1,
        k_end_heat=state.k_end_heat,
        heat_ended=state.heat_ended,
        ema_norm=ema_new,
    )
    return next_state, norm, ema_new, gamma
