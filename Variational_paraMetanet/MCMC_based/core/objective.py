import jax
import jax.numpy as jnp
from Variational_paraMetanet.MCMC_based.simulator import parametanet
from .parameter_handling import Theta, map_to_physical_params
from .adam import (
    LearningState,
    adam_update,
    adam_apply,
    exponential_warmup_scheduler,
    three_phase_scheduler,
)


def log_likelihood(
    theta: Theta,
    z: parametanet.ParaNetworkLatentParameters,
    params_static: parametanet.ParaNetworkStaticParameters,
    scales_scalar: parametanet.ParaNetworkScalarParameters,
    init_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
) -> jax.Array:

    candidate_params = map_to_physical_params(
        unconstrained_latent=z,
        unconstrained_scalar=theta.scalar,
        params_static=params_static,
        scales_scalar=scales_scalar,
    )

    sim = parametanet.rollout_simulation(init_state, boundaries, candidate_params)

    r2_v = jnp.exp(theta.obs_var.log_var_speed)
    r2_f = jnp.exp(theta.obs_var.log_var_flow)
    ll = (
        -0.5 * jnp.sum((jnp.log(sim.speed) - jnp.log(traj_true.speed)) ** 2) / r2_v
        - 0.5 * jnp.sum((jnp.log(sim.flow) - jnp.log(traj_true.flow)) ** 2) / r2_f
        - 0.5 * sim.speed.size * jnp.log(r2_v)
        - 0.5 * sim.flow.size * jnp.log(r2_f)
    )
    return ll


def log_prior(theta: Theta, z: parametanet.ParaNetworkLatentParameters) -> jax.Array:
    var_prior = jnp.exp(theta.prior.log_var)
    x = theta.prior.corr

    e = jnp.stack(
        [
            z.alpha - theta.prior.mean[0],
            z.critical_density - theta.prior.mean[1],
            z.free_flow_speed - theta.prior.mean[2],
        ]
    )
    N = e.shape[1]

    log_one_minus_corr_sq = jnp.log(4.0) + 2 * x - 2 * jax.nn.softplus(2 * x)

    log_det_penalty = (
        -0.5 * N * theta.prior.log_var - 0.5 * (N - 1) * log_one_minus_corr_sq
    )

    t1 = 0.25 * (2.0 + jnp.exp(2 * x) + jnp.exp(-2 * x))
    t2 = 0.5 * (jnp.exp(2 * x) + jnp.exp(-2 * x))
    t3 = 0.25 * (jnp.exp(2 * x) - jnp.exp(-2 * x))

    edge_terms = t1 * (e[:, 0] ** 2 + e[:, -1] ** 2)
    inner_terms = t2 * jnp.sum(e[:, 1:-1] ** 2, axis=1)
    cross_terms = 2.0 * t3 * jnp.sum(e[:, :-1] * e[:, 1:], axis=1)

    quad = (edge_terms + inner_terms - cross_terms) / var_prior

    lp = jnp.sum(log_det_penalty) - 0.5 * jnp.sum(quad)

    return lp


def log_joint(
    theta: Theta,
    z: parametanet.ParaNetworkLatentParameters,
    params_static: parametanet.ParaNetworkStaticParameters,
    scales_scalar: parametanet.ParaNetworkScalarParameters,
    init_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
) -> jax.Array:

    ll = log_likelihood(
        theta,
        z,
        params_static,
        scales_scalar,
        init_state,
        traj_true,
        boundaries,
    )

    lp = log_prior(theta, z)

    return ll + lp


log_joint_jit = jax.jit(log_joint)


@jax.jit
def pure_update_step(
    state: LearningState,
    z_new: parametanet.ParaNetworkLatentParameters,
    mcmc_var_new: jax.Array,
    gamma_0: float,
    params_static: parametanet.ParaNetworkStaticParameters,
    scales_scalar: parametanet.ParaNetworkScalarParameters,
    init_net_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
) -> tuple[LearningState, jax.Array, jax.Array]:
    k = state.k
    eps = state.adam_eps

    k_end_heat_val = jnp.array(7000)

    is_heat_ended = k >= k_end_heat_val

    gamma = three_phase_scheduler(
        k=k, gamma_max=gamma_0, k_end_heat=k_end_heat_val, heat_ended=is_heat_ended
    )

    initial_corr = state.theta.prior.corr

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
    blocked_scalar = g.scalar._replace(kappa=0.0)
    g = g._replace(scalar=blocked_scalar)

    adam_new = adam_update(state.adam, g)

    direction = adam_apply(adam_new, eps, k)
    theta_new = jax.tree.map(lambda t, d: t + gamma * d, state.theta, direction)

    reset_prior = theta_new.prior._replace(corr=initial_corr)
    theta_new = theta_new._replace(prior=reset_prior)

    next_state = LearningState(
        theta=theta_new,
        z=z_new,
        adam=adam_new,
        adam_eps=eps,
        mcmc_variance=mcmc_var_new,
        k=k + 1,
    )

    return next_state, gamma, g
