import jax
import jax.numpy as jnp
import numpy as np
import parametanet
from inference import parameter_handling


def find_stable_noise_samples(
    base_key: jax.Array,
    frozen_params: parameter_handling.ExplicitLayerParameters,
    num_mc_samples: int = 1,  # <-- NEW
) -> jax.Array:

    target_shape = frozen_params.variational_posterior.mean.shape

    if num_mc_samples > 1:
        target_shape = (num_mc_samples,) + target_shape

    noise_sample = jax.random.normal(base_key, shape=target_shape)

    return noise_sample


def expected_nll(
    params: parameter_handling.ExplicitLayerParameters,
    safe_noise: jax.Array,
    traj_true: parametanet.SimulationTrajectory,
    static_params: parametanet.ParaNetworkStaticParameters,
    init_state: parametanet.NetworkState,
    boundaries: parametanet.BoundarySequence,
    scales: parametanet.ParaNetworkParameters,
    scales_array: jax.Array,
) -> float:

    def single_sample_nll(single_noise: jax.Array) -> float:
        candidate = parameter_handling.map_to_physical_params(
            params.variational_posterior.mean,
            params.variational_posterior.log_var,
            single_noise,
            params.fis_parameters,
            static_params,
            scales,
            scales_array,
        )
        sim = parametanet.rollout_simulation(init_state, boundaries, candidate)
        M = traj_true.speed.size
        r2_v = jnp.exp(params.data_log_variance.log_var_speed) + 1e-8
        r2_f = jnp.exp(params.data_log_variance.log_var_flow) + 1e-8

        nll_v = (
            0.5 * M * params.data_log_variance.log_var_speed
            + 0.5
            * jnp.sum(
                (jnp.log(sim.speed + 1e-8) - jnp.log(traj_true.speed + 1e-8)) ** 2
            )
            / r2_v
        )
        nll_f = (
            0.5 * M * params.data_log_variance.log_var_flow
            + 0.5
            * jnp.sum((jnp.log(sim.flow + 1e-8) - jnp.log(traj_true.flow + 1e-8)) ** 2)
            / r2_f
        )
        return nll_v + nll_f

    # 2. Automatically vectorise across the MC samples if a batch is provided
    if safe_noise.ndim == 3:
        batched_nlls = jax.vmap(single_sample_nll)(safe_noise)
        return jnp.mean(batched_nlls)
    else:
        # Zero overhead standard pass
        return single_sample_nll(safe_noise)


def kl_divergence(params: parameter_handling.ExplicitLayerParameters) -> float:
    N = params.variational_posterior.mean.shape[1]
    var_prior = jnp.exp(params.prior.log_var)
    var_posterior = jnp.exp(params.variational_posterior.log_var)

    corr_phys = jax.nn.sigmoid(params.prior.corr) * 0.9999
    coeff = (var_prior * (1 - corr_phys**2) + 1e-8) ** (-1)

    e = params.variational_posterior.mean - params.prior.mean.reshape(-1, 1)
    einve = coeff * (
        e[:, 0] ** 2
        + e[:, -1] ** 2
        + (1 + corr_phys**2) * jnp.sum(e[:, 1:-1] ** 2, axis=1)
        - 2 * corr_phys * jnp.sum(e[:, :-1] * e[:, 1:], axis=1)
    )
    tr = coeff * (
        var_posterior[:, 0]
        + var_posterior[:, -1]
        + (1 + corr_phys**2) * jnp.sum(var_posterior[:, 1:-1], axis=1)
    )
    lsig2 = N * params.prior.log_var + (N - 1) * jnp.log(1 - corr_phys**2 + 1e-8)
    lsig1 = jnp.sum(params.variational_posterior.log_var, axis=1)

    return jnp.sum(0.5 * (tr + einve - N + lsig2 - lsig1))
