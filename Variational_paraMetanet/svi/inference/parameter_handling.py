import typing
import jax
import jax.numpy as jnp
import parametanet


# ---------------------------------------------------------
# SVI Data Structures
# ---------------------------------------------------------
class DataLogVariance(typing.NamedTuple):
    log_var_flow: jax.Array
    log_var_speed: jax.Array


class PriorParameters(typing.NamedTuple):
    mean: jax.Array
    log_var: jax.Array
    corr: jax.Array


class VariationalPosteriorParameters(typing.NamedTuple):
    mean: jax.Array
    log_var: jax.Array


class ExplicitLayerParameters(typing.NamedTuple):
    fis_parameters: parametanet.ParaNetworkScalarParameters
    prior: PriorParameters
    variational_posterior: VariationalPosteriorParameters
    data_log_variance: DataLogVariance


# ---------------------------------------------------------
# Mathematical Helpers
# ---------------------------------------------------------
def inv_softplus(x: jax.Array, scale: float) -> jax.Array:

    def logexpm1(v):
        return v + jnp.log(-jnp.expm1(-v))

    return logexpm1(x / scale)


def map_to_physical_params(
    mean_2d: jax.Array,
    log_var_2d: jax.Array,
    noise_2d: jax.Array,
    latent_scalars: parametanet.ParaNetworkScalarParameters,
    static_params: parametanet.ParaNetworkStaticParameters,
    scales: parametanet.ParaNetworkParameters,
    scales_array: jax.Array,
) -> parametanet.ParaNetworkParameters:
    """Links latent unconstrained variables to physical R+ constraints."""
    z = mean_2d + jnp.exp(0.5 * log_var_2d) * noise_2d

    alpha_phys = (jax.nn.softplus(z[0]) + 1e-8) * scales_array[0]
    rho_cr_phys = (jax.nn.softplus(z[1]) + 1e-8) * scales_array[1]
    v_free_phys = (jax.nn.softplus(z[2]) + 1e-8) * scales_array[2]

    beta_phys = (
        jax.nn.softplus(latent_scalars.beta) + 1e-8
    ) * scales.scalar_params.beta
    mu_phys = (jax.nn.softplus(latent_scalars.mu) + 1e-8) * scales.scalar_params.mu
    kappa_phys = (
        jax.nn.softplus(latent_scalars.kappa) + 1e-8
    ) * scales.scalar_params.kappa
    gamma_phys = (
        jax.nn.softplus(latent_scalars.gamma) + 1e-8
    ) * scales.scalar_params.gamma

    return parametanet.ParaNetworkParameters(
        static_params=static_params,
        scalar_params=parametanet.ParaNetworkScalarParameters(
            beta=beta_phys, mu=mu_phys, kappa=kappa_phys, gamma=gamma_phys
        ),
        latent_params=parametanet.ParaNetworkLatentParameters(
            alpha=alpha_phys, critical_density=rho_cr_phys, free_flow_speed=v_free_phys
        ),
    )


def get_physical_std_2d(
    mean_2d: jax.Array, log_var_2d: jax.Array, scales_array: jax.Array
) -> jax.Array:
    """Delta-method approximation for physical standard deviations."""
    std_latent = jnp.exp(0.5 * log_var_2d)
    grad_f = jax.nn.sigmoid(mean_2d) * scales_array[:, None]
    return jnp.abs(grad_f) * std_latent
