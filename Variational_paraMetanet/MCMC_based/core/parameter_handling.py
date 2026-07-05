import typing
import jax
import jax.numpy as jnp
from Variational_paraMetanet.MCMC_based.simulator import parametanet


# ---------------------------------------------------------
# Shared Data Structures
# ---------------------------------------------------------
class DataLogVariance(typing.NamedTuple):
    log_var_flow: jax.Array
    log_var_speed: jax.Array


class PriorParameters(typing.NamedTuple):
    mean: jax.Array
    log_var: jax.Array
    corr: jax.Array


class Theta(typing.NamedTuple):
    """The single learnable PyTree for the MCMC Optimizer (unconstrained space)"""

    latent: parametanet.ParaNetworkLatentParameters
    scalar: parametanet.ParaNetworkScalarParameters
    prior: PriorParameters
    obs_var: DataLogVariance


def map_to_physical_params(
    unconstrained_latent: parametanet.ParaNetworkLatentParameters,
    unconstrained_scalar: parametanet.ParaNetworkScalarParameters,
    params_static: parametanet.ParaNetworkStaticParameters,
    scales_scalar: parametanet.ParaNetworkScalarParameters,
    eps: float = 1e-6,
) -> parametanet.ParaNetworkParameters:
    """
    Maps unconstrained variables to strictly positive physical constraints using softplus.
    """
    # 1. Map Spatial (Latent) Parameters
    alpha_phys = (jax.nn.softplus(unconstrained_latent.alpha) + eps) * 1.0
    rho_cr_phys = (jax.nn.softplus(unconstrained_latent.critical_density) + eps) * 10.0
    v_free_phys = (jax.nn.softplus(unconstrained_latent.free_flow_speed) + eps) * 100.0

    # 2. Map Global (Scalar) Parameters
    beta_phys = (jax.nn.softplus(unconstrained_scalar.beta) + eps) * scales_scalar.beta
    mu_phys = (jax.nn.softplus(unconstrained_scalar.mu) + eps) * scales_scalar.mu
    kappa_phys = (
        jax.nn.softplus(unconstrained_scalar.kappa) + eps
    ) * scales_scalar.kappa
    gamma_phys = (
        jax.nn.softplus(unconstrained_scalar.gamma) + eps
    ) * scales_scalar.gamma

    return parametanet.ParaNetworkParameters(
        static_params=params_static,
        scalar_params=parametanet.ParaNetworkScalarParameters(
            beta=beta_phys, mu=mu_phys, kappa=kappa_phys, gamma=gamma_phys
        ),
        latent_params=parametanet.ParaNetworkLatentParameters(
            alpha=alpha_phys, critical_density=rho_cr_phys, free_flow_speed=v_free_phys
        ),
    )


def inverse_softplus(y_over_scale: jax.Array, eps: float = 1e-6) -> jax.Array:

    return jnp.log(jnp.expm1(y_over_scale - eps))


def to_unconstrained(physical_value: jax.Array, scale, eps: float = 1e-6) -> jax.Array:

    return inverse_softplus(physical_value / scale, eps)
