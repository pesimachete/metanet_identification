import os
from interruptible_list import interruptible_list

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"

import jax
import jax.numpy as jnp
import jax.scipy as jsp

jax.config.update("jax_enable_x64", True)

import parametanet
import parapersistentExitationSimulation as peSim
from inference import parameter_handling
from inference import svi_losses_us as svi_losses
from inference import optimizer
from utils import visualization


def disturb_measurment(key: jax.Array, measurment: jax.Array):
    """Applies multiplicative noise to the simulation measurements."""
    ssq = 10e-1
    return jnp.exp(jax.random.normal(key, measurment.shape) * ssq) * measurment


if __name__ == "__main__":
    print("Generating ground truth simulation...")
    traj_true, full_p_true, boundaries, init_stat = peSim.simulate_example()

    keys = jax.random.split(jax.random.PRNGKey(1137), 2)
    print("Applying measurement noise...")
    new_flow = disturb_measurment(keys[0], traj_true.flow)
    new_speed = disturb_measurment(keys[1], traj_true.speed)

    trj_dis = parametanet.SimulationTrajectory(
        density=new_flow / new_speed * full_p_true.static_params.lambda_,
        speed=new_speed,
        flow=new_flow,
    )

    scales = parametanet.ParaNetworkParameters(
        static_params=parametanet.ParaNetworkStaticParameters(
            T=1.0, L=1.0, lambda_=1.0
        ),
        scalar_params=parametanet.ParaNetworkScalarParameters(
            beta=10 ** jnp.floor(jnp.log10(full_p_true.scalar_params.beta)),
            mu=10 ** jnp.floor(jnp.log10(full_p_true.scalar_params.mu)),
            kappa=10 ** jnp.floor(jnp.log10(full_p_true.scalar_params.kappa)),
            gamma=10 ** jnp.floor(jnp.log10(full_p_true.scalar_params.gamma)),
        ),
        latent_params=parametanet.ParaNetworkLatentParameters(
            alpha=1.0,
            critical_density=10.0,
            free_flow_speed=100.0,
        ),
    )

    scales_array = jnp.array(
        [
            scales.latent_params.alpha,
            scales.latent_params.critical_density,
            scales.latent_params.free_flow_speed,
        ]
    )

    print(
        "Initializing variational posterior exactly on true unconstrained parameters..."
    )

    # Exact mapping of physical truth back to the unconstrained space
    alpha_lat = parameter_handling.inv_softplus(
        full_p_true.latent_params.alpha, scales.latent_params.alpha
    )
    rho_cr_lat = parameter_handling.inv_softplus(
        full_p_true.latent_params.critical_density,
        scales.latent_params.critical_density,
    )
    v_free_lat = parameter_handling.inv_softplus(
        full_p_true.latent_params.free_flow_speed, scales.latent_params.free_flow_speed
    )

    # Establish the exact unconstrained spatial profile for the posterior mean
    initial_mean_2d = jnp.array([alpha_lat, rho_cr_lat, v_free_lat])

    # Establish the global prior mean using the spatial averages
    prior_mean_latent = jnp.array(
        [
            jnp.mean(alpha_lat),
            jnp.mean(rho_cr_lat),
            jnp.mean(v_free_lat),
        ]
    )

    # Configure unconstrained prior variances directly
    unconstrained_prior_stds = jnp.array([0.005, 0.005, 0.005])
    prior_log_var = 2.0 * jnp.log(unconstrained_prior_stds)

    # Deterministically map the scalar parameters to the unconstrained space
    latent_scalars = parametanet.ParaNetworkScalarParameters(
        beta=parameter_handling.inv_softplus(
            full_p_true.scalar_params.beta, scales.scalar_params.beta
        ),
        mu=parameter_handling.inv_softplus(
            full_p_true.scalar_params.mu, scales.scalar_params.mu
        ),
        kappa=parameter_handling.inv_softplus(
            full_p_true.scalar_params.kappa, scales.scalar_params.kappa
        ),
        gamma=parameter_handling.inv_softplus(
            full_p_true.scalar_params.gamma, scales.scalar_params.gamma
        ),
    )

    # Assemble the explicit layer parameters structure
    learnable_params = parameter_handling.ExplicitLayerParameters(
        fis_parameters=latent_scalars,
        prior=parameter_handling.PriorParameters(
            mean=jnp.array([1.62, 3.16, 0.84]),
            log_var=prior_log_var,
            corr=jnp.array(
                [
                    jsp.special.logit(0.95),
                    jsp.special.logit(0.95),
                    jsp.special.logit(0.95),
                ]
            ),
        ),
        variational_posterior=parameter_handling.VariationalPosteriorParameters(
            mean=initial_mean_2d,
            log_var=jnp.array(
                [
                    2.0
                    * jnp.log(1e-3 * jnp.mean(alpha_lat))
                    * jnp.ones_like(alpha_lat),
                    2.0
                    * jnp.log(1e-3 * jnp.mean(rho_cr_lat))
                    * jnp.ones_like(rho_cr_lat),
                    2.0
                    * jnp.log(1e-3 * jnp.mean(v_free_lat))
                    * jnp.ones_like(v_free_lat),
                ]
            ),
        ),
        data_log_variance=parameter_handling.DataLogVariance(
            log_var_flow=jnp.log(1e-3), log_var_speed=jnp.log(1e-3)
        ),
    )

    # Validate the exact initialization by computing the initial loss
    test_noise = jnp.zeros_like(initial_mean_2d)
    initial_loss = svi_losses.expected_nll(
        learnable_params,
        test_noise,
        trj_dis,
        full_p_true.static_params,
        init_stat,
        boundaries,
        scales,
        scales_array,
    )

    print(
        f"-> Exact initialization completed. Initial Negative Log-Likelihood Loss: {initial_loss:.2f}"
    )

    print("Starting SVI Optimization Loop...")
    results = interruptible_list(
        optimizer.optimization_generator(
            learnable_params,
            trj_dis,
            full_p_true.static_params,
            init_stat,
            boundaries,
            scales,
            scales_array,
            log_every=100,
            num_mc_samples=1,
            patience_epochs=5000,  # Set your early stopping patience here
            tolerance=1e-3,  # Minimum required improvement
            ema_alpha=0.05,  # Weight of newest value in the Exponential Moving Avg
        ),
        save_whole=True,
        callback_whole=lambda res: visualization.print_whole(
            res, p_true=full_p_true, block=True
        ),
    )

    print("Optimization finished. Generating visualizations...")
    visualization.print_whole(results, p_true=full_p_true, block=True)
