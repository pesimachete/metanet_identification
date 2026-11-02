import os
import pickle
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# --- MEMORY OVERRIDES ---
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

jax.config.update("jax_enable_x64", True)

from Variational_paraMetanet.MCMC_based.simulator import parametanet
from Variational_paraMetanet.MCMC_based.simulator import (
    parapersistentExitationSimulation as peSim,
)
from Variational_paraMetanet.MCMC_based.core.objective import (
    log_joint_jit,
    log_likelihood,
    log_prior,
)
from Variational_paraMetanet.MCMC_based.core.parameter_handling import (
    PriorParameters,
    DataLogVariance,
    Theta,
    to_unconstrained,
)

if __name__ == "__main__":
    traj_true, full_p_true, boundaries, init_net_state = peSim.simulate_example()

    def disturb_measurment(key: jax.Array, measurment: jax.Array):
        ssq = 10e-2
        return (
            jnp.exp(jax.random.normal(key, measurment.shape) * jnp.sqrt(ssq))
            * measurment
        )

    keys = jax.random.split(jax.random.PRNGKey(1137), 3)

    print("Applying measurement noise to trajectories...")
    new_flow = disturb_measurment(keys[0], traj_true.flow)
    new_speed = disturb_measurment(keys[1], traj_true.speed)

    trj_dis = parametanet.SimulationTrajectory(
        density=new_flow / (new_speed * full_p_true.static_params.lambda_),
        speed=new_speed,
        flow=new_flow,
    )

    N = full_p_true.static_params.L.shape[0]

    print("Setting up unconstrained parameter constants...")
    alpha_uncon = to_unconstrained(full_p_true.latent_params.alpha, 1.0)
    rho_cr_uncon = to_unconstrained(full_p_true.latent_params.critical_density, 10.0)
    v_free_uncon = to_unconstrained(full_p_true.latent_params.free_flow_speed, 100.0)

    beta_uncon = to_unconstrained(
        full_p_true.scalar_params.beta, full_p_true.scalar_params.beta
    )
    mu_uncon = to_unconstrained(
        full_p_true.scalar_params.mu, full_p_true.scalar_params.mu
    )
    kappa_uncon = to_unconstrained(
        full_p_true.scalar_params.kappa, full_p_true.scalar_params.kappa
    )
    gamma_uncon = to_unconstrained(
        full_p_true.scalar_params.gamma, full_p_true.scalar_params.gamma
    )

    true_scalar_uncon = parametanet.ParaNetworkScalarParameters(
        beta=beta_uncon, mu=mu_uncon, kappa=kappa_uncon, gamma=gamma_uncon
    )
    true_latent_uncon = parametanet.ParaNetworkLatentParameters(
        alpha=alpha_uncon,
        critical_density=rho_cr_uncon,
        free_flow_speed=(v_free_uncon - jnp.mean(v_free_uncon)) / 1
        + jnp.mean(v_free_uncon),
    )

    theta_base = Theta(
        latent=true_latent_uncon,
        scalar=true_scalar_uncon,
        prior=PriorParameters(
            mean=jnp.array([1.62, 3.16, 0.84]),
            log_var=jnp.log(jnp.array([0.005, 0.005, 0.005])),
            corr=jnp.arctanh(jnp.array(0.95)),
        ),
        obs_var=DataLogVariance(
            log_var_flow=jnp.log(jnp.array(10e-4)),
            log_var_speed=jnp.log(jnp.array(10e-4)),
        ),
    )
    log_prior_jit = jax.jit(log_prior)
    ll_jit = jax.jit(log_likelihood)

    def evaluate_z3_var(z3_log_var_uncon, true_z):

        new_log_var = jnp.array(
            [
                jnp.log(0.005),
                jnp.log(0.005),
                z3_log_var_uncon,
            ]
        )

        theta_test = Theta(
            latent=true_z,
            scalar=true_scalar_uncon,
            prior=PriorParameters(
                mean=theta_base.prior.mean,
                log_var=new_log_var,
                corr=theta_base.prior.corr,
            ),
            obs_var=theta_base.obs_var,
        )

        return log_prior_jit(
            theta_test,
            true_z,
        )

    # --- CORRECTION: Set up proper JAX Parallelization using vmap ---
    # in_axes=(0, None) means we vectorize over z3_log_var_uncon but keep true_z fixed
    eval_samples_vmap_jit = jax.jit(jax.vmap(evaluate_z3_var, in_axes=(0, None)))

    # Execute Simulation
    num_vars = 1000000

    # Sweeping UNCONSTRAINED log-variance
    test_vars_uncon = jnp.linspace(-20, 4, num_vars)

    print(f"Executing {num_vars} unconstrained variances utilizing jax.vmap...")

    # --- CORRECTION: Replaced slow for-loop with a single vectorized call ---
    results_jax = eval_samples_vmap_jit(test_vars_uncon, true_latent_uncon)
    results_jax.block_until_ready()
    # ... [Keep everything before results_jax.block_until_ready()] ...

    results = np.asarray(results_jax)

    # --- 1. Find the Maximum ---
    max_val = jnp.max(results)
    best_idx = int(jnp.argmax(results))
    best_log_var = test_vars_uncon[best_idx]

    # --- 2. Calculate 1% Tolerance Interval ---
    # We use absolute value because dropping 1% from a negative log-likelihood
    # means subtracting 1% of its magnitude.
    tolerance = 0.01 * jnp.abs(max_val)
    threshold = max_val - tolerance

    # Find all indices where the result is greater than or equal to the threshold
    valid_indices = jnp.where(results >= threshold)[0]

    # Extract the lower and upper bounds of this interval
    lower_bound_log = test_vars_uncon[valid_indices[0]]
    upper_bound_log = test_vars_uncon[valid_indices[-1]]

    print(f"Optimal Variance: {jnp.exp(best_log_var):.3e} (Peak Value: {max_val:.4f})")
    print(f"1% Tolerance Threshold: {threshold:.4f}")
    print(f"Log-Variance 1% Interval: [{lower_bound_log:.4f}, {upper_bound_log:.4f}]")
    print(
        f"Actual Variance (σ²) Interval: [{jnp.exp(lower_bound_log):.3e}, {jnp.exp(upper_bound_log):.3e}]"
    )

    # --- 3. Visualization ---
    print("Optimization finished. Plotting results...")
    plt.figure(figsize=(12, 7))

    plt.plot(
        jnp.exp(test_vars_uncon),
        results,
        color="crimson",
        linewidth=3,
        label="Log-Prior",
    )

    # Shade the 1% tolerance interval on the plot
    plt.axvspan(
        jnp.exp(lower_bound_log),
        jnp.exp(upper_bound_log),
        color="gray",
        alpha=0.3,
        label="1% Tolerance Interval",
    )

    # Mark the exact optimal point
    plt.axvline(
        jnp.exp(best_log_var), color="black", linestyle="--", label="Optimal Variance"
    )

    plt.title(
        r"Log-Prior profile over $Z_3$ Variance",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel(
        r"Marginal Variance $(\sigma^2)$",
        fontsize=12,
    )
    plt.xscale("log")

    plt.ylabel("Log Prior", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    plt.savefig("z3_variance_unconstrained_profile.svg")
    print(
        "✅ Test complete. Results plotted and saved to 'z3_variance_unconstrained_profile.svg'"
    )
