import os
import pickle
import typing
from datetime import datetime

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import parametanet
import parapersistentExitationSimulation as peSim


class PriorParameters(typing.NamedTuple):
    mean: jax.Array
    log_var: jax.Array
    corr: jax.Array


class DataLogVariance(typing.NamedTuple):
    log_var_flow: jax.Array
    log_var_speed: jax.Array


def save_mcmc_results(filepath, results_dict):
    with open(filepath, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"\n[+] Successfully saved all MCMC data to: {filepath}")


def Hasting_within_gibbs_sampling(
    params_fis: parametanet.ParaNetworkScalarParameters,
    params_lat: PriorParameters,
    params_static: parametanet.ParaNetworkStaticParameters,
    data_log_variance: DataLogVariance,
    init_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
    mcmc_variance: jax.Array,
    iterations: int,
    adapt_alpha: float,
    adapt_beta: float,
    num_blocks: int = 4,
) -> tuple[list[jax.Array], list[jax.Array], list[jax.Array], jax.Array]:
    key = jax.random.PRNGKey(67)
    params_history = []
    variance_history = []

    # Initialize with the prior mean
    params_lat_sample = parametanet.ParaNetworkLatentParameters(
        alpha=params_lat.mean[0],
        critical_density=params_lat.mean[1],
        free_flow_speed=params_lat.mean[2],
    )

    def log_likelihood(latent_sample: parametanet.ParaNetworkLatentParameters) -> float:

        phys_alpha = (jax.nn.softplus(latent_sample.alpha) + 1e-6) * 1.0
        phys_rho_cr = (jax.nn.softplus(latent_sample.critical_density) + 1e-6) * 10.0
        phys_v_free = (jax.nn.softplus(latent_sample.free_flow_speed) + 1e-6) * 100.0
        phys_latent_sample = parametanet.ParaNetworkLatentParameters(
            alpha=phys_alpha,
            critical_density=phys_rho_cr,
            free_flow_speed=phys_v_free,
        )
        candidate = parametanet.ParaNetworkParameters(
            static_params=params_static,
            scalar_params=params_fis,
            latent_params=phys_latent_sample,
        )
        sim = parametanet.rollout_simulation(init_state, boundaries, candidate)

        r2_v = jnp.exp(data_log_variance.log_var_speed) + 1e-8
        r2_f = jnp.exp(data_log_variance.log_var_flow) + 1e-8

        ll_v = (
            -0.5
            * jnp.sum(
                (jnp.log(sim.speed + 1e-8) - jnp.log(traj_true.speed + 1e-8)) ** 2
            )
            / r2_v
        )
        ll_f = (
            -0.5
            * jnp.sum((jnp.log(sim.flow + 1e-8) - jnp.log(traj_true.flow + 1e-8)) ** 2)
            / r2_f
        )
        return ll_v + ll_f

    def log_prior(latent_sample: parametanet.ParaNetworkLatentParameters) -> jax.Array:
        var_prior = jnp.exp(params_lat.log_var)
        coeff = (var_prior * (1 - params_lat.corr**2) + 1e-8) ** (-1)
        e = jnp.array(
            [
                latent_sample.alpha - params_lat.mean[0],
                latent_sample.critical_density - params_lat.mean[1],
                latent_sample.free_flow_speed - params_lat.mean[2],
            ]
        )
        einve = coeff * (
            e[:, 0] ** 2
            + e[:, -1] ** 2
            + (1 + params_lat.corr**2) * jnp.sum(e[:, 1:-1] ** 2, axis=1)
            - 2 * params_lat.corr * jnp.sum(e[:, :-1] * e[:, 1:], axis=1)
        )
        return -0.5 * einve

    @jax.jit
    def log_posterior(latent_sample):
        return log_likelihood(latent_sample) + jnp.sum(
            log_prior(latent_sample), dtype=float
        )

    @jax.jit
    def gibbs_step(
        current_sample: parametanet.ParaNetworkLatentParameters,
        current_log_post: float,
        variances: jax.Array,  # Shape: (3, num_blocks)
        key: jax.Array,
    ) -> tuple[parametanet.ParaNetworkLatentParameters, float, jax.Array]:
        N = current_sample.alpha.shape[0]

        keys = jax.random.split(key, 3 * num_blocks * 2)
        acceptances = jnp.zeros((3, num_blocks), dtype=jnp.float32)

        curr_samp = current_sample
        curr_lp = current_log_post

        key_idx = 0

        for p_idx in range(3):
            for b in range(num_blocks):

                mask = (jnp.arange(N) * num_blocks // N) == b

                noise = jax.random.normal(keys[key_idx], shape=(N,))
                u = jax.random.uniform(keys[key_idx + 1])
                key_idx += 2

                # 0: Alpha, 1: Rho, 2: V_free

                if p_idx == 0:
                    curr_val = curr_samp.alpha
                    prop_val = jnp.where(
                        mask,
                        curr_val + noise * variances[0, b],
                        curr_val,
                    )
                    prop_samp = parametanet.ParaNetworkLatentParameters(
                        prop_val, curr_samp.critical_density, curr_samp.free_flow_speed
                    )
                elif p_idx == 1:
                    curr_val = curr_samp.critical_density
                    prop_val = jnp.where(
                        mask,
                        curr_val + noise * variances[1, b],
                        curr_val,
                    )
                    prop_samp = parametanet.ParaNetworkLatentParameters(
                        curr_samp.alpha, prop_val, curr_samp.free_flow_speed
                    )
                else:
                    curr_val = curr_samp.free_flow_speed
                    prop_val = jnp.where(
                        mask,
                        curr_val + noise * variances[2, b],
                        curr_val,
                    )
                    prop_samp = parametanet.ParaNetworkLatentParameters(
                        curr_samp.alpha, curr_samp.critical_density, prop_val
                    )

                lp_prop = log_posterior(prop_samp)
                accept = jnp.log(u) < (lp_prop - curr_lp)

                curr_samp = jax.lax.cond(
                    accept, lambda _: prop_samp, lambda _: curr_samp, None
                )
                curr_lp = jax.lax.cond(
                    accept, lambda _: lp_prop, lambda _: curr_lp, None
                )
                acceptances = acceptances.at[p_idx, b].set(accept)

        return curr_samp, curr_lp, acceptances

    log_post = log_posterior(params_lat_sample)
    total_acceptances = jnp.zeros((3, num_blocks))
    filtered_ar = jnp.zeros((3, num_blocks), dtype=jnp.float32)
    ar_history = []

    print(f"Starting Block MCMC Sampling with {num_blocks} spatial partitions...")
    for i in tqdm(range(iterations), desc="MCMC Progress"):

        params_history.append(params_lat_sample)
        variance_history.append(mcmc_variance)

        key, subkey = jax.random.split(key)
        params_lat_sample, log_post, accepted = gibbs_step(
            params_lat_sample, log_post, mcmc_variance, subkey
        )

        total_acceptances += accepted

        filtered_ar = filtered_ar + 0.01 * (accepted - filtered_ar)
        ar_history.append(filtered_ar)

        # Adapt variances in a completely vectorized manner (no for-loops necessary here)
        mcmc_variance = jnp.where(
            accepted == 1.0, mcmc_variance * adapt_alpha, mcmc_variance * adapt_beta
        )

    return params_history, variance_history, ar_history, total_acceptances / iterations


def disturb_measurment(key: jax.Array, measurment: jax.Array):
    """Applies multiplicative noise to the simulation measurements."""
    ssq = 10e-2
    return jnp.exp(jax.random.normal(key, measurment.shape) * ssq) * measurment


def compute_acf_jax(x: jax.Array, max_lag: int = 100):
    """Computes the ACF in pure JAX using cross-correlation."""
    n = x.shape[0]
    mean = jnp.mean(x)
    var = jnp.var(x)
    x_centered = x - mean

    cov = jnp.correlate(x_centered, x_centered, mode="full")

    acf = jax.lax.dynamic_slice(cov, (n - 1,), (max_lag,)) / (var * n)

    return jnp.where(var == 0.0, jnp.zeros(max_lag), acf)


def plot_variance_evolution(variance_history):
    """Plots the evolution of the MCMC proposal variances across blocks."""
    var_hist = jnp.array(variance_history)  # Shape: (iterations, 3, num_blocks)
    num_blocks = var_hist.shape[2]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for b in range(num_blocks):
        axs[0].plot(var_hist[:, 0, b], label=f"Block {b}", alpha=0.8)
        axs[1].plot(var_hist[:, 1, b], label=f"Block {b}", alpha=0.8)
        axs[2].plot(var_hist[:, 2, b], label=f"Block {b}", alpha=0.8)

    axs[0].set_title(r"Proposal Variance Evolution for $\alpha$")
    axs[0].set_ylabel("Variance multiplier")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title(r"Proposal Variance Evolution for $\rho_{cr}$")
    axs[1].set_ylabel("Variance multiplier")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].set_title(r"Proposal Variance Evolution for $v_{free}$")
    axs[2].set_xlabel("MCMC Iteration")
    axs[2].set_ylabel("Variance multiplier")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_mcmc_trajectories(params_history, true_params, num_blocks):
    """Plots the trajectory of the latent parameters taking a representative center-section from each block."""
    N = len(params_history[0].alpha)
    fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    for b in range(num_blocks):
        # Calculate roughly the center index of block b for representative plotting
        idx = ((b * N // num_blocks) + ((b + 1) * N // num_blocks)) // 2

        alphas = [(jax.nn.softplus(p.alpha[idx]) + 1e-6) * 1.0 for p in params_history]
        rhos = [
            (jax.nn.softplus(p.critical_density[idx]) + 1e-6) * 10.0
            for p in params_history
        ]
        vs = [
            (jax.nn.softplus(p.free_flow_speed[idx]) + 1e-6) * 100.0
            for p in params_history
        ]

        # --- Alpha Plot ---
        axs[0].plot(alphas, alpha=0.5, label=f"MCMC Sec {idx} (Block {b})")
        axs[0].axhline(
            true_params.latent_params.alpha[idx], linestyle="--", linewidth=2
        )

        # --- Rho_cr Plot ---
        axs[1].plot(rhos, alpha=0.5, label=f"MCMC Sec {idx} (Block {b})")
        axs[1].axhline(
            true_params.latent_params.critical_density[idx], linestyle="--", linewidth=2
        )

        # --- V_free Plot ---
        axs[2].plot(vs, alpha=0.5, label=f"MCMC Sec {idx} (Block {b})")
        axs[2].axhline(
            true_params.latent_params.free_flow_speed[idx], linestyle="--", linewidth=2
        )

    axs[0].set_title(r"Trajectory of $\alpha$")
    axs[0].set_ylabel(r"$\alpha$")
    axs[0].legend(loc="upper right", ncol=min(num_blocks, 3))
    axs[0].grid(True)

    axs[1].set_title(r"Trajectory of $\rho_{cr}$")
    axs[1].set_ylabel(r"$\rho_{cr}$ (veh/km/lane)")
    axs[1].legend(loc="upper right", ncol=min(num_blocks, 3))
    axs[1].grid(True)

    axs[2].set_title(r"Trajectory of $v_{free}$")
    axs[2].set_xlabel("MCMC Iteration")
    axs[2].set_ylabel(r"$v_{free}$ (km/h)")
    axs[2].legend(loc="upper right", ncol=min(num_blocks, 3))
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_acf(params_history, num_blocks, burn_in=2000, max_lag=100):
    """Extracts post-burn-in samples and plots ACF from a representative section of each block."""
    if len(params_history) <= burn_in:
        print(
            f"Warning: Chain length ({len(params_history)}) is <= burn-in ({burn_in})."
        )
        return

    post_burn_in = params_history[burn_in:]
    N = len(post_burn_in[0].alpha)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    lags = np.arange(max_lag)

    for b in range(num_blocks):
        idx = ((b * N // num_blocks) + ((b + 1) * N // num_blocks)) // 2

        alphas = jnp.array([p.alpha[idx] for p in post_burn_in])
        rhos = jnp.array([p.critical_density[idx] for p in post_burn_in])
        vs = jnp.array([p.free_flow_speed[idx] for p in post_burn_in])

        axs[0].plot(
            lags, compute_acf_jax(alphas, max_lag), label=f"Sec {idx} (Block {b})"
        )
        axs[1].plot(
            lags, compute_acf_jax(rhos, max_lag), label=f"Sec {idx} (Block {b})"
        )
        axs[2].plot(lags, compute_acf_jax(vs, max_lag), label=f"Sec {idx} (Block {b})")

    axs[0].set_title(rf"ACF of $\alpha$ | Post burn-in: {burn_in}")
    axs[0].set_ylabel("Autocorrelation")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title(rf"ACF of $\rho_{{cr}}$ | Post burn-in: {burn_in}")
    axs[1].set_ylabel("Autocorrelation")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].set_title(rf"ACF of $v_{{free}}$ | Post burn-in: {burn_in}")
    axs[2].set_xlabel("Lag")
    axs[2].set_ylabel("Autocorrelation")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_acceptance_rate_evolution(ar_history, target_ratio=0.3):
    """Plots the exponentially weighted moving average of the acceptance rates."""
    ar_hist = jnp.array(ar_history)  # Shape: (iterations, 3, num_blocks)
    num_blocks = ar_hist.shape[2]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for b in range(num_blocks):
        axs[0].plot(ar_hist[:, 0, b], label=f"Block {b}", alpha=0.8)
        axs[1].plot(ar_hist[:, 1, b], label=f"Block {b}", alpha=0.8)
        axs[2].plot(ar_hist[:, 2, b], label=f"Block {b}", alpha=0.8)

    for ax in axs:
        ax.axhline(
            target_ratio,
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"Target ({target_ratio})",
        )

    axs[0].set_title(r"Filtered Acceptance Rate for $\alpha$")
    axs[0].set_ylabel("Acceptance Rate")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title(r"Filtered Acceptance Rate for $\rho_{cr}$")
    axs[1].set_ylabel("Acceptance Rate")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].set_title(r"Filtered Acceptance Rate for $v_{free}$")
    axs[2].set_xlabel("MCMC Iteration")
    axs[2].set_ylabel("Acceptance Rate")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    print("Setting up Ground Truth Simulation...")
    keys = jax.random.split(jax.random.PRNGKey(1137), 2)
    traj_true, full_p_true, boundaries, init_stat = peSim.simulate_example()
    print("Applying measurement noise...")
    new_flow = disturb_measurment(keys[0], traj_true.flow)
    new_speed = disturb_measurment(keys[1], traj_true.speed)

    trj_dis = parametanet.SimulationTrajectory(
        density=new_flow / (new_speed * full_p_true.static_params.lambda_),
        speed=new_speed,
        flow=new_flow,
    )
    N = full_p_true.static_params.L.shape[0]

    data_log_variance = DataLogVariance(
        log_var_flow=jnp.log(jnp.array(0.01)), log_var_speed=jnp.log(jnp.array(0.01))
    )

    prior_mean = jnp.array([jnp.full(N, 1.62), jnp.full(N, 3.16), jnp.full(N, 0.84)])
    prior_log_var = jnp.log(jnp.array([0.005, 0.005, 0.005]))

    params_lat_prior = PriorParameters(
        mean=prior_mean, log_var=prior_log_var, corr=jnp.array(0.95)
    )

    num_blocks = N // 20

    mcmc_variance_init = jnp.full((3, int(num_blocks)), 0.05)

    iterations = 50000
    expected_ratio = 0.45
    alpha = 1.01
    beta = (1 / jnp.power(alpha, 100 * expected_ratio)) ** (
        1 / (100 - 100 * expected_ratio)
    )
    print(f"Acceptance ratio : {expected_ratio:.2%} ")
    history, var_history, ar_history, accept_rates = Hasting_within_gibbs_sampling(
        params_fis=full_p_true.scalar_params,
        params_lat=params_lat_prior,
        params_static=full_p_true.static_params,
        data_log_variance=data_log_variance,
        init_state=init_stat,
        traj_true=trj_dis,
        boundaries=boundaries,
        mcmc_variance=mcmc_variance_init,
        iterations=iterations,
        adapt_alpha=alpha,
        adapt_beta=beta,
        num_blocks=num_blocks,  # Passing the dynamic dimension here
    )

    results_to_save = {
        "history": history,
        "var_history": var_history,
        "ar_history": ar_history,
        "accept_rates": accept_rates,
        "num_blocks": num_blocks,
        "iterations": iterations,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mcmc_results_{num_blocks}blocks_{timestamp}.pkl"

    save_mcmc_results(filename, results_to_save)

    print("\n" + "=" * 40)
    print("           FINAL MCMC RESULTS")
    print("=" * 40)

    print("\n--- Acceptance Rates ---")
    for b in range(num_blocks):
        print(
            f"Block {b:02d}: Alpha = {accept_rates[0, b]:.2%}  |  Rho_cr = {accept_rates[1, b]:.2%}  |  V_free = {accept_rates[2, b]:.2%}"
        )

    final_vars = var_history[-1]
    print("\n--- Final Tuned Variances ---")
    for b in range(num_blocks):
        print(
            f"Block {b:02d}: Alpha = {final_vars[0, b]:.6f}  |  Rho_cr = {final_vars[1, b]:.6f}  |  V_free = {final_vars[2, b]:.6f}"
        )
    print("=" * 40 + "\n")

    print("Plotting Parameter Trajectories...")
    plot_mcmc_trajectories(history, full_p_true, num_blocks)

    print("Plotting Variance Evolution...")
    plot_variance_evolution(var_history)

    print("Plotting Filtered Acceptance Rate Evolution...")
    plot_acceptance_rate_evolution(ar_history, expected_ratio)

    print("Plotting Autocorrelation (ACF) after Burn-in...")
    plot_acf(history, num_blocks, burn_in=1000, max_lag=1000)


if __name__ == "__main__":
    main()
