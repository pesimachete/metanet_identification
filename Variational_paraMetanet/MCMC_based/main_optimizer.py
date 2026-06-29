import os
import itertools
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# Import the interruptible list (assuming it is at the root or accessible)
from interruptible_list import interruptible_list

from Variational_paraMetanet.MCMC_based.simulator import parametanet
from Variational_paraMetanet.MCMC_based.simulator import (
    parapersistentExitationSimulation as peSim,
)

from Variational_paraMetanet.MCMC_based.core.parameter_handling import (
    Theta,
    PriorParameters,
    DataLogVariance,
)
from Variational_paraMetanet.MCMC_based.core.adagrad import (
    LearningState,
    init_adagrad,
    compute_lr,
)
from Variational_paraMetanet.MCMC_based.core.objective import (
    pure_update_step,
    log_joint,
)
from Variational_paraMetanet.MCMC_based.sampling.hwg_met import (
    Hasting_within_gibbs_sampling,
)
from Variational_paraMetanet.MCMC_based.utils import visualization


def disturb_measurment(key: jax.Array, measurment: jax.Array):
    """Applies multiplicative noise to the simulation measurements."""
    ssq = 10e-2  # Adjust based on desired noise level (e.g., 10e-1 for higher noise)
    return jnp.exp(jax.random.normal(key, measurment.shape) * ssq) * measurment


def mcmc_draw(
    state: LearningState,
    params_static: parametanet.ParaNetworkStaticParameters,
    scales_scalar: parametanet.ParaNetworkScalarParameters,
    init_net_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
    num_blocks: int,
    sampler_iters: int,
    adapt_alpha: float,
    adapt_beta: float,
) -> tuple[parametanet.ParaNetworkLatentParameters, jax.Array]:

    history, var_history, _, _ = Hasting_within_gibbs_sampling(
        params_fis=state.theta.scalar,
        params_lat=state.theta.prior,
        params_static=params_static,
        scales_scalar=scales_scalar,
        data_log_variance=state.theta.obs_var,
        init_state=init_net_state,
        traj_true=traj_true,
        boundaries=boundaries,
        mcmc_variance=state.mcmc_variance,
        iterations=sampler_iters,
        adapt_alpha=adapt_alpha,
        adapt_beta=adapt_beta,
        num_blocks=num_blocks,
    )
    return history[-1], var_history[-1]


def optimization_generator(
    theta_init: Theta,
    params_static: parametanet.ParaNetworkStaticParameters,
    scales_scalar: parametanet.ParaNetworkScalarParameters,
    init_net_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
    log_every: int = 10,
    k_pre: int = 1000,
    gamma_0: float = 1e-4,
    alpha_decay: float = 2.0 / 3.0,
    c_heat: float = 1e-3,
    adagrad_eps: float = 1e-8,
    num_blocks: int = 4,
    sampler_iters: int = 10,
    adapt_alpha: float = 1.01,
    patience_epochs: int = 5000,  # Early stopping patience
):

    adapt_beta = float(
        (1.0 / adapt_alpha ** (100 * 0.45)) ** (1.0 / (100 - 100 * 0.45))
    )

    state = LearningState(
        theta=theta_init,
        z=theta_init.latent,
        adagrad=init_adagrad(theta_init),
        adagrad_eps=adagrad_eps,
        mcmc_variance=jnp.full((3, num_blocks), 0.05),
        k=0,
        k_end_heat=None,
        ema_norm=0.0,
    )

    latent_names = ["alpha", "critical_density", "free_flow_speed"]
    scalar_names = ["beta", "mu", "kappa", "gamma"]

    best_norm = float("inf")
    steps_without_improvement = 0

    pbar = tqdm(itertools.count(), desc="Optimizing MCMC-AdaGrad")

    for epoch in pbar:
        k = state.k
        gamma = compute_lr(k, k_pre, state.k_end_heat, gamma_0, alpha_decay)

        # 1. MCMC Step (Draw z)
        z_new, mcmc_var_new = mcmc_draw(
            state,
            params_static,
            scales_scalar,
            init_net_state,
            traj_true,
            boundaries,
            num_blocks,
            sampler_iters,
            adapt_alpha,
            adapt_beta,
        )

        # 2. Gradient Step
        state, norm_jax, ema_new_jax = pure_update_step(
            state,
            z_new,
            mcmc_var_new,
            gamma,
            k_pre,
            c_heat,
            params_static,
            scales_scalar,
            init_net_state,
            traj_true,
            boundaries,
        )

        norm = float(norm_jax)
        ema_new = float(ema_new_jax)

        # Heating / Cooling phase transition logic
        if k >= k_pre and state.k_end_heat is None:
            if ema_new > best_norm:
                state = state._replace(k_end_heat=k)
                pbar.write(f"[SGD] Heating ended at k={k}. Starting cooling phase.")
            else:
                best_norm = ema_new

        # Early stopping logic (monitors the EMA norm during cooling)
        if state.k_end_heat is not None:
            if ema_new < best_norm:
                best_norm = ema_new
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

        phase = (
            "pre-heat"
            if k < k_pre
            else ("heat" if state.k_end_heat is None else "cool")
        )
        pbar.set_description(
            f"Phase: {phase} | ‖g‖ EMA: {ema_new:.4f} | Wait: {steps_without_improvement}/{patience_epochs}"
        )

        # Yield exactly the format the visualization logic expects
        if epoch % log_every == 0 or steps_without_improvement >= patience_epochs:
            # Calculate Joint Log Likelihood for diagnostic plotting
            current_log_joint = float(
                log_joint(
                    state.theta,
                    z_new,
                    params_static,
                    scales_scalar,
                    init_net_state,
                    traj_true,
                    boundaries,
                )
            )

            cached_curr_p = {
                name: np.array(getattr(state.theta.latent, name))
                for name in latent_names
            }

            # The MCMC optimizer only has MAP point estimates for theta (no variational stds)
            cached_curr_s = {
                name: np.zeros_like(cached_curr_p[name]) for name in latent_names
            }

            cached_curr_pr_p = {
                name: float(state.theta.prior.mean[i])
                for i, name in enumerate(latent_names)
            }
            cached_curr_pr_s = {
                name: float(jnp.exp(0.5 * state.theta.prior.log_var[i]))
                for i, name in enumerate(latent_names)
            }
            cached_curr_pr_corr = {
                name: float(state.theta.prior.corr) for name in latent_names
            }

            for field in scalar_names:
                cached_curr_p[field] = float(getattr(state.theta.scalar, field))
                cached_curr_s[field] = 0.0
                cached_curr_pr_p[field] = None
                cached_curr_pr_s[field] = None
                cached_curr_pr_corr[field] = None

            yield {
                "epoch": epoch,
                "loss": -current_log_joint,  # Negative log joint (to look like a loss curve)
                "ema_norm": ema_new,
                "lr": gamma,
                "params": cached_curr_p,
                "stds": cached_curr_s,
                "prior_params": cached_curr_pr_p,
                "prior_stds": cached_curr_pr_s,
                "prior_corr": cached_curr_pr_corr,
                "data_var_flow": float(jnp.exp(state.theta.obs_var.log_var_flow)),
                "data_var_speed": float(jnp.exp(state.theta.obs_var.log_var_speed)),
            }

        if steps_without_improvement >= patience_epochs:
            print(f"\nConverged early at epoch {epoch} (EMA Norm: {ema_new:.4f})")
            break


if __name__ == "__main__":
    print("Simulating Base Network...")
    traj_true, full_p_true, boundaries, init_net_state = peSim.simulate_example()

    # ---------------------------------------------------------
    # DISTURBANCE LOGIC
    # ---------------------------------------------------------
    keys = jax.random.split(
        jax.random.PRNGKey(1137), 3
    )  # 3 keys: flow, speed, and parameter noise

    print("Applying measurement noise to trajectories...")
    new_flow = disturb_measurment(keys[0], traj_true.flow)
    new_speed = disturb_measurment(keys[1], traj_true.speed)

    trj_dis = parametanet.SimulationTrajectory(
        density=new_flow / (new_speed * full_p_true.static_params.lambda_),
        speed=new_speed,
        flow=new_flow,
    )

    print("Perturbing parameters for initialization...")
    N = full_p_true.static_params.L.shape[0]
    key_lat, key_sca = jax.random.split(keys[2])

    # Add Gaussian noise directly in the unconstrained space
    # (Scale of 0.5 creates a noticeable disturbance but keeps it roughly within bounds)
    lat_noise = jax.random.normal(key_lat, (3, N)) * 0.5
    disturbed_latent = parametanet.ParaNetworkLatentParameters(
        alpha=full_p_true.latent_params.alpha + lat_noise[0],
        critical_density=full_p_true.latent_params.critical_density + lat_noise[1],
        free_flow_speed=full_p_true.latent_params.free_flow_speed + lat_noise[2],
    )

    sca_noise = jax.random.normal(key_sca, (4,)) * 0.5
    disturbed_scalar = parametanet.ParaNetworkScalarParameters(
        beta=full_p_true.scalar_params.beta + sca_noise[0],
        mu=full_p_true.scalar_params.mu + sca_noise[1],
        kappa=full_p_true.scalar_params.kappa + sca_noise[2],
        gamma=full_p_true.scalar_params.gamma + sca_noise[3],
    )

    # ---------------------------------------------------------
    # INITIALIZATION & EXECUTION
    # ---------------------------------------------------------
    theta_init = Theta(
        latent=disturbed_latent,
        scalar=disturbed_scalar,
        prior=PriorParameters(
            mean=jnp.array([jnp.full(N, 1.62), jnp.full(N, 3.16), jnp.full(N, 0.84)]),
            log_var=jnp.log(jnp.array([0.005, 0.005, 0.005])),
            corr=jnp.array(0.95),
        ),
        obs_var=DataLogVariance(
            log_var_flow=jnp.log(jnp.array(0.01)),
            log_var_speed=jnp.log(jnp.array(0.01)),
        ),
    )

    print("Starting MCMC Optimization Loop...")
    results = interruptible_list(
        optimization_generator(
            theta_init=theta_init,
            params_static=full_p_true.static_params,
            scales_scalar=full_p_true.scalar_params,
            init_net_state=init_net_state,
            traj_true=trj_dis,  # Use the disturbed trajectory
            boundaries=boundaries,
            log_every=5,
            k_pre=100,
            num_blocks=20,
            sampler_iters=10,
            patience_epochs=500,  # Stops if norm doesn't decrease for 500 steps
        ),
        save_whole=True,
        callback_whole=lambda res: visualization.print_whole(
            res, p_true=full_p_true, block=True
        ),
    )

    print("Optimization finished. Generating visualizations...")
    visualization.print_whole(results, p_true=full_p_true, block=True)
