import os
import itertools
import pickle
from datetime import datetime

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import numpy as np
from tqdm import tqdm

from interruptible_list import interruptible_list

from Variational_paraMetanet.MCMC_based.simulator import parametanet
from Variational_paraMetanet.MCMC_based.simulator import (
    paraLongHighway as peSim,
)

from Variational_paraMetanet.MCMC_based.core.parameter_handling import (
    Theta,
    PriorParameters,
    DataLogVariance,
    to_unconstrained,
)
from Variational_paraMetanet.MCMC_based.core.adam import (
    LearningState,
    init_adam,
)
from Variational_paraMetanet.MCMC_based.core.objective import (
    pure_update_step,
    log_joint_jit,
)
from Variational_paraMetanet.MCMC_based.sampling.hwg_met import (
    mcmc_advance,
)
from Variational_paraMetanet.MCMC_based.utils import visualization


def disturb_measurment(key: jax.Array, measurment: jax.Array):
    """Applies multiplicative noise to the simulation measurements."""
    ssq = 10e-2
    return jnp.exp(jax.random.normal(key, measurment.shape) * ssq) * measurment


def mcmc_draw(
    state: LearningState,
    key: jax.Array,
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

    log_post0 = log_joint_jit(
        state.theta,
        state.z,
        params_static,
        scales_scalar,
        init_net_state,
        traj_true,
        boundaries,
    )

    z_new, _, var_new, _ = mcmc_advance(
        z_init=state.z,
        log_post_init=log_post0,
        mcmc_variance=state.mcmc_variance,
        key=key,
        theta=state.theta,
        params_static=params_static,
        scales_scalar=scales_scalar,
        init_state=init_net_state,
        traj_true=traj_true,
        boundaries=boundaries,
        iterations=sampler_iters,
        adapt_alpha=adapt_alpha,
        adapt_beta=adapt_beta,
        num_blocks=num_blocks,
    )
    return z_new, var_new


def optimization_generator(
    theta_init: Theta,
    params_static: parametanet.ParaNetworkStaticParameters,
    scales_scalar: parametanet.ParaNetworkScalarParameters,
    init_net_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
    log_every: int = 10,
    gamma_0: float = 1e-4,
    adagrad_eps: float = 1e-7,
    num_blocks: int = 4,
    sampler_iters: int = 10,
    adapt_alpha: float = 1.01,
    mcmc_key: jax.Array = jax.random.PRNGKey(0),
):

    adapt_beta = float(
        (1.0 / adapt_alpha ** (100 * 0.45)) ** (1.0 / (100 - 100 * 0.45))
    )

    state = LearningState(
        theta=theta_init,
        z=theta_init.latent,
        adam=init_adam(theta_init),
        adam_eps=adagrad_eps,
        mcmc_variance=jnp.full((3, num_blocks), 0.05),
        k=jnp.array(1, dtype=jnp.int32),
    )

    latent_names = ["alpha", "critical_density", "free_flow_speed"]
    scalar_names = ["beta", "mu", "kappa", "gamma"]

    pbar = tqdm(itertools.count(), desc="Optimizing MCMC-AdaGrad")

    for epoch in pbar:
        epoch_key = jax.random.fold_in(mcmc_key, epoch)

        z_new, mcmc_var_new = mcmc_draw(
            state,
            epoch_key,
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

        state, gamma_jax, grad_theta = pure_update_step(
            state,
            z_new,
            mcmc_var_new,
            gamma_0,
            params_static,
            scales_scalar,
            init_net_state,
            traj_true,
            boundaries,
        )

        gamma = float(gamma_jax)
        pbar.set_description(f"Epoch: {epoch} | lr: {gamma:.4e}")

        if epoch % log_every == 0:
            current_log_joint = float(
                log_joint_jit(
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
                name: np.array(getattr(z_new, name)) for name in latent_names
            }
            cached_curr_s = {
                name: np.zeros_like(cached_curr_p[name]) for name in latent_names
            }

            # --- FIX: Append Scalar Parameters to the cache ---
            for name in scalar_names:
                cached_curr_p[name] = float(getattr(state.theta.scalar, name))
                cached_curr_s[name] = 0.0
            # --------------------------------------------------

            cached_curr_pr_p = {
                name: np.array(state.theta.prior.mean[i])
                for i, name in enumerate(latent_names)
            }
            cached_curr_pr_s = {
                name: float(jnp.exp(0.5 * state.theta.prior.log_var[i]))
                for i, name in enumerate(latent_names)
            }
            cached_curr_pr_corr = {
                name: float(jnp.tanh(state.theta.prior.corr)) for name in latent_names
            }

            # --- Cache the gradients ---
            cached_curr_grads = {
                name: np.array(getattr(grad_theta.latent, name))
                for name in latent_names
            }
            # Extract scalar gradients
            for field in scalar_names:
                cached_curr_grads[field] = float(getattr(grad_theta.scalar, field))

            # Extract prior and observation variance gradients
            cached_curr_grads["prior_mean"] = np.array(grad_theta.prior.mean)
            cached_curr_grads["prior_log_var"] = np.array(grad_theta.prior.log_var)
            cached_curr_grads["prior_corr"] = float(grad_theta.prior.corr)
            cached_curr_grads["obs_var_flow"] = float(grad_theta.obs_var.log_var_flow)
            cached_curr_grads["obs_var_speed"] = float(grad_theta.obs_var.log_var_speed)

            yield {
                "epoch": epoch,
                "loss": -current_log_joint,
                "ema_norm": 0.0,
                "lr": gamma,
                "params": cached_curr_p,
                "stds": cached_curr_s,
                "prior_params": cached_curr_pr_p,
                "prior_stds": cached_curr_pr_s,
                "prior_corr": cached_curr_pr_corr,
                "data_var_flow": float(jnp.exp(state.theta.obs_var.log_var_flow)),
                "data_var_speed": float(jnp.exp(state.theta.obs_var.log_var_speed)),
                "grads": cached_curr_grads,
            }


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
    key_lat, key_sca, key_pri = jax.random.split(keys[2], 3)
    alpha_uncon = to_unconstrained(full_p_true.latent_params.alpha, 1.0)
    rho_cr_uncon = to_unconstrained(full_p_true.latent_params.critical_density, 10.0)
    v_free_uncon = to_unconstrained(full_p_true.latent_params.free_flow_speed, 100.0)

    lat_noise = jax.random.normal(key_lat, (3, N)) * 1e-1
    disturbed_latent = parametanet.ParaNetworkLatentParameters(
        alpha=alpha_uncon + lat_noise[0],
        critical_density=rho_cr_uncon + lat_noise[1],
        free_flow_speed=v_free_uncon + lat_noise[2],
    )

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

    sca_noise = jax.random.normal(key_sca, (4,)) * 1e-2
    disturbed_scalar = parametanet.ParaNetworkScalarParameters(
        beta=beta_uncon + sca_noise[0],
        mu=mu_uncon + sca_noise[1],
        kappa=kappa_uncon,
        gamma=gamma_uncon + sca_noise[3],
    )

    # ---------------------------------------------------------
    # INITIALIZATION & EXECUTION
    # ---------------------------------------------------------
    sca_pri = jax.random.normal(key_pri, (3,)) * 10e-2
    theta_init = Theta(
        latent=disturbed_latent,
        scalar=disturbed_scalar,
        prior=PriorParameters(
            mean=jnp.array([1.62 + sca_pri[0], 3.16 + sca_pri[1], 0.84 + sca_pri[2]]),
            log_var=jnp.log(jnp.array([0.005, 0.005, 0.005])),
            corr=jnp.arctanh(jnp.array(0.95)),
        ),
        obs_var=DataLogVariance(
            log_var_flow=jnp.log(jnp.array(0.01)),
            log_var_speed=jnp.log(jnp.array(0.01)),
        ),
    )
    print("letsgo")

    print("Starting MCMC Optimization Loop...")
    results = interruptible_list(
        optimization_generator(
            theta_init=theta_init,
            params_static=full_p_true.static_params,
            scales_scalar=full_p_true.scalar_params,
            init_net_state=init_net_state,
            traj_true=trj_dis,
            boundaries=boundaries,
            log_every=5,
            num_blocks=20,
            sampler_iters=1,
            gamma_0=1e-1,
        ),
        save_whole=True,
        callback_whole=lambda res: visualization.print_whole(
            res, p_true=full_p_true, block=True
        ),
    )

    print("Optimization finished. Generating visualizations...")
    visualization.print_whole(results, p_true=full_p_true, block=True)
    print("Saving optimization results to pickle...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pickle_filename = f"mcmc_results_{timestamp}.pkl"

    with open(pickle_filename, "wb") as f:
        pickle.dump(list(results), f)

    print(f"✅ Full results data successfully saved to {pickle_filename}")
