import os
import itertools

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"

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
    to_unconstrained,
)
from Variational_paraMetanet.MCMC_based.core.adagrad import (
    LearningState,
    init_adagrad,
)
from Variational_paraMetanet.MCMC_based.core.objective import (
    pure_update_step,
    log_joint_jit,
)
from Variational_paraMetanet.MCMC_based.sampling.hwg_met import (
    log_posterior,
    mcmc_advance,
)
from Variational_paraMetanet.MCMC_based.utils import visualization


def disturb_measurment(key: jax.Array, measurment: jax.Array):
    """Applies multiplicative noise to the simulation measurements."""
    ssq = 10e-2  # Adjust based on desired noise level (e.g., 10e-1 for higher noise)
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

    log_post0 = log_posterior(
        state.z,
        state.theta.scalar,
        state.theta.prior,
        params_static,
        scales_scalar,
        state.theta.obs_var,
        init_net_state,
        traj_true,
        boundaries,
    )

    z_new, _, var_new, _ = mcmc_advance(
        z_init=state.z,
        log_post_init=log_post0,
        mcmc_variance=state.mcmc_variance,
        key=key,
        params_fis=state.theta.scalar,
        params_lat=state.theta.prior,
        params_static=params_static,
        scales_scalar=scales_scalar,
        data_log_variance=state.theta.obs_var,
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
    k_pre: int = 1000,
    gamma_0: float = 1e-4,
    alpha_decay: float = 2.0 / 3.0,
    c_heat: float = 1e-3,
    adagrad_eps: float = 1e-8,
    num_blocks: int = 4,
    sampler_iters: int = 10,
    adapt_alpha: float = 1.01,
    patience_epochs: int = 5000,
    mcmc_key: jax.Array = jax.random.PRNGKey(0),
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
        k=jnp.array(0, dtype=jnp.int32),
        k_end_heat=jnp.array(0, dtype=jnp.int32),
        heat_ended=jnp.array(False),
        ema_norm=jnp.array(0.0),
    )

    latent_names = ["alpha", "critical_density", "free_flow_speed"]
    scalar_names = ["beta", "mu", "kappa", "gamma"]

    best_norm = float("inf")
    steps_without_improvement = 0

    pbar = tqdm(itertools.count(), desc="Optimizing MCMC-AdaGrad")

    for epoch in pbar:
        k = state.k
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

        state, norm_jax, ema_new_jax, gamma_jax = pure_update_step(
            state,
            z_new,
            mcmc_var_new,
            k_pre,
            gamma_0,
            alpha_decay,
            c_heat,
            params_static,
            scales_scalar,
            init_net_state,
            traj_true,
            boundaries,
        )

        norm = float(norm_jax)
        ema_new = float(ema_new_jax)
        gamma = float(gamma_jax)
        k_int = int(k)

        if k_int >= k_pre and not bool(state.heat_ended):
            if ema_new > best_norm:
                state = state._replace(
                    k_end_heat=jnp.array(k_int, dtype=jnp.int32),
                    heat_ended=jnp.array(True),
                )
                pbar.write(f"[SGD] Heating ended at k={k_int}. Starting cooling phase.")
            else:
                best_norm = ema_new

        if bool(state.heat_ended):
            if ema_new < best_norm:
                best_norm = ema_new
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

        phase = (
            "pre-heat"
            if k_int < k_pre
            else ("heat" if not bool(state.heat_ended) else "cool")
        )
        pbar.set_description(
            f"Phase: {phase} | ‖g‖ EMA: {ema_new:.4f} | Wait: {steps_without_improvement}/{patience_epochs}"
        )

        # Yield exactly the format the visualization logic expects
        if epoch % log_every == 0 or steps_without_improvement >= patience_epochs:
            # Calculate Joint Log Likelihood for diagnostic plotting
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

            # The MCMC optimizer only has MAP point estimates for theta (no variational stds)
            cached_curr_s = {
                name: np.zeros_like(cached_curr_p[name]) for name in latent_names
            }

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

    # full_p_true.latent_params / scalar_params are PHYSICAL values.
    # z / theta.latent / theta.scalar live in the unconstrained pre-softplus
    # space, so the true values must go through the inverse transform BEFORE
    # noise is added — otherwise softplus(physical) * scale blows up.
    alpha_uncon = to_unconstrained(full_p_true.latent_params.alpha, 1.0)
    rho_cr_uncon = to_unconstrained(full_p_true.latent_params.critical_density, 10.0)
    v_free_uncon = to_unconstrained(full_p_true.latent_params.free_flow_speed, 100.0)

    lat_noise = jax.random.normal(key_lat, (3, N)) * 0.5
    disturbed_latent = parametanet.ParaNetworkLatentParameters(
        alpha=alpha_uncon + lat_noise[0],
        critical_density=rho_cr_uncon + lat_noise[1],
        free_flow_speed=v_free_uncon + lat_noise[2],
    )

    # scales_scalar == full_p_true.scalar_params, so physical/scale == 1 for
    # every field at zero noise — this correctly recovers the true value.
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

    sca_noise = jax.random.normal(key_sca, (4,)) * 0.5
    disturbed_scalar = parametanet.ParaNetworkScalarParameters(
        beta=beta_uncon + sca_noise[0],
        mu=mu_uncon + sca_noise[1],
        kappa=kappa_uncon + sca_noise[2],
        gamma=gamma_uncon + sca_noise[3],
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
            corr=jnp.arctanh(jnp.array(0.95)),
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
            sampler_iters=1,
            patience_epochs=2000,
        ),
        save_whole=True,
        callback_whole=lambda res: visualization.print_whole(
            res, p_true=full_p_true, block=True
        ),
    )

    print("Optimization finished. Generating visualizations...")
    visualization.print_whole(results, p_true=full_p_true, block=True)
