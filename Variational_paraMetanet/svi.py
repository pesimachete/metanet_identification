import typing
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from interruptible_list import interruptible_list

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

import parametanet as parametanet
import parapersistentExitationSimulation as peSim


# ---------------------------------------------------------
# SVI Data Structures
# ---------------------------------------------------------
class DataLogVariance(typing.NamedTuple):
    var_flow: jax.Array
    var_speed: jax.Array


class PriorParameters(typing.NamedTuple):
    mean: jax.Array
    var: jax.Array
    corr: jax.Array


class VariationalPosteriorParameters(typing.NamedTuple):
    mean: jax.Array
    var: jax.Array


class ExplicitLayerParameters(typing.NamedTuple):
    fis_parameters: parametanet.ParaNetworkScalarParameters
    prior: PriorParameters
    variational_posterior: VariationalPosteriorParameters
    data_variance: DataLogVariance


# ---------------------------------------------------------
# Mathematical Helpers & Physical Mapping
# ---------------------------------------------------------
def inv_softplus(x, scale):
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
):
    z = mean_2d + jnp.exp(0.5 * log_var_2d) * noise_2d

    alpha_phys = (jax.nn.softplus(z[0]) + 1e-6) * scales_array[0]
    rho_cr_phys = (jax.nn.softplus(z[1]) + 1e-6) * scales_array[1]
    v_free_phys = (jax.nn.softplus(z[2]) + 1e-6) * scales_array[2]

    beta_phys = (
        jax.nn.softplus(latent_scalars.beta) + 1e-6
    ) * scales.scalar_params.beta
    mu_phys = (jax.nn.softplus(latent_scalars.mu) + 1e-6) * scales.scalar_params.mu
    kappa_phys = (
        jax.nn.softplus(latent_scalars.kappa) + 1e-6
    ) * scales.scalar_params.kappa
    gamma_phys = (
        jax.nn.softplus(latent_scalars.gamma) + 1e-6
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
):
    std_latent = jnp.exp(0.5 * log_var_2d)
    grad_f = jax.nn.sigmoid(mean_2d) * scales_array[:, None]
    return jnp.abs(grad_f) * std_latent


# ---------------------------------------------------------
# Safe Noise Search
# ---------------------------------------------------------
@jax.jit
def evaluate_noise_batch(
    noise_batch,
    frozen_params,
    static_params,
    init_state,
    boundaries,
    scales,
    scales_array,
):
    def run_sim(noise):
        candidate = map_to_physical_params(
            frozen_params.variational_posterior.mean,
            frozen_params.variational_posterior.log_var,
            noise,
            frozen_params.fis_parameters,
            static_params,
            scales,
            scales_array,
        )
        return parametanet.rollout_simulation(init_state, boundaries, candidate)

    sims = jax.vmap(run_sim)(noise_batch)
    leaves = jax.tree_util.tree_leaves(sims)
    bad_per_leaf = [
        jnp.any(jnp.isnan(l) | jnp.isinf(l), axis=tuple(range(1, l.ndim)))
        for l in leaves
    ]
    return jnp.any(jnp.stack(bad_per_leaf, axis=0), axis=0)


def find_stable_noise_samples(
    base_key,
    frozen_params,
    static_params,
    init_state,
    boundaries,
    scales,
    scales_array,
    mc_samples=3,
    max_attempts=50,
):
    all_indices = jnp.arange(mc_samples)

    def make_noise(attempt, indices):
        keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i * 1000 + attempt))(
            indices
        )
        return jax.vmap(
            lambda k: jax.random.normal(
                k, shape=frozen_params.variational_posterior.mean.shape
            )
        )(keys)

    noise_samples = make_noise(0, all_indices)
    bad_mask = evaluate_noise_batch(
        noise_samples,
        frozen_params,
        static_params,
        init_state,
        boundaries,
        scales,
        scales_array,
    )

    for attempt in range(1, max_attempts):
        if not np.any(bad_mask):
            break

        bad_indices = np.where(bad_mask)[0]
        new_noise = make_noise(attempt, bad_indices)
        new_mask = evaluate_noise_batch(
            new_noise,
            frozen_params,
            static_params,
            init_state,
            boundaries,
            scales,
            scales_array,
        )

        noise_samples = noise_samples.at[bad_indices].set(new_noise)
        bad_mask = bad_mask.at[bad_indices].set(new_mask)

    if np.any(bad_mask):
        print("Warning: Could not find completely stable noise. Loss might spike.")

    return noise_samples


# ---------------------------------------------------------
# SVI Core: Expected NLL & KL Divergence
# ---------------------------------------------------------
def expected_nll(
    params: ExplicitLayerParameters,
    safe_noise: jax.Array,
    traj_true,
    static_params,
    init_state,
    boundaries,
    scales,
    scales_array,
):
    candidate = map_to_physical_params(
        params.variational_posterior.mean,
        params.variational_posterior.log_var,
        safe_noise,
        params.fis_parameters,
        static_params,
        scales,
        scales_array,
    )
    sim = parametanet.rollout_simulation(init_state, boundaries, candidate)

    M = traj_true.speed.size
    r2_v = jnp.exp(params.data_log_variance.log_var_speed)
    r2_f = jnp.exp(params.data_log_variance.log_var_flow)

    nll_v = 0.5 * M * params.data_log_variance.log_var_speed + 0.5 * jnp.sum(
        (jnp.log(sim.speed + 1e-5) - jnp.log(traj_true.speed + 1e-5)) ** 2
    ) / (r2_v + 1e-7)
    nll_f = 0.5 * M * params.data_log_variance.log_var_flow + 0.5 * jnp.sum(
        (jnp.log(sim.flow + 1e-5) - jnp.log(traj_true.flow + 1e-5)) ** 2
    ) / (r2_f + 1e-7)

    return nll_v + nll_f


def kl_divergence(params: ExplicitLayerParameters) -> float:
    N = params.variational_posterior.mean.shape[1]
    var_prior = jnp.exp(params.prior.log_var)
    var_posterior = jnp.exp(params.variational_posterior.log_var)
    coeff = (var_prior * (1 - params.prior.corr**2)) ** (-1)

    tr = coeff * (
        var_posterior[:, 0]
        + var_posterior[:, -1]
        + (1 + params.prior.corr**2) * jnp.sum(var_posterior[:, 1:-1], axis=1)
    )
    e = params.variational_posterior.mean - params.prior.mean.reshape(-1, 1)
    einve = coeff * (
        e[:, 0] ** 2
        + e[:, -1] ** 2
        + (1 + params.prior.corr**2) * jnp.sum(e[:, 1:-1] ** 2, axis=1)
        - 2 * params.prior.corr * jnp.sum(e[:, :-1] * e[:, 1:], axis=1)
    )

    lsig2 = N * params.prior.log_var + (N - 1) * jnp.log(1 - params.prior.corr**2)
    lsig1 = jnp.sum(params.variational_posterior.log_var, axis=1)

    return jnp.sum(1 / 2 * (tr + einve - N + lsig2 - lsig1))


# ---------------------------------------------------------
# SVI Optimization Step (Adam)
# ---------------------------------------------------------
optimizer = optax.adam(learning_rate=5e-2)


@jax.jit
def svi_update_step(
    params: ExplicitLayerParameters,
    opt_state,
    safe_noise_batch,
    traj_true,
    static_params,
    init_state,
    boundaries,
    scales,
    scales_array,
):
    def elbo_loss(p):
        nlls = jax.vmap(
            lambda noise: expected_nll(
                p,
                noise,
                traj_true,
                static_params,
                init_state,
                boundaries,
                scales,
                scales_array,
            )
        )(safe_noise_batch)
        expected_nll_val = jnp.mean(nlls)
        kl_val = kl_divergence(p)
        return expected_nll_val + kl_val

    loss, grads = jax.value_and_grad(elbo_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


# ---------------------------------------------------------
# Main Generator and Execution
# ---------------------------------------------------------
def optimization_generator(
    initial_params,
    opt_state,
    base_key,
    trj_dis,
    static_params,
    init_stat,
    boundaries,
    scales,
    scales_array,
):
    params = initial_params
    prev_loss = jnp.float64("inf")
    loss_memo = 0.0
    mem_loss = 0.8

    pbar = tqdm.tqdm(itertools.count(), desc="Optimizing SVI")
    latent_names = ["alpha", "critical_density", "free_flow_speed"]
    scalar_names = ["beta", "mu", "kappa", "gamma"]

    for epoch in pbar:
        step_key = jax.random.fold_in(base_key, epoch)
        safe_noise_batch = find_stable_noise_samples(
            step_key,
            jax.lax.stop_gradient(params),
            static_params,
            init_stat,
            boundaries,
            scales,
            scales_array,
            mc_samples=10,
        )

        params, opt_state, loss = svi_update_step(
            params,
            opt_state,
            safe_noise_batch,
            trj_dis,
            static_params,
            init_stat,
            boundaries,
            scales,
            scales_array,
        )

        current_loss = float(loss)
        loss_diff = abs(prev_loss - current_loss)
        if loss_diff < float("inf"):
            loss_memo = loss_diff + mem_loss * loss_memo
        prev_loss = current_loss

        # Calculate Posteriors
        physical_means = (
            jax.nn.softplus(params.variational_posterior.mean) * scales_array[:, None]
        )
        physical_stds = get_physical_std_2d(
            params.variational_posterior.mean,
            params.variational_posterior.log_var,
            scales_array,
        )

        # Calculate Priors (FIXED: Correct 1D array extraction for the global prior)
        physical_prior_means = jax.nn.softplus(params.prior.mean) * scales_array
        physical_prior_stds = jnp.abs(
            jax.nn.sigmoid(params.prior.mean) * scales_array
        ) * jnp.exp(0.5 * params.prior.log_var)

        current_params = {
            name: np.array(physical_means[i]) for i, name in enumerate(latent_names)
        }
        current_stds = {
            name: np.array(physical_stds[i]) for i, name in enumerate(latent_names)
        }

        # Save priors as single floats representing the global values
        current_prior_params = {
            name: float(physical_prior_means[i]) for i, name in enumerate(latent_names)
        }
        current_prior_stds = {
            name: float(physical_prior_stds[i]) for i, name in enumerate(latent_names)
        }

        # Add scalars (No prior exists for scalars)
        for field in scalar_names:
            val = getattr(params.fis_parameters, field)
            phys_val = jax.nn.softplus(val) * getattr(scales.scalar_params, field)
            current_params[field] = np.array(phys_val)
            current_stds[field] = np.zeros_like(current_params[field])
            current_prior_params[field] = None
            current_prior_stds[field] = None

        if epoch % 50 == 0:
            pbar.set_description(
                f"Neg ELBO: {current_loss:.4f} | ΔLoss: {loss_diff:.4e}"
            )

        yield {
            "epoch": epoch,
            "loss": current_loss,
            "params": current_params,
            "stds": current_stds,
            "prior_params": current_prior_params,
            "prior_stds": current_prior_stds,
        }


def print_whole(results, p_true=None):
    if not results:
        print("No data collected.")
        return

    nll_loss_history = [step["loss"] for step in results]
    learnable_fields = list(results[0]["params"].keys())

    param_histories = {
        field: np.array([step["params"][field] for step in results])
        for field in learnable_fields
    }
    std_histories = {
        field: np.array([step["stds"][field] for step in results])
        for field in learnable_fields
    }

    prior_param_histories = {
        field: np.array([step["prior_params"][field] for step in results])
        for field in learnable_fields
        if results[0]["prior_params"][field] is not None
    }
    prior_std_histories = {
        field: np.array([step["prior_stds"][field] for step in results])
        for field in learnable_fields
        if results[0]["prior_stds"][field] is not None
    }

    cols = 3
    rows = (len(learnable_fields) + 1) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i, field in enumerate(learnable_fields):
        ax = axes[i]
        history = param_histories[field]
        stds = std_histories[field]
        epochs = np.arange(len(history))

        if history.ndim == 2:  # Spatial Parameters (Vector)
            for segment_idx in range(history.shape[1]):
                # 1. Plot Posterior (Solid Line & Darker Shade)
                (post_line,) = ax.plot(
                    epochs,
                    history[:, segment_idx],
                    alpha=0.9,
                    linewidth=1.5,
                    label="Posterior" if segment_idx == 0 else "",
                )
                ax.fill_between(
                    epochs,
                    history[:, segment_idx] - stds[:, segment_idx],
                    history[:, segment_idx] + stds[:, segment_idx],
                    color=post_line.get_color(),
                    alpha=0.25,
                )

                # 2. True Values
                if p_true is not None and hasattr(p_true.latent_params, field):
                    ax.axhline(
                        y=np.array(getattr(p_true.latent_params, field))[segment_idx],
                        color=post_line.get_color(),
                        linestyle=":",
                        alpha=0.8,
                    )

            # 3. Plot Global Prior ONCE (FIXED)
            if field in prior_param_histories:
                prior_hist = prior_param_histories[
                    field
                ]  # Now 1D array of shape (epochs,)
                prior_std = prior_std_histories[field]
                (prior_line,) = ax.plot(
                    epochs,
                    prior_hist,
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                    color="black",
                    label="Global Prior",
                )
                ax.fill_between(
                    epochs,
                    prior_hist - prior_std,
                    prior_hist + prior_std,
                    color="black",
                    alpha=0.1,
                )

            # Prevent massive duplicate legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        else:  # Scalar Parameters (No Prior)
            (line,) = ax.plot(
                epochs, history, color="blue", linewidth=2, label="Estimated Point"
            )
            if p_true is not None and hasattr(p_true.scalar_params, field):
                ax.axhline(
                    y=float(getattr(p_true.scalar_params, field)),
                    color="red",
                    linestyle="--",
                    label="True Value",
                )
            ax.legend()

        ax.set_title(f"Parameter: {field}")
        ax.set_xlabel("Epochs")
        ax.grid(True, linestyle=":", alpha=0.5)

    ax_loss = axes[len(learnable_fields)]
    ax_loss.plot(nll_loss_history, color="black", linewidth=1.5)
    ax_loss.set_title("Optimization: Negative ELBO")
    ax_loss.set_xlabel("Epochs")
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    for j in range(len(learnable_fields) + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def perturb_and_inv_softplus(key, true_val, scale, d=0.4):
    perturbed = jax.random.uniform(
        key, jnp.shape(true_val), minval=(1 - d) * true_val, maxval=(1 + d) * true_val
    )
    return inv_softplus(perturbed, scale).astype(jnp.float64)


if __name__ == "__main__":
    traj_true, full_p_true, boundaries, init_stat = peSim.simulate_example()
    ssq = 10e-3
    keys = jax.random.split(jax.random.PRNGKey(1131), 2)

    new_flow = (
        jnp.mean(traj_true.flow)
        * jax.random.t(keys[0], df=15, shape=traj_true.flow.shape)
        * ssq
        + traj_true.flow
    )
    new_speed = (
        jnp.mean(traj_true.speed)
        * jax.random.t(keys[1], df=15, shape=traj_true.speed.shape)
        * ssq
        + traj_true.speed
    )
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
            alpha=10 ** jnp.floor(jnp.log10(jnp.mean(full_p_true.latent_params.alpha))),
            critical_density=10
            ** jnp.floor(
                jnp.log10(jnp.mean(full_p_true.latent_params.critical_density))
            ),
            free_flow_speed=10
            ** jnp.floor(
                jnp.log10(jnp.mean(full_p_true.latent_params.free_flow_speed))
            ),
        ),
    )
    scales_array = jnp.array(
        [
            scales.latent_params.alpha,
            scales.latent_params.critical_density,
            scales.latent_params.free_flow_speed,
        ]
    )

    print("Searching for a stable initial parameter seed...")
    seed_offset = 0
    stable_params_found = False
    learnable_params = None

    while not stable_params_found:
        current_seed = 2447 + seed_offset
        init_keys = jax.random.split(jax.random.PRNGKey(current_seed), 7)

        alpha_lat = perturb_and_inv_softplus(
            init_keys[0], full_p_true.latent_params.alpha, scales.latent_params.alpha
        )
        rho_cr_lat = perturb_and_inv_softplus(
            init_keys[1],
            full_p_true.latent_params.critical_density,
            scales.latent_params.critical_density,
        )
        v_free_lat = perturb_and_inv_softplus(
            init_keys[2],
            full_p_true.latent_params.free_flow_speed,
            scales.latent_params.free_flow_speed,
        )
        initial_mean_2d = jnp.array([alpha_lat, rho_cr_lat, v_free_lat])

        latent_scalars = parametanet.ParaNetworkScalarParameters(
            beta=perturb_and_inv_softplus(
                init_keys[3], full_p_true.scalar_params.beta, scales.scalar_params.beta
            ),
            mu=perturb_and_inv_softplus(
                init_keys[4], full_p_true.scalar_params.mu, scales.scalar_params.mu
            ),
            kappa=perturb_and_inv_softplus(
                init_keys[5],
                full_p_true.scalar_params.kappa,
                scales.scalar_params.kappa,
            ),
            gamma=perturb_and_inv_softplus(
                init_keys[6],
                full_p_true.scalar_params.gamma,
                scales.scalar_params.gamma,
            ),
        )

        test_params = ExplicitLayerParameters(
            fis_parameters=latent_scalars,
            prior=PriorParameters(
                mean=jnp.array(
                    [jnp.mean(alpha_lat), jnp.mean(rho_cr_lat), jnp.mean(v_free_lat)]
                ),
                log_var=jnp.array(
                    [
                        jnp.log(1e-4) * jnp.mean(alpha_lat),
                        jnp.log(1e-4) * jnp.mean(rho_cr_lat),
                        jnp.log(1e-4) * jnp.mean(v_free_lat),
                    ]
                ),
                corr=jnp.array([0.8, 0.8, 0.8]),
            ),
            variational_posterior=VariationalPosteriorParameters(
                mean=initial_mean_2d,
                log_var=jnp.full_like(initial_mean_2d, jnp.log(1e-1)),
            ),
            data_log_variance=DataLogVariance(
                log_var_flow=jnp.log(1.0), log_var_speed=jnp.log(1.0)
            ),
        )

        test_noise = jnp.zeros_like(initial_mean_2d)
        test_loss = expected_nll(
            test_params,
            test_noise,
            trj_dis,
            full_p_true.static_params,
            init_stat,
            boundaries,
            scales,
            scales_array,
        )

        if not jnp.isnan(test_loss) and not jnp.isinf(test_loss):
            print(
                f"-> Found stable initialization at seed {current_seed}. Loss: {test_loss:.2f}"
            )
            learnable_params = test_params
            stable_params_found = True
        else:
            seed_offset += 13

    # 4. Setup Optimizer
    opt_state = optimizer.init(learnable_params)
    base_key = jax.random.PRNGKey(42)

    # 5. Run SVI Loop
    results = interruptible_list(
        optimization_generator(
            learnable_params,
            opt_state,
            base_key,
            trj_dis,
            full_p_true.static_params,
            init_stat,
            boundaries,
            scales,
            scales_array,
        ),
        save_whole=True,
        callback_whole=lambda res: print_whole(res, p_true=full_p_true),
    )

    print_whole(results, p_true=full_p_true)
