import functools
import jax
import jax.numpy as jnp
import optax
import numpy as np
import tqdm
import itertools
from typing import Callable, Mapping, Any
import math


import parametanet
from inference import parameter_handling
from inference import svi_losses_us as svi_losses


def make_custom_schedule(
    min_lr: float = 1e-6,
    max_lr: float = 1e-3,
    warmup_end: float = 3000,
    decay_start: float = 10000,
    decay_rate: float = 0.51,
):
    def schedule(step):
        ratio = max_lr / min_lr
        warmup = min_lr * jnp.power(ratio, step / warmup_end)
        constant = max_lr
        decay = max_lr / jnp.power((step - decay_start + 1.0), decay_rate)
        return jnp.where(
            step < warmup_end, warmup, jnp.where(step < decay_start, constant, decay)
        )

    return schedule


def make_log_smoothed_schedule(
    initial_lr: float,
    log_offset: float = math.e,
) -> Callable[[Any], jnp.ndarray]:
    """
    Generates a learning rate schedule that applies logarithmic smoothing
    to a square root decay, ensuring an infinite sum and finite squared sum.
    """

    def schedule(step) -> jnp.ndarray:
        # Adding 1.0 to the step prevents division by zero in the square root.
        # Utilizing math.e as the default offset ensures that jnp.log(step + math.e)
        # evaluates to 1.0 at step 0, thereby returning the exact initial_lr.
        denominator = jnp.sqrt(step + 1.0) * jnp.log(step + log_offset)
        decay = initial_lr / denominator
        return decay

    return schedule


def build_multi_optimizer() -> optax.GradientTransformation:
    optimizers: Mapping[Any, optax.GradientTransformation] = {
        "variational": optax.adam(
            learning_rate=make_custom_schedule(warmup_end=8000, decay_start=15000)
        ),
        "fis": optax.adam(
            learning_rate=make_custom_schedule(
                warmup_end=10000, decay_start=30000, max_lr=1e-1
            )
        ),
        "prior": optax.adam(
            learning_rate=make_custom_schedule(warmup_end=10000, decay_start=70000)
        ),
        "log_var": optax.adam(learning_rate=make_custom_schedule(decay_start=70000)),
        "zero": optax.set_to_zero(),
    }

    def label_fn(
        params: parameter_handling.ExplicitLayerParameters,
    ) -> parameter_handling.ExplicitLayerParameters:
        # Reconstruct the ExplicitLayerParameters structure to apply granular labels
        return parameter_handling.ExplicitLayerParameters(
            variational_posterior=parameter_handling.VariationalPosteriorParameters(
                mean=jax.tree_util.tree_map(
                    lambda _: "variational", params.variational_posterior.mean
                ),
                log_var=jax.tree_util.tree_map(
                    lambda _: "variational", params.variational_posterior.log_var
                ),
            ),
            fis_parameters=jax.tree_util.tree_map(
                lambda _: "fis", params.fis_parameters
            ),
            prior=parameter_handling.PriorParameters(
                mean=jax.tree_util.tree_map(lambda _: "prior", params.prior.mean),
                log_var=jax.tree_util.tree_map(lambda _: "zero", params.prior.log_var),
                corr=jax.tree_util.tree_map(lambda _: "zero", params.prior.corr),
            ),
            data_log_variance=jax.tree_util.tree_map(
                lambda _: "zero", params.data_log_variance
            ),
        )

    return optax.multi_transform(optimizers, label_fn)


adam_optimizer = build_multi_optimizer()


def _svi_update_step_impl(
    optimizer: optax.GradientTransformation,
    params: parameter_handling.ExplicitLayerParameters,
    opt_state: optax.OptState,
    noise_batch: jax.Array,
    traj_true: parametanet.SimulationTrajectory,
    static_params: parametanet.ParaNetworkStaticParameters,
    init_state: parametanet.NetworkState,
    boundaries: parametanet.BoundarySequence,
    scales: parametanet.ParaNetworkParameters,
    scales_array: jax.Array,
) -> tuple[parameter_handling.ExplicitLayerParameters, optax.OptState, jax.Array]:
    def elbo_loss(p: parameter_handling.ExplicitLayerParameters) -> float:
        nll = svi_losses.expected_nll(
            p,
            noise_batch,
            traj_true,
            static_params,
            init_state,
            boundaries,
            scales,
            scales_array,
        )
        return nll + svi_losses.kl_divergence(p)

    loss, grads = jax.value_and_grad(elbo_loss)(params)
    updates, opt_state_new = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state_new, loss


# Bind the concrete optimizer instance; jax.jit sees a stable, hashable callable.
svi_update_step: Any = jax.jit(functools.partial(_svi_update_step_impl, adam_optimizer))


@jax.jit
def get_metrics(
    params: parameter_handling.ExplicitLayerParameters,
    scales: parametanet.ParaNetworkParameters,
    scales_array: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Return raw unconstrained mean and standard deviation for accurate bound propagation
    unconstrained_means = params.variational_posterior.mean
    unconstrained_stds = jnp.exp(0.5 * params.variational_posterior.log_var)

    unconstrained_prior_means = params.prior.mean
    unconstrained_prior_stds = jnp.exp(0.5 * params.prior.log_var)

    return (
        unconstrained_means,
        unconstrained_stds,
        unconstrained_prior_means,
        unconstrained_prior_stds,
    )


def optimization_generator(
    initial_params: parameter_handling.ExplicitLayerParameters,
    trj_dis: parametanet.SimulationTrajectory,
    static_params: parametanet.ParaNetworkStaticParameters,
    init_stat: parametanet.NetworkState,
    boundaries: parametanet.BoundarySequence,
    scales: parametanet.ParaNetworkParameters,
    scales_array: jax.Array,
    log_every: int = 10,
    num_mc_samples: int = 1,
    patience_epochs: int = 5000,
    tolerance: float = 1e-3,
    ema_alpha: float = 0.05,
):
    params = initial_params
    opt_state = adam_optimizer.init(initial_params)
    base_key = jax.random.PRNGKey(42)

    pbar = tqdm.tqdm(itertools.count(), desc="Optimizing SVI")
    latent_names = ["alpha", "critical_density", "free_flow_speed"]
    scalar_names = ["beta", "mu", "kappa", "gamma"]

    accumulated_resamples = 0
    prev_loss = float("inf")

    cached_curr_p: dict = {}
    cached_curr_s: dict = {}
    cached_curr_pr_p: dict = {}
    cached_curr_pr_s: dict = {}
    cached_curr_pr_corr: dict = {}
    cached_loss: float = float("inf")
    cached_var_flow: float = 0.0
    cached_var_speed: float = 0.0

    # Early stopping and smoothed loss variables
    smoothed_loss = None
    best_loss = float("inf")
    steps_without_improvement = 0
    chunk_losses = []

    for epoch in pbar:
        step_key = jax.random.fold_in(base_key, epoch)

        noise_batch = svi_losses.find_stable_noise_samples(
            step_key,
            jax.lax.stop_gradient(params),
            num_mc_samples=num_mc_samples,
        )

        params, opt_state, loss = svi_update_step(
            params,
            opt_state,
            noise_batch,
            trj_dis,
            static_params,
            init_stat,
            boundaries,
            scales,
            scales_array,
        )

        # Store loss for the chunk
        chunk_losses.append(float(loss))

        if epoch % log_every == 0:
            current_mean_loss = float(np.mean(chunk_losses))
            chunk_losses = []  # Reset chunk

            if smoothed_loss is None:
                smoothed_loss = current_mean_loss
            else:
                smoothed_loss = (ema_alpha * current_mean_loss) + (
                    (1 - ema_alpha) * smoothed_loss
                )

            if smoothed_loss < (best_loss - tolerance):
                best_loss = smoothed_loss
                steps_without_improvement = 0
            else:
                # Add log_every steps since we evaluate in chunks
                steps_without_improvement += log_every if epoch > 0 else 1

            loss_diff = abs(prev_loss - current_mean_loss)
            prev_loss = current_mean_loss
            cached_loss = current_mean_loss

            # <-- NEW: Display Beta, Smoothed Loss, and Patience in the progress bar
            pbar.set_description(
                f"Neg ELBO: {current_mean_loss:.2f} | Smth: {smoothed_loss:.2f} | Wait: {steps_without_improvement}/{patience_epochs}"
            )

            p_m, p_s, pr_m, pr_s = get_metrics(params, scales, scales_array)
            phys_corr = jax.nn.sigmoid(params.prior.corr) * 0.99

            cached_curr_p = {
                name: np.array(p_m[i]) for i, name in enumerate(latent_names)
            }
            cached_curr_s = {
                name: np.array(p_s[i]) for i, name in enumerate(latent_names)
            }
            cached_curr_pr_p = {
                name: float(pr_m[i]) for i, name in enumerate(latent_names)
            }
            cached_curr_pr_s = {
                name: float(pr_s[i]) for i, name in enumerate(latent_names)
            }
            cached_curr_pr_corr = {
                name: float(phys_corr[i]) for i, name in enumerate(latent_names)
            }

            for field in scalar_names:
                val = getattr(params.fis_parameters, field)
                phys_val = jax.nn.softplus(val) * getattr(scales.scalar_params, field)
                cached_curr_p[field] = np.array(phys_val)
                cached_curr_s[field] = np.zeros_like(cached_curr_p[field])
                cached_curr_pr_p[field] = None
                cached_curr_pr_s[field] = None
                cached_curr_pr_corr[field] = None

            # Map log-variance to physical variance for the dataset
            cached_var_flow = float(jnp.exp(params.data_log_variance.log_var_flow))
            cached_var_speed = float(jnp.exp(params.data_log_variance.log_var_speed))

            yield {
                "epoch": epoch,
                "loss": cached_loss,
                "smoothed_loss": smoothed_loss,
                "params": cached_curr_p,
                "stds": cached_curr_s,
                "prior_params": cached_curr_pr_p,
                "prior_stds": cached_curr_pr_s,
                "prior_corr": cached_curr_pr_corr,
                "data_var_flow": cached_var_flow,
                "data_var_speed": cached_var_speed,
            }

            if steps_without_improvement >= patience_epochs:
                print(
                    f"\nConverged early at epoch {epoch} (Smoothed ELBO: {smoothed_loss:.3f})"
                )
                break