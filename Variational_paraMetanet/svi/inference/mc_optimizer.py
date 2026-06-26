import functools
import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Callable, Mapping, Any

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
                lambda _: "log_var", params.data_log_variance
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


@functools.partial(jax.jit, static_argnames=["chunk_size", "num_mc_samples"])
def train_chunk_svi(
    params,
    opt_state,
    start_epoch,
    base_key,
    traj_true,
    static_params,
    init_state,
    boundaries,
    scales,
    scales_array,
    chunk_size,
    num_mc_samples,
):
    def step(carry, xs):
        p, state = carry
        epoch = start_epoch + xs
        step_key = jax.random.fold_in(base_key, epoch)

        noise_batch = svi_losses.find_stable_noise_samples(
            step_key, jax.lax.stop_gradient(p), num_mc_samples=num_mc_samples
        )

        new_p, new_state, loss = _svi_update_step_impl(
            adam_optimizer,
            p,
            state,
            noise_batch,
            traj_true,
            static_params,
            init_state,
            boundaries,
            scales,
            scales_array,
        )
        return (new_p, new_state), (loss, new_p)

    carry_final, (losses, params_history) = jax.lax.scan(
        step, (params, opt_state), jnp.arange(chunk_size)
    )
    return carry_final, losses, params_history


def optimize_svi(
    initial_params: parameter_handling.ExplicitLayerParameters,
    trj_dis: parametanet.SimulationTrajectory,
    static_params: parametanet.ParaNetworkStaticParameters,
    init_stat: parametanet.NetworkState,
    boundaries: parametanet.BoundarySequence,
    scales: parametanet.ParaNetworkParameters,
    scales_array: jax.Array,
    mc_run_id: int,
    max_iterations: int = 150000,
    chunk_size: int = 500,
    num_mc_samples: int = 1,
    patience_epochs: int = 5000,
    tolerance: float = 1e-4,
):
    params = initial_params
    opt_state = adam_optimizer.init(initial_params)
    base_key = jax.random.PRNGKey(42 + mc_run_id)

    num_chunks = max_iterations // chunk_size

    smoothed_loss = None
    ema_alpha = 0.1
    best_loss = float("inf")
    steps_without_improvement = 0
    converged_epoch = max_iterations

    loss_history_all = []
    param_history_all = {
        "mean": [],
        "log_var": [],
        "fis_scalars": [],
        "var_flow": [],
        "var_speed": [],  # <--- NEW: Tracking Variances
    }

    for chunk_idx in range(num_chunks):
        start_epoch = chunk_idx * chunk_size

        (params, opt_state), losses, params_hist = train_chunk_svi(
            params,
            opt_state,
            start_epoch,
            base_key,
            trj_dis,
            static_params,
            init_stat,
            boundaries,
            scales,
            scales_array,
            chunk_size,
            num_mc_samples,
        )

        losses_np = np.array(jax.device_get(losses))
        loss_history_all.append(losses_np)

        param_history_all["mean"].append(
            np.array(jax.device_get(params_hist.variational_posterior.mean))
        )
        param_history_all["log_var"].append(
            np.array(jax.device_get(params_hist.variational_posterior.log_var))
        )
        param_history_all["var_flow"].append(
            np.array(jax.device_get(params_hist.data_log_variance.log_var_flow))
        )
        param_history_all["var_speed"].append(
            np.array(jax.device_get(params_hist.data_log_variance.log_var_speed))
        )

        param_history_all["fis_scalars"].append(
            {
                k: np.array(jax.device_get(getattr(params_hist.fis_parameters, k)))
                for k in params_hist.fis_parameters._fields
            }
        )

        current_mean_loss = float(np.mean(losses_np))
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
            steps_without_improvement += chunk_size

        if steps_without_improvement >= patience_epochs:
            converged_epoch = (chunk_idx + 1) * chunk_size
            print(
                f"MC Run {mc_run_id}: Converged early at epoch {converged_epoch} (Smoothed ELBO: {smoothed_loss:.3f})"
            )
            break

    final_loss_history = np.concatenate(loss_history_all)

    flat_param_history = {
        "mean": np.concatenate(param_history_all["mean"], axis=0),
        "log_var": np.concatenate(param_history_all["log_var"], axis=0),
        "fis_scalars": {
            k: np.concatenate(
                [chunk[k] for chunk in param_history_all["fis_scalars"]], axis=0
            )
            for k in param_history_all["fis_scalars"][0].keys()
        },
        # Convert from log-variance to true variance here
        "variance_flow": np.exp(np.concatenate(param_history_all["var_flow"], axis=0)),
        "variance_speed": np.exp(
            np.concatenate(param_history_all["var_speed"], axis=0)
        ),
    }

    return params, final_loss_history, flat_param_history, converged_epoch
