import typing
import matplotlib.pyplot as plt
import itertools
import numpy as np
import tqdm
import os
import json
from interruptible_list import interruptible_list

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax


import parametanet
import parasimulationMetanet as peSim


print(jax.devices())
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class LearningParams(typing.NamedTuple):
    metanet: parametanet.ParaNetworkParameters
    log_vars: jax.Array


def inv_softplus(x, scale):
    def logexpm1(x):
        return x + jnp.log(-jnp.expm1(-x))

    return logexpm1(x / scale)


def get_physical_params(
    raw_metanet_params: parametanet.ParaNetworkParameters,
    scales: parametanet.ParaNetworkParameters,
):
    return parametanet.ParaNetworkParameters(
        T=raw_metanet_params.T,
        L=raw_metanet_params.L,
        lambda_=raw_metanet_params.lambda_,
        beta=jax.nn.softplus(raw_metanet_params.beta) * scales.beta,
        mu=jax.nn.softplus(raw_metanet_params.mu) * scales.mu,
        kappa=jax.nn.softplus(raw_metanet_params.kappa) * scales.kappa,
        gamma=jax.nn.softplus(raw_metanet_params.gamma) * scales.gamma,
        alpha=jax.nn.softplus(raw_metanet_params.alpha) * scales.alpha,
        critical_density=jax.nn.softplus(raw_metanet_params.critical_density)
        * scales.critical_density,
        free_flow_speed=jax.nn.softplus(raw_metanet_params.free_flow_speed)
        * scales.free_flow_speed,
    )


def nll_loss(
    params: LearningParams,
    traj_true: parametanet.SimulationTrajectory,
    initial_state: parametanet.NetworkState,
    boundaries: parametanet.BoundarySequence,
    scales: parametanet.ParaNetworkParameters,
    penalty_weight: float = 0.5,
):

    physical_metanet = get_physical_params(params.metanet, scales)

    traj_pred = parametanet.rollout_simulation(
        initial_state, boundaries, physical_metanet
    )

    var_q = jnp.exp(params.log_vars[0])
    var_v = jnp.exp(params.log_vars[1])

    n = traj_pred.flow.shape[0]

    loss_q = 0.5 * n * params.log_vars[0] + 0.5 * jnp.sum(
        (jnp.log(traj_pred.flow) - jnp.log(traj_true.flow)) ** 2
    ) / (var_q + 1e-7)
    loss_v = 0.5 * n * params.log_vars[1] + 0.5 * jnp.sum(
        (jnp.log(traj_pred.speed) - jnp.log(traj_true.speed)) ** 2
    ) / (var_v + 1e-7)

    base_loss = loss_q + loss_v

    reg_loss = 0.0

    empirical_fields = [
        "alpha",
        "critical_density",
        "free_flow_speed",
    ]

    """  
    "tau",
    "nu",
    "kappa",
    "delta",
    """

    for field in physical_metanet._fields:
        if field in empirical_fields:
            p_pred = getattr(physical_metanet, field)
            p_diff = p_pred[:-1] - p_pred[1:]
            p_norm = 0.5 * (jnp.abs(p_pred[:-1] + p_pred[1:]) + 1e-8)
            reg_loss += jnp.sum((p_diff / p_norm) ** 2)

    return base_loss + 1 / 2 * (penalty_weight * reg_loss)


traj_true, p_true, boundaries, init_stat = peSim.simulate_example()


ssq = 10e-2


def disturb_measurment(key, measurment):
    return jnp.exp(jax.random.normal(key, measurment.shape) * ssq) * measurment


keys = jax.random.split(jax.random.PRNGKey(1129), 2)
new_flow = disturb_measurment(keys[0], traj_true.flow)
new_speed = disturb_measurment(keys[1], traj_true.speed)
new_density = new_flow / new_speed * p_true.lambda_
trj_dis = parametanet.SimulationTrajectory(
    density=new_density,
    speed=new_speed,
    flow=new_flow,
)

K = traj_true.density.shape[0]

d = 0.6


def perturb_and_inv_softplus_vec(key, true_val, scale, d=0.6):
    perturbed = jax.random.uniform(
        key,
        true_val.shape,
        minval=(1 - d) * true_val,
        maxval=(1 + d) * true_val,
    )
    return inv_softplus(perturbed, scale)


def perturb_and_inv_softplus_float(key, true_val, scale, d=0.3):
    perturbed = jax.random.uniform(
        key,
        (),
        minval=(1 - d) * true_val,
        maxval=(1 + d) * true_val,
    ).astype(jnp.float64)
    return inv_softplus(perturbed, scale)


keys = jax.random.split(jax.random.PRNGKey(2441), 7)
scales = parametanet.ParaNetworkParameters(
    beta=10 ** (jnp.floor(jnp.log10(p_true.beta))),
    free_flow_speed=10 ** (jnp.floor(jnp.log10(p_true.free_flow_speed))),
    kappa=10 ** (jnp.floor(jnp.log10(p_true.kappa))),
    gamma=10 ** (jnp.floor(jnp.log10(p_true.gamma))),
    critical_density=10 ** (jnp.floor(jnp.log10(p_true.critical_density))),
    alpha=10 ** (jnp.floor(jnp.log10(p_true.alpha))),
    mu=10 ** (jnp.floor(jnp.log10(p_true.mu))),
    L=1.0,
    lambda_=1.0,
    T=1.0,
)


def initialize_params(p_true, scales, keys):
    return parametanet.ParaNetworkParameters(
        beta=perturb_and_inv_softplus_float(keys[0], p_true.beta, scales.beta),
        free_flow_speed=perturb_and_inv_softplus_vec(
            keys[1], p_true.free_flow_speed, scales.free_flow_speed
        ),
        gamma=perturb_and_inv_softplus_float(keys[2], p_true.gamma, scales.gamma),
        mu=perturb_and_inv_softplus_float(keys[3], p_true.mu, scales.mu),
        critical_density=perturb_and_inv_softplus_vec(
            keys[4], p_true.critical_density, scales.critical_density
        ),
        alpha=perturb_and_inv_softplus_vec(keys[5], p_true.alpha, scales.alpha),
        kappa=perturb_and_inv_softplus_float(keys[6], p_true.kappa, scales.kappa),
        L=p_true.L,  # FROZEN
        lambda_=p_true.lambda_,  # FROZEN
        T=p_true.T,  # FROZEN
    )


initial_metanet_params = initialize_params(p_true, scales, keys)

frac = jnp.sum(traj_true.flow) / jnp.sum(
    traj_true.speed
)  # This does not work in this fake data.

initial_params = LearningParams(
    metanet=initial_metanet_params,
    log_vars=jnp.array([jnp.log(1), jnp.log(1 * frac)]),
)

mask = LearningParams(
    metanet=parametanet.ParaNetworkParameters(
        beta=1,
        free_flow_speed=jnp.ones_like(p_true.free_flow_speed),
        kappa=1,
        mu=1,
        critical_density=jnp.ones_like(p_true.critical_density),
        alpha=jnp.ones_like(p_true.alpha),
        gamma=1,
        L=jnp.zeros_like(p_true.L),  # FROZEN
        lambda_=jnp.zeros_like(p_true.lambda_),  # FROZEN
        T=0,  # FROZEN
    ),
    log_vars=jnp.ones_like(initial_params.log_vars),
)


learning_rate = 1e-4
optimizer = optax.adam(learning_rate)

opt_state = optimizer.init(initial_params)


# Define a JIT-compiled update step for speed
@jax.jit
def update_step(params, opt_state, traj_true, initial_state, boundaries, scales, mask):

    penalty_weight = 10

    loss, grads = jax.value_and_grad(nll_loss)(
        params, traj_true, initial_state, boundaries, scales, penalty_weight
    )

    grads = jax.tree_util.tree_map(
        lambda g, m: jnp.where(m == 1.0, g, 0.0), grads, mask
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, loss


params = initial_params


prev_loss = jnp.float64("inf")


def optimization_generator(
    params, opt_state, trj_dis, init_stat, boundaries, scales, mask
):
    prev_loss = jnp.float64("inf")
    loss_memo = 0.0
    mem_loss = 0.9

    pbar = tqdm.tqdm(itertools.count(), desc="Optimizing")

    for epoch in pbar:
        # 1. Perform optimization step
        params, opt_state, loss = update_step(
            params, opt_state, trj_dis, init_stat, boundaries, scales, mask
        )
        current_loss = float(loss)

        # 2. Check for convergence
        loss_diff = abs(prev_loss - current_loss)
        if loss_diff < float("inf"):
            loss_memo = loss_diff + mem_loss * loss_memo

        prev_loss = current_loss

        # 3. Calculate physical parameters
        physical_metanet_current = get_physical_params(params.metanet, scales)

        # 4. Extract trackable parameters to save
        current_params = {}
        for field in physical_metanet_current._fields:
            if field not in ["L", "lambda_", "T"]:  # Only track learnable parameters
                current_params[field] = np.array(
                    getattr(physical_metanet_current, field)
                )

        # 5. Update progress bar UI occasionally
        if epoch % 50 == 0:
            var_rho = jnp.exp(params.log_vars[0])
            var_v = jnp.exp(params.log_vars[1])
            pbar.set_description(
                f"Loss: {current_loss:.4f} | ΔLoss: {loss_diff:.4e} | Loss Memo: {loss_memo:.4e}"
            )

        yield {"epoch": epoch, "loss": current_loss, "params": current_params}


def print_and_save_summary(
    params, scales, p_true=None, output_filename="id_params/identified_parameters.json"
):
    """
    Prints a comparison of learned vs true parameters and saves the final
    identified parameters (including frozen ones and variances) to a JSON file.

    Args:
        params: The final learned JAX parameters object.
        scales: The scales used for get_physical_params.
        p_true: (Optional) The ground truth parameter object for comparison.
        output_filename: (str) Path to save the JSON output.
    """
    print("\nIdentified METANET Parameters vs True Parameters:")

    # Recompute the final physical parameters from the current 'params'
    physical_metanet_final = get_physical_params(params.metanet, scales)

    params_to_save = {}

    # 1. Print and store physical parameters
    for field in physical_metanet_final._fields:
        learned_val = getattr(physical_metanet_final, field)
        frozen_str = "(FROZEN)" if field in ["L", "lambda_", "T"] else ""

        # Format for JSON
        if hasattr(learned_val, "tolist"):
            params_to_save[field] = learned_val.tolist()
        else:
            params_to_save[field] = float(learned_val)

        # Format for Printing
        if jnp.size(learned_val) == 1:
            l_str = f"{float(jnp.squeeze(learned_val)):.4f}"
        else:
            with np.printoptions(precision=4, suppress=True, edgeitems=2, threshold=5):
                l_str = f"{np.array(learned_val)}"

        # Handle p_true comparison safely
        if p_true is not None and hasattr(p_true, field):
            true_val = getattr(p_true, field)
            if jnp.size(true_val) == 1:
                t_str = f"{float(jnp.squeeze(true_val)):.4f}"
            else:
                with np.printoptions(
                    precision=4, suppress=True, edgeitems=2, threshold=5
                ):
                    t_str = f"{np.array(true_val)}"

            print(f"  {field:>16}: {l_str} \t(True: {t_str}) {frozen_str}")
        else:
            # Fallback if p_true isn't provided
            print(f"  {field:>16}: {l_str} \t{frozen_str}")

    # 2. Print and store learned error covariances (variances)
    print("\nLearned Error Covariances (Variances):")
    var_rho = float(jnp.exp(params.log_vars[0]))
    var_v = float(jnp.exp(params.log_vars[1]))

    print(f"  Density (rho) variance : {var_rho:.4f}")
    print(f"  Speed (v) variance     : {var_v:.4f}")

    params_to_save["learned_variance_rho"] = var_rho
    params_to_save["learned_variance_v"] = var_v

    # 3. Save to JSON
    print("\nSaving parameters to JSON...")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, "w") as f:
        json.dump(params_to_save, f, indent=4)

    print(f"Parameters successfully saved to '{output_filename}'.")


def print_whole(results):
    """
    Callback function to unpack a list of state dictionaries and plot the results.

    Args:
        results (list of dicts): The list yielded by the optimization generator.
        p_true (object, optional): Ground truth parameters for plotting reference lines.
    """
    if not results:
        print("No data was collected to plot.")
        return

    print("\nGenerating plots from collected data...")

    # 1. Unpack the history structures from the yielded results
    nll_loss_history = [step["loss"] for step in results]

    param_histories = {}
    learnable_fields = list(results[0]["params"].keys())

    for field in learnable_fields:
        param_histories[field] = [step["params"][field] for step in results]

    num_params = len(learnable_fields)

    # 2. Set up the figure grid
    cols = 3
    rows = (num_params + 1) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))

    # Safeguard for a single subplot case
    if rows * cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # 3. Plot each parameter
    for i, field in enumerate(learnable_fields):
        ax = axes[i]

        # Convert list of arrays to a 2D numpy array: Shape (Epochs, N)
        history = np.array(param_histories[field])

        if history.ndim == 2:  # It's a vector (e.g., N=20)
            lines = ax.plot(history, alpha=0.6, linewidth=1)

            # Plot True Values if p_true is provided and has this field
            if p_true is not None and hasattr(p_true, field):
                true_val = np.array(getattr(p_true, field))
                for segment_idx in range(history.shape[1]):
                    ax.axhline(
                        y=true_val[segment_idx],
                        color=lines[segment_idx].get_color(),
                        linestyle="--",
                        alpha=0.4,
                    )
            ax.set_title(f"Parameter: {field}")

        else:  # It's a scalar
            ax.plot(history, color="blue", linewidth=2, label="Estimated")

            # Plot True Value if p_true is provided and has this field
            if p_true is not None and hasattr(p_true, field):
                true_val = float(getattr(p_true, field))
                ax.axhline(y=true_val, color="red", linestyle="--", label="True Value")
                ax.legend()

            ax.set_title(f"Parameter: {field}")

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Value")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # 4. Place the NLL Loss in the final slot
    ax_loss = axes[num_params]
    ax_loss.plot(nll_loss_history, color="black", linewidth=1.5)
    ax_loss.set_title("Optimization: NLL Loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, which="both", linestyle="--", alpha=0.5)

    # 5. Clean up empty slots
    for j in range(num_params + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


results = interruptible_list(
    optimization_generator(
        params, opt_state, trj_dis, init_stat, boundaries, scales, mask
    ),
    save_whole=True,
    callback_whole=print_whole,
)

# 2. Plot the trajectory
print_whole(results)

# 3. Print summaries and save to JSON
print_and_save_summary(params, scales, p_true=p_true)
