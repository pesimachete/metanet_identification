import typing
import os
import json
import traceback
import multiprocessing as mp
import numpy as np
import matplotlib
import functools
from collections import deque

matplotlib.use("Agg")  # Safe backend for non-GUI environments
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Environment & JAX Setup (Force CPU & Prevent Thread Collisions)
# ---------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import optax
import parametanet
import parapersistentExitationSimulation as peSim

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------
# Data Structures
# ---------------------------------------------------------
class LearningParams(typing.NamedTuple):
    metanet: parametanet.ParaNetworkParameters
    log_vars: jax.Array


# ---------------------------------------------------------
# Mathematical Helpers
# ---------------------------------------------------------
def inv_softplus(x, scale):
    def logexpm1(v):
        return v + jnp.log(-jnp.expm1(-v))

    return logexpm1(x / scale)


def get_physical_params(raw_metanet_params, scales):
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


def nll_loss(params, traj_true, initial_state, boundaries, scales, penalty_weight):
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

    empirical_fields = ["alpha", "critical_density", "free_flow_speed"]
    for field in physical_metanet._fields:
        if field in empirical_fields:
            p_pred = getattr(physical_metanet, field)
            p_diff = p_pred[:-1] - p_pred[1:]
            p_norm = 0.5 * (jnp.abs(p_pred[:-1] + p_pred[1:]) + 1e-8)
            reg_loss += jnp.sum((p_diff / p_norm) ** 2)

    return base_loss + 1 / 2 * (penalty_weight * reg_loss)


# ---------------------------------------------------------
# Optimization Core
# ---------------------------------------------------------
optimizer = optax.lbfgs(1e-4)


@functools.partial(jax.jit, static_argnames=["chunk_size"])
def train_chunk(
    initial_params,
    opt_state,
    traj_true,
    initial_state,
    boundaries,
    scales,
    mask,
    chunk_size,
    penalty_weight,
):
    def step(carry, _):
        params, opt_state = carry

        def value_fn(p):
            p_masked = jax.tree_util.tree_map(
                lambda cur, orig, m: jnp.where(m == 1.0, cur, orig), p, params, mask
            )
            return nll_loss(
                p_masked, traj_true, initial_state, boundaries, scales, penalty_weight
            )

        loss, grads = jax.value_and_grad(value_fn)(params)
        updates, opt_state = optimizer.update(
            grads, opt_state, params, value=loss, grad=grads, value_fn=value_fn
        )
        new_params = optax.apply_updates(params, updates)
        return (new_params, opt_state), (loss, new_params)

    # Use None for 'xs' and define the length to run a fixed number of iterations
    final_state, history = jax.lax.scan(
        step, (initial_params, opt_state), None, length=chunk_size
    )
    return final_state, history


@jax.jit
def train_step(
    params,
    opt_state,
    traj_true,
    initial_state,
    boundaries,
    scales,
    penalty_weight,
    mask,
):
    def value_fn(p):
        p_masked = jax.tree_util.tree_map(
            lambda cur, orig, m: jnp.where(m == 1.0, cur, orig), p, params, mask
        )
        return nll_loss(
            p_masked, traj_true, initial_state, boundaries, scales, penalty_weight
        )

    loss, grads = jax.value_and_grad(value_fn)(params)
    updates, opt_state = optimizer.update(
        grads, opt_state, params, value=loss, grad=grads, value_fn=value_fn
    )
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


# ---------------------------------------------------------
# Parallel Worker Function (Monte Carlo Run)
# ---------------------------------------------------------
def run_monte_carlo_optimization(
    mc_run_id,
    mc_seed,
    penalty_weight,
    max_epochs,
    p_true,
    traj_base,
    boundaries_base,
    init_stat,
    scales,
    mask,
):
    try:
        print(f"MC Run {mc_run_id}: Starting (Seed: {mc_seed})")

        # 1. Noise Generation (Defines trj_dis)
        noise_key1, noise_key2 = jax.random.split(jax.random.PRNGKey(mc_seed))
        """
        new_flow = (
            jnp.exp(jax.random.normal(noise_key1, traj_base.flow.shape) * 0.1)
            * traj_base.flow
        )
        new_speed = (
            jnp.exp(jax.random.normal(noise_key2, traj_base.speed.shape) * 0.1)
            * traj_base.speed
        )
        """
        ssq = 0.001
        """
        new_flow = (
            jnp.mean(traj_base.flow)
            * (jax.random.normal(noise_key1, traj_base.flow.shape) * ssq)
            + traj_base.flow
        )
        new_speed = (
            jnp.mean(traj_base.speed)
            * (jax.random.normal(noise_key2, traj_base.speed.shape) * ssq)
            + traj_base.speed
        )
        """
        df = 15.0
        new_flow = (
            jnp.mean(traj_base.flow)
            * (jax.random.t(noise_key1, df, traj_base.flow.shape) * ssq)
            + traj_base.flow
        )
        new_speed = (
            jnp.mean(traj_base.speed)
            * (jax.random.t(noise_key2, df, traj_base.speed.shape) * ssq)
            + traj_base.speed
        )

        trj_dis = parametanet.SimulationTrajectory(
            density=(new_flow / new_speed) * p_true.lambda_,
            speed=new_speed,
            flow=new_flow,
        )

        # 2. Initial Parameter Setup & Stability Test
        seed_offset = 0
        while True:
            current_seed = 2443 + 2 * mc_run_id + 1 + seed_offset
            keys = jax.random.split(jax.random.PRNGKey(current_seed), 7)

            initial_metanet = parametanet.ParaNetworkParameters(
                beta=inv_softplus(
                    jax.random.uniform(
                        keys[0], (), minval=0.7 * p_true.beta, maxval=1.3 * p_true.beta
                    ),
                    scales.beta,
                ),
                free_flow_speed=inv_softplus(
                    jax.random.uniform(
                        keys[1],
                        p_true.free_flow_speed.shape,
                        minval=0.4 * p_true.free_flow_speed,
                        maxval=1.6 * p_true.free_flow_speed,
                    ),
                    scales.free_flow_speed,
                ),
                gamma=inv_softplus(
                    jax.random.uniform(
                        keys[2],
                        (),
                        minval=0.7 * p_true.gamma,
                        maxval=1.3 * p_true.gamma,
                    ),
                    scales.gamma,
                ),
                mu=inv_softplus(
                    jax.random.uniform(
                        keys[3], (), minval=0.7 * p_true.mu, maxval=1.3 * p_true.mu
                    ),
                    scales.mu,
                ),
                critical_density=inv_softplus(
                    jax.random.uniform(
                        keys[4],
                        p_true.critical_density.shape,
                        minval=0.4 * p_true.critical_density,
                        maxval=1.6 * p_true.critical_density,
                    ),
                    scales.critical_density,
                ),
                alpha=inv_softplus(
                    jax.random.uniform(
                        keys[5],
                        p_true.alpha.shape,
                        minval=0.4 * p_true.alpha,
                        maxval=1.6 * p_true.alpha,
                    ),
                    scales.alpha,
                ),
                kappa=inv_softplus(
                    jax.random.uniform(
                        keys[6],
                        (),
                        minval=0.7 * p_true.kappa,
                        maxval=1.3 * p_true.kappa,
                    ),
                    scales.kappa,
                ),
                L=p_true.L,
                lambda_=p_true.lambda_,
                T=p_true.T,
            )

            # Create the initial learning parameters
            frac_stat = jnp.sum(trj_dis.flow) / jnp.sum(trj_dis.speed)
            init_p = LearningParams(
                metanet=initial_metanet,
                log_vars=jnp.array([jnp.log(1.0), jnp.log(1.0 * frac_stat)]),
            )
            init_p = jax.tree_util.tree_map(jnp.asarray, init_p)

            # --- THE STABILITY DRY-RUN ---
            # Define a quick value function for these specific initial parameters
            def test_value_fn(p):
                p_masked = jax.tree_util.tree_map(
                    lambda cur, orig, m: jnp.where(m == 1.0, cur, orig), p, init_p, mask
                )
                return nll_loss(
                    p_masked,
                    trj_dis,
                    init_stat,
                    boundaries_base,
                    scales,
                    penalty_weight,
                )

            # Evaluate the loss once (this forces parametanet.rollout_simulation to run)
            initial_loss = test_value_fn(init_p)

            # Check if the loss exploded to NaN or Infinity
            if not jnp.isnan(initial_loss) and not jnp.isinf(initial_loss):
                print(
                    f"MC Run {mc_run_id}: Found stable parameter seed ({current_seed}). Initial Loss: {initial_loss:.2f}"
                )
                break  # Safe seed found! Exit the loop.

            # If we reach here, the simulation was unstable.
            print(
                f"MC Run {mc_run_id}: Unstable simulation (NaN/Inf) with seed {current_seed}. Incrementing seed and retrying..."
            )
            seed_offset += 100

        # Initialize the optimizer only after we have guaranteed stable parameters
        opt_state = optimizer.init(init_p)

        # ---------------------------------------------------------
        # Chunked Execution & Convergence Tracking
        # ---------------------------------------------------------
        chunk_size = 300
        num_chunks = max_epochs // chunk_size
        tolerance = 1e-7

        loss_history_all = []
        param_history_all = {
            f: [] for f in p_true._fields if f not in ["L", "lambda_", "T"]
        }
        converged_epoch = max_epochs

        current_params = init_p
        current_opt_state = opt_state

        for chunk_idx in range(num_chunks):
            (current_params, current_opt_state), history = train_chunk(
                current_params,
                current_opt_state,
                trj_dis,
                init_stat,
                boundaries_base,
                scales,
                mask,
                chunk_size,
                penalty_weight,
            )

            # Block async dispatch to safely extract this chunk's data to NumPy
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), history)
            losses, params_hist = history

            # 1. Process Loss
            losses_np = np.array(losses, dtype=np.float32)
            loss_history_all.append(losses_np)

            # 2. Process Physical Parameters
            phys_hist = get_physical_params(params_hist.metanet, scales)
            for f in param_history_all:
                vals = np.array(getattr(phys_hist, f), dtype=np.float32)
                param_history_all[f].append(vals)

            # 3. Convergence Check (using the max/min of the current 300-epoch chunk)
            diff = np.max(losses_np) - np.min(losses_np)
            if diff < tolerance:
                converged_epoch = (chunk_idx + 1) * chunk_size
                print(
                    f"MC Run {mc_run_id}: Converged early at epoch {converged_epoch} (Diff: {diff:.2e})"
                )
                break

        print(f"MC Run {mc_run_id}: Finished. Awaiting process join.")

        # Flatten the accumulated chunk data
        final_loss_history = np.concatenate(loss_history_all)
        final_param_history = {
            f: np.concatenate(param_history_all[f], axis=0) for f in param_history_all
        }

        # Get the very last parameters for the JSON output
        final_params_dict = {f: final_param_history[f][-1] for f in final_param_history}

        return {
            "status": "success",
            "mc_run_id": mc_run_id,
            "converged_epoch": converged_epoch,
            "loss_history": final_loss_history,
            "parameter_history": final_param_history,
            "final_params": final_params_dict,
        }
    except Exception as e:
        print(f"\n❌ MC RUN {mc_run_id} FAILED WITH ERROR:\n{traceback.format_exc()}\n")
        return {"status": "error", "mc_run_id": mc_run_id, "error": str(e)}


# ---------------------------------------------------------
# Data Visualization (Main Thread)
# ---------------------------------------------------------
def save_convergence_plot(mc_run_id, loss_history, param_history, p_true, save_dir):
    """Saves the convergence of individual parameters for a single MC run."""
    os.makedirs(save_dir, exist_ok=True)

    fields = list(param_history.keys())
    num_params = len(fields)
    cols = 3
    rows = (num_params + 1) // cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, field in enumerate(fields):
        ax = axes[i]
        history = np.array(param_history[field])

        if history.ndim == 2:  # Vector Parameter
            lines = ax.plot(history, alpha=0.6, linewidth=1)
            if hasattr(p_true, field):
                true_val = np.array(getattr(p_true, field))
                for idx, val in enumerate(true_val):
                    ax.axhline(
                        y=val, color=lines[idx].get_color(), linestyle="--", alpha=0.4
                    )
        else:  # Scalar Parameter
            ax.plot(history, color="blue", linewidth=2)
            if hasattr(p_true, field):
                ax.axhline(y=float(getattr(p_true, field)), color="red", linestyle="--")

        ax.set_title(f"{field} Convergence")
        ax.grid(True, linestyle=":", alpha=0.5)

    # NLL Loss Plot
    ax_loss = axes[num_params]
    ax_loss.plot(loss_history, color="black", linewidth=1.5)
    ax_loss.set_title(f"Optimization: NLL Loss (MC Run {mc_run_id})")
    ax_loss.set_xlabel("Epochs")
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    for j in range(num_params + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"convergence_mc_{mc_run_id}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_mc_bias_plot(aggregated_params, p_true, save_dir="mc_results"):
    """
    Creates boxplots of the estimated parameters across all MC runs
    and overlays the true parameter values.
    """
    os.makedirs(save_dir, exist_ok=True)

    fields = list(aggregated_params.keys())
    num_params = len(fields)
    cols = 3
    rows = (num_params - 1) // cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, field in enumerate(fields):
        ax = axes[i]
        data = np.array(
            aggregated_params[field]
        )  # Shape: (N_runs,) or (N_runs, N_links)
        true_val = np.array(getattr(p_true, field))

        if data.ndim == 2:
            num_links = data.shape[1]
            positions = np.arange(1, num_links + 1)

            ax.boxplot(
                data,
                positions=positions,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue", alpha=0.6),
                medianprops=dict(color="red", linewidth=1.5),
            )
            ax.plot(
                positions,
                true_val,
                "r--",
                marker="o",
                markersize=6,
                label="True Value",
                linewidth=2,
            )
            ax.set_xticks(positions)
            ax.set_xlabel("Link Index")

        else:
            ax.boxplot(
                data,
                positions=[1],
                patch_artist=True,
                widths=0.5,
                boxprops=dict(facecolor="lightblue", color="blue", alpha=0.6),
                medianprops=dict(color="red", linewidth=1.5),
            )
            ax.axhline(
                y=float(true_val),
                color="red",
                linestyle="--",
                label="True Value",
                linewidth=2,
            )
            ax.set_xticks([1])
            ax.set_xticklabels(["Global"])

        ax.set_title(f"{field} Estimation")
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend()

    for j in range(num_params, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "estimator_bias_distribution.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"-> Saved Monte Carlo Bias Plot: {save_path}")


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Create main directories
    os.makedirs("mc_results", exist_ok=True)
    individual_runs_dir = "mc_results/individual_runs"
    os.makedirs(individual_runs_dir, exist_ok=True)

    print("Generating base dataset...")
    traj_true_base, p_true, boundaries_base, init_stat = peSim.simulate_example()

    def get_scale(val):
        return 10 ** (jnp.floor(jnp.log10(jnp.mean(val))))

    scales = parametanet.ParaNetworkParameters(
        beta=get_scale(p_true.beta),
        free_flow_speed=get_scale(p_true.free_flow_speed),
        kappa=get_scale(p_true.kappa),
        gamma=get_scale(p_true.gamma),
        critical_density=get_scale(p_true.critical_density),
        alpha=get_scale(p_true.alpha),
        mu=get_scale(p_true.mu),
        L=1.0,
        lambda_=1.0,
        T=1.0,
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
            L=0,
            lambda_=0,
            T=0,
        ),
        log_vars=jnp.ones(2),
    )

    # ---------------------------------------------------------
    # Monte Carlo Study Setup
    # ---------------------------------------------------------
    num_mc_runs = 20
    max_epochs = 300000
    fixed_penalty_weight = 50

    base_seed = 1000
    mc_seeds = [base_seed + i for i in range(num_mc_runs)]

    pool_args = [
        (
            i,
            seed,
            fixed_penalty_weight,
            max_epochs,
            p_true,
            traj_true_base,
            boundaries_base,
            init_stat,
            scales,
            mask,
        )
        for i, seed in enumerate(mc_seeds)
    ]

    print(
        f"\nDispatching {num_mc_runs} Monte Carlo tests to evaluate Estimator Bias..."
    )

    with mp.Pool(processes=min(num_mc_runs, mp.cpu_count())) as pool:
        all_results = pool.starmap(run_monte_carlo_optimization, pool_args)

    # ---------------------------------------------------------
    # Estimator Bias Calculation & Processing
    # ---------------------------------------------------------
    print("\nProcessing Monte Carlo results...")

    successful_runs = [r for r in all_results if r["status"] == "success"]
    if not successful_runs:
        print("All MC runs failed.")
        exit(1)

    aggregated_params = {f: [] for f in successful_runs[0]["final_params"].keys()}

    # Save individual thread data and aggregate for global bias
    for result in successful_runs:
        mc_id = result["mc_run_id"]

        # 1. Save Convergence Plot for this thread
        save_convergence_plot(
            mc_id,
            result["loss_history"],
            result["parameter_history"],
            p_true,
            save_dir=individual_runs_dir,
        )

        # 2. Save JSON Final Parameters for this thread
        with open(f"{individual_runs_dir}/mc_run_{mc_id}_params.json", "w") as f_out:
            serializable_params = {
                k: v.tolist() if isinstance(v, np.ndarray) else float(v)
                for k, v in result["final_params"].items()
            }
            serializable_params["converged_epoch"] = result["converged_epoch"]
            json.dump(serializable_params, f_out, indent=4)

        # 3. Aggregate for global distribution
        for field, final_val in result["final_params"].items():
            aggregated_params[field].append(final_val)

    # --- PLOT THE GLOBAL BIAS DISTRIBUTION ---
    save_mc_bias_plot(aggregated_params, p_true, "mc_results")

    # Console Summary & JSON Export
    print("\n--- ESTIMATOR BIAS REPORT ---")
    empirical_distributions = {}

    for field, vals in aggregated_params.items():
        vals_array = np.array(vals)
        empirical_mean = np.mean(vals_array, axis=0)
        empirical_std = np.std(vals_array, axis=0)

        true_val = np.array(getattr(p_true, field))
        bias = empirical_mean - true_val

        print(f"\nParameter: {field}")
        print(f"  True Value    : {true_val}")
        print(f"  Empirical Mean: {empirical_mean}")
        print(f"  Bias (E[θ] - θ): {bias}")

        empirical_distributions[field] = {
            "mean": (
                empirical_mean.tolist()
                if isinstance(empirical_mean, np.ndarray)
                else float(empirical_mean)
            ),
            "std": (
                empirical_std.tolist()
                if isinstance(empirical_std, np.ndarray)
                else float(empirical_std)
            ),
            "bias": bias.tolist() if isinstance(bias, np.ndarray) else float(bias),
        }

    with open("mc_results/estimator_bias_summary.json", "w") as f:
        json.dump(empirical_distributions, f, indent=4)

    print(
        "\nDone! All individual thread data and global empirical distributions saved to 'mc_results/'."
    )
