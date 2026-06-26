import typing
import os
import json
import traceback
import multiprocessing as mp
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Safe backend for non-GUI environments
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Environment & JAX Setup (Force CPU & Prevent Thread Collisions)
# ---------------------------------------------------------
# CRITICAL: Prevent underlying C++ math libraries from thrashing each other across processes
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
# Mathematical Helpers (UNTOUCHED)
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


def nll_loss(params, traj_true, initial_state, boundaries, scales, penalty_weight=10.0):
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
# Optimization Core (lax.scan)
# ---------------------------------------------------------
optimizer = optax.lbfgs(1e-4)


@jax.jit(static_argnames=["num_epochs"])
def train_loop(
    initial_params,
    opt_state,
    traj_true,
    initial_state,
    boundaries,
    scales,
    mask,
    num_epochs,
):
    def step(carry, _):
        params, opt_state = carry

        def value_fn(p):
            p_masked = jax.tree_util.tree_map(
                lambda cur, orig, m: jnp.where(m == 1.0, cur, orig), p, params, mask
            )
            return nll_loss(p_masked, traj_true, initial_state, boundaries, scales)

        loss, grads = jax.value_and_grad(value_fn)(params)
        updates, opt_state = optimizer.update(
            grads, opt_state, params, value=loss, grad=grads, value_fn=value_fn
        )
        new_params = optax.apply_updates(params, updates)
        return (new_params, opt_state), (loss, new_params)

    final_state, history = jax.lax.scan(
        step, (initial_params, opt_state), jnp.arange(num_epochs)
    )
    return final_state, history


# ---------------------------------------------------------
# Parallel Worker Function
# ---------------------------------------------------------
def run_fraction_optimization(
    run_id, f, num_epochs, p_true, traj_base, boundaries_base, init_stat, scales, mask
):
    try:
        frac_str = f"{f:.2f}"
        K_base = boundaries_base.q_0.shape[0]
        new_K = max(1, int(K_base * f))

        print(f"Worker {run_id}: Starting Fraction {frac_str} (Steps: {new_K})")

        # Truncate and Disturb
        new_boundaries = parametanet.BoundarySequence(
            q_0=boundaries_base.q_0[:new_K],
            v_0=boundaries_base.v_0[:new_K],
            rho_N_plus_1=boundaries_base.rho_N_plus_1[:new_K],
            r=boundaries_base.r[:new_K, :],
            s=boundaries_base.s[:new_K, :],
        )

        noise_key1, noise_key2 = jax.random.split(jax.random.PRNGKey(1129))
        new_flow = (
            jnp.exp(jax.random.normal(noise_key1, traj_base.flow[:new_K].shape) * 0.1)
            * traj_base.flow[:new_K]
        )
        new_speed = (
            jnp.exp(jax.random.normal(noise_key2, traj_base.speed[:new_K].shape) * 0.1)
            * traj_base.speed[:new_K]
        )
        trj_dis = parametanet.SimulationTrajectory(
            density=(new_flow / new_speed) * p_true.lambda_,
            speed=new_speed,
            flow=new_flow,
        )

        # Initialize Params
        keys = jax.random.split(jax.random.PRNGKey(2441), 7)
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
                    keys[2], (), minval=0.7 * p_true.gamma, maxval=1.3 * p_true.gamma
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
                    keys[6], (), minval=0.7 * p_true.kappa, maxval=1.3 * p_true.kappa
                ),
                scales.kappa,
            ),
            L=p_true.L,
            lambda_=p_true.lambda_,
            T=p_true.T,
        )

        frac_stat = jnp.sum(trj_dis.flow) / jnp.sum(trj_dis.speed)
        init_p = LearningParams(
            metanet=initial_metanet,
            log_vars=jnp.array([jnp.log(1.0), jnp.log(1.0 * frac_stat)]),
        )

        init_p = jax.tree_util.tree_map(jnp.asarray, init_p)
        opt_state = optimizer.init(init_p)

        # Execute XLA compiled loop
        final_state, history = train_loop(
            init_p,
            opt_state,
            trj_dis,
            init_stat,
            new_boundaries,
            scales,
            mask,
            num_epochs,
        )

        # --- THE FIX: DEEP ASYNC BLOCKING ---
        # Force Python to wait until absolutely every element in the tree has finished computing
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), final_state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), history)

        losses, params_hist = history
        phys_hist = get_physical_params(params_hist.metanet, scales)

        # --- THE FIX: HARD MEMORY DECOUPLING ---
        # Convert to numpy and use .copy() to completely sever the memory pointer from JAX/XLA
        param_dict = {
            f: np.array(getattr(phys_hist, f), dtype=np.float32).copy()
            for f in phys_hist._fields
            if f not in ["L", "lambda_", "T"]
        }

        loss_array = np.array(losses, dtype=np.float32).copy()

        print(f"Worker {run_id}: Finished successfully. Awaiting process join.")
        return {
            "status": "success",
            "fraction": f,
            "loss_history": loss_array,
            "parameter_history": param_dict,
            "final_variances": {
                "var_rho": float(np.exp(np.array(final_state[0].log_vars[0]))),
                "var_v": float(np.exp(np.array(final_state[0].log_vars[1]))),
            },
        }
    except Exception as e:
        print(f"\n❌ WORKER {run_id} FAILED WITH ERROR:\n{traceback.format_exc()}\n")
        return {"status": "error", "fraction": f, "error": str(e)}


# ---------------------------------------------------------
# Data Visualization (Main Thread)
# ---------------------------------------------------------
def save_convergence_plot(
    frac, loss_history, param_history, p_true, save_dir="convergence_plots"
):
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
        history = param_history[field]

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
    ax_loss.set_title("Optimization: NLL Loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    for j in range(num_params + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"convergence_time_{frac:.2f}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"-> Saved plot: {save_path}")


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.makedirs("id_params", exist_ok=True)

    print("Generating simulation baseline...")
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

    fractions = [0.67, 0.8, 0.9, 1.0]
    num_epochs = 2500

    pool_args = [
        (
            i,
            f,
            num_epochs,
            p_true,
            traj_true_base,
            boundaries_base,
            init_stat,
            scales,
            mask,
        )
        for i, f in enumerate(fractions)
    ]

    print(f"\nDispatching {len(fractions)} parallel tests...")

    # 1. Run optimization on workers
    with mp.Pool(processes=len(fractions)) as pool:
        all_results = pool.starmap(run_fraction_optimization, pool_args)

    # 2. Process results on main thread
    print("\nProcessing results and saving individual files...")

    for result in all_results:
        f = result["fraction"]
        frac_str = f"{f:.2f}"

        if result["status"] == "error":
            print(f"Skipping saving for fraction {frac_str} due to worker error.")
            continue

        # Draw and save the plot
        save_convergence_plot(
            f, result["loss_history"], result["parameter_history"], p_true
        )

        # Build individual JSON parameter payload
        params_to_save = {}
        for field, array_data in result["parameter_history"].items():
            final_vals = array_data[-1]

            # Save raw final values
            if isinstance(final_vals, np.ndarray):
                params_to_save[field] = final_vals.tolist()
            else:
                params_to_save[field] = float(final_vals)

        params_to_save["learned_variance_rho"] = result["final_variances"]["var_rho"]
        params_to_save["learned_variance_v"] = result["final_variances"]["var_v"]

        # Save independent JSON file for this fraction
        json_path = f"id_params/identified_parameters_time_{frac_str}.json"
        with open(json_path, "w") as f_out:
            json.dump(params_to_save, f_out, indent=4)

        print(f"-> Saved data: {json_path}")

    print("\nDone! All processing complete.")
