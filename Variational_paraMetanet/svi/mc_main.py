import os
import json
import traceback
import multiprocessing as mp
import numpy as np
import matplotlib
import functools

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.scipy as jsp

jax.config.update("jax_enable_x64", True)

import parametanet
import parapersistentExitationSimulation as peSim
from inference import parameter_handling
from inference import mc_optimizer as svi_optimizer


def disturb_measurement(key: jax.Array, measurement: jax.Array, scale=10e-1):
    return jnp.exp(jax.random.normal(key, measurement.shape) * scale) * measurement


def save_individual_convergence_plot(
    mc_id, loss_history, phys_param_history, p_true_flat, converged_epoch, save_dir
):
    fields = list(phys_param_history.keys())
    num_params = len(fields)
    cols = 3
    rows = (num_params + 1) // cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = np.array([axes]) if rows * cols == 1 else axes.flatten()

    for i, field in enumerate(fields):
        ax = axes[i]
        history = np.array(phys_param_history[field])

        # Calculate X-axis specifically for the sliced parameter snippet
        start_epoch = max(0, converged_epoch - len(history))
        x_axis = np.arange(start_epoch, converged_epoch)

        if history.ndim == 2:
            lines = ax.plot(x_axis, history, alpha=0.6, linewidth=1)
            if field in p_true_flat:
                true_val = np.array(p_true_flat[field])
                for idx, val in enumerate(true_val):
                    ax.axhline(
                        y=val, color=lines[idx].get_color(), linestyle="--", alpha=0.4
                    )
        else:
            ax.plot(x_axis, history, color="blue", linewidth=2)
            if field in p_true_flat:
                ax.axhline(y=float(p_true_flat[field]), color="red", linestyle="--")

        ax.set_title(f"{field} (Last {len(history)} Epochs)")
        ax.grid(True, linestyle=":", alpha=0.5)

    # NLL/ELBO Loss Plot (Plots the FULL history)
    ax_loss = axes[num_params]
    ax_loss.plot(loss_history, color="black", linewidth=1.5)
    ax_loss.set_title(f"SVI ELBO Loss (Full History)")
    ax_loss.set_xlabel("Epochs")
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    for j in range(num_params + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"convergence_mc_{mc_id}.png"), dpi=150)
    plt.close(fig)


def run_monte_carlo_svi_worker(
    mc_run_id,
    mc_seed,
    p_true,
    traj_true_base,
    boundaries_base,
    init_stat,
    max_iterations,
):
    try:
        print(f"MC Run {mc_run_id}: Starting (Seed: {mc_seed})")

        p_lambda = (
            p_true.static_params.lambda_
            if hasattr(p_true, "static_params")
            else p_true.lambda_
        )

        if hasattr(p_true, "scalar_params"):
            p_beta = p_true.scalar_params.beta
            p_mu = p_true.scalar_params.mu
            p_kappa = p_true.scalar_params.kappa
            p_gamma = p_true.scalar_params.gamma
            p_alpha = p_true.latent_params.alpha
            p_rho = p_true.latent_params.critical_density
            p_v = p_true.latent_params.free_flow_speed
            static_p = p_true.static_params
        else:
            p_beta = p_true.beta
            p_mu = p_true.mu
            p_kappa = p_true.kappa
            p_gamma = p_true.gamma
            p_alpha = p_true.alpha
            p_rho = p_true.critical_density
            p_v = p_true.free_flow_speed
            static_p = p_true

        keys = jax.random.split(jax.random.PRNGKey(mc_seed), 2)
        new_flow = disturb_measurement(keys[0], traj_true_base.flow)
        new_speed = disturb_measurement(keys[1], traj_true_base.speed)

        trj_dis = parametanet.SimulationTrajectory(
            density=new_flow / new_speed * p_lambda,
            speed=new_speed,
            flow=new_flow,
        )

        def get_scale(val):
            return 10 ** (jnp.floor(jnp.log10(jnp.mean(val))))

        scales = parametanet.ParaNetworkParameters(
            static_params=parametanet.ParaNetworkStaticParameters(
                T=1.0, L=1.0, lambda_=1.0
            ),
            scalar_params=parametanet.ParaNetworkScalarParameters(
                beta=get_scale(p_beta),
                mu=get_scale(p_mu),
                kappa=get_scale(p_kappa),
                gamma=get_scale(p_gamma),
            ),
            latent_params=parametanet.ParaNetworkLatentParameters(
                alpha=get_scale(p_alpha),
                critical_density=get_scale(p_rho),
                free_flow_speed=get_scale(p_v),
            ),
        )
        scales_array = jnp.array(
            [
                scales.latent_params.alpha,
                scales.latent_params.critical_density,
                scales.latent_params.free_flow_speed,
            ]
        )

        alpha_lat = parameter_handling.inv_softplus(p_alpha, scales.latent_params.alpha)
        rho_cr_lat = parameter_handling.inv_softplus(
            p_rho, scales.latent_params.critical_density
        )
        v_free_lat = parameter_handling.inv_softplus(
            p_v, scales.latent_params.free_flow_speed
        )

        initial_mean_2d = jnp.array([alpha_lat, rho_cr_lat, v_free_lat])
        prior_log_var = 2.0 * jnp.log(jnp.array([0.005, 0.005, 0.005]))

        latent_scalars = parametanet.ParaNetworkScalarParameters(
            beta=parameter_handling.inv_softplus(p_beta, scales.scalar_params.beta),
            mu=parameter_handling.inv_softplus(p_mu, scales.scalar_params.mu),
            kappa=parameter_handling.inv_softplus(p_kappa, scales.scalar_params.kappa),
            gamma=parameter_handling.inv_softplus(p_gamma, scales.scalar_params.gamma),
        )

        initial_params = parameter_handling.ExplicitLayerParameters(
            fis_parameters=latent_scalars,
            prior=parameter_handling.PriorParameters(
                mean=jnp.array([1.62, 3.16, 0.84]),
                log_var=prior_log_var,
                corr=jnp.array([jsp.special.logit(0.95)] * 3),
            ),
            variational_posterior=parameter_handling.VariationalPosteriorParameters(
                mean=initial_mean_2d,
                log_var=jnp.array(
                    [
                        2.0
                        * jnp.log(1e-3 * jnp.mean(alpha_lat))
                        * jnp.ones_like(alpha_lat),
                        2.0
                        * jnp.log(1e-3 * jnp.mean(rho_cr_lat))
                        * jnp.ones_like(rho_cr_lat),
                        2.0
                        * jnp.log(1e-3 * jnp.mean(v_free_lat))
                        * jnp.ones_like(v_free_lat),
                    ]
                ),
            ),
            data_log_variance=parameter_handling.DataLogVariance(
                log_var_flow=jnp.log(1e-3), log_var_speed=jnp.log(1e-3)
            ),
        )

        final_params, loss_history, raw_param_history, converged_epoch = (
            svi_optimizer.optimize_svi(
                initial_params=initial_params,
                trj_dis=trj_dis,
                static_params=static_p,
                init_stat=init_stat,
                boundaries=boundaries_base,
                scales=scales,
                scales_array=scales_array,
                mc_run_id=mc_run_id,
                max_iterations=max_iterations,
            )
        )

        # --- SLICE TO LAST 5000 EPOCHS ---
        keep_last = 5000
        phys_param_history = {}

        # 1. Slice Latents
        z_mean = raw_param_history["mean"][-keep_last:]
        phys_param_history["alpha"] = (
            np.log1p(np.exp(z_mean[:, 0, :])) + 1e-8
        ) * float(scales_array[0])
        phys_param_history["critical_density"] = (
            np.log1p(np.exp(z_mean[:, 1, :])) + 1e-8
        ) * float(scales_array[1])
        phys_param_history["free_flow_speed"] = (
            np.log1p(np.exp(z_mean[:, 2, :])) + 1e-8
        ) * float(scales_array[2])

        # 2. Slice Scalars
        for field in raw_param_history["fis_scalars"]:
            scale_val = float(getattr(scales.scalar_params, field))
            unconstrained_vals = raw_param_history["fis_scalars"][field][-keep_last:]
            phys_param_history[field] = (
                np.log1p(np.exp(unconstrained_vals)) + 1e-8
            ) * scale_val

        # 3. Slice Variances (Converted in optimize_svi)
        phys_param_history["variance_flow"] = raw_param_history["variance_flow"][
            -keep_last:
        ]
        phys_param_history["variance_speed"] = raw_param_history["variance_speed"][
            -keep_last:
        ]

        # Extract final params from the end of the sliced histories
        final_dict = {f: phys_param_history[f][-1] for f in phys_param_history}

        print(f"MC Run {mc_run_id}: Finished. Awaiting process join.")
        return {
            "status": "success",
            "mc_run_id": mc_run_id,
            "converged_epoch": converged_epoch,
            "loss_history": loss_history,
            "final_params": final_dict,
            "phys_param_history": phys_param_history,
        }

    except Exception as e:
        print(f"\n❌ MC RUN {mc_run_id} FAILED WITH ERROR:\n{traceback.format_exc()}\n")
        return {"status": "error", "mc_run_id": mc_run_id, "error": str(e)}


def save_mc_bias_plot(aggregated_params, p_true_flat, save_dir="mc_results"):
    os.makedirs(save_dir, exist_ok=True)
    fields = list(aggregated_params.keys())
    num_params = len(fields)
    cols = 3
    rows = (num_params - 1) // cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array([axes]) if rows * cols == 1 else axes.flatten()

    for i, field in enumerate(fields):
        ax = axes[i]
        data = np.array(aggregated_params[field])
        true_val = np.array(p_true_flat[field]) if field in p_true_flat else None

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
            if true_val is not None:
                ax.plot(
                    positions,
                    true_val,
                    "r--",
                    marker="o",
                    markersize=6,
                    label="True Value",
                )
            ax.set_xticks(positions)
        else:
            ax.boxplot(
                data,
                positions=[1],
                patch_artist=True,
                widths=0.5,
                boxprops=dict(facecolor="lightblue", color="blue", alpha=0.6),
                medianprops=dict(color="red", linewidth=1.5),
            )
            if true_val is not None:
                ax.axhline(
                    y=float(true_val), color="red", linestyle="--", label="True Value"
                )
            ax.set_xticks([1])
            ax.set_xticklabels(["Global"])

        ax.set_title(f"{field} Estimation")
        ax.grid(True, linestyle=":", alpha=0.7)
        if true_val is not None:
            ax.legend()

    for j in range(num_params, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "svi_estimator_bias_distribution.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    os.makedirs("mc_results", exist_ok=True)
    individual_runs_dir = "mc_results/individual_runs"
    os.makedirs(individual_runs_dir, exist_ok=True)

    print("Generating base dataset...")
    traj_true_base, p_true, boundaries_base, init_stat = peSim.simulate_example()

    num_mc_runs = 20
    max_iterations = 100000
    base_seed = 1000

    pool_args = [
        (
            i,
            base_seed + i,
            p_true,
            traj_true_base,
            boundaries_base,
            init_stat,
            max_iterations,
        )
        for i in range(num_mc_runs)
    ]

    print(f"\nDispatching {num_mc_runs} parallel SVI Monte Carlo tests...")
    with mp.Pool(processes=min(num_mc_runs, mp.cpu_count())) as pool:
        all_results = pool.starmap(run_monte_carlo_svi_worker, pool_args)

    print("\nProcessing Monte Carlo results...")
    successful_runs = [r for r in all_results if r["status"] == "success"]

    if not successful_runs:
        print("All MC runs failed.")
        exit(1)

    aggregated_params = {f: [] for f in successful_runs[0]["final_params"].keys()}

    p_true_flat = {}
    if hasattr(p_true, "scalar_params"):
        for field in p_true.scalar_params._fields:
            p_true_flat[field] = getattr(p_true.scalar_params, field)
        for field in p_true.latent_params._fields:
            p_true_flat[field] = getattr(p_true.latent_params, field)
    else:
        for field in p_true._fields:
            if field not in ["L", "lambda_", "T"]:
                p_true_flat[field] = getattr(p_true, field)

    for result in successful_runs:
        mc_id = result["mc_run_id"]

        save_individual_convergence_plot(
            mc_id,
            result["loss_history"],
            result["phys_param_history"],
            p_true_flat,
            result["converged_epoch"],
            save_dir=individual_runs_dir,
        )

        json_export_data = {
            "converged_epoch": result["converged_epoch"],
            "final_parameters": {
                k: v.tolist() if isinstance(v, np.ndarray) else float(v)
                for k, v in result["final_params"].items()
            },
            "parameter_history": {
                k: v.tolist() if isinstance(v, np.ndarray) else float(v)
                for k, v in result["phys_param_history"].items()
            },
        }

        with open(f"{individual_runs_dir}/mc_run_{mc_id}_params.json", "w") as f_out:
            json.dump(json_export_data, f_out, indent=4)

        for field, final_val in result["final_params"].items():
            aggregated_params[field].append(final_val)

    save_mc_bias_plot(aggregated_params, p_true_flat, "mc_results")
    print(
        "\nDone! Global empirical distributions and individual parameter trajectories saved to 'mc_results/'."
    )
