from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
import os
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

from Variational_paraMetanet.MCMC_based.simulator import parametanet
from Variational_paraMetanet.MCMC_based.simulator import paraLongHighway as peSim
from Variational_paraMetanet.MCMC_based.core.parameter_handling import (
    map_to_physical_params,
)

BURN_IN_EPOCH = 400_000
PERCENTILES = (2.5, 25.0, 50.0, 75.0, 97.5)


def load_results(pickle_path: Path):
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "results" in payload:
        return payload["results"], payload
    return payload, None


def get_parameter_context(payload):
    """Recover the exact scales used by map_to_physical_params()."""
    if payload is not None:
        params_static = payload.get("params_static")
        scalar_scales = payload.get("scalar_scales")
        p_true = payload.get("true_parameters")

        if params_static is not None and scalar_scales is not None:
            return params_static, scalar_scales, p_true
        if p_true is not None:
            return p_true.static_params, p_true.scalar_params, p_true

    # Equivalent deterministic setup used by main_optimizer.py.
    _, full_p_true, _, _ = peSim.simulate_example()
    return full_p_true.static_params, full_p_true.scalar_params, full_p_true


def step_to_physical(step, params_static, scalar_scales):
    """Use the exact parameter_handling.py conversion used during learning."""
    params = step["params"]

    unconstrained_latent = parametanet.ParaNetworkLatentParameters(
        alpha=jnp.asarray(params["alpha"]),
        critical_density=jnp.asarray(params["critical_density"]),
        free_flow_speed=jnp.asarray(params["free_flow_speed"]),
    )
    unconstrained_scalar = parametanet.ParaNetworkScalarParameters(
        beta=jnp.asarray(params["beta"]),
        mu=jnp.asarray(params["mu"]),
        kappa=jnp.asarray(params["kappa"]),
        gamma=jnp.asarray(params["gamma"]),
    )

    return map_to_physical_params(
        unconstrained_latent=unconstrained_latent,
        unconstrained_scalar=unconstrained_scalar,
        params_static=params_static,
        scales_scalar=scalar_scales,
    )


def physical_latent_histories(results, params_static, scalar_scales):
    """Convert all saved iterations into physical latent-parameter histories."""
    physical_steps = [
        step_to_physical(step, params_static, scalar_scales) for step in results
    ]

    return {
        "alpha": np.asarray(
            [np.asarray(step.latent_params.alpha) for step in physical_steps]
        ),
        "critical_density": np.asarray(
            [np.asarray(step.latent_params.critical_density) for step in physical_steps]
        ),
        "free_flow_speed": np.asarray(
            [np.asarray(step.latent_params.free_flow_speed) for step in physical_steps]
        ),
    }


def chain_statistics(history, retained_mask):
    """Compute mean and requested posterior percentiles independently per chain."""
    samples = np.asarray(history)[retained_mask]
    if samples.ndim != 2:
        raise ValueError("Expected a two-dimensional [saved_step, chain] history.")

    return {
        "mean": np.mean(samples, axis=0),
        "p02_5": np.percentile(samples, 2.5, axis=0),
        "p25": np.percentile(samples, 25.0, axis=0),
        "p50": np.percentile(samples, 50.0, axis=0),
        "p75": np.percentile(samples, 75.0, axis=0),
        "p97_5": np.percentile(samples, 97.5, axis=0),
        "n_samples": samples.shape[0],
    }


def write_chain_summary_csv(output_path, field, stats, truth):
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "parameter",
                "chain",
                "n_post_burnin_samples",
                "mean",
                "percentile_2_5",
                "percentile_25",
                "percentile_50",
                "percentile_75",
                "percentile_97_5",
                "true_value",
                "mean_minus_true",
            ],
        )
        writer.writeheader()
        for chain_index in range(len(stats["mean"])):
            true_value = float(truth[chain_index]) if truth is not None else np.nan
            writer.writerow(
                {
                    "parameter": field,
                    "chain": chain_index + 1,
                    "n_post_burnin_samples": stats["n_samples"],
                    "mean": stats["mean"][chain_index],
                    "percentile_2_5": stats["p02_5"][chain_index],
                    "percentile_25": stats["p25"][chain_index],
                    "percentile_50": stats["p50"][chain_index],
                    "percentile_75": stats["p75"][chain_index],
                    "percentile_97_5": stats["p97_5"][chain_index],
                    "true_value": true_value,
                    "mean_minus_true": stats["mean"][chain_index] - true_value,
                }
            )


def plot_all_chains_summary(output_path, field, stats, truth, burn_in_epoch):
    """
    One vector plot per parameter. Each chain shows:
    - 95% interval (2.5th--97.5th percentile),
    - interquartile interval (25th--75th percentile),
    - posterior mean,
    - true value.
    """
    n_chains = len(stats["mean"])
    chain_numbers = np.arange(1, n_chains + 1)

    fig_height = max(7.0, min(18.0, 0.12 * n_chains + 4.5))
    fig, ax = plt.subplots(figsize=(15, fig_height), constrained_layout=True)

    ax.vlines(
        chain_numbers,
        stats["p02_5"],
        stats["p97_5"],
        color="steelblue",
        linewidth=1.2,
        alpha=0.72,
        label="95% interval (2.5–97.5%)",
        zorder=1,
    )
    ax.vlines(
        chain_numbers,
        stats["p25"],
        stats["p75"],
        color="royalblue",
        linewidth=4.0,
        alpha=0.9,
        label="IQR (25–75%)",
        zorder=2,
    )
    ax.scatter(
        chain_numbers,
        stats["mean"],
        color="navy",
        marker="o",
        s=26,
        label="Post-burn-in mean",
        zorder=4,
    )

    if truth is not None:
        ax.scatter(
            chain_numbers,
            truth,
            color="crimson",
            marker="x",
            s=40,
            linewidths=1.6,
            label="True value",
            zorder=5,
        )

    ax.set_title(f"Post-burn-in chain summaries — {field}")
    ax.set_xlabel("Chain number")
    ax.set_ylabel("Physical parameter value")
    ax.set_xlim(0, n_chains + 1)
    ax.set_xticks(np.arange(0, n_chains + 1, 10))
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=True)

    fig.text(
        0.5,
        0.01,
        f"Retained samples: epochs ≥ {burn_in_epoch:,}; mean and percentiles are computed independently for every chain.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_mean_vs_truth(output_path, field, stats, truth, burn_in_epoch):
    """Compact companion plot of posterior mean versus true value for all chains."""
    if truth is None:
        return

    n_chains = len(stats["mean"])
    chain_numbers = np.arange(1, n_chains + 1)

    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    ax.plot(
        chain_numbers,
        stats["mean"],
        color="navy",
        marker="o",
        markersize=3.8,
        linewidth=1.1,
        label="Post-burn-in mean",
    )
    ax.plot(
        chain_numbers,
        truth,
        color="crimson",
        marker="x",
        markersize=4.2,
        linewidth=1.1,
        linestyle="--",
        label="True value",
    )
    ax.set_title(f"Post-burn-in means versus true values — {field}")
    ax.set_xlabel("Chain number")
    ax.set_ylabel("Physical parameter value")
    ax.set_xlim(0, n_chains + 1)
    ax.set_xticks(np.arange(0, n_chains + 1, 10))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.text(
        0.5,
        0.01,
        f"Mean calculated from all saved epochs ≥ {burn_in_epoch:,}.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-chain posterior means and percentiles after burn-in, "
            "then compare them with true values in vector SVG figures."
        )
    )
    parser.add_argument("pickle_file", type=Path, help="Path to mcmc_results_*.pkl")
    parser.add_argument(
        "--burn-in",
        type=int,
        default=BURN_IN_EPOCH,
        help="Only use saved iterations at or after this epoch (default: 400000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("post_burnin_chain_analysis"),
        help="Directory for SVG figures and per-parameter CSV summaries",
    )
    args = parser.parse_args()

    results, payload = load_results(args.pickle_file)
    if not results:
        raise ValueError("The pickle contains no optimization results.")

    epochs = np.asarray([step["epoch"] for step in results])
    retained_mask = epochs >= args.burn_in
    if not np.any(retained_mask):
        raise ValueError(
            f"No saved records have epoch ≥ {args.burn_in:,}. "
            f"The last saved epoch is {int(epochs.max()):,}."
        )

    params_static, scalar_scales, p_true = get_parameter_context(payload)
    histories = physical_latent_histories(results, params_static, scalar_scales)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for field, history in histories.items():
        stats = chain_statistics(history, retained_mask)
        truth = None
        if p_true is not None and hasattr(p_true.latent_params, field):
            truth = np.asarray(getattr(p_true.latent_params, field), dtype=float)

        write_chain_summary_csv(
            args.output_dir / f"summary_{field}.csv", field, stats, truth
        )
        plot_all_chains_summary(
            args.output_dir / f"summary_intervals_{field}.svg",
            field,
            stats,
            truth,
            args.burn_in,
        )
        plot_mean_vs_truth(
            args.output_dir / f"summary_mean_vs_truth_{field}.svg",
            field,
            stats,
            truth,
            args.burn_in,
        )

        for chain_index in range(len(stats["mean"])):
            all_rows.append(
                {
                    "parameter": field,
                    "chain": chain_index + 1,
                    "n_post_burnin_samples": stats["n_samples"],
                    "mean": stats["mean"][chain_index],
                    "percentile_2_5": stats["p02_5"][chain_index],
                    "percentile_25": stats["p25"][chain_index],
                    "percentile_50": stats["p50"][chain_index],
                    "percentile_75": stats["p75"][chain_index],
                    "percentile_97_5": stats["p97_5"][chain_index],
                    "true_value": (
                        float(truth[chain_index]) if truth is not None else np.nan
                    ),
                    "mean_minus_true": (
                        stats["mean"][chain_index] - float(truth[chain_index])
                        if truth is not None
                        else np.nan
                    ),
                }
            )

    combined_csv = args.output_dir / "summary_all_parameters.csv"
    with combined_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"Retained {int(retained_mask.sum())} saved records with epoch ≥ {args.burn_in:,}."
    )
    print(f"Wrote post-burn-in summaries to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
