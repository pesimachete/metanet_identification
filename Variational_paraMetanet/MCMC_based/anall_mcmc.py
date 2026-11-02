from __future__ import annotations

import argparse
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

SELECTED_CHAINS = (1, 5, 11, 17, 24, 28, 37, 43, 49, 55, 61, 68, 74, 81, 87, 94, 100)


def save_figure(fig, output_path: Path, extra_bottom: float = 0.0) -> None:
    """Save a vector SVG, optionally reserving space below axes for a legend."""
    if extra_bottom > 0.0:
        fig.subplots_adjust(bottom=extra_bottom)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def load_results(pickle_path: Path):
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"], payload
    return payload, None


def get_parameter_context(payload):
    """Get arguments required by the project physical parameter mapper."""
    if payload is not None:
        params_static = payload.get("params_static")
        scalar_scales = payload.get("scalar_scales")
        p_true = payload.get("true_parameters")
        if params_static is not None and scalar_scales is not None:
            return params_static, scalar_scales, p_true
        if p_true is not None:
            return p_true.static_params, p_true.scalar_params, p_true

    _, full_p_true, _, _ = peSim.simulate_example()
    return full_p_true.static_params, full_p_true.scalar_params, full_p_true


def step_to_physical(step, params_static, scalar_scales):
    """Use exactly map_to_physical_params() from parameter_handling.py."""
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


def softplus_np(values):
    values = np.asarray(values, dtype=float)
    return np.where(values > 20.0, values, np.log1p(np.exp(values)))


def physical_prior_history(unconstrained_mean, unconstrained_std, scale):
    mean = softplus_np(unconstrained_mean) * scale
    lower = softplus_np(unconstrained_mean - unconstrained_std) * scale
    upper = softplus_np(unconstrained_mean + unconstrained_std) * scale
    return mean, lower, upper


def plot_all_chains_with_bottom_legend(
    output_path,
    epochs,
    history,
    true_values,
    prior_mean_z,
    prior_std_z,
    scale,
    field,
):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    history = np.asarray(history)
    if history.ndim == 1:
        history = history[:, None]

    n_chains = history.shape[1]
    fig, ax = plt.subplots(figsize=(15, 7.2), constrained_layout=True)

    # A continuous colormap can represent all 100 chains without a huge legend.
    cmap = plt.get_cmap("turbo")
    normalizer = mpl.colors.Normalize(vmin=1, vmax=n_chains)

    for chain_index in range(n_chains):
        color = cmap(normalizer(chain_index + 1))
        ax.plot(
            epochs,
            history[:, chain_index],
            color=color,
            linewidth=0.75,
            alpha=0.62,
            rasterized=False,
        )
        if true_values is not None and chain_index < len(true_values):
            ax.axhline(
                true_values[chain_index],
                color=color,
                linestyle=":",
                linewidth=0.6,
                alpha=0.48,
                rasterized=False,
            )

    if prior_mean_z is not None and prior_std_z is not None:
        prior_mean_z = np.asarray(prior_mean_z)
        if prior_mean_z.ndim == 1:
            prior_mean_z = prior_mean_z[:, None]

        # Plot a single representative prior trajectory to avoid visual clutter.
        mean, lower, upper = physical_prior_history(
            prior_mean_z[:, 0], prior_std_z, scale
        )
        ax.plot(
            epochs,
            mean,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            label="Prior mean (chain 1)",
        )
        ax.fill_between(
            epochs,
            lower,
            upper,
            color="black",
            alpha=0.07,
            label="Prior mean ± 1 std. (chain 1)",
        )

    # A small two-entry legend explains line styles, while the color bar
    # identifies the chain index.
    map_proxy = mpl.lines.Line2D(
        [0], [0], color="gray", linewidth=1.4, label="MAP trajectory"
    )
    truth_proxy = mpl.lines.Line2D(
        [0], [0], color="gray", linestyle=":", linewidth=1.4, label="True value"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [map_proxy, truth_proxy, *handles],
        ["MAP trajectory", "True value", *labels],
        loc="upper right",
        fontsize=9,
        frameon=True,
    )

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=normalizer, cmap=cmap),
        ax=ax,
        pad=0.02,
        fraction=0.035,
    )
    colorbar.set_label("Chain number")
    colorbar.set_ticks([1, 20, 40, 60, 80, 100])

    ax.set_title(f"Physical MAP and true values — all chains — {field}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Physical value")
    ax.grid(True, linestyle=":", alpha=0.45)

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_selected_chain(
    output_path,
    epochs,
    history,
    chain_number,
    true_values,
    field,
):
    """Create one clear SVG for one requested 1-based chain number."""
    history = np.asarray(history)
    if history.ndim == 1:
        history = history[:, None]

    chain_index = chain_number - 1
    if chain_index < 0 or chain_index >= history.shape[1]:
        print(
            f"Skipping {field}, chain {chain_number}: only {history.shape[1]} chains are available."
        )
        return

    fig, ax = plt.subplots(figsize=(12, 5.8))
    line = ax.plot(
        epochs,
        history[:, chain_index],
        color="royalblue",
        linewidth=1.8,
        label=f"MAP chain {chain_number}",
    )[0]

    if true_values is not None and chain_index < len(true_values):
        ax.axhline(
            true_values[chain_index],
            color="crimson",
            linestyle="--",
            linewidth=1.7,
            label=f"True value: {true_values[chain_index]:.6g}",
        )

    ax.set_title(f"Physical MAP history — {field}, chain {chain_number}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Physical value")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="best")
    save_figure(fig, output_path)


def export_learning_figures(results, params_static, scalar_scales, p_true, output_dir):
    if not results:
        raise ValueError("The supplied pickle contains no optimization results.")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = output_dir / "selected_chains"
    selected_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.asarray([step["epoch"] for step in results])
    losses = np.asarray([step["loss"] for step in results], dtype=float)
    lrs = np.asarray([step["lr"] for step in results], dtype=float)
    ema_norms = np.asarray(
        [step.get("ema_norm", np.nan) for step in results], dtype=float
    )

    physical_steps = [
        step_to_physical(step, params_static, scalar_scales) for step in results
    ]

    latent_histories = {
        "alpha": np.asarray(
            [np.asarray(item.latent_params.alpha) for item in physical_steps]
        ),
        "critical_density": np.asarray(
            [np.asarray(item.latent_params.critical_density) for item in physical_steps]
        ),
        "free_flow_speed": np.asarray(
            [np.asarray(item.latent_params.free_flow_speed) for item in physical_steps]
        ),
    }
    scalar_histories = {
        "beta": np.asarray([float(item.scalar_params.beta) for item in physical_steps]),
        "mu": np.asarray([float(item.scalar_params.mu) for item in physical_steps]),
        "kappa": np.asarray(
            [float(item.scalar_params.kappa) for item in physical_steps]
        ),
        "gamma": np.asarray(
            [float(item.scalar_params.gamma) for item in physical_steps]
        ),
    }
    latent_scales = {
        "alpha": 1.0,
        "critical_density": 10.0,
        "free_flow_speed": 100.0,
    }

    true_latent = {}
    true_scalar = {}
    if p_true is not None:
        true_latent = {
            name: np.asarray(getattr(p_true.latent_params, name))
            for name in latent_histories
            if hasattr(p_true.latent_params, name)
        }
        true_scalar = {
            name: float(getattr(p_true.scalar_params, name))
            for name in scalar_histories
            if hasattr(p_true.scalar_params, name)
        }

    # All-chain spatial figures: legend placed underneath the image.
    for field, history in latent_histories.items():
        prior_mean_z = np.asarray(
            [step["prior_params"][field] for step in results], dtype=float
        )
        prior_std_z = np.asarray(
            [step["prior_stds"][field] for step in results], dtype=float
        )

        plot_all_chains_with_bottom_legend(
            output_dir / f"01_map_prior_all_chains_{field}.svg",
            epochs,
            history,
            true_latent.get(field),
            prior_mean_z,
            prior_std_z,
            latent_scales[field],
            field,
        )

        # Exactly three separate images per spatial parameter: chains 1, 10, and 19.
        for chain_number in SELECTED_CHAINS:
            plot_selected_chain(
                selected_dir / f"map_{field}_chain_{chain_number}.svg",
                epochs,
                history,
                chain_number,
                true_latent.get(field),
                field,
            )

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(epochs, prior_std_z**2, color="purple", linewidth=2)
        ax.set_title(f"Unconstrained prior variance — {field}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Variance")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(True, linestyle=":", alpha=0.55)
        save_figure(fig, output_dir / f"02_prior_variance_{field}.svg")

        corr = np.asarray([step["prior_corr"][field] for step in results], dtype=float)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(epochs, corr, color="forestgreen", linewidth=2)
        ax.set_title(f"Prior correlation (tanh space) — {field}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Correlation")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, linestyle=":", alpha=0.55)
        save_figure(fig, output_dir / f"03_prior_correlation_{field}.svg")

    # One scalar SVG per scalar parameter.
    for field, history in scalar_histories.items():
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(epochs, history, color="royalblue", linewidth=2, label="Estimated MAP")
        if field in true_scalar:
            ax.axhline(
                true_scalar[field],
                color="crimson",
                linestyle="--",
                linewidth=1.5,
                label="True value",
            )
        ax.set_title(f"Physical scalar parameter history — {field}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Physical value")
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(loc="best")
        save_figure(fig, output_dir / f"04_scalar_map_{field}.svg")

    diagnostics = [
        (
            "gradient_norm_ema",
            ema_norms,
            "Gradient norm (EMA)",
            "Gradient norm",
            "crimson",
            True,
        ),
        ("learning_rate", lrs, "Learning-rate schedule", "Learning rate", "teal", True),
        (
            "negative_log_joint",
            losses,
            "Negative log-joint (proxy loss)",
            "Loss",
            "black",
            False,
        ),
    ]
    for filename, values, title, ylabel, color, log_scale in diagnostics:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(epochs, values, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if log_scale and np.all(np.isfinite(values)) and np.all(values > 0):
            ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.55)
        save_figure(fig, output_dir / f"05_diagnostic_{filename}.svg")

    if "data_var_flow" in results[0] and "data_var_speed" in results[0]:
        flow_variance = np.asarray(
            [step["data_var_flow"] for step in results], dtype=float
        )
        speed_variance = np.asarray(
            [step["data_var_speed"] for step in results], dtype=float
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            epochs,
            flow_variance,
            color="dodgerblue",
            linewidth=2,
            label="Flow variance",
        )
        ax.plot(
            epochs, speed_variance, color="tomato", linewidth=2, label="Speed variance"
        )
        ax.set_title("Observation variances")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Variance")
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend()
        save_figure(fig, output_dir / "05_diagnostic_observation_variances.svg")

    if "grads" in results[0]:
        available_grads = results[0]["grads"]

        for field in latent_histories:
            if field not in available_grads:
                continue
            history = np.asarray([step["grads"][field] for step in results])
            if history.ndim == 1:
                history = history[:, None]
            fig, ax = plt.subplots(figsize=(12, 5))
            for chain_index in range(history.shape[1]):
                ax.plot(epochs, history[:, chain_index], linewidth=1.0, alpha=0.7)
            ax.set_title(f"Gradient history — {field}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Gradient")
            ax.grid(True, linestyle=":", alpha=0.55)
            save_figure(fig, output_dir / f"06_gradient_spatial_{field}.svg")

        for field in scalar_histories:
            if field not in available_grads:
                continue
            history = np.asarray(
                [step["grads"][field] for step in results], dtype=float
            )
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(epochs, history, color="darkorange", linewidth=2)
            ax.set_title(f"Gradient history — {field}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Gradient")
            ax.grid(True, linestyle=":", alpha=0.55)
            save_figure(fig, output_dir / f"07_gradient_scalar_{field}.svg")

        special_gradients = {
            "prior_mean": "Gradient — prior mean",
            "prior_log_var": "Gradient — prior log variance",
            "prior_corr": "Gradient — prior correlation",
            "obs_var_flow": "Gradient — observation variance (flow)",
            "obs_var_speed": "Gradient — observation variance (speed)",
        }
        spatial_names = ["alpha", "critical_density", "free_flow_speed"]
        for key, title in special_gradients.items():
            if key not in available_grads:
                continue
            history = np.asarray([step["grads"][key] for step in results])
            fig, ax = plt.subplots(figsize=(12, 5))
            if history.ndim == 1:
                ax.plot(epochs, history, linewidth=2)
            else:
                for component in range(history.shape[1]):
                    label = (
                        spatial_names[component]
                        if component < len(spatial_names)
                        else f"Component {component + 1}"
                    )
                    ax.plot(epochs, history[:, component], linewidth=1.5, label=label)
                ax.legend()
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Gradient")
            ax.grid(True, linestyle=":", alpha=0.55)
            save_figure(fig, output_dir / f"08_gradient_{key}.svg")


def main():
    parser = argparse.ArgumentParser(
        description="Export individual vector SVG learning plots from an MCMC results pickle."
    )
    parser.add_argument("pickle_file", type=Path, help="Path to mcmc_results_*.pkl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("learning_figures_svg"),
        help="Directory where individual SVG files are written",
    )
    args = parser.parse_args()

    results, payload = load_results(args.pickle_file)
    params_static, scalar_scales, p_true = get_parameter_context(payload)
    export_learning_figures(
        results, params_static, scalar_scales, p_true, args.output_dir
    )

    image_count = len(list(args.output_dir.rglob("*.svg")))
    print(f"Saved {image_count} vector SVG figures in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
