"""Create SVG figures directly from parapersistentExitationSimulation.py.

Run from the project environment:
    python plot_parapersistent_simulation_svg.py --output-dir figures

Synthetic observations use exactly the multiplicative log-normal measurement
noise used in main_optimizer.py:
    y_noisy = exp(N(0, 1e-2)) * y_true
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Variational_paraMetanet.MCMC_based.simulator.parapersistentExitationSimulation import (
    simulate_example,
)


def save_figure(fig: plt.Figure, filename: Path) -> None:
    fig.tight_layout()
    fig.savefig(filename, format="svg", bbox_inches="tight")
    plt.close(fig)


def disturb_measurement(key: jax.Array, measurement: jax.Array) -> jax.Array:
    """Same noise model and scale as main_optimizer.disturbmeasurment."""
    ssq = 10e-2
    return jnp.exp(jax.random.normal(key, measurement.shape) * ssq) * measurement


def spatial_profiles(params, output_dir: Path) -> None:
    latent = params.latent_params
    profiles = [
        (r"$\alpha$", np.asarray(latent.alpha), "–", "#0072B2"),
        (
            r"$\rho_{\mathrm{cr}}$",
            np.asarray(latent.critical_density),
            "veh/km/lane",
            "#D55E00",
        ),
        (r"$v_f$", np.asarray(latent.free_flow_speed), "km/h", "#009E73"),
    ]
    sections = np.arange(1, profiles[0][1].size + 1)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    for ax, (label, values, units, color) in zip(axes, profiles):
        ax.plot(sections, values, "o-", color=color, lw=1.8, ms=4)
        ax.set_ylabel(f"{label} [{units}]")
        ax.grid(alpha=0.28)
    axes[-1].set_xlabel("Freeway section")
    fig.suptitle("Ground-truth spatial parameter profiles", y=1.01)
    save_figure(fig, output_dir / "ground_truth_spatial_profiles.svg")


def heatmap(
    values: np.ndarray, title: str, colorbar_label: str, cmap: str, filename: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    image = ax.imshow(
        values.T, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap
    )
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Freeway section")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(colorbar_label)
    save_figure(fig, filename)


def traffic_state_heatmaps(
    density: np.ndarray, speed: np.ndarray, flow: np.ndarray, output_dir: Path
) -> None:
    entries = [
        (density, "Density", "veh/km/lane", "Reds"),
        (speed, "Speed", "km/h", "turbo"),
        (flow, "Flow", "veh/h", "Blues"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    for ax, (values, title, unit, cmap) in zip(axes, entries):
        image = ax.imshow(
            values.T, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap
        )
        ax.set_title(title)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Freeway section")
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label(unit)
    fig.suptitle("Noise-free traffic-state heatmaps")
    fig.savefig(
        output_dir / "noise_free_density_speed_flow_heatmaps.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig)


def noisy_trajectory(
    speed: np.ndarray, output_dir: Path, section: int, stride: int, seed: int
) -> None:
    if not 1 <= section <= speed.shape[1]:
        raise ValueError(f"section must be between 1 and {speed.shape[1]}")

    clean = jnp.asarray(speed[:, section - 1])
    noisy = np.asarray(disturb_measurement(jax.random.PRNGKey(seed), clean))
    clean_np = np.asarray(clean)
    observed_steps = np.arange(0, clean_np.size, max(stride, 1))

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.plot(
        np.arange(clean_np.size),
        clean_np,
        color="#0072B2",
        lw=1.7,
        label="Noise-free speed",
    )
    ax.scatter(
        observed_steps,
        noisy[observed_steps],
        s=10,
        color="#D55E00",
        alpha=0.68,
        label=r"Noisy observations: $\exp(\varepsilon)y$, $\varepsilon\sim\mathcal{N}(0,10^{-2})$",
        zorder=3,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title(f"Noise-free and noisy speed trajectory — section {section}")
    ax.grid(alpha=0.28)
    ax.legend(frameon=False)
    save_figure(fig, output_dir / "noise_free_vs_noisy_speed_trajectory.svg")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create SVG figures from parapersistentExitationSimulation.py"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--section",
        type=int,
        default=None,
        help="One-based section for noisy-speed illustration",
    )
    parser.add_argument(
        "--observation-stride",
        type=int,
        default=20,
        help="Show every nth noisy observation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1137,
        help="JAX PRNG seed used for measurement noise",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traj, params, boundaries, _ = simulate_example()
    density = np.asarray(traj.density)
    speed = np.asarray(traj.speed)
    flow = np.asarray(traj.flow)

    spatial_profiles(params, args.output_dir)
    heatmap(
        np.asarray(boundaries.r),
        "On-ramp demand heatmap",
        "Demand [veh/h]",
        "magma",
        args.output_dir / "on_ramp_demand_heatmap.svg",
    )
    traffic_state_heatmaps(density, speed, flow, args.output_dir)
    noisy_trajectory(
        speed,
        args.output_dir,
        args.section or (speed.shape[1] // 2 + 1),
        args.observation_stride,
        args.seed,
    )

    off_ramp_extraction = np.asarray(boundaries.s)
    if np.any(off_ramp_extraction != 0.0):
        heatmap(
            off_ramp_extraction,
            "Off-ramp extraction heatmap — Dataset III",
            "Extraction [veh/h]",
            "viridis",
            args.output_dir / "dataset_iii_off_ramp_extraction_heatmap.svg",
        )
    else:
        print(
            "No off-ramp extraction SVG: boundaries.s is identically zero in this simulation."
        )

    print(f"SVG figures written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
