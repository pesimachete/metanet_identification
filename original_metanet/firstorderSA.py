import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import jax
import jax.numpy as jnp


import metanet
import simulationMetanet as pes


os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def compute_separated_sensitivities(
    p0: metanet.NetworkParameters,
    init_state: metanet.NetworkState,
    boundaries: metanet.BoundarySequence,
    baseline_traj: metanet.SimulationTrajectory,
):
    # Pre-compute normalization factors to keep sensitivities dimensionless

    # Define the three isolated loss functions
    def loss_density(p: metanet.NetworkParameters):
        traj = metanet.rollout_simulation(init_state, boundaries, p)
        return jnp.mean(jnp.square(traj.density - baseline_traj.density))

    def loss_speed(p: metanet.NetworkParameters):
        traj = metanet.rollout_simulation(init_state, boundaries, p)
        return jnp.mean(jnp.square(traj.speed - baseline_traj.speed))

    def loss_flow(p: metanet.NetworkParameters):
        traj = metanet.rollout_simulation(init_state, boundaries, p)
        return jnp.mean(jnp.square(traj.flow - baseline_traj.flow))

    H_rho_tree = jax.jit(jax.jacfwd(jax.jacrev(loss_density)))(p0)
    H_v_tree = jax.jit(jax.jacfwd(jax.jacrev(loss_speed)))(p0)
    H_q_tree = jax.jit(jax.jacfwd(jax.jacrev(loss_flow)))(p0)

    empirical_fields = [
        "tau",
        "nu",
        "kappa",
        "delta",
        "alpha",
        "critical_density",
        "free_flow_speed",
    ]

    sensitivities = {"density": {}, "speed": {}, "flow": {}}

    # Extract and scale the diagonals for each parameter across the 3 Hessians
    for field in empirical_fields:
        p0_val = getattr(p0, field)
        scaling_factor = jnp.square(p0_val)

        # Helper to extract the diagonal and scale it
        def extract_and_scale(tree):
            cross_tree = getattr(tree, field)
            hessian_matrix = getattr(cross_tree, field)
            return jnp.diag(hessian_matrix) * scaling_factor

        sensitivities["density"][field] = extract_and_scale(H_rho_tree)
        sensitivities["speed"][field] = extract_and_scale(H_v_tree)
        sensitivities["flow"][field] = extract_and_scale(H_q_tree)

    return sensitivities


def plot_separated_importances(sensitivities, N: int):
    fields = list(sensitivities["density"].keys())
    states = ["density", "speed", "flow"]
    titles = ["Density Sensitivity", "Speed Sensitivity", "Flow Sensitivity"]
    cmaps = ["Reds", "Greens", "Blues"]  # Match your trajectory plot colors!

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, state in enumerate(states):
        ax = axes[idx]

        # Stack into a 2D matrix (num_parameters x N)
        data_matrix = np.abs(np.array([sensitivities[state][f] for f in fields]))

        # Heatmap with LogNorm. Add a tiny epsilon to avoid log(0).
        cax = ax.imshow(
            data_matrix + 1e-12, cmap=cmaps[idx], aspect="auto", norm=LogNorm()
        )

        # Formatting
        ax.set_xticks(np.arange(N))
        ax.set_xlabel("Network Section Index (i)", fontsize=12, fontweight="bold")
        ax.set_title(titles[idx], fontsize=14)

        # Add a colorbar for each subplot
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        if idx == 2:
            cbar.set_label("Relative Sensitivity Magnitude (Log Scale)", fontsize=11)

    # Set Y-axis labels only on the leftmost plot
    axes[0].set_yticks(np.arange(len(fields)))
    axes[0].set_yticklabels(fields, fontsize=11)
    axes[0].set_ylabel("Empirical Parameter", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Parameter Sensitivity Breakdown by State Variable",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("1. Running baseline simulation...")
    baseline_traj, p0, boundaries, init_state = pes.simulate_example()
    N = len(p0.L)

    print(
        "\n2. JIT-compiling and calculating 3 isolated Hessians (this will take a moment)..."
    )
    sensitivities = compute_separated_sensitivities(
        p0, init_state, boundaries, baseline_traj
    )

    print("\n3. Computation complete. Generating plots...")
    plot_separated_importances(sensitivities, N)
