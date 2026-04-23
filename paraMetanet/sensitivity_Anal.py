import os
import numpy as np
import matplotlib.pyplot as plt

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.flatten_util

import parametanet
import parapersistentExitationSimulation as pes

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def compute_full_hessian(
    p0: parametanet.ParaNetworkParameters,
    init_state: parametanet.NetworkState,
    boundaries: parametanet.BoundarySequence,
    baseline_traj: parametanet.SimulationTrajectory,
):
    # 1. Define the parameters we want to analyze
    empirical_fields = [
        "beta",
        "mu",
        "gamma",
        "kappa",
        "alpha",
        "critical_density",
        "free_flow_speed",
    ]

    active_params = {field: getattr(p0, field) for field in empirical_fields}
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(active_params)
    block_sizes = {k: jnp.size(v) for k, v in active_params.items()}

    # 2. Define the exact Base Loss
    def loss_fn(flat_p):
        p_dict = unflatten_fn(flat_p)
        p_eval = parametanet.ParaNetworkParameters(
            T=p0.T, L=p0.L, lambda_=p0.lambda_, **p_dict
        )

        traj_pred = parametanet.rollout_simulation(init_state, boundaries, p_eval)

        loss_q = 0.5 * jnp.sum(
            (jnp.log(traj_pred.flow + 1e-7) - jnp.log(baseline_traj.flow + 1e-7)) ** 2
        )
        loss_v = 0.5 * jnp.sum(
            (jnp.log(traj_pred.speed + 1e-7) - jnp.log(baseline_traj.speed + 1e-7)) ** 2
        )

        return loss_q + loss_v

    # 3. Compute the full Hessian matrix (2nd derivative)
    print("   -> Compiling and computing full Hessian...")
    H_matrix = jax.jit(jax.jacfwd(jax.jacrev(loss_fn)))(flat_params)

    # 4. Scale the Hessian: H_ij / sqrt(H_ii * H_jj)
    # Extract the main diagonal
    H_diag = jnp.diag(H_matrix)

    # Secure the diagonal against numerical noise (tiny negative values or exact zeros)
    # by taking the absolute value and clipping to a very small positive number.
    H_diag_safe = jnp.clip(jnp.abs(H_diag), a_min=1e-12)

    # Compute the denominator matrix where element (i,j) is sqrt(H_ii * H_jj)
    denominator = jnp.sqrt(jnp.outer(H_diag_safe, H_diag_safe))

    # Calculate the normalized Hessian
    scaled_H_matrix = H_matrix / denominator

    return scaled_H_matrix, block_sizes, empirical_fields


def plot_full_hessian(H_matrix, block_sizes, fields):
    H_np = np.array(H_matrix)

    boundaries = [0]
    ticks = []
    current_idx = 0

    for field in fields:
        size = block_sizes[field]
        ticks.append(current_idx + size / 2 - 0.5)
        current_idx += size
        boundaries.append(current_idx)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Removed SymLogNorm because the values are now strictly bounded [-1, 1]
    # We can use standard vmin and vmax.
    im = ax.imshow(
        H_np,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=1.5)
        ax.axvline(b - 0.5, color="black", linewidth=1.5)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(fields, rotation=45, ha="right", fontsize=12, fontweight="bold")
    ax.set_yticklabels(fields, fontsize=12, fontweight="bold")

    ax.set_title(
        "Normalized Hessian (Parameter Correlation Heatmap)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Updated Colorbar text
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hessian Correlation: H_ij / sqrt(H_ii * H_jj)", fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("1. Running baseline simulation...")
    baseline_traj, p0, boundaries_seq, init_state = pes.simulate_example()

    print("\n2. Calculating 2nd-order parameter dependencies (Complete Hessian)...")
    H_matrix, block_sizes, fields = compute_full_hessian(
        p0, init_state, boundaries_seq, baseline_traj
    )

    print("\n3. Computation complete. Generating dependency heatmap...")
    plot_full_hessian(H_matrix, block_sizes, fields)
