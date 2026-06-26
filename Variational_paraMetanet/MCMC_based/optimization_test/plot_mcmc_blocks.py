import os
import pickle
import typing
from datetime import datetime

from matplotlib.pylab import float32

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import parametanet
import parapersistentExitationSimulation as peSim

# parametanet must be imported so pickle can successfully deserialize the custom classes
try:
    import parametanet
except ImportError:
    print(
        "[-] Error: Could not import 'parametanet'. Please ensure it is in the same directory or your PYTHONPATH."
    )
    sys.exit(1)


def compute_acf(x, max_lag=1000):
    """Computes the Autocorrelation Function using numpy."""
    x = np.asarray(x)
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    if var == 0.0:
        return np.zeros(max_lag)

    x_centered = x - mean
    cov = np.correlate(x_centered, x_centered, mode="full")[n - 1 :]
    acf = cov[:max_lag] / (var * n)
    return acf


def main():
    # 1. Provide the path to your generated pickle file here
    filepath = input(
        "Enter the path to the MCMC pickle file (e.g., mcmc_results_...pkl): "
    ).strip()

    if not os.path.exists(filepath):
        print(f"[-] Error: File '{filepath}' not found.")
        sys.exit(1)

    # 2. Load the data
    print(f"[+] Loading data from {filepath}...")
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    history = data["history"]
    ar_history = np.array(data["ar_history"])  # Shape: (iterations, 3, num_blocks)
    accept_rates = np.array(data["accept_rates"])  # Shape: (3, num_blocks)
    num_blocks = data["num_blocks"]
    iterations = data["iterations"]

    # 3. Create the dynamically named folder
    current_date = datetime.now().strftime("%Y-%m-%d")
    overall_ar = np.mean(accept_rates)

    folder_name = (
        f"{current_date}_{num_blocks}Blocks_{iterations}Iters_{overall_ar:.2%}AR"
    )
    # Clean up percentage sign for file system compatibility
    folder_name = folder_name.replace("%", "pct")

    os.makedirs(folder_name, exist_ok=True)
    print(f"[+] Created output directory: {folder_name}")

    N = len(history[0].alpha)
    burn_in = 1  # Defaulting burn-in to 1000 iterations
    max_lag = iterations - burn_in

    # 4. Generate and save one image per block
    print("[+] Generating block summary images...")
    for b in range(num_blocks):
        # Find the center index for this block to act as the representative sample
        idx = ((b * N // num_blocks) + ((b + 1) * N // num_blocks)) // 2

        # Extract trajectories with the softplus transformatmcmc_results_10blocks_20260616_143645.pklions applied in your original code
        alphas = np.array(
            [(jax.nn.softplus(p.alpha[idx]) + 1e-6) * 1.0 for p in history]
        )
        rhos = np.array(
            [(jax.nn.softplus(p.critical_density[idx]) + 1e-6) * 10.0 for p in history]
        )
        vs = np.array(
            [(jax.nn.softplus(p.free_flow_speed[idx]) + 1e-6) * 100.0 for p in history]
        )

        # Extract ACF (post burn-in)
        acf_alpha = compute_acf(alphas[burn_in:], max_lag=max_lag)
        acf_rho = compute_acf(rhos[burn_in:], max_lag=max_lag)
        acf_v = compute_acf(vs[burn_in:], max_lag=max_lag)
        lags = np.arange(len(acf_alpha))

        # Setup figure: 3 rows (Evolution, ACF, Filtered AR), 3 columns (Alpha, Rho_cr, V_free)
        fig, axs = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(
            f"MCMC Diagnostics - Block {b} (Representative Section {idx})", fontsize=16
        )

        # --- ROW 1: EVOLUTION (TRAJECTORY) ---
        axs[0, 0].plot(alphas, color="blue", alpha=0.7)
        axs[0, 0].set_title(r"Evolution of $\alpha$")
        axs[0, 0].set_ylabel("Value")
        axs[0, 0].grid(True)

        axs[0, 1].plot(rhos, color="orange", alpha=0.7)
        axs[0, 1].set_title(r"Evolution of $\rho_{cr}$")
        axs[0, 1].grid(True)

        axs[0, 2].plot(vs, color="green", alpha=0.7)
        axs[0, 2].set_title(r"Evolution of $v_{free}$")
        axs[0, 2].grid(True)

        # --- ROW 2: AUTOCORRELATION (ACF) ---
        axs[1, 0].plot(lags, acf_alpha, color="blue")
        axs[1, 0].set_title(f"ACF of $\\alpha$ (burn-in: {burn_in})")
        axs[1, 0].set_ylabel("Autocorrelation")
        axs[1, 0].grid(True)

        axs[1, 1].plot(lags, acf_rho, color="orange")
        axs[1, 1].set_title(f"ACF of $\\rho_{{cr}}$ (burn-in: {burn_in})")
        axs[1, 1].grid(True)

        axs[1, 2].plot(lags, acf_v, color="green")
        axs[1, 2].set_title(f"ACF of $v_{{free}}$ (burn-in: {burn_in})")
        axs[1, 2].grid(True)

        # --- ROW 3: FILTERED ACCEPTANCE RATE ---
        # ar_history shape is (iterations, 3, num_blocks) -> index 0 is alpha, 1 is rho, 2 is v_free
        axs[2, 0].plot(ar_history[:, 0, b], color="blue")
        axs[2, 0].axhline(0.3, color="red", linestyle=":", label="Target (0.3)")
        axs[2, 0].set_title(r"Filtered AR of $\alpha$")
        axs[2, 0].set_xlabel("Iteration")
        axs[2, 0].set_ylabel("Acceptance Rate")
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        axs[2, 1].plot(ar_history[:, 1, b], color="orange")
        axs[2, 1].axhline(0.3, color="red", linestyle=":")
        axs[2, 1].set_title(r"Filtered AR of $\rho_{cr}$")
        axs[2, 1].set_xlabel("Iteration")
        axs[2, 1].grid(True)

        axs[2, 2].plot(ar_history[:, 2, b], color="green")
        axs[2, 2].axhline(0.3, color="red", linestyle=":")
        axs[2, 2].set_title(r"Filtered AR of $v_{free}$")
        axs[2, 2].set_xlabel("Iteration")
        axs[2, 2].grid(True)

        plt.tight_layout()

        # Save the figure to the newly created directory
        save_path = os.path.join(folder_name, f"block_{b:02d}_diagnostics.svg")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)  # Close the figure to free up memory

    print(
        f"[+] Successfully saved {num_blocks} images to the directory '{folder_name}'."
    )


if __name__ == "__main__":
    main()
