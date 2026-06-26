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


def main():
    # 1. Provide the path to your generated pickle file
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
    accept_rates = np.array(data["accept_rates"])
    num_blocks = data["num_blocks"]
    iterations = data["iterations"]

    # 3. Create/Locate the dynamically named folder
    current_date = datetime.now().strftime("%Y-%m-%d")
    overall_ar = np.mean(accept_rates)

    folder_name = f"{current_date}_{num_blocks}Blocks_{iterations}Iters_{overall_ar:.2%}AR".replace(
        "%", "pct"
    )
    os.makedirs(folder_name, exist_ok=True)
    print(f"[+] Saving separated histograms to directory: {folder_name}")

    N = len(history[0].alpha)
    burn_in = 1000  # Discard the first 10% of samples

    print(f"[-] Applying a burn-in of {burn_in} iterations.")
    post_burn_in = history[burn_in:]

    # 4. Generate and save one image per block with separated subplots
    print("[+] Generating block histogram images...")

    for b in range(num_blocks):
        # Determine the start and end section indices for this specific block
        start_idx = b * N // num_blocks
        end_idx = (b + 1) * N // num_blocks
        num_sections = end_idx - start_idx

        # Create a grid: rows = number of sections in this block, columns = 3 parameters
        # Height scales dynamically based on how many sections are in the block (4 inches per section)
        fig, axs = plt.subplots(num_sections, 3, figsize=(18, 4 * num_sections))

        # Ensure axs is always a 2D array even if there is only 1 section in the block
        axs = np.atleast_2d(axs)

        fig.suptitle(
            f"Posterior Distributions - Block {b} (Sections {start_idx} to {end_idx - 1})",
            fontsize=18,
            y=0.99,
        )

        # Iterate through every section in this block
        for row_idx, sec_idx in enumerate(range(start_idx, end_idx)):

            # Extract samples and apply the physical transformations from your simulation
            alphas = np.array(
                [(jax.nn.softplus(p.alpha[sec_idx]) + 1e-6) * 1.0 for p in post_burn_in]
            )
            rhos = np.array(
                [
                    (jax.nn.softplus(p.critical_density[sec_idx]) + 1e-6) * 10.0
                    for p in post_burn_in
                ]
            )
            vs = np.array(
                [
                    (jax.nn.softplus(p.free_flow_speed[sec_idx]) + 1e-6) * 100.0
                    for p in post_burn_in
                ]
            )

            # Plot filled histograms since they no longer overlap
            axs[row_idx, 0].hist(alphas, bins=40, color="blue", alpha=0.7)
            axs[row_idx, 1].hist(rhos, bins=40, color="orange", alpha=0.7)
            axs[row_idx, 2].hist(vs, bins=40, color="green", alpha=0.7)

            # Format titles to clearly indicate which section we are looking at
            axs[row_idx, 0].set_title(rf"Section {sec_idx}: $\alpha$")
            axs[row_idx, 1].set_title(rf"Section {sec_idx}: $\rho_{{cr}}$")
            axs[row_idx, 2].set_title(rf"Section {sec_idx}: $v_{{free}}$")

            # Apply standard formatting to all subplots in this row
            for col_idx in range(3):
                axs[row_idx, col_idx].grid(axis="y", alpha=0.3)
                axs[row_idx, col_idx].set_ylabel("Frequency")

            # Add x-axis labels only to the bottom row to keep it clean
            if row_idx == num_sections - 1:
                axs[row_idx, 0].set_xlabel("Value")
                axs[row_idx, 1].set_xlabel("Density (veh/km/lane)")
                axs[row_idx, 2].set_xlabel("Speed (km/h)")

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(folder_name, f"block_{b:02d}_separated_histograms.svg")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[+] Successfully saved {num_blocks} separated histogram images.")


if __name__ == "__main__":
    main()
