import json
import os
import numpy as np
import matplotlib.pyplot as plt
import parapersistentExitationSimulation as peSim


def calculate_l2_errors():
    print("Loading true parameters from simulation baseline...")
    _, p_true, _, _ = peSim.simulate_example()

    # The fractions you tested and the parameters you learned
    fractions = [0.10, 0.20, 0.33, 0.50, 0.67, 0.80, 0.90, 1.00]
    learnable_params = [
        "beta",
        "mu",
        "kappa",
        "gamma",
        "alpha",
        "critical_density",
        "free_flow_speed",
    ]

    # Dictionary to hold the L2 error history for each parameter
    l2_errors = {p: [] for p in learnable_params}
    valid_fractions = []

    print("Reading JSON files and calculating L2 Norms...")
    for f in fractions:
        frac_str = f"{f:.2f}"
        file_path = f"id_params/identified_parameters_time_{frac_str}.json"

        if not os.path.exists(file_path):
            print(f"  [!] Warning: {file_path} not found. Skipping.")
            continue

        with open(file_path, "r") as file:
            data = json.load(file)

        valid_fractions.append(f)

        for p in learnable_params:
            # Cast both to numpy arrays for vectorized math (handles scalars and vectors automatically)
            est_val = np.array(data[p])
            true_val = np.array(getattr(p_true, p))

            # Calculate the L2 Norm of the difference: || x_est - x_true ||_2
            l2_norm = np.linalg.norm(est_val - true_val)
            l2_errors[p].append(l2_norm)

    return valid_fractions, l2_errors


def plot_error_trends(fractions, errors):
    if not fractions:
        print("No valid data found to plot!")
        return

    print("Generating L2 Error trend plots...")
    num_params = len(errors)
    cols = 4
    rows = (num_params + cols - 1) // cols  # Calculate required rows

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for i, (param, err_history) in enumerate(errors.items()):
        ax = axes[i]

        # Plot the error trend
        ax.plot(
            fractions,
            err_history,
            marker="o",
            markersize=8,
            linewidth=2,
            color="#e74c3c",
        )

        ax.set_title(f"L2 Error: {param}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Data Time Fraction", fontsize=10)
        ax.set_ylabel("L2 Norm ($|| \hat{\\theta} - \\theta ||_2$)", fontsize=10)

        # Clean up the axes and add a grid
        ax.set_xticks(fractions)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Optional: Force y-axis to start near zero if it makes sense visually
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(max(0, ymin - (ymax - ymin) * 0.1), ymax + (ymax - ymin) * 0.1)

    # Turn off any unused subplots
    for j in range(num_params, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # Save the plot
    save_dir = "convergence_plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "estimation_l2_error_trends.png")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"-> Success! Saved error plot to: '{save_path}'")


if __name__ == "__main__":
    fractions, errors = calculate_l2_errors()
    plot_error_trends(fractions, errors)
