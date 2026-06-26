import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import parapersistentExitationSimulation as peSim
import glob
import re


def generate_plots():
    # 1. Get true parameters
    print("Loading true parameters...")
    # We call simulate_example() to get p_true for the L2 error calculation
    _, p_true, _, _ = peSim.simulate_example()

    # Parameters to analyze
    params_keys = [
        "beta",
        "free_flow_speed",
        "gamma",
        "mu",
        "critical_density",
        "alpha",
        "kappa",
    ]
    vectorial_keys = ["free_flow_speed", "critical_density", "alpha"]

    # 2. Find all json files in id_params folder
    json_files = glob.glob("id_params/identified_parameters_reg_*.json")
    if not json_files:
        print("No JSON files found in 'id_params' directory.")
        return

    results = {}
    for jf in json_files:
        match = re.search(r"reg_([0-9.]+)\.json", jf)
        if match:
            pw = float(match.group(1))
            with open(jf, "r") as f:
                results[pw] = json.load(f)

    sorted_pws = sorted(results.keys())
    print(f"Found results for penalty weights: {sorted_pws}")

    # 3. Series 1: Vectorial Trends (over 20 sections)
    out_dir_vec = "vectorial_trends"
    os.makedirs(out_dir_vec, exist_ok=True)

    sections = np.arange(1, 21)  # X-axis: 1-20 sections

    for pw in sorted_pws:
        data = results[pw]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Vectorial Variables over 20 Sections (Penalty Weight: {pw})", fontsize=16
        )

        for i, vkey in enumerate(vectorial_keys):
            ax = axes[i]
            est_val = np.array(data[vkey])
            true_val = np.array(getattr(p_true, vkey))

            ax.plot(sections, est_val, marker="o", label="Estimated", color="blue")
            ax.plot(
                sections,
                true_val,
                marker="x",
                linestyle="--",
                label="True",
                color="black",
            )
            ax.set_title(vkey.replace("_", " ").title())
            ax.set_xlabel("Section Index")
            ax.set_ylabel("Value")
            ax.set_xticks(sections)

            # Prevent overlapping x-ticks
            if len(sections) == 20:
                ax.set_xticks(np.arange(1, 21, 2))

            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(out_dir_vec, f"vectorial_reg_{pw}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"-> Saved Section Trend Plot: {save_path}")

    # 4. Series 2: L2 Error Trend (1 Single Image)
    l2_errors = {k: [] for k in params_keys}

    for pw in sorted_pws:
        data = results[pw]
        for k in params_keys:
            est_val = np.array(data[k])
            true_val = np.array(getattr(p_true, k))
            # Compute L2 error: ||est - true||_2
            err = np.linalg.norm(est_val - true_val)
            l2_errors[k].append(err)

    fig_err, axes_err = plt.subplots(3, 3, figsize=(15, 12))
    fig_err.suptitle("L2 Error Trend as Regularization Increases", fontsize=18, y=0.98)
    axes_err = axes_err.flatten()

    for i, k in enumerate(params_keys):
        ax = axes_err[i]

        # Plotting the L2 error vs penalty weights
        ax.plot(sorted_pws, l2_errors[k], marker="o", color="crimson", linewidth=2)
        ax.set_title(f"L2 Error: {k.replace('_', ' ').title()}", fontsize=12)
        ax.set_xlabel("Penalty Weight")
        ax.set_ylabel("L2 Error")

        # Set X-axis to logarithmic since weights are non-linear (0.1, 1, 200, 500, 1000)
        ax.set_xscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)

    # Hide unused subplots (since there are 7 parameters, we have 2 empty spots in a 3x3 grid)
    for j in range(len(params_keys), len(axes_err)):
        axes_err[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust top layout so title doesn't overlap
    err_plot_path = "L2_error_trend.png"
    plt.savefig(err_plot_path, dpi=200)
    plt.close()
    print(f"-> Saved Error Plot: {err_plot_path}")


if __name__ == "__main__":
    print("Starting post-processing visualization...")
    generate_plots()
    print("\nAll plotting complete!")
