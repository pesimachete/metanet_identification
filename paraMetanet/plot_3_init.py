import json
import matplotlib.pyplot as plt
import numpy as np

import parasimulationMetanet as peSim


def plot_full_convergence(
    json_filename="all_runs_LBFGS_results.json", target_run_id=None
):
    """
    Plots a 3x3 grid of parameter convergence and NLL loss.

    Args:
        json_filename: Path to the generated JSON results.
        target_run_id: The ID of the worker to plot (0, 1, or 2).
                       Set to None to overlay all runs (warning: can be messy for vectors!).
    """
    print(f"Loading data from {json_filename}...")

    try:
        with open(json_filename, "r") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_filename}.")
        return

    tau = 0.08
    nu = 50.0
    kappa = 40.0
    delta = 0.012
    true_values = {
        "beta": 1.0 / tau,
        "mu": nu / tau,
        "kappa": kappa,
        "gamma": delta / tau,
        "alpha": 1.8,
        "critical_density": 32,
        "free_flow_speed": 120.0,
    }

    # Extract the parameter names from the first run
    param_names = list(all_results[0]["parameter_history"].keys())

    # We want a subplot for each parameter + 1 for the Loss
    total_plots = len(param_names) + 1
    cols = 3
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    # Colors for runs if plotting multiple
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Filter runs based on target_run_id
    runs_to_plot = (
        all_results
        if target_run_id is None
        else [r for r in all_results if r["run_id"] == target_run_id]
    )

    if not runs_to_plot:
        print(f"Error: Run ID {target_run_id} not found in the JSON.")
        return

    # 1. Plot the Parameters
    for i, param in enumerate(param_names):
        ax = axes[i]

        for run_idx, run_data in enumerate(runs_to_plot):
            history = np.array(run_data["parameter_history"][param])

            # Use blue if plotting one run, otherwise cycle colors
            color = (
                "blue" if target_run_id is not None else colors[run_idx % len(colors)]
            )
            label_prefix = (
                "Estimated"
                if target_run_id is not None
                else f"Run {run_data['run_id']}"
            )

            # Handle Scalars (1D array)
            if history.ndim == 1:
                ax.plot(history, color=color, linewidth=2, label=label_prefix)

            # Handle Vectors (2D array, e.g., alpha, free_flow_speed)
            elif history.ndim == 2:
                for j in range(history.shape[1]):
                    # Only add label to the first segment so the legend isn't massive
                    ax.plot(
                        history[:, j],
                        color=color,
                        alpha=0.4,
                        linewidth=1,
                        label=label_prefix if j == 0 else "",
                    )

        # Plot True Values (if provided)
        if true_values.get(param) is not None:
            t_val = np.array(true_values[param])
            if t_val.ndim == 0 or t_val.size == 1:  # Scalar
                ax.axhline(
                    y=float(t_val),
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    label="True Value",
                )
            else:  # Vector
                for j, val in enumerate(t_val):
                    ax.axhline(
                        y=float(val),
                        color="red",
                        linestyle="--",
                        alpha=0.3,
                        label="True Value" if j == 0 else "",
                    )

        ax.set_title(f"Parameter: {param}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Value")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="upper right")

    # 2. Plot the NLL Loss in the final slot
    ax_loss = axes[len(param_names)]
    for run_idx, run_data in enumerate(runs_to_plot):
        loss = run_data["loss_history"]
        color = "black" if target_run_id is not None else colors[run_idx % len(colors)]
        label = (
            "Loss" if target_run_id is not None else f"Run {run_data['run_id']} Loss"
        )
        ax_loss.plot(loss, color=color, linewidth=1.5, label=label)

    ax_loss.set_title("Optimization: NLL Loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_loss.legend(loc="upper right")

    # 3. Clean up empty subplots (like the bottom right corner)
    for j in range(len(param_names) + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Change target_run_id to 1 or 2 to see the other workers, or None to see them all combined!
    plot_full_convergence()
