import numpy as np
import matplotlib.pyplot as plt


def softplus_np(x):
    """Numerically stable numpy implementation of the softplus function."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def to_physical(z, scale):
    """Maps unconstrained variables to the physical domain."""
    return (softplus_np(z) + 1e-6) * scale


def print_whole(results, p_true=None, block=False):
    """
    block=False allows the script to continue running immediately after drawing the plots.
    If you call this at the very end of your script, you might want block=True so the
    windows don't immediately close when the program terminates.
    """
    if not results:
        print("No data collected.")
        return

    # Using the FULL dataset, no downsampling
    epochs_list = [step["epoch"] for step in results]
    nll_loss_history = [step["loss"] for step in results]

    if "lr" in results[0]:
        lr_history = [step["lr"] for step in results]
    else:
        lr_history = [5e-2 / np.power((e + 1.0), 0.7) for e in epochs_list]

    learnable_fields = list(results[0]["params"].keys())

    latent_fields = [
        f
        for f in learnable_fields
        if results[0].get("prior_params", {}).get(f) is not None
    ]
    scalar_fields = [
        f for f in learnable_fields if results[0].get("prior_params", {}).get(f) is None
    ]

    param_histories = {
        field: np.array([step["params"][field] for step in results])
        for field in learnable_fields
    }
    std_histories = {
        field: np.array([step["stds"][field] for step in results])
        for field in learnable_fields
    }

    prior_param_histories = {
        field: np.array([step["prior_params"][field] for step in results])
        for field in latent_fields
    }
    prior_std_histories = {
        field: np.array([step["prior_stds"][field] for step in results])
        for field in latent_fields
    }

    has_corr = "prior_corr" in results[0]
    if has_corr:
        prior_corr_histories = {
            field: np.array([step["prior_corr"][field] for step in results])
            for field in latent_fields
        }

    scales_dict = {}
    if p_true is not None:
        scales_dict = {
            "alpha": 10 ** np.floor(np.log10(np.mean(p_true.latent_params.alpha))),
            "critical_density": 10
            ** np.floor(np.log10(np.mean(p_true.latent_params.critical_density))),
            "free_flow_speed": 10
            ** np.floor(np.log10(np.mean(p_true.latent_params.free_flow_speed))),
        }

    # =====================================================================
    # FIGURE 1: Spatial Parameters
    # =====================================================================
    fig1, axes1 = plt.subplots(
        len(latent_fields), 3, figsize=(24, 4.5 * len(latent_fields))
    )
    fig1.suptitle(
        "Spatial Parameters: Physical Means & Unconstrained Variances",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for i, field in enumerate(latent_fields):
        ax_mean = axes1[i, 0]
        ax_var = axes1[i, 1]
        ax_pvar = axes1[i, 2]

        scale = scales_dict.get(field, 1.0)

        history_z = param_histories[field]
        stds_z = std_histories[field]

        for segment_idx in range(history_z.shape[1]):
            z_upper = history_z[:, segment_idx] + stds_z[:, segment_idx]
            z_lower = history_z[:, segment_idx] - stds_z[:, segment_idx]

            phys_mean = to_physical(history_z[:, segment_idx], scale)
            phys_upper = to_physical(z_upper, scale)
            phys_lower = to_physical(z_lower, scale)

            (post_line,) = ax_mean.plot(
                epochs_list,
                phys_mean,
                alpha=0.9,
                linewidth=1.5,
                label="Posterior Estimate" if segment_idx == 0 else "",
            )
            ax_mean.fill_between(
                epochs_list,
                phys_lower,
                phys_upper,
                color=post_line.get_color(),
                alpha=0.20,
            )

            if p_true is not None and hasattr(p_true.latent_params, field):
                ax_mean.axhline(
                    y=np.array(getattr(p_true.latent_params, field))[segment_idx],
                    color=post_line.get_color(),
                    linestyle=":",
                    alpha=0.8,
                )

        prior_hist_z = prior_param_histories[field]
        prior_std_z = prior_std_histories[field]

        prior_z_upper = prior_hist_z + prior_std_z
        prior_z_lower = prior_hist_z - prior_std_z

        phys_prior_mean = to_physical(prior_hist_z, scale)
        phys_prior_upper = to_physical(prior_z_upper, scale)
        phys_prior_lower = to_physical(prior_z_lower, scale)

        ax_mean.plot(
            epochs_list,
            phys_prior_mean,
            linestyle="--",
            alpha=0.8,
            linewidth=2,
            color="black",
            label="Global Prior",
        )
        ax_mean.fill_between(
            epochs_list,
            phys_prior_lower,
            phys_prior_upper,
            color="black",
            alpha=0.1,
        )

        ax_mean.set_title(f"Physical Means: {field}")
        ax_mean.set_xlabel("Epochs")
        ax_mean.grid(True, linestyle=":", alpha=0.5)

        handles, labels = ax_mean.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # Hardcoded legend location to prevent massive lag
        ax_mean.legend(by_label.values(), by_label.keys(), loc="upper right")

        for segment_idx in range(history_z.shape[1]):
            ax_var.plot(
                epochs_list, stds_z[:, segment_idx] ** 2, alpha=0.8, linewidth=1.5
            )
        ax_var.set_title(f"Unconstrained Posterior Variance: {field}")
        ax_var.set_xlabel("Epochs")
        ax_var.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax_var.grid(True, linestyle=":", alpha=0.5)

        ax_pvar.plot(epochs_list, prior_std_z**2, color="purple", linewidth=2)
        ax_pvar.set_title(f"Unconstrained Prior Variance: {field}")
        ax_pvar.set_xlabel("Epochs")
        ax_pvar.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax_pvar.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=block)
    if not block:
        plt.pause(0.01)

    # =====================================================================
    # FIGURE 2: Scalar Parameters
    # =====================================================================
    if scalar_fields:
        cols = min(2, len(scalar_fields))
        rows = (len(scalar_fields) + cols - 1) // cols
        fig2, axes2 = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        fig2.suptitle("Scalar Parameters", fontsize=16, fontweight="bold", y=0.95)
        axes2 = np.atleast_1d(axes2).flatten()

        for i, field in enumerate(scalar_fields):
            ax = axes2[i]
            history = param_histories[field]
            ax.plot(
                epochs_list, history, color="blue", linewidth=2, label="Estimated Point"
            )
            if p_true is not None and hasattr(p_true.scalar_params, field):
                ax.axhline(
                    y=float(getattr(p_true.scalar_params, field)),
                    color="red",
                    linestyle="--",
                    label="True Value",
                )

            # Hardcoded legend location
            ax.legend(loc="upper right")
            ax.set_title(f"Scalar Mean: {field}")
            ax.set_xlabel("Epochs")
            ax.grid(True, linestyle=":", alpha=0.5)

        for j in range(len(scalar_fields), len(axes2)):
            axes2[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show(block=block)
        if not block:
            plt.pause(0.01)

    # =====================================================================
    # FIGURE 3: Global Diagnostics
    # =====================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(20, 10))
    fig3.suptitle(
        "Optimization & Global Diagnostics", fontsize=16, fontweight="bold", y=0.95
    )
    axes3 = axes3.flatten()

    ax_loss = axes3[0]
    ax_loss.plot(epochs_list, nll_loss_history, color="black", linewidth=1.5)
    ax_loss.set_title("Optimization: Negative ELBO")
    ax_loss.set_xlabel("Epochs")
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    ax_lr = axes3[1]
    ax_lr.plot(epochs_list, lr_history, color="green", linewidth=2)
    ax_lr.set_title("Learning Rate Schedule")
    ax_lr.set_xlabel("Epochs")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.grid(True, linestyle=":", alpha=0.5)

    ax_corr = axes3[2]
    if has_corr:
        for field in latent_fields:
            ax_corr.plot(
                epochs_list, prior_corr_histories[field], label=field, linewidth=2
            )
        # Hardcoded legend location
        ax_corr.legend(loc="upper right")
    else:
        ax_corr.text(0.5, 0.5, "No Correlation Data", ha="center", va="center")
    ax_corr.set_title("Global Prior Correlation")
    ax_corr.set_xlabel("Epochs")
    ax_corr.grid(True, linestyle=":", alpha=0.5)

    ax_dvar = axes3[3]
    if "data_var_flow" in results[0] and "data_var_speed" in results[0]:
        var_flow_hist = [step["data_var_flow"] for step in results]
        var_speed_hist = [step["data_var_speed"] for step in results]

        ax_dvar.plot(
            epochs_list,
            var_flow_hist,
            label="Flow Variance",
            color="dodgerblue",
            linewidth=2,
        )
        ax_dvar.plot(
            epochs_list,
            var_speed_hist,
            label="Speed Variance",
            color="tomato",
            linewidth=2,
        )
        ax_dvar.set_title("Data Observation Variances")
        ax_dvar.set_xlabel("Epochs")
        ax_dvar.set_yscale("log")
        ax_dvar.grid(True, linestyle=":", alpha=0.5)

        # Hardcoded legend location
        ax_dvar.legend(loc="upper right")
    else:
        ax_dvar.text(0.5, 0.5, "No Data Variance Tracked", ha="center", va="center")
        ax_dvar.set_title("Data Observation Variances")
        ax_dvar.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show(block=block)
    if not block:
        plt.pause(0.01)

    # =====================================================================
    # FIGURE 4: Final State Comparison
    # =====================================================================
    if p_true is not None:
        fig4, axes4 = plt.subplots(
            1, len(latent_fields), figsize=(6 * len(latent_fields), 5)
        )
        fig4.suptitle(
            "Final Estimated vs True State of Spatial Parameters Across Sections",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        axes4 = np.atleast_1d(axes4)
        for i, field in enumerate(latent_fields):
            ax = axes4[i]
            scale = scales_dict.get(field, 1.0)

            final_estimate_z = param_histories[field][-1]
            final_estimate_phys = to_physical(final_estimate_z, scale)

            true_vals = np.array(getattr(p_true.latent_params, field))
            sections = np.arange(len(true_vals))

            ax.plot(
                sections,
                true_vals,
                color="red",
                linestyle="--",
                marker="o",
                label="True State",
            )
            ax.plot(
                sections,
                final_estimate_phys,
                color="blue",
                linestyle="-",
                marker="x",
                label="Final Estimate",
            )

            ax.set_title(f"State Across Sections: {field}")
            ax.set_xlabel("Section Index")
            ax.set_ylabel("Physical Parameter Value")
            ax.set_xticks(sections)

            # Hardcoded legend location
            ax.legend(loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show(block=block)
        if not block:
            plt.pause(0.01)
