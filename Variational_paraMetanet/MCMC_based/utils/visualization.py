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
    Plots the full optimization history collected by the MCMC generator.
    """
    if not results:
        print("No data collected.")
        return

    epochs_list = [step["epoch"] for step in results]
    neg_joint_history = [step["loss"] for step in results]
    ema_norm_history = [step["ema_norm"] for step in results]
    lr_history = [step["lr"] for step in results]

    learnable_fields = list(results[0]["params"].keys())

    # Sort fields into spatial (latent) and global (scalars)
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
    prior_param_histories = {
        field: np.array([step["prior_params"][field] for step in results])
        for field in latent_fields
    }
    prior_std_histories = {
        field: np.array([step["prior_stds"][field] for step in results])
        for field in latent_fields
    }
    prior_corr_histories = {
        field: np.array([step["prior_corr"][field] for step in results])
        for field in latent_fields
    }

    scales_dict = {}
    if p_true is not None:
        scales_dict = {
            "alpha": 1.0,
            "critical_density": 10.0,
            "free_flow_speed": 100.0,
            "beta": getattr(p_true.scalar_params, "beta", 1.0),
            "mu": getattr(p_true.scalar_params, "mu", 1.0),
            "kappa": getattr(p_true.scalar_params, "kappa", 1.0),
            "gamma": getattr(p_true.scalar_params, "gamma", 1.0),
        }

    # =====================================================================
    # FIGURE 1: Spatial Parameters
    # =====================================================================
    fig1, axes1 = plt.subplots(
        len(latent_fields), 2, figsize=(16, 4.5 * len(latent_fields))
    )
    fig1.suptitle(
        "MCMC Spatial Parameters: MAP Estimates & Priors",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    if len(latent_fields) == 1:
        axes1 = [axes1]

    for i, field in enumerate(latent_fields):
        ax_mean = axes1[i][0]
        ax_pvar = axes1[i][1]

        scale = scales_dict.get(field, 1.0)
        history_z = param_histories[field]

        # 1. Plot MAP Trajectory for each segment
        for segment_idx in range(history_z.shape[1]):
            phys_mean = to_physical(history_z[:, segment_idx], scale)
            (post_line,) = ax_mean.plot(
                epochs_list,
                phys_mean,
                alpha=0.5,
                linewidth=1.5,
                label="MAP Trajectories" if segment_idx == 0 else "",
            )
            if p_true is not None and hasattr(p_true.latent_params, field):
                ax_mean.axhline(
                    y=np.array(getattr(p_true.latent_params, field))[segment_idx],
                    color=post_line.get_color(),
                    linestyle=":",
                    alpha=0.8,
                )

        # 2. Plot Segment-wise Prior
        prior_hist_z = prior_param_histories[field]
        prior_std_z = prior_std_histories[field]

        # Ensure it's 2D for the loop just in case
        if prior_hist_z.ndim == 1:
            prior_hist_z = prior_hist_z.reshape(-1, 1)

        # Loop through each of the 20 segments for the prior
        for segment_idx in range(prior_hist_z.shape[1]):
            p_hist_z_seg = prior_hist_z[:, segment_idx]

            phys_prior_mean = to_physical(p_hist_z_seg, scale)
            phys_prior_upper = to_physical(p_hist_z_seg + prior_std_z, scale)
            phys_prior_lower = to_physical(p_hist_z_seg - prior_std_z, scale)

            ax_mean.plot(
                epochs_list,
                phys_prior_mean,
                linestyle="--",
                alpha=0.6,
                linewidth=1.5,
                color="black",
                label="Segment Prior Mean" if segment_idx == 0 else "",
            )
            ax_mean.fill_between(
                epochs_list,
                phys_prior_lower,
                phys_prior_upper,
                color="black",
                alpha=0.03,  # Reduced alpha so 20 overlapping shades don't block out the plot
                label="Prior Std Dev" if segment_idx == 0 else "",
            )
        # Prior Variances
        ax_pvar.plot(epochs_list, prior_std_z**2, color="purple", linewidth=2)
        ax_pvar.set_title(f"Unconstrained Prior Variance: {field}")
        ax_pvar.set_xlabel("Epochs")
        ax_pvar.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax_pvar.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=False)
    plt.pause(0.01)

    # =====================================================================
    # FIGURE 2: Scalar Parameters
    # =====================================================================
    if scalar_fields:
        cols = min(2, len(scalar_fields))
        rows = (len(scalar_fields) + cols - 1) // cols
        fig2, axes2 = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        fig2.suptitle(
            "Global Scalar Parameters (MAP)", fontsize=16, fontweight="bold", y=0.95
        )
        axes2 = np.atleast_1d(axes2).flatten()

        for i, field in enumerate(scalar_fields):
            ax = axes2[i]
            scale = scales_dict.get(field, 1.0)

            # Map unconstrained scalar MAP to physical value
            phys_history = to_physical(param_histories[field], scale)

            ax.plot(
                epochs_list,
                phys_history,
                color="blue",
                linewidth=2,
                label="Estimated MAP",
            )
            if p_true is not None and hasattr(p_true.scalar_params, field):
                ax.axhline(
                    y=float(getattr(p_true.scalar_params, field)),
                    color="red",
                    linestyle="--",
                    label="True Value",
                )

            ax.legend(loc="upper right")
            ax.set_title(f"Scalar Value: {field}")
            ax.set_xlabel("Epochs")
            ax.grid(True, linestyle=":", alpha=0.5)

        for j in range(len(scalar_fields), len(axes2)):
            axes2[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show(block=False)
        plt.pause(0.01)

    # =====================================================================
    # FIGURE 3: Optimizer Diagnostics
    # =====================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(20, 10))
    fig3.suptitle(
        "AdaGrad Optimizer Diagnostics", fontsize=16, fontweight="bold", y=0.95
    )
    axes3 = axes3.flatten()

    ax_norm = axes3[0]
    ax_norm.plot(epochs_list, ema_norm_history, color="crimson", linewidth=2)
    ax_norm.set_title("Gradient Norm (EMA)")
    ax_norm.set_xlabel("Epochs")
    ax_norm.set_yscale("log")
    ax_norm.grid(True, linestyle="--", alpha=0.5)

    ax_lr = axes3[1]
    ax_lr.plot(epochs_list, lr_history, color="teal", linewidth=2)
    ax_lr.set_title("Learning Rate Schedule ($\gamma$)")
    ax_lr.set_xlabel("Epochs")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.grid(True, linestyle=":", alpha=0.5)

    ax_loss = axes3[2]
    ax_loss.plot(epochs_list, neg_joint_history, color="black", linewidth=1.5)
    ax_loss.set_title("Negative Log-Joint (Proxy Loss)")
    ax_loss.set_xlabel("Epochs")
    ax_loss.grid(True, linestyle=":", alpha=0.5)

    ax_dvar = axes3[3]
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
        epochs_list, var_speed_hist, label="Speed Variance", color="tomato", linewidth=2
    )
    ax_dvar.set_title("Data Observation Variances")
    ax_dvar.set_xlabel("Epochs")
    ax_dvar.set_yscale("log")
    ax_dvar.grid(True, linestyle=":", alpha=0.5)
    ax_dvar.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show(block=block)
