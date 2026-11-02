from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
import os

import matplotlib

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"


import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from Variational_paraMetanet.MCMC_based.simulator import parametanet
from Variational_paraMetanet.MCMC_based.simulator import paraLongHighway as peSim
from Variational_paraMetanet.MCMC_based.core.parameter_handling import (
    DataLogVariance,
    PriorParameters,
    Theta,
)
from Variational_paraMetanet.MCMC_based.core.objective import log_joint, log_likelihood

SCALAR_FIELDS = ("beta", "mu", "kappa", "gamma")
BURN_IN_EPOCH = 400_000
Z_95 = 1.959963984540054


def disturb_measurement(key: jax.Array, measurement: jax.Array) -> jax.Array:
    """Same multiplicative measurement perturbation used in main_optimizer.py."""
    ssq = 10e-2
    return jnp.exp(jax.random.normal(key, measurement.shape) * ssq) * measurement


def load_results(pickle_path: Path):
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"], payload
    return payload, None


def recreate_problem():
    """Recreate the simulation, boundaries, and noisy data used during fitting."""
    traj_true, full_p_true, boundaries, init_net_state = peSim.simulate_example()
    keys = jax.random.split(jax.random.PRNGKey(1137), 3)

    new_flow = disturb_measurement(keys[0], traj_true.flow)
    new_speed = disturb_measurement(keys[1], traj_true.speed)
    disturbed_trajectory = parametanet.SimulationTrajectory(
        density=new_flow / (new_speed * full_p_true.static_params.lambda_),
        speed=new_speed,
        flow=new_flow,
    )
    return disturbed_trajectory, full_p_true, boundaries, init_net_state


def scalar_vector_to_namedtuple(values: jax.Array):
    return parametanet.ParaNetworkScalarParameters(
        beta=values[0], mu=values[1], kappa=values[2], gamma=values[3]
    )


def scalar_vector_from_step(step) -> np.ndarray:
    params = step["params"]
    return np.asarray(
        [params["beta"], params["mu"], params["kappa"], params["gamma"]],
        dtype=float,
    )


def latent_draw_from_step(step):
    """The cached latent parameter entries are the MCMC draw z at that log step."""
    params = step["params"]
    return parametanet.ParaNetworkLatentParameters(
        alpha=jnp.asarray(params["alpha"]),
        critical_density=jnp.asarray(params["critical_density"]),
        free_flow_speed=jnp.asarray(params["free_flow_speed"]),
    )


def theta_template_from_step(step):
    """Rebuild all fixed/nuisance Theta components from one logged result."""
    latent_names = ("alpha", "critical_density", "free_flow_speed")
    prior_mean = jnp.asarray(
        [np.asarray(step["prior_params"][name]) for name in latent_names]
    )
    prior_std = jnp.asarray([float(step["prior_stds"][name]) for name in latent_names])

    # Cache contains tanh(corr); Theta stores its unconstrained representation.
    stored_corr = float(step["prior_corr"]["alpha"])
    stored_corr = np.clip(stored_corr, -1.0 + 1e-12, 1.0 - 1e-12)

    return Theta(
        latent=latent_draw_from_step(step),
        scalar=scalar_vector_to_namedtuple(jnp.asarray(scalar_vector_from_step(step))),
        prior=PriorParameters(
            mean=prior_mean,
            log_var=2.0 * jnp.log(prior_std),
            corr=jnp.asarray(np.arctanh(stored_corr)),
        ),
        obs_var=DataLogVariance(
            log_var_flow=jnp.log(jnp.asarray(float(step["data_var_flow"]))),
            log_var_speed=jnp.log(jnp.asarray(float(step["data_var_speed"]))),
        ),
    )


def replace_scalar(theta_template: Theta, scalar_vector: jax.Array) -> Theta:
    return theta_template._replace(scalar=scalar_vector_to_namedtuple(scalar_vector))


def softplus_np(values):
    values = np.asarray(values, dtype=float)
    return np.where(values > 20.0, values, np.log1p(np.exp(values)))


def sigmoid_np(values):
    values = np.asarray(values, dtype=float)
    return np.where(
        values >= 0.0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def physical_values_and_jacobian(unconstrained_scalar, scalar_scales):
    scales = np.asarray(
        [
            float(scalar_scales.beta),
            float(scalar_scales.mu),
            float(scalar_scales.kappa),
            float(scalar_scales.gamma),
        ]
    )
    physical = scales * softplus_np(unconstrained_scalar)
    jacobian = np.diag(scales * sigmoid_np(unconstrained_scalar))
    return physical, jacobian


def project_to_spd(matrix: np.ndarray, floor: float):
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floored = np.maximum(eigenvalues, floor)
    spd = (eigenvectors * floored) @ eigenvectors.T
    return spd, eigenvalues, floored


def compute_fisher_identity_estimate(
    retained_steps,
    scalar_hat,
    theta_template,
    params_static,
    scalar_scales,
    init_state,
    trajectory,
    boundaries,
    objective_kind,
):
    """
    Estimate FIM from final latent draws using Fisher's identity machinery.

    The complete-data score s_m is evaluated at the same fixed scalar estimate
    eta_hat for each retained latent draw z_m:

        s_m = grad_eta log f(y, z_m; eta_hat)

    The average score estimates the observed-data score by Fisher's identity.
    For the non-independent model, Algorithm 2 of Baey et al. recommends
    averaging final complete-data Hessians. This routine therefore reports both:

      1) score outer-product estimate: mean_m(s_m s_m^T),
      2) Louis-style complete-data curvature estimate: -mean_m Hessian_m.

    The curvature matrix is used for confidence intervals by default because it
    remains full-rank in the single globally coupled trajectory setting.
    """
    if objective_kind == "likelihood":

        def complete_objective(scalar_vector, z_draw):
            theta = replace_scalar(theta_template, scalar_vector)
            return log_likelihood(
                theta,
                z_draw,
                params_static,
                scalar_scales,
                init_state,
                trajectory,
                boundaries,
            )

    elif objective_kind == "joint":

        def complete_objective(scalar_vector, z_draw):
            theta = replace_scalar(theta_template, scalar_vector)
            return log_joint(
                theta,
                z_draw,
                params_static,
                scalar_scales,
                init_state,
                trajectory,
                boundaries,
            )

    else:
        raise ValueError("objective_kind must be 'likelihood' or 'joint'.")

    score_function = jax.jit(jax.grad(complete_objective, argnums=0))
    hessian_function = jax.jit(jax.hessian(complete_objective, argnums=0))

    scores = []
    negative_hessians = []
    for index, step in enumerate(retained_steps, start=1):
        z_draw = latent_draw_from_step(step)
        score = np.asarray(score_function(jnp.asarray(scalar_hat), z_draw), dtype=float)
        hessian = np.asarray(
            hessian_function(jnp.asarray(scalar_hat), z_draw), dtype=float
        )
        scores.append(score)
        negative_hessians.append(-0.5 * (hessian + hessian.T))
        if index % 25 == 0 or index == len(retained_steps):
            print(f"Processed Fisher draw {index}/{len(retained_steps)}")

    scores = np.asarray(scores)
    negative_hessians = np.asarray(negative_hessians)

    mean_score = np.mean(scores, axis=0)
    score_outer_product = (scores.T @ scores) / len(scores)
    score_covariance = np.cov(scores, rowvar=False, ddof=1)
    mean_negative_hessian = np.mean(negative_hessians, axis=0)

    # Louis identity for observed information:
    # I_obs = E[-H_complete | y] - Cov(score_complete | y).
    louis_observed_information = mean_negative_hessian - score_covariance

    return {
        "scores": scores,
        "mean_score": mean_score,
        "score_outer_product": score_outer_product,
        "score_covariance": score_covariance,
        "mean_negative_hessian": mean_negative_hessian,
        "louis_observed_information": louis_observed_information,
    }


def confidence_interval_from_information(information, jacobian, floor):
    information_spd, raw_eigenvalues, used_eigenvalues = project_to_spd(
        information, floor
    )
    covariance_unconstrained = np.linalg.inv(information_spd)
    covariance_physical = jacobian @ covariance_unconstrained @ jacobian.T
    standard_errors = np.sqrt(np.maximum(np.diag(covariance_physical), 0.0))
    return {
        "information_spd": information_spd,
        "raw_eigenvalues": raw_eigenvalues,
        "used_eigenvalues": used_eigenvalues,
        "covariance_unconstrained": covariance_unconstrained,
        "covariance_physical": covariance_physical,
        "standard_errors": standard_errors,
    }


def plot_ci(output_path, physical, standard_errors, truth, title):
    positions = np.arange(len(SCALAR_FIELDS))
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.errorbar(
        positions,
        physical,
        yerr=Z_95 * standard_errors,
        fmt="o",
        markersize=8,
        capsize=7,
        capthick=1.7,
        color="navy",
        ecolor="royalblue",
        elinewidth=2.0,
        label="Estimate ± 95% Fisher/Louis CI",
        zorder=3,
    )
    ax.scatter(
        positions,
        truth,
        marker="x",
        color="crimson",
        linewidths=2.0,
        s=70,
        label="True value",
        zorder=4,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(SCALAR_FIELDS)
    ax.set_ylabel("Physical parameter value")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax.legend(loc="best")
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_relative_ci(output_path, physical, standard_errors, truth, title):
    positions = np.arange(len(SCALAR_FIELDS))
    relative = physical / truth
    relative_se = standard_errors / np.abs(truth)

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.errorbar(
        positions,
        relative,
        yerr=Z_95 * relative_se,
        fmt="o",
        markersize=8,
        capsize=7,
        capthick=1.7,
        color="navy",
        ecolor="royalblue",
        elinewidth=2.0,
        label="Estimate ± 95% Fisher/Louis CI divided by truth",
    )
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.8, label="True value")
    ax.set_xticks(positions)
    ax.set_xticklabels(SCALAR_FIELDS)
    ax.set_ylabel("Estimate / true value")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax.legend(loc="best")
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_matrix(output_path, matrix, title, cmap="coolwarm", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(SCALAR_FIELDS)))
    ax.set_yticks(np.arange(len(SCALAR_FIELDS)))
    ax.set_xticklabels(SCALAR_FIELDS)
    ax.set_yticklabels(SCALAR_FIELDS)
    ax.set_title(title)
    for row in range(len(SCALAR_FIELDS)):
        for col in range(len(SCALAR_FIELDS)):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:.2e}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(image, ax=ax)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_matrix_csv(output_path, name, matrix):
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([name, *SCALAR_FIELDS])
        for field, row in zip(SCALAR_FIELDS, matrix):
            writer.writerow([field, *row])


def write_summary_csv(
    output_path, scalar_hat, physical, ci_result, truth, mean_score, n_draws
):
    standard_errors = ci_result["standard_errors"]
    lower = physical - Z_95 * standard_errors
    upper = physical + Z_95 * standard_errors
    fieldnames = [
        "parameter",
        "unconstrained_estimate",
        "physical_estimate",
        "fisher_louis_standard_error_physical",
        "fisher_louis_95_ci_lower",
        "fisher_louis_95_ci_upper",
        "true_value",
        "true_value_inside_ci",
        "mean_complete_data_score",
        "n_post_burnin_latent_draws",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, field in enumerate(SCALAR_FIELDS):
            writer.writerow(
                {
                    "parameter": field,
                    "unconstrained_estimate": scalar_hat[index],
                    "physical_estimate": physical[index],
                    "fisher_louis_standard_error_physical": standard_errors[index],
                    "fisher_louis_95_ci_lower": lower[index],
                    "fisher_louis_95_ci_upper": upper[index],
                    "true_value": truth[index],
                    "true_value_inside_ci": bool(
                        lower[index] <= truth[index] <= upper[index]
                    ),
                    "mean_complete_data_score": mean_score[index],
                    "n_post_burnin_latent_draws": n_draws,
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate scalar Fisher information using final latent MCMC draws, "
            "Fisher's identity, Louis' identity, and complete-data Hessians."
        )
    )
    parser.add_argument("pickle_file", type=Path, help="Path to mcmc_results_*.pkl")
    parser.add_argument(
        "--burn-in",
        type=int,
        default=BURN_IN_EPOCH,
        help="Use logged epochs at or after this value (default: 400000).",
    )
    parser.add_argument(
        "--max-draws",
        type=int,
        default=0,
        help="Optional evenly spaced cap on retained draws; 0 uses every retained draw.",
    )
    parser.add_argument(
        "--scalar-estimate",
        choices=("last", "mean"),
        default="last",
        help="Fixed scalar eta_hat: last retained state or mean over retained states.",
    )
    parser.add_argument(
        "--objective",
        choices=("likelihood", "joint"),
        default="joint",
        help="Complete-data objective: likelihood or joint (default: joint).",
    )
    parser.add_argument(
        "--information",
        choices=("louis", "mean-hessian", "score-op"),
        default="louis",
        help=(
            "Information used for CIs: Louis observed information, mean negative "
            "complete-data Hessian, or score outer product (default: louis)."
        ),
    )
    parser.add_argument(
        "--eigenvalue-floor",
        type=float,
        default=1e-8,
        help="Smallest eigenvalue allowed before information inversion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fisher_identity_ci"),
        help="Folder for SVG plots and CSV matrices.",
    )
    args = parser.parse_args()

    results, _ = load_results(args.pickle_file)
    if not results:
        raise ValueError("The pickle contains no logged optimization results.")

    epochs = np.asarray([step["epoch"] for step in results])
    retained_indices = np.flatnonzero(epochs >= args.burn_in)
    if len(retained_indices) == 0:
        raise ValueError(
            f"No logged steps have epoch >= {args.burn_in:,}; final epoch is {int(epochs.max()):,}."
        )

    if args.max_draws > 0 and len(retained_indices) > args.max_draws:
        positions = np.linspace(0, len(retained_indices) - 1, args.max_draws, dtype=int)
        retained_indices = retained_indices[positions]

    retained_steps = [results[index] for index in retained_indices]
    scalar_history = np.asarray(
        [scalar_vector_from_step(step) for step in retained_steps]
    )
    scalar_hat = (
        scalar_history[-1]
        if args.scalar_estimate == "last"
        else np.mean(scalar_history, axis=0)
    )

    # Use final retained nuisance state while averaging over the retained latent draws.
    theta_template = theta_template_from_step(retained_steps[-1])
    trajectory, full_p_true, boundaries, init_state = recreate_problem()

    fisher_parts = compute_fisher_identity_estimate(
        retained_steps=retained_steps,
        scalar_hat=scalar_hat,
        theta_template=theta_template,
        params_static=full_p_true.static_params,
        scalar_scales=full_p_true.scalar_params,
        init_state=init_state,
        trajectory=trajectory,
        boundaries=boundaries,
        objective_kind=args.objective,
    )

    information_options = {
        "louis": fisher_parts["louis_observed_information"],
        "mean-hessian": fisher_parts["mean_negative_hessian"],
        "score-op": fisher_parts["score_outer_product"],
    }
    information_raw = information_options[args.information]

    physical, jacobian = physical_values_and_jacobian(
        scalar_hat, full_p_true.scalar_params
    )
    ci_result = confidence_interval_from_information(
        information_raw, jacobian, args.eigenvalue_floor
    )

    truth = np.asarray(
        [
            float(full_p_true.scalar_params.beta),
            float(full_p_true.scalar_params.mu),
            float(full_p_true.scalar_params.kappa),
            float(full_p_true.scalar_params.gamma),
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    title_base = f"Fisher/Louis information ({args.information}; {args.objective})"
    plot_ci(
        args.output_dir / "fisher_identity_95_ci_physical_scalars.svg",
        physical,
        ci_result["standard_errors"],
        truth,
        title_base + " — physical scalar parameters",
    )
    plot_relative_ci(
        args.output_dir / "fisher_identity_95_ci_relative_to_truth.svg",
        physical,
        ci_result["standard_errors"],
        truth,
        title_base + " — relative to truth",
    )
    plot_matrix(
        args.output_dir / "louis_observed_information.svg",
        fisher_parts["louis_observed_information"],
        "Louis observed-information estimate (unconstrained scalar space)",
    )
    plot_matrix(
        args.output_dir / "mean_negative_complete_hessian.svg",
        fisher_parts["mean_negative_hessian"],
        "Mean negative complete-data Hessian",
    )
    plot_matrix(
        args.output_dir / "score_outer_product.svg",
        fisher_parts["score_outer_product"],
        "Complete-data score outer-product estimate",
    )

    write_summary_csv(
        args.output_dir / "fisher_identity_scalar_summary.csv",
        scalar_hat,
        physical,
        ci_result,
        truth,
        fisher_parts["mean_score"],
        len(retained_steps),
    )
    write_matrix_csv(
        args.output_dir / "score_outer_product.csv",
        "score_outer_product",
        fisher_parts["score_outer_product"],
    )
    write_matrix_csv(
        args.output_dir / "score_covariance.csv",
        "score_covariance",
        fisher_parts["score_covariance"],
    )
    write_matrix_csv(
        args.output_dir / "mean_negative_complete_hessian.csv",
        "mean_negative_complete_hessian",
        fisher_parts["mean_negative_hessian"],
    )
    write_matrix_csv(
        args.output_dir / "louis_observed_information_raw.csv",
        "louis_observed_information_raw",
        fisher_parts["louis_observed_information"],
    )
    write_matrix_csv(
        args.output_dir / "information_used_spd.csv",
        "information_used_spd",
        ci_result["information_spd"],
    )
    write_matrix_csv(
        args.output_dir / "covariance_unconstrained.csv",
        "covariance_unconstrained",
        ci_result["covariance_unconstrained"],
    )
    write_matrix_csv(
        args.output_dir / "covariance_physical.csv",
        "covariance_physical",
        ci_result["covariance_physical"],
    )
    np.savetxt(
        args.output_dir / "information_raw_eigenvalues.txt",
        ci_result["raw_eigenvalues"],
    )
    np.savetxt(
        args.output_dir / "information_used_eigenvalues.txt",
        ci_result["used_eigenvalues"],
    )

    print(f"Retained latent draws: {len(retained_steps)}")
    print(f"Epoch range: {retained_steps[0]['epoch']} to {retained_steps[-1]['epoch']}")
    print(
        "Mean complete-data score (Fisher identity estimate):",
        fisher_parts["mean_score"],
    )
    print("Raw selected-information eigenvalues:", ci_result["raw_eigenvalues"])
    print(f"Output written to: {args.output_dir.resolve()}")
    print(
        "CIs are conditional on fixed final nuisance parameters and use the selected information estimator."
    )


if __name__ == "__main__":
    main()
