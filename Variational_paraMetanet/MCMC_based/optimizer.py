import os
import typing

from tqdm import trange

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import parametanet
from hwg_met import DataLogVariance, PriorParameters, Hasting_within_gibbs_sampling

# Short aliases
Latent = parametanet.ParaNetworkLatentParameters
Scalar = parametanet.ParaNetworkScalarParameters
Static = parametanet.ParaNetworkStaticParameters


# ---------------------------------------------------------------------------
# Theta — the single learnable pytree
# ---------------------------------------------------------------------------


class Theta(typing.NamedTuple):
    latent: Latent  # per-section: alpha, rho_cr, v_free
    scalar: Scalar  # global: beta, mu, kappa, gamma
    prior: PriorParameters  # hyperparams: mean (3,N), log_var (3,), corr ()
    obs_var: DataLogVariance  # noise: log_var_flow, log_var_speed


class AdaGradState(typing.NamedTuple):
    r: Theta  # cumulative squared gradients


class LearningState(typing.NamedTuple):
    theta: Theta
    z: Latent
    adagrad: AdaGradState
    adagrad_eps: float
    mcmc_variance: jax.Array
    k: int
    k_end_heat: int
    ema_norm: float


class StepDiagnostic(typing.NamedTuple):
    k: int
    gamma: float
    grad_norm: float
    ema_norm: float
    k_end_heat: int | None


class RunResult(typing.NamedTuple):
    state: LearningState
    theta_history: list[Theta]
    lr_history: list[float]
    grad_norm_history: list[float]
    diagnostics: list[StepDiagnostic]


# ---------------------------------------------------------------------------
# Log-joint  log p(y, z | theta)
# ---------------------------------------------------------------------------


def log_joint(
    theta: Theta,
    z: Latent,  # (unconstrained)
    params_static: Static,
    init_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
) -> jax.Array:

    z_phys = Latent(
        alpha=(jax.nn.softplus(z.alpha) + 1e-6) * 1.0,
        critical_density=(jax.nn.softplus(z.critical_density) + 1e-6) * 10.0,
        free_flow_speed=(jax.nn.softplus(z.free_flow_speed) + 1e-6) * 100.0,
    )

    sim = parametanet.rollout_simulation(
        init_state,
        boundaries,
        parametanet.ParaNetworkParameters(
            static_params=params_static,
            scalar_params=theta.scalar,
            latent_params=z_phys,
        ),
    )

    r2_v = jnp.exp(theta.obs_var.log_var_speed) + 1e-8
    r2_f = jnp.exp(theta.obs_var.log_var_flow) + 1e-8
    ll = (
        -0.5
        * jnp.sum((jnp.log(sim.speed + 1e-8) - jnp.log(traj_true.speed + 1e-8)) ** 2)
        / r2_v
        - 0.5
        * jnp.sum((jnp.log(sim.flow + 1e-8) - jnp.log(traj_true.flow + 1e-8)) ** 2)
        / r2_f
        # log-normalisation terms for the Gaussian likelihood
        - 0.5 * sim.speed.size * jnp.log(r2_v)
        - 0.5 * sim.flow.size * jnp.log(r2_f)
    )

    # AR(1) spatial prior on z (using theta.prior hyperparams)
    var_prior = jnp.exp(theta.prior.log_var)  # (3,)
    coeff = (var_prior * (1 - theta.prior.corr**2) + 1e-8) ** (-1)
    e = jnp.stack(
        [
            z.alpha - theta.prior.mean[0],
            z.critical_density - theta.prior.mean[1],
            z.free_flow_speed - theta.prior.mean[2],
        ]
    )  # (3, N)
    quad = coeff * (
        e[:, 0] ** 2
        + e[:, -1] ** 2
        + (1 + theta.prior.corr**2) * jnp.sum(e[:, 1:-1] ** 2, axis=1)
        - 2 * theta.prior.corr * jnp.sum(e[:, :-1] * e[:, 1:], axis=1)
    )
    lp = -0.5 * jnp.sum(quad)

    return ll + lp


# ---------------------------------------------------------------------------
# AdaGrad preconditioner
# ---------------------------------------------------------------------------


def init_adagrad(theta: Theta) -> AdaGradState:
    return AdaGradState(r=jax.tree.map(jnp.zeros_like, theta))


def adagrad_update(state: AdaGradState, g: Theta) -> AdaGradState:
    return AdaGradState(r=jax.tree.map(lambda r, gi: r + gi**2, state.r, g))


def adagrad_apply(state: AdaGradState, g: Theta, eps: float) -> Theta:
    return jax.tree.map(lambda r, gi: gi / (jnp.sqrt(r) + eps), state.r, g)


def adagrad_apply_stabilised(
    state: AdaGradState, g: Theta, gamma: float, eps: float
) -> Theta:
    def _blend(r, gi):
        scale = (1.0 - gamma) * jnp.maximum(1.0, jnp.mean(r)) + gamma * r
        return gi / (jnp.sqrt(scale) + eps)

    return jax.tree.map(_blend, state.r, g)


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------


def compute_lr(
    k: int,
    k_pre: int,
    k_end_heat: int | None,
    gamma_0: float,
    alpha_decay: float,
) -> float:
    if k < k_pre:
        return float(gamma_0 ** (1.0 - k / k_pre))
    elif k_end_heat is None:
        return 1.0
    else:
        return float((k - k_end_heat) ** (-alpha_decay))


# ---------------------------------------------------------------------------
# JIT-compiled Pure Optimization Step
# ---------------------------------------------------------------------------


@jax.jit
def pure_update_step(
    state: LearningState,
    z_new: Latent,
    mcmc_var_new: jax.Array,
    gamma: float,
    k_pre: int,
    c_heat: float,
    params_static: Static,
    init_net_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
) -> tuple[LearningState, jax.Array, jax.Array]:
    """
    Pure mathematical step decorated with JIT execution.
    Completely decoupled from printing and side-effect histories.
    """
    k = state.k
    eps = state.adagrad_eps

    # 1. Score calculation via autograd
    score_fn = jax.grad(log_joint)
    g = score_fn(
        state.theta,
        z_new,
        params_static,
        init_net_state,
        traj_true,
        boundaries,
    )

    # 2. Update AdaGrad accumulation
    adagrad_new = adagrad_update(state.adagrad, g)

    # 3. Apply preconditioned updates
    direction = jax.lax.cond(
        k < k_pre,
        lambda _: adagrad_apply_stabilised(adagrad_new, g, gamma, eps),
        lambda _: adagrad_apply(adagrad_new, g, eps),
        operand=None,
    )

    theta_new = jax.tree.map(lambda t, d: t + gamma * d, state.theta, direction)

    # 4. Process gradient norms and EMA updates
    leaves = jax.tree.leaves(g)
    norm = jnp.sqrt(sum(jnp.sum(l**2) for l in leaves))
    ema_new = state.ema_norm + c_heat * (norm - state.ema_norm)

    next_state = LearningState(
        theta=theta_new,
        z=z_new,
        adagrad=adagrad_new,
        adagrad_eps=eps,
        mcmc_variance=mcmc_var_new,
        k=k + 1,
        k_end_heat=state.k_end_heat,
        ema_norm=ema_new,
    )

    return next_state, norm, ema_new


# ---------------------------------------------------------------------------
# MWG transition kernel  Π_theta(z' | z, y)
# ---------------------------------------------------------------------------


def mcmc_draw(
    state: LearningState,
    params_static: Static,
    init_net_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
    num_blocks: int,
    sampler_iters: int,
    adapt_alpha: float,
    adapt_beta: float,
) -> tuple[Latent, jax.Array]:
    history, var_history, _, _ = Hasting_within_gibbs_sampling(
        params_fis=state.theta.scalar,
        params_lat=state.theta.prior,
        params_static=params_static,
        data_log_variance=state.theta.obs_var,
        init_state=init_net_state,
        traj_true=traj_true,
        boundaries=boundaries,
        mcmc_variance=state.mcmc_variance,
        iterations=sampler_iters,
        adapt_alpha=adapt_alpha,
        adapt_beta=adapt_beta,
        num_blocks=num_blocks,
    )
    return history[-1], var_history[-1]


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def run(
    theta_init: Theta,
    params_static: Static,
    init_net_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
    total_iterations: int,
    k_pre: int = 1000,
    gamma_0: float = 1e-4,
    alpha_decay: float = 2.0 / 3.0,
    c_heat: float = 1e-3,
    adagrad_eps: float = 1e-8,
    num_blocks: int = 4,
    sampler_iters: int = 10,
    adapt_alpha: float = 1.01,
) -> RunResult:
    """Runs the complete optimization loop safely on the host side with tqdm."""
    adapt_beta = float(
        (1.0 / adapt_alpha ** (100 * 0.45)) ** (1.0 / (100 - 100 * 0.45))
    )

    N = params_static.L.shape[0]

    # Initialize pure state variables
    state = LearningState(
        theta=theta_init,
        z=theta_init.latent,
        adagrad=init_adagrad(theta_init),
        adagrad_eps=adagrad_eps,
        mcmc_variance=jnp.full((3, num_blocks), 0.05),
        k=0,
        k_end_heat=None,
        ema_norm=0.0,
    )

    # Initialize histories outside the pure updates
    theta_history = [state.theta]
    lr_history = []
    grad_norm_history = []
    all_diag: list[StepDiagnostic] = []

    print(
        f"\n{'='*55}\n"
        f"  AdaGrad-SGD | N={N} sections\n"
        f"  k_pre={k_pre}  γ₀={gamma_0}  α={alpha_decay:.3f}\n"
        f"  MCMC {sampler_iters} sweeps/step | {num_blocks} blocks\n"
        f"{'='*55}\n"
    )

    # Use trange for the progress bar
    pbar = trange(total_iterations, desc="Optimizing")

    for _ in pbar:
        k = state.k

        gamma = compute_lr(k, k_pre, state.k_end_heat, gamma_0, alpha_decay)

        z_new, mcmc_var_new = mcmc_draw(
            state,
            params_static,
            init_net_state,
            traj_true,
            boundaries,
            num_blocks,
            sampler_iters,
            adapt_alpha,
            adapt_beta,
        )

        state, norm_jax, ema_new_jax = pure_update_step(
            state,
            z_new,
            mcmc_var_new,
            gamma,
            k_pre,
            c_heat,
            params_static,
            init_net_state,
            traj_true,
            boundaries,
        )
        norm = float(norm_jax)
        ema_new = float(ema_new_jax)

        if k >= k_pre and state.k_end_heat is None and len(grad_norm_history) > 1:
            if ema_new > grad_norm_history[-1]:
                state = state._replace(k_end_heat=k)
                # Use pbar.write to prevent overwriting the progress bar
                pbar.write(f"[SGD] Heating ended at k={k}.")

        # Record histories
        theta_history.append(state.theta)
        lr_history.append(gamma)
        grad_norm_history.append(ema_new)

        diag = StepDiagnostic(
            k=k,
            gamma=gamma,
            grad_norm=norm,
            ema_norm=ema_new,
            k_end_heat=state.k_end_heat,
        )
        all_diag.append(diag)

        # Determine phase for the progress bar metrics
        phase = (
            "pre-heat"
            if k < k_pre
            else ("heat" if state.k_end_heat is None else "cool")
        )

        # Update progress bar suffix metrics
        pbar.set_postfix(
            {
                "γ": f"{gamma:.2e}",
                "‖g‖": f"{norm:.4f}",
                "EMA": f"{ema_new:.4f}",
                "Phase": phase,
            }
        )

    print(f"\n[SGD] Done. k={state.k}")

    return RunResult(
        state=state,
        theta_history=theta_history,
        lr_history=lr_history,
        grad_norm_history=grad_norm_history,
        diagnostics=all_diag,
    )


import matplotlib.pyplot as plt


def plot_optimizer_history(result: RunResult, p_true=None):
    """
    Plots the full optimization history of latent and scalar parameters,
    along with optimizer diagnostics.
    """
    epochs = range(len(result.theta_history))

    # 1. Extract physical trajectories for spatial parameters
    # Shape will be (epochs, N)
    alphas = np.array(
        [((jax.nn.softplus(t.latent.alpha) + 1e-6) * 1.0) for t in result.theta_history]
    )
    rhos = np.array(
        [
            ((jax.nn.softplus(t.latent.critical_density) + 1e-6) * 10.0)
            for t in result.theta_history
        ]
    )
    vs = np.array(
        [
            ((jax.nn.softplus(t.latent.free_flow_speed) + 1e-6) * 100.0)
            for t in result.theta_history
        ]
    )

    N = alphas.shape[1]
    mid_idx = N // 2  # Highlight the middle section for clarity

    # =====================================================================
    # FIGURE 1: Spatial Parameters (Latent)
    # =====================================================================
    fig1, axs1 = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig1.suptitle(
        "Optimization Trajectory: Spatial Parameters (Physical Space)",
        fontsize=15,
        fontweight="bold",
        y=0.96,
    )

    # Alpha
    axs1[0].plot(
        epochs, alphas, color="dodgerblue", alpha=0.1
    )  # Background spread of all sections
    axs1[0].plot(
        epochs,
        alphas[:, mid_idx],
        color="navy",
        linewidth=2,
        label=f"Section {mid_idx} (MAP Estimate)",
    )
    if p_true:
        axs1[0].axhline(
            p_true.latent_params.alpha[mid_idx],
            color="red",
            linestyle="--",
            linewidth=2,
            label="True Value",
        )
    axs1[0].set_title(r"Trajectory of $\alpha$")
    axs1[0].legend(loc="upper right")
    axs1[0].grid(True, linestyle=":", alpha=0.6)

    # Critical Density
    axs1[1].plot(epochs, rhos, color="mediumseagreen", alpha=0.1)
    axs1[1].plot(
        epochs,
        rhos[:, mid_idx],
        color="darkgreen",
        linewidth=2,
        label=f"Section {mid_idx} (MAP Estimate)",
    )
    if p_true:
        axs1[1].axhline(
            p_true.latent_params.critical_density[mid_idx],
            color="red",
            linestyle="--",
            linewidth=2,
            label="True Value",
        )
    axs1[1].set_title(r"Trajectory of $\rho_{cr}$")
    axs1[1].legend(loc="upper right")
    axs1[1].grid(True, linestyle=":", alpha=0.6)

    # Free Flow Speed
    axs1[2].plot(epochs, vs, color="coral", alpha=0.1)
    axs1[2].plot(
        epochs,
        vs[:, mid_idx],
        color="darkred",
        linewidth=2,
        label=f"Section {mid_idx} (MAP Estimate)",
    )
    if p_true:
        axs1[2].axhline(
            p_true.latent_params.free_flow_speed[mid_idx],
            color="red",
            linestyle="--",
            linewidth=2,
            label="True Value",
        )
    axs1[2].set_title(r"Trajectory of $v_{free}$")
    axs1[2].set_xlabel("Optimization Step")
    axs1[2].legend(loc="upper right")
    axs1[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)

    # =====================================================================
    # FIGURE 2: Scalar Parameters
    # =====================================================================
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig2.suptitle(
        "Optimization Trajectory: Global Scalar Parameters",
        fontsize=15,
        fontweight="bold",
        y=0.96,
    )
    axs2 = axs2.flatten()

    for i, field in enumerate(["beta", "mu", "kappa", "gamma"]):
        vals = [getattr(t.scalar, field) for t in result.theta_history]
        axs2[i].plot(epochs, vals, linewidth=2, color="purple", label="Estimated Point")
        if p_true and hasattr(p_true.scalar_params, field):
            axs2[i].axhline(
                float(getattr(p_true.scalar_params, field)),
                color="red",
                linestyle="--",
                linewidth=2,
                label="True Value",
            )
        axs2[i].set_title(field)
        axs2[i].legend(loc="upper right")
        axs2[i].grid(True, linestyle=":", alpha=0.6)
        if i >= 2:
            axs2[i].set_xlabel("Optimization Step")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)

    # =====================================================================
    # FIGURE 3: Optimizer Diagnostics
    # =====================================================================
    fig3, axs3 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig3.suptitle(
        "AdaGrad Optimizer Diagnostics", fontsize=15, fontweight="bold", y=0.96
    )

    # Learning Rate
    axs3[0].plot(result.lr_history, color="teal", linewidth=2)
    axs3[0].set_title("Learning Rate Schedule ($\gamma$)")
    axs3[0].set_ylabel("Learning Rate")
    axs3[0].grid(True, linestyle=":", alpha=0.6)

    # Gradients
    axs3[1].plot(
        result.grad_norm_history, color="crimson", linewidth=2, label="EMA Norm"
    )

    # Mark where cooling started
    k_end_heat = result.diagnostics[-1].k_end_heat
    if k_end_heat is not None:
        axs3[1].axvline(
            k_end_heat,
            color="black",
            linestyle="--",
            label=f"Cooling starts (k={k_end_heat})",
        )

    axs3[1].set_title("Gradient Norm (EMA)")
    axs3[1].set_xlabel("Optimization Step")
    axs3[1].set_ylabel("Norm Value")
    axs3[1].legend(loc="upper right")
    axs3[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import parapersistentExitationSimulation as peSim
    from hwg_met import disturb_measurment

    # 1. Setup PRNG keys and simulate base data
    keys = jax.random.split(jax.random.PRNGKey(1137), 2)
    traj_true, full_p_true, boundaries, init_net_state = peSim.simulate_example()

    # 2. Disturb measurements to create noisy observation trajectory
    new_flow = disturb_measurment(keys[0], traj_true.flow)
    new_speed = disturb_measurment(keys[1], traj_true.speed)
    trj_dis = parametanet.SimulationTrajectory(
        density=new_flow / (new_speed * full_p_true.static_params.lambda_),
        speed=new_speed,
        flow=new_flow,
    )

    N = full_p_true.static_params.L.shape[0]

    # 3. Initialize Theta
    theta_init = Theta(
        latent=full_p_true.latent_params,
        scalar=full_p_true.scalar_params,
        prior=PriorParameters(
            mean=jnp.array([jnp.full(N, 1.62), jnp.full(N, 3.16), jnp.full(N, 0.84)]),
            log_var=jnp.log(jnp.array([0.005, 0.005, 0.005])),
            corr=jnp.array(0.95),
        ),
        obs_var=DataLogVariance(
            log_var_flow=jnp.log(jnp.array(0.01)),
            log_var_speed=jnp.log(jnp.array(0.01)),
        ),
    )

    # 4. Execute the refactored run loop (returns RunResult NamedTuple)
    result = run(
        theta_init=theta_init,
        params_static=full_p_true.static_params,
        init_net_state=init_net_state,
        traj_true=trj_dis,
        boundaries=boundaries,
        total_iterations=3,
        k_pre=10,
        gamma_0=1e-4,
        alpha_decay=2.0 / 3.0,
        c_heat=1e-3,
        adagrad_eps=1e-8,
        num_blocks=20,
        sampler_iters=10,
    )

    # 5. Extract and print final results cleanly using dot notation
    z_final = result.state.theta.latent

    print("\n" + "-" * 55)
    print(" FINAL LATENT ESTIMATES (PHYSICAL SPACE)")
    print("-" * 55)
    print(
        f"  α      : {np.array((jax.nn.softplus(z_final.alpha)            + 1e-6) * 1.0)}"
    )
    print(
        f"  ρ_cr   : {np.array((jax.nn.softplus(z_final.critical_density) + 1e-6) * 10.0)}"
    )
    print(
        f"  v_free : {np.array((jax.nn.softplus(z_final.free_flow_speed)  + 1e-6) * 100.0)}"
    )

    s = result.state.theta.scalar
    print("\n" + "-" * 55)
    print(" FINAL SCALAR ESTIMATES")
    print("-" * 55)
    print(f"  beta  = {s.beta:.4f}")
    print(f"  mu    = {s.mu:.4f}")
    print(f"  kappa = {s.kappa:.4f}")
    print(f"  gamma = {s.gamma:.4f}")

    p = result.state.theta.prior
    print("\n" + "-" * 55)
    print(" FINAL PRIOR HYPERPARAMETER ESTIMATES")
    print("-" * 55)
    print(f"  log_var = {np.array(p.log_var)}")
    print(f"  corr    = {float(p.corr):.4f}\n")

    plot_optimizer_history(result, p_true=full_p_true)
