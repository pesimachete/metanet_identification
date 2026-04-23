import typing
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import os
import json

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"

import jax
import jax.numpy as jnp
import optax


import metanet
import persistentExitationSimulation as peSim

import os


print(jax.devices())
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class LearningParams(typing.NamedTuple):
    metanet: metanet.NetworkParameters
    log_vars: jax.Array


def inv_softplus(x, scale):
    def logexpm1(x):
        return x + jnp.log(-jnp.expm1(-x))

    return logexpm1(x / scale)


def get_physical_params(raw_metanet_params, scales):
    return metanet.NetworkParameters(
        tau=jax.nn.softplus(raw_metanet_params.tau) * scales.tau,
        free_flow_speed=jax.nn.softplus(raw_metanet_params.free_flow_speed)
        * scales.free_flow_speed,
        kappa=jax.nn.softplus(raw_metanet_params.kappa) * scales.kappa,
        nu=jax.nn.softplus(raw_metanet_params.nu) * scales.nu,
        critical_density=jax.nn.softplus(raw_metanet_params.critical_density)
        * scales.critical_density,
        alpha=jax.nn.softplus(raw_metanet_params.alpha) * scales.alpha,
        delta=jax.nn.softplus(raw_metanet_params.delta) * scales.delta,
        L=raw_metanet_params.L,
        lambda_=raw_metanet_params.lambda_,
        T=raw_metanet_params.T,
    )


def calc_param_errors(params, p_true, mask, scales):

    errors = {}
    physical_metanet = get_physical_params(params.metanet, scales)

    for field in physical_metanet._fields:
        m = getattr(mask.metanet, field)

        if jnp.any(m > 0):
            p_pred = getattr(physical_metanet, field)
            p_tgt = getattr(p_true, field)

            rel_diff = jnp.abs((p_pred - p_tgt) / (p_tgt + 1e-8))

            error_term = jnp.sum(rel_diff) / (jnp.sum(m) + 1e-8)
            errors[field] = error_term

    return errors


def nll_loss(
    params: LearningParams,
    traj_true: metanet.SimulationTrajectory,
    initial_state: metanet.NetworkState,
    boundaries: metanet.BoundarySequence,
    scales: metanet.NetworkParameters,
    penalty_weight: float = 0.5,
):

    physical_metanet = get_physical_params(params.metanet, scales)

    traj_pred = metanet.rollout_simulation(initial_state, boundaries, physical_metanet)

    var_q = jnp.exp(params.log_vars[0])
    var_v = jnp.exp(params.log_vars[1])

    n = traj_pred.flow.shape[0]

    loss_q = 0.5 * n * params.log_vars[0] + 0.5 * jnp.sum(
        (jnp.log(traj_pred.flow) - jnp.log(traj_true.flow)) ** 2
    ) / (var_q + 1e-7)
    loss_v = 0.5 * n * params.log_vars[1] + 0.5 * jnp.sum(
        (jnp.log(traj_pred.speed) - jnp.log(traj_true.speed)) ** 2
    ) / (var_v + 1e-7)

    base_loss = loss_q + loss_v

    reg_loss = 0.0

    empirical_fields = [
        "alpha",
        "critical_density",
        "free_flow_speed",
    ]

    """  
    "tau",
    "nu",
    "kappa",
    "delta",
    """

    for field in physical_metanet._fields:
        if field in empirical_fields:
            p_pred = getattr(physical_metanet, field)
            p_diff = p_pred[:-1] - p_pred[1:]
            p_norm = 0.5 * (jnp.abs(p_pred[:-1] + p_pred[1:]) + 1e-8)
            reg_loss += jnp.sum((p_diff / p_norm) ** 2)

    return base_loss + 1 / 2 * (penalty_weight * reg_loss)


traj_true, p_true, boundaries, init_stat = peSim.simulate_example()


ssq = 10e-2


def disturb_measurment(key, measurment):
    return jnp.exp(jax.random.normal(key, measurment.shape) * ssq) * measurment


keys = jax.random.split(jax.random.PRNGKey(1129), 2)
new_flow = disturb_measurment(keys[0], traj_true.flow)
new_speed = disturb_measurment(keys[1], traj_true.speed)
new_density = new_flow / new_speed * p_true.lambda_
trj_dis = metanet.SimulationTrajectory(
    density=new_density,
    speed=new_speed,
    flow=new_flow,
)

K = traj_true.density.shape[0]

d = 0.6


def perturb_and_inv_softplus_vec(key, true_val, scale, d=1.0):
    perturbed = jax.random.uniform(
        key,
        true_val.shape,
        minval=(1 - d) * true_val,
        maxval=(1 + d) * true_val,
    )
    return inv_softplus(perturbed, scale)


def perturb_and_inv_softplus_float(key, true_val, scale, d=0.5):
    perturbed = jax.random.uniform(
        key,
        (),
        minval=(1 - d) * true_val,
        maxval=(1 + d) * true_val,
    ).astype(jnp.float32)
    return inv_softplus(perturbed, scale)


keys = jax.random.split(jax.random.PRNGKey(1187), 7)
scales = metanet.NetworkParameters(
    tau=10 ** (jnp.floor(jnp.log10(p_true.tau))),
    free_flow_speed=10 ** (jnp.floor(jnp.log10(p_true.free_flow_speed))),
    kappa=10 ** (jnp.floor(jnp.log10(p_true.kappa))),
    nu=10 ** (jnp.floor(jnp.log10(p_true.nu))),
    critical_density=10 ** (jnp.floor(jnp.log10(p_true.critical_density))),
    alpha=10 ** (jnp.floor(jnp.log10(p_true.alpha))),
    delta=10 ** (jnp.floor(jnp.log10(p_true.delta))),
    L=1.0,
    lambda_=1.0,
    T=1.0,
)
initial_metanet_params = metanet.NetworkParameters(
    tau=perturb_and_inv_softplus_float(keys[0], p_true.tau, scales.tau),
    free_flow_speed=perturb_and_inv_softplus_vec(
        keys[1], p_true.free_flow_speed, scales.free_flow_speed
    ),
    kappa=perturb_and_inv_softplus_float(keys[2], p_true.kappa, scales.kappa),
    nu=perturb_and_inv_softplus_float(keys[3], p_true.nu, scales.nu),
    critical_density=perturb_and_inv_softplus_vec(
        keys[4], p_true.critical_density, scales.critical_density
    ),
    alpha=perturb_and_inv_softplus_vec(keys[5], p_true.alpha, scales.alpha),
    delta=perturb_and_inv_softplus_float(keys[6], p_true.delta, scales.delta),
    L=p_true.L,  # FROZEN
    lambda_=p_true.lambda_,  # FROZEN
    T=p_true.T,  # FROZEN
)

print("Initial METANET Parameters (after perturbation):")
initial_phyical_metanet = get_physical_params(initial_metanet_params, scales)
for field in initial_metanet_params._fields:
    val = getattr(initial_phyical_metanet, field)
    if jnp.size(val) == 1:
        print(f"  {field:>20}: {float(jnp.squeeze(val)):.4f}")
    else:
        with np.printoptions(precision=4, suppress=True, edgeitems=2, threshold=5):
            print(f"  {field:>20}: {np.array(val)}")


frac = jnp.sum(traj_true.flow) / jnp.sum(
    traj_true.speed
)  # This does not work in this fake data.
initial_params = LearningParams(
    metanet=initial_metanet_params,
    log_vars=jnp.array([jnp.log(1), jnp.log(1 * frac)]),
)

mask = LearningParams(
    metanet=metanet.NetworkParameters(
        tau=1,
        free_flow_speed=jnp.ones_like(p_true.free_flow_speed),
        kappa=1,
        nu=1,
        critical_density=jnp.ones_like(p_true.critical_density),
        alpha=jnp.ones_like(p_true.alpha),
        delta=1,
        L=jnp.zeros_like(p_true.L),  # FROZEN
        lambda_=jnp.zeros_like(p_true.lambda_),  # FROZEN
        T=0,  # FROZEN
    ),
    log_vars=jnp.ones_like(initial_params.log_vars),
)


learning_rate = 1e-3
optimizer = optax.adam(learning_rate)

opt_state = optimizer.init(initial_params)


# Define a JIT-compiled update step for speed
@jax.jit
def update_step(params, opt_state, traj_true, initial_state, boundaries, scales, mask):

    penalty_weight = 10

    loss, grads = jax.value_and_grad(nll_loss)(
        params, traj_true, initial_state, boundaries, scales, penalty_weight
    )

    grads = jax.tree_util.tree_map(
        lambda g, m: jnp.where(m == 1.0, g, 0.0), grads, mask
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, loss


params = initial_params

learning_rate = 1e-3
optimizer = optax.adam(learning_rate)

opt_state = optimizer.init(initial_params)


# Define a JIT-compiled update step for speed
@jax.jit
def update_step(params, opt_state, traj_true, initial_state, boundaries, scales, mask):

    penalty_weight = 10

    loss, grads = jax.value_and_grad(nll_loss)(
        params, traj_true, initial_state, boundaries, scales, penalty_weight
    )

    grads = jax.tree_util.tree_map(
        lambda g, m: jnp.where(m == 1.0, g, 0.0), grads, mask
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, loss


params = initial_params

# Configuration
max_epochs = 3000000
tolerance = 1e-2  # Your threshold: 10^-2
prev_loss = float("inf")

# Initialize lists
nll_loss_history = []
param_histories = {}

print("\nStarting parameter identification with early stopping...")
pbar = tqdm.tqdm(range(max_epochs))
mem_loss = 0.3

num_epochs = 0
loss_memo = 0
for epoch in pbar:
    # 1. Perform optimization step
    params, opt_state, loss = update_step(
        params, opt_state, trj_dis, init_stat, boundaries, scales, mask
    )

    current_loss = float(loss)

    # 2. Check for convergence (Early Stopping)
    # We check the absolute difference between the previous and current loss

    loss_diff = abs(prev_loss - current_loss)

    if loss_diff < float("inf"):
        loss_memo = loss_diff + mem_loss * loss_memo
    num_epochs = epoch
    if epoch > 0 and loss_memo < tolerance:
        num_epochs = epoch
        print(
            f"\nConverged at epoch {epoch}: Loss difference {loss_diff:.6e} < {tolerance}, Loss Memory {loss_memo:.6e}"
        )
        break

    prev_loss = current_loss

    # 3.Calculate physical parameters for logging
    physical_metanet_current = get_physical_params(params.metanet, scales)

    # Store metrics
    nll_loss_history.append(current_loss)

    for field in physical_metanet_current._fields:
        if field not in ["L", "lambda_", "T"]:  # Only track learnable parameters
            if field not in param_histories:
                param_histories[field] = []

            val = getattr(physical_metanet_current, field)
            # Store the full vector (shape (20,)) or scalar as a numpy array
            param_histories[field].append(np.array(val))

    # 5. Update progress bar
    if epoch % 50 == 0:
        var_rho = jnp.exp(params.log_vars[0])
        var_v = jnp.exp(params.log_vars[1])
        pbar.set_description(
            f"Loss: {current_loss:.4f} | ΔLoss: {loss_diff:.4e} | Loss Memo: {loss_memo:.4e}"
        )

print("\nIdentified METANET Parameters vs True Parameters:")

physical_metanet_final = get_physical_params(params.metanet, scales)

for field in physical_metanet_final._fields:
    learned_val = getattr(physical_metanet_final, field)
    true_val = getattr(p_true, field)
    frozen_str = "(FROZEN)" if field in ["L", "lambda_", "T"] else ""

    if jnp.size(learned_val) == 1:
        l_str = f"{float(jnp.squeeze(learned_val)):.4f}"
        t_str = f"{float(jnp.squeeze(true_val)):.4f}"
    else:
        with np.printoptions(precision=4, suppress=True, edgeitems=2, threshold=5):
            l_str = f"{np.array(learned_val)}"
            t_str = f"{np.array(true_val)}"

    print(f"  {field:>16}: {l_str} \t(True: {t_str}) {frozen_str}")

print("\nLearned Error Covariances (Variances):")
print(f"  Density (rho) variance : {float(jnp.exp(params.log_vars[0])):.4f}")
print(f"  Speed (v) variance     : {float(jnp.exp(params.log_vars[1])):.4f}")


print("\nSaving parameters to JSON...")

# Create a dictionary to hold the data
params_to_save = {}

# Iterate through the physical parameters and convert them to standard Python types
for field in physical_metanet_final._fields:
    val = getattr(physical_metanet_final, field)

    # If it's a JAX/NumPy array, convert it to a standard Python list or float
    if hasattr(val, "tolist"):
        params_to_save[field] = val.tolist()
    else:
        params_to_save[field] = float(val)

# It is also helpful to save your learned variances
params_to_save["learned_variance_rho"] = float(jnp.exp(params.log_vars[0]))
params_to_save["learned_variance_v"] = float(jnp.exp(params.log_vars[1]))

# Create directory if it doesn't exist
os.makedirs("id_params", exist_ok=True)
output_filename = "id_params/identified_parameters.json"

with open(output_filename, "w") as f:
    json.dump(params_to_save, f, indent=4)

print(f"Parameters successfully saved to '{output_filename}'.")


# ==========================================
# 4. PLOT THE RESULTS (All 20 Segments)
# ==========================================
learnable_fields = [f for f in param_histories.keys()]
num_params = len(learnable_fields)

# Set up the grid: rows x 3 columns
cols = 3
rows = (num_params + 1) // cols + 1
fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
axes = axes.flatten()

for i, field in enumerate(learnable_fields):
    ax = axes[i]
    # Convert list of arrays to a 2D numpy array: Shape (Epochs, N)
    history = np.array(param_histories[field])

    # Get True Values from p_true
    true_val = np.array(getattr(p_true, field))

    if history.ndim == 2:  # It's a vector (N=20)
        # Plot all 20 lines for the estimation
        lines = ax.plot(history, alpha=0.6, linewidth=1)
        # Plot 20 dashed lines for the true values
        for segment_idx in range(history.shape[1]):
            ax.axhline(
                y=true_val[segment_idx],
                color=lines[segment_idx].get_color(),
                linestyle="--",
                alpha=0.4,
            )
        ax.set_title(f"Parameter: {field}")
    else:  # It's a scalar
        ax.plot(history, color="blue", linewidth=2, label="Estimated")
        ax.axhline(y=float(true_val), color="red", linestyle="--", label="True Value")
        ax.set_title(f"Parameter: {field}")
        ax.legend()

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Value")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

# Place the NLL Loss in the final slot
ax_loss = axes[num_params]
ax_loss.plot(nll_loss_history, color="black", linewidth=1.5)
ax_loss.set_title("Optimization: NLL Loss")
ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True, which="both", linestyle="--", alpha=0.5)

# Clean up empty slots
for j in range(num_params + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
