import itertools
import typing
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import os


import jax
import jax.numpy as jnp
import optax


import metanet
import persistentExitationSimulation as peSim


os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class LearningParams(typing.NamedTuple):
    metanet: metanet.NetworkParameters
    log_vars: jax.Array


def inv_softplus(x):
    """Numerically stable inverse softplus for initializing raw parameters."""
    return x + jnp.log(-jnp.expm1(-x))


def get_physical_params(raw_metanet_params):
    """Applies softplus to trainable parameters to strictly enforce positivity."""
    return metanet.NetworkParameters(
        tau=jax.nn.softplus(raw_metanet_params.tau),
        free_flow_speed=jax.nn.softplus(raw_metanet_params.free_flow_speed),
        kappa=jax.nn.softplus(raw_metanet_params.kappa),
        nu=jax.nn.softplus(raw_metanet_params.nu),
        critical_density=jax.nn.softplus(raw_metanet_params.critical_density),
        alpha=jax.nn.softplus(raw_metanet_params.alpha),
        delta=jax.nn.softplus(raw_metanet_params.delta),
        # Keep frozen parameters unchanged
        L=raw_metanet_params.L,
        lambda_=raw_metanet_params.lambda_,
        T=raw_metanet_params.T,
    )


def calc_param_error(params, p_true, mask):
    """
    Calculates the Relative Mean Squared Error between predicted and true parameters.
    Only considers trainable parameters defined by the mask.
    """
    total_error = 0.0

    physical_metanet = get_physical_params(params.metanet)

    # Iterate over all fields in the metanet parameters
    for field in physical_metanet._fields:
        p_pred = getattr(physical_metanet, field)
        p_tgt = getattr(p_true, field)
        m = getattr(mask.metanet, field)

        # Calculate relative error: ((pred - true) / true)^2
        # Adding 1e-8 to avoid division by zero
        rel_diff = (p_pred - p_tgt) / (p_tgt + 1e-8)

        # Multiply by mask to ignore frozen parameters, and sum the array
        error_term = jnp.sum((rel_diff**2) * m)
        total_error += error_term

    return total_error


def nll_loss(
    params: LearningParams,
    traj_true: metanet.SimulationTrajectory,
    initial_state: metanet.NetworkState,
    boundaries: metanet.BoundarySequence,
    penalty_weight: float = 0.5,
):

    physical_metanet = get_physical_params(params.metanet)

    traj_pred = metanet.rollout_simulation(initial_state, boundaries, physical_metanet)

    var_rho = jnp.exp(params.log_vars[0])
    var_v = jnp.exp(params.log_vars[1])
    n = traj_true.density.size

    loss_rho = (
        0.5 * params.log_vars[0]
        + 0.5 * jnp.mean((traj_pred.density - traj_true.density) ** 2) / var_rho
    )
    loss_v = (
        0.5 * params.log_vars[1]
        + 0.5 * jnp.mean((traj_pred.speed - traj_true.speed) ** 2) / var_v
    )

    base_loss = loss_rho + loss_v

    reg_loss = 0.0

    empirical_fields = [
        "tau",
        "nu",
        "kappa",
        "delta",
        "alpha",
        "critical_density",
        "free_flow_speed",
    ]
    for field in physical_metanet._fields:
        if field in empirical_fields:
            p_pred = getattr(physical_metanet, field)
            for p0, p1 in itertools.pairwise(jnp.ravel(p_pred)):
                reg_loss += jnp.sum((p0 - p1) ** 2)

    return base_loss + (penalty_weight * reg_loss)


traj_true, p_true, boundaries, init_stat = peSim.simulate_example()

ssq = 10e-5


def disturb_measurment(key, measurment):
    return jax.random.normal(key, measurment.shape) * ssq + measurment


keys = jax.random.split(jax.random.PRNGKey(42), 2)
new_density = disturb_measurment(keys[0], traj_true.density)
new_speed = disturb_measurment(keys[1], traj_true.speed)
new_flow = new_density * new_speed * p_true.lambda_
trj_dis = metanet.SimulationTrajectory(
    density=new_density,
    speed=new_speed,
    flow=new_flow,
)

K = traj_true.density.shape[0]

d = 0.1


def perturb_and_inv_softplus(key, true_val):
    # Perturb the ground truth, then apply inverse softplus so that when
    # the model applies softplus, it returns exactly this perturbed value
    perturbed = jax.random.uniform(
        key,
        true_val.shape,
        minval=(1 - d) * true_val,
        maxval=(1 + d) * true_val,
    )
    return inv_softplus(perturbed)


keys = jax.random.split(jax.random.PRNGKey(42), 7)
initial_metanet_params = metanet.NetworkParameters(
    tau=perturb_and_inv_softplus(keys[0], p_true.tau),
    free_flow_speed=perturb_and_inv_softplus(keys[1], p_true.free_flow_speed),
    kappa=perturb_and_inv_softplus(keys[2], p_true.kappa),
    nu=perturb_and_inv_softplus(keys[3], p_true.nu),
    critical_density=perturb_and_inv_softplus(keys[4], p_true.critical_density),
    alpha=perturb_and_inv_softplus(keys[5], p_true.alpha),
    delta=perturb_and_inv_softplus(keys[6], p_true.delta),
    L=p_true.L,  # FROZEN
    lambda_=p_true.lambda_,  # FROZEN
    T=p_true.T,  # FROZEN
)

frac = jnp.sum(traj_true.flow) / jnp.sum(traj_true.speed)
initial_params = LearningParams(
    metanet=initial_metanet_params,
    log_vars=jnp.array([jnp.log(1), jnp.log(1 * frac)]),
)


mask = LearningParams(
    metanet=metanet.NetworkParameters(
        tau=jnp.ones_like(p_true.tau),
        free_flow_speed=jnp.ones_like(p_true.free_flow_speed),
        kappa=jnp.ones_like(p_true.kappa),
        nu=jnp.ones_like(p_true.nu),
        critical_density=jnp.ones_like(p_true.critical_density),
        alpha=jnp.ones_like(p_true.alpha),
        delta=jnp.ones_like(p_true.delta),
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
def update_step(params, opt_state, traj_true, initial_state, boundaries, mask):

    penalty_weight = 0.8

    loss, grads = jax.value_and_grad(nll_loss)(
        params, traj_true, initial_state, boundaries, penalty_weight
    )

    # SAFELY zero out frozen gradients using jnp.where
    grads = jax.tree_util.tree_map(
        lambda g, m: jnp.where(m == 1.0, g, 0.0), grads, mask
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, loss


num_epochs = 10
params = initial_params

# Initialize lists to store the history for plotting
nll_loss_history = []
param_error_history = []

print("\nStarting parameter identification...")
pbar = tqdm.tqdm(range(num_epochs))
for epoch in pbar:
    # Perform optimization step
    params, opt_state, loss = update_step(
        params, opt_state, trj_dis, init_stat, boundaries, mask
    )

    # Calculate parameter error against ground truth (p_true)
    p_err = calc_param_error(params, p_true, mask)

    # Store metrics
    nll_loss_history.append(float(loss))
    param_error_history.append(float(p_err))

    # Update progress bar every 50 epochs
    if epoch % 50 == 0:
        var_rho = jnp.exp(params.log_vars[0])
        var_v = jnp.exp(params.log_vars[1])
        pbar.set_description(
            f"Loss: {loss:.4f} | Param Err: {p_err:.4f} | Var Rho: {var_rho:.3f} | Var V: {var_v:.3f}"
        )


print("\nIdentified METANET Parameters vs True Parameters:")

physical_metanet_final = get_physical_params(params.metanet)

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

    print(f"  {field:>10}: {l_str} \t(True: {t_str}) {frozen_str}")

print("\nLearned Error Covariances (Variances):")
print(f"  Density (rho) variance : {float(jnp.exp(params.log_vars[0])):.4f}")
print(f"  Speed (v) variance     : {float(jnp.exp(params.log_vars[1])):.4f}")

import json

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

# Write the dictionary to a JSON file with pretty formatting
output_filename = "id_params/identified_parameters.json"
with open(output_filename, "w") as f:
    json.dump(params_to_save, f, indent=4)

print(f"Parameters successfully saved to '{output_filename}'.")


# ==========================================
# 4. PLOT THE RESULTS
# ==========================================
plt.figure(figsize=(12, 5))

# Plot 1: The real vs predicted parameter error (Relative MSE)
plt.subplot(1, 2, 1)
plt.plot(
    range(num_epochs),
    param_error_history,
    label="Relative Parameter Error",
    color="red",
)
plt.xlabel("Epochs")
plt.ylabel("Parameter Error (Log Scale)")
plt.title("Parameter Error vs. Epochs")
plt.yscale("log")  # Log scale is usually best to visualize parameter convergence
plt.grid(True, which="both", ls="--")
plt.legend()

# Plot 2: The standard NLL loss
plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), nll_loss_history, label="NLL Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("NLL Loss")
plt.title("Negative Log-Likelihood Loss vs. Epochs")
plt.grid(True, which="both", ls="--")
plt.legend()

plt.tight_layout()
plt.show()
