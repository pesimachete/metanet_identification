import typing
import os
import json
import multiprocessing as mp

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

import parametanet
import parapersistentExitationSimulation as peSim

print("Available devices:", jax.devices())


# ---------------------------------------------------------
# Data Structures
# ---------------------------------------------------------
class LearningParams(typing.NamedTuple):
    metanet: parametanet.ParaNetworkParameters
    log_vars: jax.Array


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def inv_softplus(x, scale):
    def logexpm1(v):
        return v + jnp.log(-jnp.expm1(-v))

    return logexpm1(x / scale)


def get_physical_params(
    raw_metanet_params: parametanet.ParaNetworkParameters,
    scales: parametanet.ParaNetworkParameters,
):
    return parametanet.ParaNetworkParameters(
        T=raw_metanet_params.T,
        L=raw_metanet_params.L,
        lambda_=raw_metanet_params.lambda_,
        beta=jax.nn.softplus(raw_metanet_params.beta) * scales.beta,
        mu=jax.nn.softplus(raw_metanet_params.mu) * scales.mu,
        kappa=jax.nn.softplus(raw_metanet_params.kappa) * scales.kappa,
        gamma=jax.nn.softplus(raw_metanet_params.gamma) * scales.gamma,
        alpha=jax.nn.softplus(raw_metanet_params.alpha) * scales.alpha,
        critical_density=jax.nn.softplus(raw_metanet_params.critical_density)
        * scales.critical_density,
        free_flow_speed=jax.nn.softplus(raw_metanet_params.free_flow_speed)
        * scales.free_flow_speed,
    )


def nll_loss(
    params: LearningParams,
    traj_true: parametanet.SimulationTrajectory,
    initial_state: parametanet.NetworkState,
    boundaries: parametanet.BoundarySequence,
    scales: parametanet.ParaNetworkParameters,
    penalty_weight: float = 0.5,
):
    physical_metanet = get_physical_params(params.metanet, scales)
    traj_pred = parametanet.rollout_simulation(
        initial_state, boundaries, physical_metanet
    )

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

    empirical_fields = ["alpha", "critical_density", "free_flow_speed"]

    for field in physical_metanet._fields:
        if field in empirical_fields:
            p_pred = getattr(physical_metanet, field)
            p_diff = p_pred[:-1] - p_pred[1:]
            p_norm = 0.5 * (jnp.abs(p_pred[:-1] + p_pred[1:]) + 1e-8)
            reg_loss += jnp.sum((p_diff / p_norm) ** 2)

    return base_loss + 1 / 2 * (penalty_weight * reg_loss)


def disturb_measurment(key, measurment, ssq):
    return jnp.exp(jax.random.normal(key, measurment.shape) * ssq) * measurment


def perturb_and_inv_softplus_vec(key, true_val, scale, d=0.5):
    perturbed = jax.random.uniform(
        key, true_val.shape, minval=(1 - d) * true_val, maxval=(1 + d) * true_val
    )
    return inv_softplus(perturbed, scale)


def perturb_and_inv_softplus_float(key, true_val, scale, d=0.2):
    perturbed = jax.random.uniform(
        key, (), minval=(1 - d) * true_val, maxval=(1 + d) * true_val
    ).astype(jnp.float64)
    return inv_softplus(perturbed, scale)


def initialize_params(p_true, scales, keys):
    return parametanet.ParaNetworkParameters(
        beta=perturb_and_inv_softplus_float(keys[0], p_true.beta, scales.beta),
        free_flow_speed=perturb_and_inv_softplus_vec(
            keys[1], p_true.free_flow_speed, scales.free_flow_speed
        ),
        gamma=perturb_and_inv_softplus_float(keys[2], p_true.gamma, scales.gamma),
        mu=perturb_and_inv_softplus_float(keys[3], p_true.mu, scales.mu),
        critical_density=perturb_and_inv_softplus_vec(
            keys[4], p_true.critical_density, scales.critical_density
        ),
        alpha=perturb_and_inv_softplus_vec(keys[5], p_true.alpha, scales.alpha),
        kappa=perturb_and_inv_softplus_float(keys[6], p_true.kappa, scales.kappa),
        L=p_true.L,  # FROZEN
        lambda_=p_true.lambda_,  # FROZEN
        T=p_true.T,  # FROZEN
    )


# Global optimizer defined so the compiled loop can use it
learning_rate = 1e-4
optimizer = optax.lbfgs(learning_rate)


# ---------------------------------------------------------
# The Accelerated Training Loop (JAX lax.scan + L-BFGS)
# ---------------------------------------------------------
@jax.jit(static_argnames=["num_epochs"])
def train_loop(
    initial_params,
    opt_state,
    traj_true,
    initial_state,
    boundaries,
    scales,
    mask,
    num_epochs,
):

    def step(carry, epoch_idx):
        params, opt_state = carry
        penalty_weight = 100.0

        # L-BFGS requires a value function for its internal line search
        def value_fn(p):
            # Apply the mask dynamically inside the closure
            p_masked = jax.tree_util.tree_map(
                lambda current, original, m: jnp.where(m == 1.0, current, original),
                p,
                params,
                mask,
            )
            return nll_loss(
                p_masked, traj_true, initial_state, boundaries, scales, penalty_weight
            )

        loss, grads = jax.value_and_grad(value_fn)(params)

        updates, opt_state = optimizer.update(
            grads, opt_state, params, value=loss, grad=grads, value_fn=value_fn
        )

        new_params = optax.apply_updates(params, updates)

        # Carry forward the state, and accumulate loss & new_params for history
        return (new_params, opt_state), (loss, new_params)

    # Run the loop natively in C++/XLA
    final_state, history = jax.lax.scan(
        step, (initial_params, opt_state), jnp.arange(num_epochs)
    )

    return final_state, history


# ---------------------------------------------------------
# The Multiprocessing Worker
# ---------------------------------------------------------
def run_optimization(
    run_id, num_epochs, p_true, trj_dis, init_stat, boundaries, scales, mask
):
    print(f"Worker {run_id}: Starting {num_epochs} epochs using L-BFGS...")

    # Unique initialization per run
    keys = jax.random.split(jax.random.PRNGKey(2441 + run_id + 42), 7)
    initial_metanet_params = initialize_params(p_true, scales, keys)

    frac = jnp.sum(trj_dis.flow) / jnp.sum(trj_dis.speed)
    initial_params = LearningParams(
        metanet=initial_metanet_params,
        log_vars=jnp.array([jnp.log(1.0), jnp.log(1.0 * frac)]),
    )

    initial_params = jax.tree_util.tree_map(jnp.asarray, initial_params)
    opt_state = optimizer.init(initial_params)

    # 1. Execute XLA-compiled Loop
    final_state, history = train_loop(
        initial_params,
        opt_state,
        trj_dis,
        init_stat,
        boundaries,
        scales,
        mask,
        num_epochs,
    )

    # --- NEW: Force Python to wait for the math to ACTUALLY finish ---
    jax.block_until_ready(history)

    losses, params_history = history
    print(f"Worker {run_id}: Math actually finished. Processing history...")

    # 2. Vectorized conversion of raw history -> physical parameters
    physical_history = get_physical_params(params_history.metanet, scales)

    # 3. Downcast to Float32 and format for JSON using strictly NumPy
    import numpy as np  # Import standard numpy to safely exit JAX memory

    def to_list_f32(jax_arr):
        # np.array acts as a bridge, preventing XLA deadlocks during the cast
        return np.array(jax_arr, dtype=np.float32).tolist()

    param_dict = {}
    for field in physical_history._fields:
        if field not in ["L", "lambda_", "T"]:  # Exclude frozen
            param_dict[field] = to_list_f32(getattr(physical_history, field))

    # 4. Extract final variances
    final_log_vars = final_state[0].log_vars
    var_rho = float(jnp.exp(final_log_vars[0]))
    var_v = float(jnp.exp(final_log_vars[1]))

    print(f"Worker {run_id}: Finished all tasks.")
    return {
        "run_id": run_id,
        "loss_history": to_list_f32(losses),
        "parameter_history": param_dict,
        "final_variances": {"var_rho": var_rho, "var_v": var_v},
    }


# ---------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # Force 'spawn' to prevent XLA context locks between processes
    mp.set_start_method("spawn", force=True)

    # -------------------------------------------------------------
    # OPTIONAL: If it crashes with 3 workers, uncomment this line
    # to force JAX to only allocate memory as it needs it!
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # -------------------------------------------------------------

    print("Generating simulation data...")
    traj_true, p_true, boundaries, init_stat = peSim.simulate_example()

    ssq = 10e-2
    keys = jax.random.split(jax.random.PRNGKey(1129), 2)
    new_flow = disturb_measurment(keys[0], traj_true.flow, ssq)
    new_speed = disturb_measurment(keys[1], traj_true.speed, ssq)
    new_density = new_flow / new_speed * p_true.lambda_

    trj_dis = parametanet.SimulationTrajectory(
        density=new_density,
        speed=new_speed,
        flow=new_flow,
    )

    scales = parametanet.ParaNetworkParameters(
        beta=10 ** (jnp.floor(jnp.log10(p_true.beta))),
        free_flow_speed=10 ** (jnp.floor(jnp.log10(p_true.free_flow_speed))),
        kappa=10 ** (jnp.floor(jnp.log10(p_true.kappa))),
        gamma=10 ** (jnp.floor(jnp.log10(p_true.gamma))),
        critical_density=10 ** (jnp.floor(jnp.log10(p_true.critical_density))),
        alpha=10 ** (jnp.floor(jnp.log10(p_true.alpha))),
        mu=10 ** (jnp.floor(jnp.log10(p_true.mu))),
        L=1.0,
        lambda_=1.0,
        T=1.0,
    )

    mask = LearningParams(
        metanet=parametanet.ParaNetworkParameters(
            beta=1,
            free_flow_speed=jnp.ones_like(p_true.free_flow_speed),
            kappa=1,
            mu=1,
            critical_density=jnp.ones_like(p_true.critical_density),
            alpha=jnp.ones_like(p_true.alpha),
            gamma=1,
            L=jnp.zeros_like(p_true.L),
            lambda_=jnp.zeros_like(p_true.lambda_),
            T=0,
        ),
        log_vars=jnp.ones(2),
    )

    num_epochs = 2500
    num_runs = 3

    # Package arguments for the worker pool
    pool_args = [
        (i, num_epochs, p_true, trj_dis, init_stat, boundaries, scales, mask)
        for i in range(num_runs)
    ]

    print(f"\nDispatching {num_runs} runs to Multiprocessing Pool...")

    with mp.Pool(processes=num_runs) as pool:
        all_results = pool.starmap(run_optimization, pool_args)

    output_file = "all_runs_LBFGS_results.json"
    print(f"Writing all results to {output_file}...")

    with open(output_file, "w") as f:
        json.dump(all_results, f)

    print("Done! Data is ready for analysis.")
