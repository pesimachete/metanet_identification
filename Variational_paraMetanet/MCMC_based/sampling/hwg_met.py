import os
import pickle
import typing
import functools
from datetime import datetime
import matplotlib.pyplot as plt

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from Variational_paraMetanet.MCMC_based.simulator import parametanet
from Variational_paraMetanet.MCMC_based.simulator import (
    parapersistentExitationSimulation as peSim,
)
from Variational_paraMetanet.MCMC_based.core.objective import (
    log_joint_jit,
)

from Variational_paraMetanet.MCMC_based.core.parameter_handling import (
    PriorParameters,
    DataLogVariance,
    Theta,
)


def save_mcmc_results(filepath, results_dict):
    with open(filepath, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"\n[+] Successfully saved all MCMC data to: {filepath}")


@functools.partial(jax.jit, static_argnames=("num_blocks",))
def gibbs_step(
    current_sample: parametanet.ParaNetworkLatentParameters,
    current_log_post: jax.Array,
    variances: jax.Array,
    key: jax.Array,
    theta: Theta,
    params_static: parametanet.ParaNetworkStaticParameters,
    scales_scalar: parametanet.ParaNetworkScalarParameters,
    init_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
    num_blocks: int,
) -> tuple[parametanet.ParaNetworkLatentParameters, jax.Array, jax.Array]:
    N = current_sample.alpha.shape[0]
    keys = jax.random.split(key, 3 * num_blocks * 2)
    acceptances = jnp.zeros((3, num_blocks), dtype=jnp.float32)

    curr_samp = current_sample
    curr_lp = current_log_post
    key_idx = 0

    for p_idx in range(3):
        for b in range(num_blocks):
            mask = (jnp.arange(N) * num_blocks // N) == b
            noise = jax.random.normal(keys[key_idx], shape=(N,))
            u = jax.random.uniform(keys[key_idx + 1])
            key_idx += 2

            if p_idx == 0:
                prop_val = jnp.where(
                    mask, curr_samp.alpha + noise * variances[0, b], curr_samp.alpha
                )
                prop_samp = parametanet.ParaNetworkLatentParameters(
                    prop_val, curr_samp.critical_density, curr_samp.free_flow_speed
                )
            elif p_idx == 1:
                prop_val = jnp.where(
                    mask,
                    curr_samp.critical_density + noise * variances[1, b],
                    curr_samp.critical_density,
                )
                prop_samp = parametanet.ParaNetworkLatentParameters(
                    curr_samp.alpha, prop_val, curr_samp.free_flow_speed
                )
            else:
                prop_val = jnp.where(
                    mask,
                    curr_samp.free_flow_speed + noise * variances[2, b],
                    curr_samp.free_flow_speed,
                )
                prop_samp = parametanet.ParaNetworkLatentParameters(
                    curr_samp.alpha, curr_samp.critical_density, prop_val
                )

            lp_prop = log_joint_jit(
                theta,
                prop_samp,
                params_static,
                scales_scalar,
                init_state,
                traj_true,
                boundaries,
            )
            accept = jnp.log(u) < (lp_prop - curr_lp)

            curr_samp, curr_lp = jax.lax.cond(
                accept,
                lambda _: (prop_samp, lp_prop),
                lambda _: (curr_samp, curr_lp),
                None,
            )
            acceptances = acceptances.at[p_idx, b].set(accept)

    return curr_samp, curr_lp, acceptances


@functools.partial(jax.jit, static_argnames=("iterations", "num_blocks"))
def mcmc_advance(
    z_init: parametanet.ParaNetworkLatentParameters,
    log_post_init: jax.Array,
    mcmc_variance: jax.Array,
    key: jax.Array,
    theta: Theta,
    params_static: parametanet.ParaNetworkStaticParameters,
    scales_scalar: parametanet.ParaNetworkScalarParameters,
    init_state: parametanet.NetworkState,
    traj_true: parametanet.SimulationTrajectory,
    boundaries: parametanet.BoundarySequence,
    iterations: int,
    adapt_alpha: float,
    adapt_beta: float,
    num_blocks: int,
):
    def scan_body(carry, subkey):
        samp, lp, variances = carry
        samp_new, lp_new, accepted = gibbs_step(
            samp,
            lp,
            variances,
            subkey,
            theta,
            params_static,
            scales_scalar,
            init_state,
            traj_true,
            boundaries,
            num_blocks=num_blocks,
        )
        variances_new = jnp.where(
            accepted == 1.0, variances * adapt_alpha, variances * adapt_beta
        )
        return (samp_new, lp_new, variances_new), accepted

    keys = jax.random.split(key, iterations)
    (z_final, lp_final, var_final), acc_hist = jax.lax.scan(
        scan_body, (z_init, log_post_init, mcmc_variance), keys
    )
    return z_final, lp_final, var_final, jnp.mean(acc_hist, axis=0)
