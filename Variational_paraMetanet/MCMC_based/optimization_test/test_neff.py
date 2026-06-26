import os
import pickle
from datetime import datetime

# Force JAX to use CPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ==========================================
# Core JAX Computation Functions
# ==========================================


@jax.jit(static_argnames=["max_lag"])
def compute_acf_jax_1d(x: jax.Array, max_lag: int = 1000):
    """Computes the ACF for a single 1D trace."""
    n = x.shape[0]
    mean = jnp.mean(x)
    var = jnp.var(x)
    x_centered = x - mean

    cov = jnp.correlate(x_centered, x_centered, mode="full")
    acf = jax.lax.dynamic_slice(cov, (n - 1,), (max_lag,)) / (var * n)

    return jnp.where(var == 0.0, jnp.zeros(max_lag), acf)


@jax.jit(static_argnames=["max_lag"])
def calc_neff_ar1_approx(trace: jax.Array, max_lag: int = 1000):
    """Calculates Neff for a single trace using the AR(1) rho < 0.4 approximation."""
    num_samples = trace.shape[0]
    acf = compute_acf_jax_1d(trace, max_lag)

    # Find the first lag k where acf < 0.4
    condition = acf < 0.4
    k = jnp.argmax(condition)

    # Fallback to max_lag - 1 if it never drops below 0.4
    k_safe = jnp.where(k == 0, max_lag - 1, k)
    rho_k = acf[k_safe]

    # Ensure a positive base for the fractional exponent
    rho_k_safe = jnp.maximum(rho_k, 1e-6)

    # AR(1) approximation
    rho_1_approx = jnp.power(rho_k_safe, 1.0 / k_safe)

    # ESS Calculation
    neff = num_samples * (1.0 - rho_1_approx) / (1.0 + rho_1_approx)

    return neff, rho_1_approx, k_safe


# Vectorize the function to run across the targeted spatial sections
calc_neff_vectorized = jax.vmap(calc_neff_ar1_approx, in_axes=(0, None))


# ==========================================
# File Processing and Text Output
# ==========================================


def analyze_and_save(
    pickle_filename: str,
    target_sections: list,
    burn_in: int = 1000,
    max_lag: int = 1000,
):
    print(f"Loading data from {pickle_filename}...")

    with open(pickle_filename, "rb") as f:
        data = pickle.load(f)

    history = data["history"]
    if len(history) <= burn_in:
        raise ValueError(
            f"Chain length ({len(history)}) is less than burn-in ({burn_in})."
        )

    post_burn_in = history[burn_in:]
    N_samples = len(post_burn_in)

    # Extract Metadata
    num_blocks = data.get("num_blocks", "Unknown")
    iterations = data.get("iterations", len(history))

    # Calculate overall acceptance rate
    if "accept_rates" in data:
        mean_acceptance = float(jnp.mean(data["accept_rates"])) * 100
        ar_str = f"{mean_acceptance:.1f}%"
        ar_filename_str = f"{mean_acceptance:.1f}"
    else:
        ar_str = "Unknown"
        ar_filename_str = "Unknown"

    # Convert target_sections list to a JAX array for indexing
    idx_array = jnp.array(target_sections)

    # Stack histories to shape: (Spatial Sections N, Iterations), then slice ONLY the targets
    alphas = jnp.stack([p.alpha for p in post_burn_in], axis=1)[idx_array, :]
    rhos = jnp.stack([p.critical_density for p in post_burn_in], axis=1)[idx_array, :]
    v_frees = jnp.stack([p.free_flow_speed for p in post_burn_in], axis=1)[idx_array, :]

    print(
        f"Processing chains... (Samples: {N_samples}, Targeted Sections: {target_sections})"
    )

    # Compute ESS in JAX for the subset
    neff_a, rho1_a, k_a = calc_neff_vectorized(alphas, max_lag)
    neff_r, rho1_r, k_r = calc_neff_vectorized(rhos, max_lag)
    neff_v, rho1_v, k_v = calc_neff_vectorized(v_frees, max_lag)

    # Convert to NumPy for text formatting
    neff_a, rho1_a, k_a = np.array(neff_a), np.array(rho1_a), np.array(k_a)
    neff_r, rho1_r, k_r = np.array(neff_r), np.array(rho1_r), np.array(k_r)
    neff_v, rho1_v, k_v = np.array(neff_v), np.array(rho1_v), np.array(k_v)

    # Generate Output Text File
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = f"ess_report_AR_{ar_filename_str}_{timestamp}.txt"

    with open(txt_filename, "w") as txt_file:
        txt_file.write("====================================================\n")
        txt_file.write(f"      TARGETED MCMC ESS REPORT (AR: {ar_str})       \n")
        txt_file.write("====================================================\n")
        txt_file.write(f"Source File      : {pickle_filename}\n")
        txt_file.write(f"Target Sections  : {target_sections}\n")
        txt_file.write(f"Num Blocks       : {num_blocks}\n")
        txt_file.write(f"Total Iterations : {iterations}\n")
        txt_file.write(f"Burn-in Distance : {burn_in}\n")
        txt_file.write(f"Acceptance Rate  : {ar_str}\n")
        txt_file.write(f"Post-burn Samples: {N_samples}\n")
        txt_file.write("====================================================\n\n")

        def write_parameter_block(param_name, neff, rho1, k_lags):
            txt_file.write(f"--- PARAMETER: {param_name.upper()} ---\n")
            txt_file.write(
                f"{'Section':<10} | {'N_eff':<12} | {'Rho_1_approx':<15} | {'Lag to <0.4':<12}\n"
            )
            txt_file.write("-" * 55 + "\n")

            # Map the results back to the original section numbers
            for idx, section_num in enumerate(target_sections):
                txt_file.write(
                    f"{section_num:<10} | {neff[idx]:<12.2f} | {rho1[idx]:<15.6f} | {k_lags[idx]:<12}\n"
                )

            min_idx = np.argmin(neff)
            max_idx = np.argmax(neff)

            txt_file.write("-" * 55 + "\n")
            txt_file.write(f"MEAN N_eff : {np.mean(neff):.2f}\n")
            txt_file.write(
                f"MIN N_eff  : {np.min(neff):.2f} (Section {target_sections[min_idx]})\n"
            )
            txt_file.write(
                f"MAX N_eff  : {np.max(neff):.2f} (Section {target_sections[max_idx]})\n\n\n"
            )

        write_parameter_block("Alpha", neff_a, rho1_a, k_a)
        write_parameter_block("Critical Density (Rho_cr)", neff_r, rho1_r, k_r)
        write_parameter_block("Free Flow Speed (V_free)", neff_v, rho1_v, k_v)

    print(f"\n[+] Success! Targeted ESS report saved to: {txt_filename}")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # USER CONFIGURATION
    # ---------------------------------------------------------
    TARGET_FILE = "mcmc_results_20blocks_20260625_001915.pkl"
    SECTIONS_TO_ANALYZE = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19]
    BURN_IN = 10000
    MAX_LAG = 1000

    try:
        analyze_and_save(
            TARGET_FILE,
            target_sections=SECTIONS_TO_ANALYZE,
            burn_in=BURN_IN,
            max_lag=MAX_LAG,
        )
    except FileNotFoundError:
        print(f"[!] Error: Could not find the file '{TARGET_FILE}'.")
        print("    Please verify the filename and path.")
