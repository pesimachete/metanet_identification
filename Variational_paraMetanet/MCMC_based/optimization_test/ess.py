import pickle
import jax
import jax.numpy as jnp
import numpy as np


def compute_acf_jax(x: jax.Array, max_lag: int = 1000):
    """Computes the ACF in pure JAX using cross-correlation."""
    n = x.shape[0]
    mean = jnp.mean(x)
    var = jnp.var(x)
    x_centered = x - mean

    cov = jnp.correlate(x_centered, x_centered, mode="full")
    acf = jax.lax.dynamic_slice(cov, (n - 1,), (max_lag,)) / (var * n)

    return jnp.where(var == 0.0, jnp.zeros(max_lag), acf)


def calculate_effective_contribution(acf_values):
    """
    Calculates the effective contribution per epoch.
    Truncates the sum at the first negative ACF value to eliminate noise.
    """
    mask = jnp.cumsum(acf_values < 0) == 0
    truncated_sum = jnp.sum(acf_values * mask)
    denominator = -1.0 + 2.0 * truncated_sum
    return jnp.maximum(1.0 / denominator, 0.0)


def compute_ess_per_block(post_burn_in, num_blocks, n_samples, max_lag):
    """
    Computes ESS using a single representative index per block (fast, approximate).
    Returns a dict with keys 'alpha', 'rho', 'v' each of shape (num_blocks,).
    """
    N = len(post_burn_in[0].alpha)

    print("\n[1] PER-BLOCK ESS (representative center index)")
    print("-" * 60)
    print(f"{'Block':<10} | {'Parameter':<10} | {'Eff. Contribution':<20} | {'ESS'}")
    print("-" * 60)

    block_ess = {"alpha": [], "rho": [], "v": []}

    for b in range(num_blocks):
        idx = ((b * N // num_blocks) + ((b + 1) * N // num_blocks)) // 2

        alphas = jnp.array([p.alpha[idx] for p in post_burn_in])
        rhos = jnp.array([p.critical_density[idx] for p in post_burn_in])
        vs = jnp.array([p.free_flow_speed[idx] for p in post_burn_in])

        eff_alpha = calculate_effective_contribution(compute_acf_jax(alphas, max_lag))
        eff_rho = calculate_effective_contribution(compute_acf_jax(rhos, max_lag))
        eff_v = calculate_effective_contribution(compute_acf_jax(vs, max_lag))

        block_ess["alpha"].append(float(eff_alpha * n_samples))
        block_ess["rho"].append(float(eff_rho * n_samples))
        block_ess["v"].append(float(eff_v * n_samples))

        print(
            f"Block {b:<4} | {'Alpha':<10} | {eff_alpha:<20.4f} | {eff_alpha * n_samples:.1f}"
        )
        print(
            f"Block {b:<4} | {'Rho_cr':<10} | {eff_rho:<20.4f} | {eff_rho * n_samples:.1f}"
        )
        print(
            f"Block {b:<4} | {'V_free':<10} | {eff_v:<20.4f} | {eff_v * n_samples:.1f}"
        )
        print("-" * 60)

    return block_ess


def compute_ess_full_vector(post_burn_in, num_blocks, n_samples, max_lag):
    """
    Computes ESS over the full joint parameter vector, averaging over all spatial
    indices within each block. Reports min/mean ESS across the whole vector,
    which is the honest mixing diagnostic for the sampled joint distribution.
    """
    N = len(post_burn_in[0].alpha)

    print("\n[2] FULL VECTOR ESS (averaged over all indices per block)")
    print("-" * 60)
    print(f"{'Block':<10} | {'Parameter':<10} | {'Mean ESS':<20} | {'Min ESS'}")
    print("-" * 60)

    all_ess = []

    for b in range(num_blocks):
        block_start = b * N // num_blocks
        block_end = (b + 1) * N // num_blocks
        block_indices = range(block_start, block_end)

        ess_alpha_list, ess_rho_list, ess_v_list = [], [], []

        for idx in block_indices:
            alphas = jnp.array([p.alpha[idx] for p in post_burn_in])
            rhos = jnp.array([p.critical_density[idx] for p in post_burn_in])
            vs = jnp.array([p.free_flow_speed[idx] for p in post_burn_in])

            ess_a = float(
                calculate_effective_contribution(compute_acf_jax(alphas, max_lag))
                * n_samples
            )
            ess_r = float(
                calculate_effective_contribution(compute_acf_jax(rhos, max_lag))
                * n_samples
            )
            ess_v = float(
                calculate_effective_contribution(compute_acf_jax(vs, max_lag))
                * n_samples
            )

            ess_alpha_list.append(ess_a)
            ess_rho_list.append(ess_r)
            ess_v_list.append(ess_v)

            all_ess.extend([ess_a, ess_r, ess_v])

        mean_a, min_a = np.mean(ess_alpha_list), np.min(ess_alpha_list)
        mean_r, min_r = np.mean(ess_rho_list), np.min(ess_rho_list)
        mean_v, min_v = np.mean(ess_v_list), np.min(ess_v_list)

        print(f"Block {b:<4} | {'Alpha':<10} | {mean_a:<20.1f} | {min_a:.1f}")
        print(f"Block {b:<4} | {'Rho_cr':<10} | {mean_r:<20.1f} | {min_r:.1f}")
        print(f"Block {b:<4} | {'V_free':<10} | {mean_v:<20.1f} | {min_v:.1f}")
        print("-" * 60)

    all_ess = np.array(all_ess)
    print(f"\n{'='*60}")
    print(f"  Global Min ESS  (bottleneck): {np.min(all_ess):.1f}")
    print(f"  Global Mean ESS             : {np.mean(all_ess):.1f}")
    print(f"  Min ESS / n_samples (ratio) : {np.min(all_ess) / n_samples:.4f}")
    print(f"{'='*60}")

    return all_ess


def main():
    filename = "mcmc_results_1blocks_20260616_180745.pkl"

    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"Successfully loaded data from {filename}")
    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Please check the filename.")
        return

    history = data["history"]
    num_blocks = data["num_blocks"]

    burn_in = 1000
    if len(history) <= burn_in:
        print(f"Warning: Chain length ({len(history)}) is <= burn-in ({burn_in}).")
        return

    post_burn_in = history[burn_in:]
    n_samples = len(post_burn_in)
    max_lag = n_samples

    print(f"\nAnalyzing {n_samples} post-burn-in samples across {num_blocks} blocks...")

    # --- Per-block ESS (fast, one representative index per block) ---
    compute_ess_per_block(post_burn_in, num_blocks, n_samples, max_lag)

    # --- Full vector ESS (all indices, honest mixing diagnostic) ---
    compute_ess_full_vector(post_burn_in, num_blocks, n_samples, max_lag)


if __name__ == "__main__":
    main()
