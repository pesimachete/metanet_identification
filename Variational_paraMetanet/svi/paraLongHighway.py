import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import parametanet


def generate_prbs_jax(
    key, length: int, amplitude: float, min_hold_steps: int, max_hold_steps: int
):
    max_flips = length // min_hold_steps + 1
    key_hold, key_val = jax.random.split(key)

    holds = jax.random.randint(
        key_hold, shape=(max_flips,), minval=min_hold_steps, maxval=max_hold_steps
    )

    flip_indices = jnp.cumsum(holds)

    vals = jax.random.choice(
        key_val, jnp.array([-amplitude, amplitude]), shape=(max_flips,)
    )

    time_steps = jnp.arange(length)
    idx_map = jnp.searchsorted(flip_indices, time_steps)

    return vals[idx_map]


def setup_demand_profile(key, K: int, N: int, steps_per_hour: int):
    """
    Constructs a demand profile bounded by the theoretical capacity of the network.
    Generates severe but mathematically stable localized congestion.
    """
    dem = jnp.zeros((K, N))
    time_hours = jnp.linspace(0, 72, K)

    major_ramps = jnp.array(np.linspace(4, N - 6, 15, dtype=int))
    minor_ramps = jnp.array([i for i in range(N) if i not in major_ramps])

    # A. Major Ramps: Calibrated Structural Trend with Phase Shift
    phase_shifts = jnp.linspace(0, 2.0 * jnp.pi, len(major_ramps))

    time_expanded = time_hours[:, None]
    phase_expanded = phase_shifts[None, :]

    # Lowered structural baseline to allow high PRBS spikes without exceeding global capacity
    base_demand = 40.0 + 30.0 * jnp.sin(
        2.0 * jnp.pi * time_expanded / 24.0 - jnp.pi / 2.0 + phase_expanded
    )
    base_demand = jnp.maximum(base_demand, 10.0)
    dem = dem.at[:, major_ramps].set(base_demand)

    # B. Major Ramps: High PRBS Excitation (Causes localized traffic jams)
    key, *subkeys = jax.random.split(key, len(major_ramps) + 1)

    vmap_prbs = jax.vmap(generate_prbs_jax, in_axes=(0, None, None, None, None))
    major_prbs = vmap_prbs(jnp.array(subkeys), K, 80.0, 12, 60)

    dem = dem.at[:, major_ramps].add(major_prbs.T)

    # C. Minor Ramps: Negligible Background Noise to prevent saturation
    key, key_base = jax.random.split(key)
    minor_bases = jax.random.uniform(
        key_base, shape=(len(minor_ramps),), minval=5.0, maxval=10.0
    )

    keys_minor = jax.random.split(key, len(minor_ramps))
    minor_prbs = vmap_prbs(keys_minor, K, 10.0, 12, 60)

    dem = dem.at[:, minor_ramps].set(minor_bases[None, :] + minor_prbs.T)

    return jnp.maximum(dem, 0.0)


def generate_noisy_latent_parameters(
    key: jax.Array, N: int = 100
) -> tuple[jax.Array, jax.Array, jax.Array]:
    section_indices = jnp.arange(N)
    vf_base = jnp.where(section_indices < N // 2, 120.0, 100.0)
    rho_cr_base = jnp.where(section_indices < N // 2, 32.0, 28.0)
    alpha_base = jnp.full(N, 1.8)

    innovations = jax.random.normal(key, shape=(3, N))
    std_devs = jnp.array([[1.0], [0.2], [0.02]])
    innovations = innovations * std_devs

    zeta = 0.85
    noise = jnp.zeros((3, N))
    noise = noise.at[:, 0].set(innovations[:, 0])

    for i in range(1, N):
        noise = noise.at[:, i].set(zeta * noise[:, i - 1] + innovations[:, i])

    vf_final = jnp.maximum(vf_base + noise[0], 10.0)
    rho_cr_final = jnp.maximum(rho_cr_base + noise[1], 5.0)
    alpha_final = jnp.maximum(alpha_base + noise[2], 0.1)

    return vf_final, rho_cr_final, alpha_final


def generate_ar1_parameters_unconstrained(
    key: jax.Array, N: int = 100, zeta: float = 0.95
) -> tuple[jax.Array, jax.Array, jax.Array]:
    z_vfree_base = 0.84
    z_rhocr_base = 3.16
    z_alpha_base = 1.62

    innovations = jax.random.normal(key, shape=(3, N))
    std_devs = jnp.array([[0.005], [0.005], [0.005]])
    innovations = innovations * std_devs

    noise = jnp.zeros((3, N))
    noise = noise.at[:, 0].set(innovations[:, 0])

    for i in range(1, N):
        noise = noise.at[:, i].set(zeta * noise[:, i - 1] + innovations[:, i])

    z_vfree = z_vfree_base + noise[0]
    z_rhocr = z_rhocr_base + noise[1]
    z_alpha = z_alpha_base + noise[2]

    return z_vfree, z_rhocr, z_alpha


def simulate_example():
    N = 100
    key = jax.random.PRNGKey(42)

    T = 10.0 / 3600.0
    steps_per_hour = int(3600 / 10)
    K = 72 * steps_per_hour

    L = jnp.full(N, 0.5)
    lambda_ = jnp.full(N, 1.0)

    key, param_key = jax.random.split(key)
    z_vfree, z_rhocr, z_alpha = generate_ar1_parameters_unconstrained(param_key, N)

    vf_noisy = (jax.nn.softplus(z_vfree) + 1e-6) * 100.0
    rho_cr_noisy = (jax.nn.softplus(z_rhocr) + 1e-6) * 10.0
    alpha_noisy = (jax.nn.softplus(z_alpha) + 1e-6) * 1.0

    physical_params = parametanet.NetworkParameters(
        T=T,
        L=L,
        lambda_=lambda_,
        tau=0.08,
        nu=50.0,
        kappa=40.0,
        delta=0.012,
        alpha=alpha_noisy,
        critical_density=rho_cr_noisy,
        free_flow_speed=vf_noisy,
    )

    params = parametanet.to_para_network(physical_params)

    key, demand_key = jax.random.split(key)
    dem = setup_demand_profile(demand_key, K, N, steps_per_hour)

    # Lowered initial boundary density to free up mainline capacity
    rho0_val = 12.0
    rho_init = jnp.full(N, rho0_val)
    v_init = vf_noisy * jnp.exp(
        -(1.0 / alpha_noisy) * (rho_init / rho_cr_noisy) ** alpha_noisy
    )
    q_init = rho_init * v_init * lambda_

    init_state = parametanet.NetworkState(
        density=rho_init,
        flow=q_init,
        speed=v_init,
    )

    v0_val = 120.0 * jnp.exp(-(1.0 / 1.8) * (rho0_val / 32.0) ** 1.8)

    # Define 15 Off-Ramps interleaved immediately downstream of the major on-ramps
    # This design safely dissipates the extreme congestion waves induced by the PRBS signals
    off_ramps = jnp.array(np.linspace(6, N - 4, 15, dtype=int))
    s_profile = jnp.zeros((K, N))

    s_profile = s_profile.at[:, off_ramps].set(120.0)

    boundaries = parametanet.BoundarySequence(
        q_0=jnp.full(K, rho0_val * v0_val),
        v_0=jnp.full(K, v0_val),
        rho_N_plus_1=jnp.full(K, rho_init[-1]),
        r=dem,
        s=s_profile,
    )

    traj = parametanet.rollout_simulation(init_state, boundaries, params)

    return traj, params, boundaries, init_state


def plot_trajectories(traj, dem, ext, N: int, K: int, section_length: float = 0.5):
    density_res = np.array(traj.density)
    speed_res = np.array(traj.speed)
    flow_res = np.array(traj.flow)
    time_grid, section_grid = np.meshgrid(np.arange(K), np.arange(N))

    fig, axs = plt.subplots(2, 3, figsize=(22, 10))

    # Density visualization adjusted to highlight the highly congested regions clearly
    mesh1 = axs[0, 0].pcolormesh(
        time_grid, section_grid, density_res.T, cmap="Reds", shading="auto", vmax=60.0
    )
    axs[0, 0].set_title("Freeway Density over Time and Sections")
    axs[0, 0].set_xlabel("Time Step (k)")
    axs[0, 0].set_ylabel("Section Index")
    fig.colorbar(mesh1, ax=axs[0, 0], label="Density (veh/km)")

    mesh2 = axs[0, 1].pcolormesh(
        time_grid, section_grid, speed_res.T, cmap="gist_heat", shading="auto"
    )
    axs[0, 1].set_title("Freeway Velocity over Time and Sections")
    axs[0, 1].set_xlabel("Time Step (k)")
    axs[0, 1].set_ylabel("Section Index")
    fig.colorbar(mesh2, ax=axs[0, 1], label="Speed (km/h)")

    mesh3 = axs[1, 0].pcolormesh(
        time_grid, section_grid, flow_res.T, cmap="Blues", shading="auto"
    )
    axs[1, 0].set_title("Freeway Flow over Time and Sections")
    axs[1, 0].set_xlabel("Time Step (k)")
    axs[1, 0].set_ylabel("Section Index")
    fig.colorbar(mesh3, ax=axs[1, 0], label="Flow (veh/h)")

    mesh4 = axs[1, 1].pcolormesh(
        time_grid, section_grid, dem.T, cmap="Reds", shading="auto"
    )
    axs[1, 1].set_title("Demand Profile (On-Ramps)")
    axs[1, 1].set_xlabel("Time Step (k)")
    axs[1, 1].set_ylabel("Section Index")
    fig.colorbar(mesh4, ax=axs[1, 1], label="Demand (veh/h)")

    mesh5 = axs[0, 2].pcolormesh(
        time_grid, section_grid, ext.T, cmap="Greens", shading="auto"
    )
    axs[0, 2].set_title("Extraction Profile (Off-Ramps)")
    axs[0, 2].set_xlabel("Time Step (k)")
    axs[0, 2].set_ylabel("Section Index")
    fig.colorbar(mesh5, ax=axs[0, 2], label="Extraction Rate (veh/h)")

    total_vehicles = np.sum(density_res, axis=1) * section_length
    time_steps_array = np.arange(K)
    axs[1, 2].plot(time_steps_array, total_vehicles, color="darkred", linewidth=1.5)
    axs[1, 2].set_title("Total Vehicle Population within Corridor")
    axs[1, 2].set_xlabel("Time Step (k)")
    axs[1, 2].set_ylabel("Total Vehicles")
    axs[1, 2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    traj, params, boundaries, init_state = simulate_example()
    print("Simulation completed. Verification metrics:")
    print("Density at t=0:", traj.density[0])

    plot_trajectories(traj, boundaries.r, boundaries.s, N=100, K=72 * 360)
