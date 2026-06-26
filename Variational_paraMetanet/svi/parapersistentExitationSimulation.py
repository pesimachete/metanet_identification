import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


import parametanet


def generate_prbs_jax(
    key, length: int, amplitude: float, min_hold_steps: int, max_hold_steps: int
):

    # Maximum possible flips if every hold was the minimum duration
    max_flips = length // min_hold_steps + 1

    key_hold, key_val = jax.random.split(key)

    # Generate random hold durations for each flip
    holds = jax.random.randint(
        key_hold, shape=(max_flips,), minval=min_hold_steps, maxval=max_hold_steps
    )

    # Cumulative sum to get the step indices where the flips occur
    flip_indices = jnp.cumsum(holds)

    # Generate random values (-amplitude or +amplitude)
    vals = jax.random.choice(
        key_val, jnp.array([-amplitude, amplitude]), shape=(max_flips,)
    )

    # Map each time step to its corresponding interval index
    time_steps = jnp.arange(length)
    idx_map = jnp.searchsorted(flip_indices, time_steps)

    return vals[idx_map]


def setup_demand_profile(key, K: int, N: int, steps_per_hour: int):
    """
    Constructs the full demand profile using JAX immutable array updates.
    """
    dem = jnp.zeros((K, N))
    time_hours = jnp.linspace(0, 144, K)

    major_ramps = jnp.array([4, 9, 14])
    # Identify minor ramps (0-19 excluding 4, 9, 14)
    minor_ramps = jnp.array([i for i in range(N) if i not in [4, 9, 14]])

    # A. Major Ramps: Structural Trend
    base_demand = 200 + 100 * jnp.sin(2 * jnp.pi * time_hours / 24 - jnp.pi / 2)
    base_demand = jnp.maximum(base_demand, 50.0)

    # In JAX, we use .at[...].set() instead of in-place mutation
    dem = dem.at[:, major_ramps].set(base_demand[:, None])

    # B. Major Ramps: Overloads
    for day in range(6):
        # Morning Rush
        start_m = int((day * 24 + 7) * steps_per_hour)
        end_m = int((day * 24 + 10) * steps_per_hour)
        dem = dem.at[start_m:end_m, 4].add(600.0)

        # Evening Rush
        start_e = int((day * 24 + 16) * steps_per_hour)
        end_e = int((day * 24 + 19) * steps_per_hour)
        dem = dem.at[start_e:end_e, 9].add(700.0)

        # Mid-day Anomaly
        start_mid = int((day * 24 + 12) * steps_per_hour)
        end_mid = int((day * 24 + 14) * steps_per_hour)
        dem = dem.at[start_mid:end_mid, 14].add(500.0)

    # C. Major Ramps: High-Amplitude PRBS Excitation
    key, k1, k2, k3 = jax.random.split(key, 4)
    prbs_4 = generate_prbs_jax(k1, K, 150.0, 12, 60)
    prbs_9 = generate_prbs_jax(k2, K, 150.0, 12, 60)
    prbs_14 = generate_prbs_jax(k3, K, 150.0, 12, 60)

    dem = dem.at[:, 4].add(prbs_4)
    dem = dem.at[:, 9].add(prbs_9)
    dem = dem.at[:, 14].add(prbs_14)

    # D. Minor Ramps: Low-Level Background Excitation
    key, key_base = jax.random.split(key)

    # Generate random baselines for all minor ramps at once
    minor_bases = jax.random.uniform(
        key_base, shape=(len(minor_ramps),), minval=20.0, maxval=50.0
    )

    # Vectorize the PRBS generator to apply to all minor ramps simultaneously!
    vmap_prbs = jax.vmap(generate_prbs_jax, in_axes=(0, None, None, None, None))
    keys_minor = jax.random.split(key, len(minor_ramps))
    minor_prbs = vmap_prbs(
        keys_minor, K, 15.0, 12, 60
    )  # Returns shape (num_minor_ramps, K)

    # Add minor baseline and transposed PRBS to the demand array
    dem = dem.at[:, minor_ramps].set(minor_bases[None, :] + minor_prbs.T)

    # Ensure no negative demand
    return jnp.maximum(dem, 0.0)


def generate_noisy_latent_parameters(
    key: jax.Array, N: int = 20
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Generates latent traffic parameters with spatial AR(1) noise using a simplified, batched approach.
    """
    # 1. Baseline Profiles (Simplified using jnp.where)
    section_indices = jnp.arange(N)
    vf_base = jnp.where(section_indices < N // 2, 120.0, 100.0)
    rho_cr_base = jnp.where(section_indices < N // 2, 32.0, 28.0)
    alpha_base = jnp.full(N, 1.8)

    # 2. Batched White Noise Generation
    # Generate all noise simultaneously: Shape (3 parameters, N sections)
    innovations = jax.random.normal(key, shape=(3, N))

    std_devs = jnp.array([[1.0], [0.2], [0.02]])
    innovations = innovations * std_devs

    # 3. Simplified AR(1) Spatial Loop
    # JAX will unroll this loop efficiently during compilation for static N
    zeta = 0.85
    noise = jnp.zeros((3, N))
    noise = noise.at[:, 0].set(innovations[:, 0])

    for i in range(1, N):
        noise = noise.at[:, i].set(zeta * noise[:, i - 1] + innovations[:, i])

    # 4. Superimpose and Constrain
    vf_final = jnp.maximum(vf_base + noise[0], 10.0)
    rho_cr_final = jnp.maximum(rho_cr_base + noise[1], 5.0)
    alpha_final = jnp.maximum(alpha_base + noise[2], 0.1)

    return vf_final, rho_cr_final, alpha_final


def generate_ar1_parameters(
    key: jax.Array, N: int = 20, zeta: float = 0.85
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Generates a pure spatial AR(1) process for latent traffic parameters,
    omitting base deterministic behavior.
    """
    # 1. Batched White Noise Generation
    # Generate all noise simultaneously: Shape (3 parameters, N sections)
    innovations = jax.random.normal(key, shape=(3, N))

    # Apply standard deviations
    std_devs = jnp.array([[1.0], [0.2], [0.02]])
    innovations = innovations * std_devs

    # 2. AR(1) Spatial Loop
    # JAX will unroll this loop efficiently during compilation for static N
    noise = jnp.zeros((3, N))
    noise = noise.at[:, 0].set(innovations[:, 0])

    for i in range(1, N):
        noise = noise.at[:, i].set(zeta * noise[:, i - 1] + innovations[:, i])

    # 3. Return the pure AR(1) series for each parameter
    return noise[0] + 120, noise[1] + 32, noise[2] + 1.8


def generate_ar1_parameters_unconstrained(
    key: jax.Array, N: int = 20, zeta: float = 0.95
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Generates a spatial AR(1) process directly in the unconstrained space.
    The base targets (0.84, 3.16, 1.62) map to physical values of 120, 32, and 1.8
    after undergoing the scaled softplus transformation.
    """
    z_vfree_base = 0.84
    z_rhocr_base = 3.16
    z_alpha_base = 1.62

    innovations = jax.random.normal(key, shape=(3, N))

    # Standard deviations are scaled down to operate appropriately within the unconstrained space
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
    N = 20

    # Initialize main JAX PRNG Key
    key = jax.random.PRNGKey(42)

    # ---------------------------
    # 1. Horizon & Geometry
    # ---------------------------
    T = 10.0 / 3600.0
    steps_per_hour = int(3600 / 10)
    K = 144 * steps_per_hour

    L = jnp.full(N, 0.5)
    lambda_ = jnp.full(N, 1.0)

    # ---------------------------
    # 2. Corrected Physical Parameters
    # ---------------------------
    key = jax.random.PRNGKey(42)

    # 1. Generate Noisy Parameters
    key, param_key = jax.random.split(key)
    z_vfree, z_rhocr, z_alpha = generate_ar1_parameters_unconstrained(param_key, N)
    vf_noisy = (jax.nn.softplus(z_vfree) + 1e-6) * 100.0
    rho_cr_noisy = (jax.nn.softplus(z_rhocr) + 1e-6) * 10.0
    alpha_noisy = (jax.nn.softplus(z_alpha) + 1e-6) * 1.0

    tau = 0.08
    nu = 50.0
    kappa = 40.0
    delta = 0.012

    physical_params = parametanet.NetworkParameters(
        T=T,
        L=L,
        lambda_=lambda_,
        tau=float(tau),
        nu=float(nu),
        kappa=float(kappa),
        delta=float(delta),
        alpha=alpha_noisy,
        critical_density=rho_cr_noisy,
        free_flow_speed=vf_noisy,
    )

    params = parametanet.to_para_network(physical_params)

    # ---------------------------
    # 3. Demand Profile
    # ---------------------------
    key, demand_key = jax.random.split(key)
    dem = setup_demand_profile(demand_key, K, N, steps_per_hour)

    # ---------------------------
    # 4. Setup Initial & Boundary Values
    # ---------------------------
    rho_init = jnp.full(N, 15.0)
    v_init = vf_noisy * jnp.exp(
        -(1.0 / alpha_noisy) * (rho_init / rho_cr_noisy) ** alpha_noisy
    )
    q_init = rho_init * v_init * lambda_

    init_state = parametanet.NetworkState(
        density=rho_init,
        flow=q_init,
        speed=v_init,
    )

    rho0_val = 15.0
    v0_val = 120.0 * jnp.exp(-(1.0 / 1.8) * (rho0_val / 32.0) ** 1.8)

    boundaries = parametanet.BoundarySequence(
        q_0=jnp.full(K, rho0_val * v0_val),
        v_0=jnp.full(K, v0_val),
        rho_N_plus_1=jnp.full(K, rho_init[-1]),
        r=dem,
        s=jnp.zeros((K, N)),
    )

    # ---------------------------
    # 5. Run Simulation
    # ---------------------------
    # If metanet.rollout_simulation is written in pure JAX, you can even JIT this call!
    traj = parametanet.rollout_simulation(init_state, boundaries, params)

    return traj, params, boundaries, init_state


def plot_trajectories(traj, dem, N: int, K: int):
    density_res = np.array(traj.density)
    speed_res = np.array(traj.speed)
    flow_res = np.array(traj.flow)
    time_grid, section_grid = np.meshgrid(np.arange(K), np.arange(N))

    # Create a 2x2 grid of subplots in one large figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Density Plot (Top-Left)
    mesh1 = axs[0, 0].pcolormesh(
        time_grid, section_grid, density_res.T, cmap="Reds", shading="auto"
    )
    axs[0, 0].set_title("Freeway Density over Time and Sections")
    axs[0, 0].set_xlabel("Time Step (k)")
    axs[0, 0].set_ylabel("Section Index")
    fig.colorbar(mesh1, ax=axs[0, 0], label="Density (veh/km)")

    # 2. Velocity Plot (Top-Right)
    mesh2 = axs[0, 1].pcolormesh(
        time_grid, section_grid, speed_res.T, cmap="gist_heat", shading="auto"
    )
    axs[0, 1].set_title("Freeway Velocity over Time and Sections")
    axs[0, 1].set_xlabel("Time Step (k)")
    axs[0, 1].set_ylabel("Section Index")
    fig.colorbar(mesh2, ax=axs[0, 1], label="Speed (km/h)")

    # 3. Flow Plot (Bottom-Left)
    mesh3 = axs[1, 0].pcolormesh(
        time_grid, section_grid, flow_res.T, cmap="Blues", shading="auto"
    )
    axs[1, 0].set_title("Freeway Flow over Time and Sections")
    axs[1, 0].set_xlabel("Time Step (k)")
    axs[1, 0].set_ylabel("Section Index")
    fig.colorbar(mesh3, ax=axs[1, 0], label="Flow (veh/h)")

    # 4. Demand Plot (Bottom-Right)
    mesh4 = axs[1, 1].pcolormesh(
        time_grid, section_grid, dem.T, cmap="Reds", shading="auto"
    )
    axs[1, 1].set_title("Demand Profile over Time and Sections")
    axs[1, 1].set_xlabel("Time Step (k)")
    axs[1, 1].set_ylabel("Section Index")
    fig.colorbar(mesh4, ax=axs[1, 1], label="Demand (veh/h)")

    plt.tight_layout()  # Ensures labels and colorbars don't overlap
    plt.show()


if __name__ == "__main__":
    traj, params, boundaries, init_state = simulate_example()
    print("Simulation completed. Sample output:")
    print("Density at t=0:", traj.density[0])
    print("Speed at t=0:", traj.speed[0])
    print("Flow at t=0:", traj.flow[0])
    plot_trajectories(
        traj, boundaries.r, N=20, K=144 * 360
    )  # Adjust K if you change the horizon
