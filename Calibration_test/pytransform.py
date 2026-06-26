import numpy as np
import scipy.io
import jax.numpy as jnp
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Import the cla-sses from your parametanet.py file
from parametanet_real import (
    NetworkState,
    BoundarySequence,
    NetworkParameters,
    to_para_network,
    rollout_simulation,
)


def load_and_prepare_metanet_data(mat_filepath: str):
    """
    Reads Input_data.mat, interpolates the data to the simulation time step,
    and constructs the JAX-compatible NetworkState, BoundarySequence, and NetworkParameters.
    """
    # 1. Load MATLAB Data
    data = scipy.io.loadmat(mat_filepath)

    # Extract arrays (flattening where necessary to ensure correct 1D/2D shapes)
    On_ramp_flow = data["On_ramp_flow"]
    Off_ramp_rate = data["Off_ramp_rate"]
    Boundary_mainstream = data["Boundary_mainstream"]

    Initial_mainstream_speed = data["Initial_mainstream_speed"].flatten()
    Initial_mainstream_flow = data["Initial_mainstream_flow"].flatten()

    # 2. Time Configuration & Interpolation (Matching INITIALIZE.m)
    Tm = 60  # Measurement interval (seconds)
    Tsim = 5  # Simulation interval (seconds)

    # Start and End time in seconds (00:00:00 to 03:59:00)
    Starting_time = 0
    Ending_time = 3 * 3600 + 59 * 60 + 0

    # Create time vectors
    time_mis = np.arange(Starting_time, Ending_time + Tm, Tm)
    time_sim = np.arange(Starting_time, Ending_time + Tsim, Tsim)
    K = len(time_sim) - 1  # Total simulation steps

    # Interpolation function helper
    def interpolate_data(meas_time, meas_data, sim_time):
        f = interp1d(meas_time, meas_data, axis=0, fill_value="extrapolate")
        return f(sim_time)

    On_ramp_flow_Interp = interpolate_data(time_mis, On_ramp_flow, time_sim)
    Off_ramp_rate_Interp = interpolate_data(time_mis, Off_ramp_rate, time_sim)
    Boundary_mainstream_Interp = interpolate_data(
        time_mis, Boundary_mainstream, time_sim
    )

    # 3. Network Configuration (Lengths and Lanes from INITIALIZE.m)
    N = 21

    # Section lengths in km (Delta from MATLAB)
    L_meters = np.array(
        [
            433,
            437,
            444,
            441,
            463,
            450,
            450,
            448,
            306,
            340,
            350,
            424,
            566,
            706,
            658,
            424,
            419,
            432,
            413,
            427,
            426,
        ]
    )
    L_km = L_meters / 1000.0

    # Number of lanes: 3 lanes for sections 1-8 (idx 0-7), 2 lanes for 9-21 (idx 8-20)
    lambda_lanes = np.array([3] * 8 + [2] * 13)

    lane_N_plus_1 = 2  # Lane downstream of the final section

    # Calculate delta_lambda (Lane drop indicator: lambda_i - lambda_i+1)
    # A positive number indicates a lane drop (which triggers the negative weaving speed penalty)
    delta_lambda = np.append(
        lambda_lanes[:-1] - lambda_lanes[1:], lambda_lanes[-1] - lane_N_plus_1
    )

    # 4. Build the Initial NetworkState (k=0)
    v_init = Initial_mainstream_speed
    q_init = Initial_mainstream_flow
    rho_init = q_init / (v_init * lambda_lanes)

    initial_state = NetworkState(
        density=jnp.array(rho_init, dtype=jnp.float32),
        flow=jnp.array(q_init, dtype=jnp.float32),
        speed=jnp.array(v_init, dtype=jnp.float32),
    )

    # 5. Build BoundarySequence
    # Boundary Mainstream: [q_in, v_in, q_out, v_out]
    q_0 = Boundary_mainstream_Interp[:, 0]
    v_0 = Boundary_mainstream_Interp[:, 1]

    # Calculate downstream density rho_{N+1} from flow and speed
    q_N_plus_1 = Boundary_mainstream_Interp[:, 2]
    v_N_plus_1 = Boundary_mainstream_Interp[:, 3]
    rho_N_plus_1 = q_N_plus_1 / (v_N_plus_1 * lane_N_plus_1)

    # Construct On-ramp arrays (Shape: K x N)
    # MATLAB specifies on-ramps at sections 11 and 17 (Python indices 9 and 15)
    r_matrix = np.zeros((len(time_sim), N))
    r_matrix[:, 9] = On_ramp_flow_Interp[:, 0]
    r_matrix[:, 15] = On_ramp_flow_Interp[:, 1]

    # Construct Off-ramp arrays (Shape: K x N)
    # MATLAB specifies off-ramps at sections 9 and 15 (Python indices 7 and 13)
    s_matrix = np.zeros((len(time_sim), N))
    s_matrix[:, 7] = Off_ramp_rate_Interp[:, 0]
    s_matrix[:, 13] = Off_ramp_rate_Interp[:, 1]

    # Because jax.lax.scan runs over K steps, we drop the very last interpolated point
    # to maintain shape matching (K steps requires K boundary values)
    boundaries = BoundarySequence(
        q_0=jnp.array(q_0[:-1]),
        v_0=jnp.array(v_0[:-1]),
        rho_N_plus_1=jnp.array(rho_N_plus_1[:-1]),
        r=jnp.array(r_matrix[:-1, :]),
        s=jnp.zeros(
            (K, N)
        ),  # Left as zero, since parametanet uses s_rate for calculation
        s_rate=jnp.array(s_matrix[:-1, :]),  # The actual splitting fraction (1 - beta)
    )

    # 6. Setup NetworkParameters (using initial guesses from INITIALIZE.m)
    params = NetworkParameters(
        T=Tsim / 3600.0,  # T in hours
        L=jnp.array(L_km),
        lambda_=jnp.array(lambda_lanes),
        delta_lambda=jnp.array(delta_lambda),  # ADDED: Lane drop indicators
        tau=10.24 / 3600.0,  # tau converted to hours
        nu=8.96,
        kappa=10.0,  # chi_1 in MATLAB
        delta=2.92,  # delta_on in MATLAB
        phi=0.0001346,  # ADDED: Weaving sensitivity (from Model_parameter1(5) in INITIALIZE.m)
        alpha=jnp.array([1.61] * N),  # aexp in MATLAB
        critical_density=jnp.array([36.69] * N),
        free_flow_speed=jnp.array([117.0] * N),
    )

    return initial_state, boundaries, params


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


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Dynamically get the absolute path to the directory this script is in
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Attach the filename to that directory path
    mat_filepath = os.path.join(script_dir, "Input_data.mat")

    # 3. Load and prepare the data
    initial_state, boundaries, original_params = load_and_prepare_metanet_data(
        mat_filepath
    )

    # 4. Convert standard parameters to the reparameterized version expected by the step function
    para_params = to_para_network(original_params)

    # 5. Run the fast JAX compiled simulation
    print("Running simulation rollout...")
    trajectory = rollout_simulation(initial_state, boundaries, para_params)

    print(f"Simulation complete!")
    print(f"Trajectory Speed Shape: {trajectory.speed.shape} (Expected: K x 21)")
    print(f"Final Speed at step K: {trajectory.speed[-1, :]}")
    plot_trajectories(trajectory, boundaries.r, N=21, K=trajectory.speed.shape[0])
