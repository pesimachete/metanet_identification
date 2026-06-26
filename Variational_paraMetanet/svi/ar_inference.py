import typing
import jax
import jax.numpy as jnp

# Import the user-provided simulation and network definitions
import parametanet
from parapersistentExitationSimulation import simulate_example


class AR1Estimates(typing.NamedTuple):
    """
    Estimated stochastic parameters for the Metanet latent variables.
    """

    mean: jax.Array
    zeta: jax.Array
    variance: jax.Array


@jax.jit
def estimate_ar1_parameters(x: jax.Array) -> AR1Estimates:
    """
    Estimates the AR(1) properties of a spatial parameter sequence
    using the one-shot least squares solution.
    """
    x_mean = jnp.mean(x)
    y = x - x_mean

    y_prev = y[:-1]
    y_curr = y[1:]

    numerator = jnp.sum(y_prev * y_curr)
    denominator = jnp.sum(y_prev**2)

    # Epsilon added to denominator to guarantee numerical stability
    zeta = numerator / (denominator + 1e-7)

    residuals = y_curr - zeta * y_prev
    process_noise_variance = jnp.mean(residuals**2)

    safe_zeta = jnp.clip(zeta, -0.99, 0.99)
    sigma_x_sq = process_noise_variance / (1.0 - safe_zeta**2)

    return AR1Estimates(mean=x_mean, zeta=zeta, variance=sigma_x_sq)


def analyze_network_stochastics(
    params: parametanet.ParaNetworkParameters,
) -> dict[str, AR1Estimates]:
    """
    Applies the AR(1) estimation block to the three latent variables
    within the network parameters.
    """
    latent = params.latent_params

    return {
        "Fundamental Diagram Shape (alpha)": estimate_ar1_parameters(latent.alpha),
        "Critical Density (rho_cr)": estimate_ar1_parameters(latent.critical_density),
        "Free Flow Speed (v_free)": estimate_ar1_parameters(latent.free_flow_speed),
    }


def generate_estimation_report():
    """
    Executes the simulation setup and reports the identified AR(1) parameters.
    """
    # 1. Retrieve the initialized parameters from the existing simulation framework
    _, params, _, _ = simulate_example()

    # 2. Execute the estimation routine
    estimates = analyze_network_stochastics(params)

    # 3. Format and output the analytical results
    print("==================================================")
    print(" LATENT VARIABLE AR(1) ESTIMATION REPORT ")
    print("==================================================\n")

    for name, est in estimates.items():
        print(f"--- {name} ---")
        print(f"  Reference Mean (x^r)        : {float(est.mean):.4f}")
        print(f"  Correlation Coeff (zeta_x)  : {float(est.zeta):.4f}")
        print(f"  Steady-State Variance       : {float(est.variance):.4f}\n")

    print("==================================================")


if __name__ == "__main__":
    generate_estimation_report()
