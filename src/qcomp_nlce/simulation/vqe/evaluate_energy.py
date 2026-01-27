import numpy as np

from qcomp_nlce.utils import (
    get_ground_energy,
    get_hva_vqe,
    TFIM,
    get_optimal_n_layers,
)
from qcomp_nlce.utils.statevector import sv_expectation_observables


def evaluate_energy(
    n_qubits: int,
    J: float,
    h: float,
    longitudinal_field: float,
    parameters: np.ndarray,
    periodic: bool = False,
    n_layers: int = None,
) -> dict:
    """
    Evaluate the energy of a VQE circuit for given parameters.

    Args:
        n_qubits: Number of qubits
        J: Coupling strength (XX interaction)
        h: Transverse field strength (Z field)
        longitudinal_field: Longitudinal field strength (X field)
        parameters: Array of circuit parameters
        periodic: Whether to use periodic boundary conditions
        n_layers: Number of VQE layers (if None, uses 2 * optimal)

    Returns:
        dict with energy, ground_energy, and relative_error
    """
    hamiltonian = TFIM(
        n_qubits, h, J, periodic=periodic, longitudinal_field=longitudinal_field
    )

    if n_layers is None:
        n_layers = get_optimal_n_layers(hamiltonian.num_qubits)

    circuit = get_hva_vqe(hamiltonian, n_layers=n_layers, unique_weights=True)

    expected_n_params = len(circuit.parameters)
    if len(parameters) != expected_n_params:
        raise ValueError(
            f"Expected {expected_n_params} parameters, got {len(parameters)}"
        )

    energy = sv_expectation_observables(
        circuit=circuit,
        observables=[hamiltonian],
        parameters=parameters,
    )[0]

    ground_energy = get_ground_energy(hamiltonian)
    relative_error = np.abs(ground_energy - energy) / np.abs(ground_energy)

    return {
        "energy": np.real(energy),
        "ground_energy": ground_energy,
        "relative_error": np.real(relative_error),
        "n_params": expected_n_params,
        "n_layers": n_layers,
        "circuit_depth": circuit.depth(),
    }


if __name__ == "__main__":
    # Example usage with the same settings as vqe_gs_simulations.py
    n_qubits = 4
    J = 1.0
    h = 1
    hl = 0.0  # longitudinal field
    periodic = False

    # Create the circuit to see how many parameters are needed
    hamiltonian = TFIM(n_qubits, h, J, periodic=periodic, longitudinal_field=hl)
    n_layers = get_optimal_n_layers(n_qubits)
    circuit = get_hva_vqe(hamiltonian, n_layers=n_layers, unique_weights=True)

    print(f"Circuit requires {len(circuit.parameters)} parameters")
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Number of layers: {n_layers}")
    print()

    # Replace this with your actual parameters
    # For demonstration, using zeros (which won't be optimal)
    julia_parameters = [0.19592065919565052,0.2621430393018819,0.195920659440307,0.6064083320382069,0.5038932744351345,0.503893274474842,0.6064083320674938,0.31098287067504593,-0.0685050814774656,0.31098287047530465,0.601795925941863,-0.18360261828864113,-0.18360261816571743,0.6017959260406702
]

    # Or load from a file:
    # your_parameters = np.load("path/to/params.npy")

    result = evaluate_energy(
        n_qubits=n_qubits,
        J=J,
        h=h,
        longitudinal_field=hl,
        parameters=julia_parameters,
        periodic=periodic,
    )

    print(f"Energy:          {result['energy']:.6f}")
    print(f"Ground energy:   {result['ground_energy']:.6f}")
    print(f"Relative error:  {result['relative_error']:.6f}")
