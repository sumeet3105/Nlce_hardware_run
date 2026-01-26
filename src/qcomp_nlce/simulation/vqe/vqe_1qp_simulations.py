import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    get_ground_energy,
    get_hva_vqe,
    TFIM,
    optimize_circuit,
    save_optimized_vqe,
    load_optimized_vqe,
    get_save_directory,
    estimate_observable_1qp_initial_state,
    get_tfim_sweep_circuit,
    circuit_to_matrix,
    apply_parameters,
    get_min_gap,
    get_optimal_n_layers,
)

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from typing import Callable

# Ground state energy of TFIM using VQE with HVA


def get_cost_offdiag(
    circuit: QuantumCircuit, hamiltonian: SparsePauliOp
) -> Callable[[np.ndarray], float]:
    def cost(xk: np.ndarray) -> float:
        a_mat = estimate_observable_1qp_initial_state(circuit, xk, hamiltonian)
        b_mat = estimate_observable_1qp_initial_state(circuit, xk, hamiltonian**2)
        return np.abs(np.trace(b_mat) - np.sum(np.abs(a_mat) ** 2))

    return cost


def get_cost_trace(
    circuit: QuantumCircuit, hamiltonian: SparsePauliOp
) -> Callable[[np.ndarray], float]:
    """
    Alternative cost function for VQE 1qp optimization.
    Minimizes the trace of the expectation matrix: sum of diagonal elements.

    Returns:
        Cost function that takes parameters and returns trace of expectation matrix
    """

    def cost(xk: np.ndarray) -> float:
        # Compute expectation matrix A = <psi|U* H U|psi>
        a_mat = estimate_observable_1qp_initial_state(circuit, xk, hamiltonian)
        # Return real part of sum of diagonal elements (trace)
        return np.real(np.trace(a_mat))

    return cost


def get_cost_sweep(
    vqe_circ: QuantumCircuit,
    ham_start: SparsePauliOp,
    ham_end: SparsePauliOp,
) -> Callable[[np.ndarray], float]:
    min_gap = get_min_gap(ham_end)
    total_time = 1 / (min_gap**2)
    n_steps = 10 * ham_end.num_qubits**2

    sweep_circuit = get_tfim_sweep_circuit(
        ham_start,
        ham_end,
        total_time=total_time,
        n_steps=int(n_steps),
    )
    sweep_mat = circuit_to_matrix(sweep_circuit)

    def cost(xk: np.ndarray) -> float:

        vqe_mat = circuit_to_matrix(apply_parameters(vqe_circ, xk))

        return np.sqrt(2**n_qubits - np.abs(np.trace(sweep_mat.conj().T @ vqe_mat)))

    return cost


def train_and_save(
    n_qubits: int,
    J: float,
    hl: float,
    dirname: str,
    which_cost: str,
    dry_run: bool,
    n_layers: int = None,
):
    print(f"{n_qubits} qubits, J/h={J}")
    periodic = False
    h = 1
    ham_j, ham_h = TFIM(n_qubits, h, J, periodic=False, split_parts=True, longitudinal_field=0.1)
    hamiltonian = ham_j + ham_h
    if n_layers is None:
        n_layers = 2*get_optimal_n_layers(hamiltonian.num_qubits)
    circuit = get_hva_vqe(hamiltonian, n_layers=n_layers, unique_weights=True)

    # circuit = get_custom_vqe(4, n_layers=2)
    ground_energy = get_ground_energy(hamiltonian)

    if which_cost == "sweep":
        cost = get_cost_sweep(circuit, ham_j, hamiltonian)
    elif which_cost == "offdiag":
        cost = get_cost_offdiag(circuit, hamiltonian)
    else:
        raise ValueError(f"Invalid cost: {which_cost}")

    print(f"The circuit depth is {circuit.depth()}")
    print(f"The number of parameters is {len(circuit.parameters)}")

    # Load optimized parameters from ground state VQE
    gs_dirname = f"gs_vqe_{int(J*100)}j_{int(hl*100)}hl"
    gs_vqe_path = get_save_directory(f"{gs_dirname}/vqe/{n_qubits}q_{n_layers}l")
    _, _, gs_weights, _ = load_optimized_vqe(gs_vqe_path)
    initial_params = np.real(gs_weights[-1])  # Take the last (final optimized) set of parameters
    # print(initial_params.shape)
    # # Original initial_params setup:
    # initial_params = 0.1 * np.ones((len(circuit.parameters),))

    cost_values, params = optimize_circuit(
        circuit,
        hamiltonian,
        cost=cost,
        initial_params=initial_params,
    )
    print(f"final cost is {np.abs(cost_values[-1])}")

    save_optimized_vqe(
        f"{dirname}/vqe/cost_{which_cost}/{n_qubits}q_{n_layers}l",
        hamiltonian,
        circuit,
        params,
        ground_energy,
        dry_run,
    )

hl=0.1
# input here "offdiag" or "sweep"
if __name__ == "__main__":
    dry_run = False
    which_cost = "offdiag"
    for n_qubits in (5,):
        for J in (0.8,):
            dirname = f"elem_vqe_{int(J*100)}j_{int(hl*100)}hl"
            train_and_save(n_qubits, J, hl, dirname, which_cost, dry_run)


