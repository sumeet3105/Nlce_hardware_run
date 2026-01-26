import os
from pathlib import Path
from typing import Literal

import numpy as np
from qcomp_nlce.utils import (
    circ_ham_expectation_values,
    compare_with_true,
    get_appropriate_fake_backend,
    get_backend,
    hamming_weight_states,
    measure_hamiltonian_elements,
    BACKEND_ALIAS,
    get_backend_name,
    BACKEND_TYPES,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


from qiskit_ibm_runtime import Session

HOME = Path(__file__).parent


def loop_measure_1qp_expectations_v2(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    which_backends: list[str],
    solve: bool = True,
    provider_name: str = "IBM",
    n_shots: int = 1e3,
):
    n_qubits = circuit.num_qubits
    comp_states = hamming_weight_states(n_qubits, 1)
    if solve:
        true_comp_vals = circ_ham_expectation_values(hamiltonian, circuit, comp_states)
    else:
        true_comp_vals = None
    d = {
        "system": {"n_qubits": circuit.num_qubits, "values": true_comp_vals},
    }
    for which_backend in which_backends:
        print(f"==== {which_backend} ====")
        item = measure_1qp_expectations_v2(
            hamiltonian, circuit, which_backend, true_comp_vals, provider_name, n_shots
        )
        d[which_backend] = item[which_backend]
    return d


def measure_1qp_expectations_v2(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    which_backend: BACKEND_TYPES,
    true_comp_vals: np.ndarray = None,
    provider_name: str = "IBM",
    n_shots: int = 1e3,
) -> np.ndarray:

    n_qubits = circuit.num_qubits
    print(f"Minimal trace error (only shot noise) {.5*(n_qubits)/np.sqrt(n_shots):.4e}")

    comp_states = hamming_weight_states(n_qubits, 1)

    if which_backend == "statevector":
        bn = which_backend
        backend = which_backend

    elif which_backend == "fake_backend" or which_backend == "real_backend":

        if which_backend == "fake_backend":
            backend = get_appropriate_fake_backend(
                circuit.num_qubits, which=provider_name
            )
        elif which_backend == "real_backend":
            backend = get_backend(circuit.num_qubits, provider_name)
        bn = backend.name

    # Session mode for faster execution

    if which_backend == "real_backend":
        with Session(backend) as s:
            comp_vals, stds = measure_hamiltonian_elements(
                hamiltonian,
                circuit,
                comp_states,
                None,
                backend,
                which_backend,
                n_shots,
            )
    else:
        comp_vals, stds = measure_hamiltonian_elements(
            hamiltonian,
            circuit,
            comp_states,
            None,
            backend,
            which_backend,
            n_shots,
        )
    print(f"{which_backend} ({bn})")
    td = compare_with_true(true_comp_vals, comp_vals, bn)

    return {
        which_backend: {
            "name": bn,
            "trace_dist": td,
            "values": comp_vals,
            "stds": stds,
        }
    }


def measure_1qp_expectation_v3(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    backend: BACKEND_ALIAS,
    which_backend: BACKEND_TYPES,
    true_comp_vals: np.ndarray = None,
    n_shots: int = 1e3,
):
    n_qubits = circuit.num_qubits
    print(f"Minimal trace error (only shot noise) {.5*(n_qubits)/np.sqrt(n_shots):.4e}")
    bn = get_backend_name(backend)
    comp_states = hamming_weight_states(n_qubits, 1)
    comp_vals, stds = measure_hamiltonian_elements(
        hamiltonian,
        circuit,
        comp_states,
        None,
        backend,
        which_backend,
        n_shots,
    )
    print(f"{which_backend} ({bn})")
    td = compare_with_true(true_comp_vals, comp_vals, bn)

    return {
        which_backend: {
            "name": bn,
            "trace_dist": td,
            "values": comp_vals,
            "stds": stds,
        }
    }
