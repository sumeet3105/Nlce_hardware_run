from qiskit import QuantumCircuit
import numpy as np
from qcomp_nlce.utils import (
    circuit_matrix_elements,
    measure_matrix_elements,
    get_appropriate_fake_backend,
    get_backend,
    hamming_weight_states,
    custom_idx_groups_of_n,
    compare_with_true,
    HADAMARD_TESTS,
    get_backend_name,
    BACKEND_ALIAS,
    circuit_to_matrix,
)
from pathlib import Path

from qiskit_ibm_runtime import Session

HOME = Path(__file__).parent


def loop_measure_1qp_elements_v2(
    circuit: QuantumCircuit,
    which_backends: list[str],
    solve: bool = True,
    provider_name: str = "IBM",
    n_shots: int = 1e3,
    group_size: int = None,
    which_hadamard_test: str = "full",
):
    n_qubits = circuit.num_qubits
    comp_states = hamming_weight_states(n_qubits, 1)

    if solve:
        true_comp_vals = circuit_matrix_elements(circuit, comp_states)
        if "stefan" in which_hadamard_test:
            oUo = circuit_to_matrix(circuit)[0, 0]
            true_comp_vals = oUo.conj() * true_comp_vals
    else:
        true_comp_vals = None
    d = {
        "system": {"n_qubits": circuit.num_qubits, "values": true_comp_vals},
    }

    for which_backend in which_backends:
        print(f"==== {which_backend} ====")
        item = measure_1qp_elements_v2(
            circuit,
            which_backend,
            true_comp_vals,
            provider_name,
            n_shots,
            group_size,
            which_hadamard_test,
        )
        d[which_backend] = item[which_backend]
    return d


def measure_1qp_elements_v2(
    circuit: QuantumCircuit,
    which_backend: str,
    true_comp_vals: np.ndarray = None,
    provider_name: str = "IBM",
    n_shots: int = 1e3,
    group_size: int = None,
    which_hadamard_test: HADAMARD_TESTS = "full",
):

    n_qubits = circuit.num_qubits
    print(f"Minimal trace error (only shot noise) {.5*(n_qubits)/np.sqrt(n_shots):.4e}")
    comp_states = hamming_weight_states(n_qubits, 1)
    true_comp_vals = circuit_matrix_elements(circuit, comp_states)

    # the matrix you get otherwise is indexed backwards wrt to what you get
    # from the circuit-to-matrix qiskit function
    true_comp_vals = true_comp_vals[::-1, ::-1]

    if which_hadamard_test != "full":
        if group_size is None:
            group_size = len(circuit)
        custom_idx = custom_idx_groups_of_n(len(circuit), group_size)
    else:
        custom_idx = None

    if which_backend == "statevector":
        bn = which_backend
        backend = bn

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
            comp_vals, stds = measure_matrix_elements(
                circuit,
                comp_states,
                circuit.parameters,
                backend,
                which_backend,
                n_shots,
                custom_idx=custom_idx,
                which_hadamard_test=which_hadamard_test,
            )
    else:
        comp_vals, stds = measure_matrix_elements(
            circuit,
            comp_states,
            circuit.parameters,
            backend,
            which_backend,
            n_shots,
            custom_idx=custom_idx,
            which_hadamard_test=which_hadamard_test,
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


def measure_1qp_elements_v3(
    circuit: QuantumCircuit,
    which_backend: str,
    backend: BACKEND_ALIAS,
    true_comp_vals: np.ndarray = None,
    n_shots: int = 1e3,
    group_size: int = None,
    which_hadamard_test: HADAMARD_TESTS = "full",
):
    bn = get_backend_name(backend)
    n_qubits = circuit.num_qubits
    print(f"Minimal trace error (only shot noise) {.5*(n_qubits)/np.sqrt(n_shots):.4e}")
    comp_states = hamming_weight_states(n_qubits, 1)
    true_comp_vals = circuit_matrix_elements(circuit, comp_states)

    # the matrix you get otherwise is indexed backwards wrt to what you get
    # from the circuit-to-matrix qiskit function
    true_comp_vals = true_comp_vals[::-1, ::-1]

    if which_hadamard_test != "full":
        if group_size is None:
            group_size = len(circuit)
        custom_idx = custom_idx_groups_of_n(len(circuit), group_size)
    else:
        custom_idx = None

    comp_vals, stds = measure_matrix_elements(
        circuit,
        comp_states,
        circuit.parameters,
        backend,
        which_backend,
        n_shots,
        custom_idx=custom_idx,
        which_hadamard_test=which_hadamard_test,
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
