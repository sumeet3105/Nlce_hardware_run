from qcomp_nlce.hw.quasiparticle.unitary_1qp_sector import loop_measure_1qp_elements_v2
from qcomp_nlce.utils import (
    load_optimized_vqe,
    res_sim_dir,
    circuit_to_matrix,
    circuit_matrix_elements,
    apply_parameters,
    hamming_weight_states,
)

from qcomp_nlce.utils.results import pretty_print_diff
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit


def get_true_values(circuit: QuantumCircuit) -> np.ndarray:
    comp_states = hamming_weight_states(circuit.num_qubits, 1)
    elems = circuit_matrix_elements(circuit, comp_states)
    mat = circuit_to_matrix(circuit)
    return mat[0, 0], elems


if __name__ == "__main__":

    elem_dirname = Path(
        res_sim_dir(),
        f"elem_vqe_100j/vqe/cost_offdiag/5q_3l",
    )
    hamiltonian1, elem_circuit, elem_weights, ground_energy = load_optimized_vqe(
        elem_dirname
    )
    elem_circuit = apply_parameters(elem_circuit, elem_weights[-1])

    which_backends = ["fake_backend", "statevector"]
    provider_name = "AQT"
    n_shots = int(2e3)

    d1 = loop_measure_1qp_elements_v2(
        elem_circuit,
        which_backends=which_backends,
        provider_name=provider_name,
        which_hadamard_test="stefan_samples",
        n_shots=n_shots,
    )
    d2 = loop_measure_1qp_elements_v2(
        elem_circuit,
        which_backends=which_backends,
        provider_name=provider_name,
        which_hadamard_test="stefan",
        n_shots=n_shots,
    )
    d3 = loop_measure_1qp_elements_v2(
        elem_circuit,
        which_backends=which_backends,
        provider_name=provider_name,
        which_hadamard_test="estimator",
        n_shots=n_shots,
    )

    oUo, true_comp_vals = get_true_values(elem_circuit)
    true_stef_vals = oUo.conj() * true_comp_vals

    stefsam = d1["fake_backend"]["values"]
    stefem = d2["fake_backend"]["values"]
    estim = d3["fake_backend"]["values"]

    sv_stefsam = d1["statevector"]["values"]
    sv_stefem = d2["statevector"]["values"]
    sv_estim = d3["statevector"]["values"]

    print("ESTIM")
    pretty_print_diff(estim, true_comp_vals)

    print("STEF ESTIM")
    pretty_print_diff(stefem, true_stef_vals)

    print("STEF SAMPLES")
    pretty_print_diff(stefsam, true_stef_vals)

    print("SV ESTIM")
    pretty_print_diff(sv_estim, true_comp_vals)

    print("SV STEF ESTIM")
    pretty_print_diff(sv_stefem, true_stef_vals)

    print("SV STEF SAMPLES")
    pretty_print_diff(sv_stefsam, true_stef_vals)

    print("SUMMARY")
    print("estimator")
    print(np.sum(np.abs(estim - true_comp_vals)))
    print("stefan estimator")
    print(np.sum(np.abs(stefem - true_stef_vals)))
    print("stefan sampling")
    print(np.sum(np.abs(stefsam - true_stef_vals)))

    ...
