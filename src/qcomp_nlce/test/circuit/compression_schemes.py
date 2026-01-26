from qcomp_nlce.utils.folders import res_sim_dir
from qcomp_nlce.utils.circuit import apply_parameters, circ_stats
from qcomp_nlce.utils.sweep import sweep_time_from_steps, get_tfim_sweep_circuit
from qcomp_nlce.utils.save import load_optimized_vqe
from qcomp_nlce.utils.hardware import (
    compile_circuit_for_backend,
    get_appropriate_fake_backend,
    compute_mean_circ_stats,
    get_backend,
)
from qcomp_nlce.utils.hamiltonian import TFIM
from pathlib import Path
from qiskit import QuantumCircuit
from qcomp_nlce.utils.overlap import build_stefan_circuits_sample


def test_compilation_schemes(circuit: QuantumCircuit | list, provider_name: str):

    if isinstance(circuit, list):
        n_qubits = circuit[0].num_qubits
    else:
        n_qubits = circuit.num_qubits

    backend = get_backend(n_qubits, provider_name)
    print("Basic, no AI")
    no_ai = compile_circuit_for_backend(circuit, backend, ai=False)
    print("with Heuristic + AI")
    mit_ai = compile_circuit_for_backend(circuit, backend, ai=True)
    ...


def stef_compression():
    print("# Stefan Hadamard tests")
    gs_dirname_l = Path(res_sim_dir(), f"gs_vqe_100j/vqe/5q_3l")
    gs_dirname_s = Path(res_sim_dir(), f"gs_vqe_100j/vqe/4q_2l")

    _, elem_circuit_l, elem_weights_l, _ = load_optimized_vqe(gs_dirname_l)
    elem_circuit_l = apply_parameters(elem_circuit_l, elem_weights_l[-1])

    _, elem_circuit_s, elem_weights_s, _ = load_optimized_vqe(gs_dirname_s)
    elem_circuit_s = apply_parameters(elem_circuit_s, elem_weights_s[-1])

    stefans = build_stefan_circuits_sample(elem_circuit_l, None, "10000", "10000")

    test_compilation_schemes(stefans)


def vqe_compression():
    print("# VQE")
    gs_dirname_l = Path(res_sim_dir(), f"gs_vqe_100j/vqe/5q_3l")
    gs_dirname_s = Path(res_sim_dir(), f"gs_vqe_100j/vqe/4q_2l")

    _, elem_circuit_l, elem_weights_l, _ = load_optimized_vqe(gs_dirname_l)
    elem_circuit_l = apply_parameters(elem_circuit_l, elem_weights_l[-1])

    _, elem_circuit_s, elem_weights_s, _ = load_optimized_vqe(gs_dirname_s)
    elem_circuit_s = apply_parameters(elem_circuit_s, elem_weights_s[-1])
    print("##IQM")
    print("## large")
    test_compilation_schemes(elem_circuit_l, "IQM")
    print("## small")
    test_compilation_schemes(elem_circuit_s, "IQM")
    print("##QEXA")
    print("## large")
    test_compilation_schemes(elem_circuit_l, "QEXA")
    print("## small")
    test_compilation_schemes(elem_circuit_s, "QEXA")


def sweep_compression():
    print("# Sweep")
    J = 1
    n_steps = 10
    n_qubits = 5
    ham_j, ham_h = TFIM(n_qubits, 1, J, periodic=False, split_parts=True)
    hamiltonian = ham_j + ham_h
    total_time = sweep_time_from_steps(ham_h, hamiltonian, n_steps)
    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=total_time,
        n_steps=n_steps,
    )

    print("## f{n_qubits} qubits")
    test_compilation_schemes(circuit)


if __name__ == "__main__":
    # stef_compression()
    vqe_compression()
    # sweep_compression()
