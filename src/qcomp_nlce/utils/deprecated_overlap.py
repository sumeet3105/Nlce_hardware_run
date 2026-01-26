import copy
from data_manager import ExperimentDataManager
from Nlce_hardware_run.src.qcomp_nlce.utils.workflow_tfim_1d import (
    heff_backend_type,
    dispersion_all_backends,
    plot_all_backends_dispersions_montecarlo,
)

from qcomp_nlce.utils.stats import mean_std, mc_disp_from_res_dict

import matplotlib.pyplot as plt
import numpy as np


def measure_matrix_element_circuits(
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    handcraft: bool = True,
) -> tuple[QuantumCircuit, QuantumCircuit]:
    circ_out = circuit_with_init(circuit, x, y)
    re_circ = hadamard_test_circuit(
        circ_out, parameters, real_part=True, handcraft=handcraft
    )
    im_circ = hadamard_test_circuit(
        circ_out, parameters, real_part=False, handcraft=handcraft
    )
    return re_circ, im_circ


def measure_matrix_elements_merge_circuits(
    circuit: QuantumCircuit,
    comp_states: list[str],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
    handcraft: bool = True,
):

    circs: list[QuantumCircuit] = []
    for state_x in comp_states:
        for state_y in comp_states:
            re_circ, im_circ = measure_matrix_element_circuits(
                circuit=circuit,
                x=state_x,
                y=state_y,
                parameters=parameters,
                handcraft=handcraft,
            )
            circs.append(re_circ)
            circs.append(im_circ)

    jobs = sample_circuit(circs, parameters, backend, n_shots)
    n_elems = len(comp_states) ** 2
    out = []

    for ind in range(0, 2 * n_elems - 1, 2):
        re_counts = get_counts_from_job(jobs, ind)
        im_counts = get_counts_from_job(jobs, ind + 1)
        v = hadam_diff(re_counts, n_shots) + 1j * hadam_diff(im_counts, n_shots)
        out.append(v)

    return np.reshape(out, shape=(len(comp_states), len(comp_states)))


def swap_test(
    circuit: QuantumCircuit,
    init_circuit: QuantumCircuit = None,
    parameters: np.ndarray = None,
    backend: BACKEND_ALIAS = "statevector",
    n_shots: int = 1000,
):
    circuit = apply_parameters(circuit, parameters)
    swap_circuit = swap_test_circuit(circuit, init_circuit)
    swap_circuit.add_register(ClassicalRegister(1, name="meas"))
    swap_circuit.measure(0, 0)
    counts = sample_counts_circuit(swap_circuit, backend, n_shots)
    p0 = counts["0"] / n_shots
    return 2 * p0 - 1


def iterative_hadamard_test(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
    backend: BACKEND_ALIAS = "statevector",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
) -> float:

    # !!! This only works for initial state ket(0)
    circuits_re = iterative_hadamard_test_circuits(
        circuit=circuit,
        parameters=parameters,
        real_part=True,
        custom_idx=custom_idx,
        handcraft=handcraft,
    )
    circuits_im = iterative_hadamard_test_circuits(
        circuit=circuit,
        parameters=parameters,
        real_part=False,
        custom_idx=custom_idx,
        handcraft=handcraft,
    )
    circuits = [*circuits_re, *circuits_im]
    half_len = len(circuits_re)

    jobs = sample_circuit(circuits, parameters, backend, n_shots)
    gs = []
    # the first qubit is the last in the bitstring
    keys = {
        "p0": "0" * (circuit.num_qubits) + "0",
        "p1": "0" * (circuit.num_qubits) + "1",
    }
    for ind in range(0, half_len):
        re_counts = get_counts_from_job(jobs, ind)
        im_counts = get_counts_from_job(jobs, ind + half_len)
        gl = hadam_diff(re_counts, keys=keys) + 1j * hadam_diff(im_counts, keys=keys)
        gs.append(gl)

    gs = np.array(gs)

    if np.any(np.isclose(gs, 0.0)):
        # the zeros cancel out
        if np.sum(np.isclose(gs[::2], 0.0)) == np.sum(np.isclose(gs[1::2], 0.0)):
            gs[(np.isclose(gs, 0.0))] = 1.0
        else:
            # technically, there also could be a mistake somewhere,
            # but fuck this
            return 0.0

    if len(gs) == 1:
        final_g = gs[0]
    elif len(gs) % 2 == 0:
        final_g = np.prod(gs[1::2]) / np.prod(np.conj(gs[0::2]))
    elif len(gs) % 2 == 1:
        final_g = np.prod(gs[0::2]) / np.prod(np.conj(gs[1::2]))

    # assert np.isclose(np.abs(final_g), 1.0)
    return final_g


def iterative_hadamard_test_estimator(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
    backend: BACKEND_ALIAS = "statevector",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
):
    # !!! This only works for initial state ket(0)

    circuits = iterative_hadamard_test_circuits(
        circuit=circuit,
        parameters=parameters,
        real_part="estimator",
        custom_idx=custom_idx,
        handcraft=handcraft,
    )

    proj = projector_ham(circuit.num_qubits)
    hams = [
        proj ^ SparsePauliOp("I"),
        proj ^ SparsePauliOp("X"),
        proj ^ SparsePauliOp("Y"),
        proj ^ SparsePauliOp("Z"),
    ]

    jobs = estimate_circuit(
        backend=backend,
        observables=hams,
        circuit=circuits,
        parameters=None,
        n_shots=n_shots,
    )

    final_g, _, _, std = get_overlaps_from_res(jobs.result())

    return final_g, std
