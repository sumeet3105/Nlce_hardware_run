from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qcomp_nlce.utils.measure import (
    sample_counts_circuit,
    circuit_estimation_job,
    circuit_sampling_job,
    get_counts_from_res,
    convert_count_keys,
    std_from_shots,
    trace_out_estimation,
)
from qcomp_nlce.utils.hamiltonian import (
    index_bits,
    projector_ham,
    trace_dist,
    get_tfim_ratio,
    HAM_NAME_VAR,
)
from qcomp_nlce.utils.circuit import (
    to_controlled_gate,
    apply_parameters,
    circuit_instruction_to_circuit,
    prep_circ_superpos,
    circuit_to_matrix,
)

from qcomp_nlce.utils.save import (
    tmp_entry_exists,
    add_tmp_entry_and_save,
    load_tmp_item,
    rebuild_matrices_from_tmp,
    tmp_filename,
)

from qcomp_nlce.utils.unscented_transform import (
    unscented_transform,
    complex_to_double_real,
    vectorize_complex_func,
)

from qcomp_nlce.utils.hardware import (
    is_hardware_run,
    BACKEND_ALIAS,
    sample_based_estimation,
)

from qcomp_nlce.utils.stats import sum_std, mean_std

from qiskit.primitives import EstimatorResult, PrimitiveResult

from typing import Union, Tuple, Literal


def custom_idx_groups_of_n(total_len: int, group_size: int = 2):
    return [
        list(range(x, min(x + group_size, total_len)))
        for x in range(0, total_len, group_size)
    ]


def compare_with_true(true_vals: np.ndarray, vals: np.ndarray, name: str) -> float:
    td = -1
    if true_vals is not None:
        td = trace_dist(true_vals, vals)
        print(f"{name} -- {td:.5e}")
    else:
        print(np.round(vals, 3))
    return td


def iterative_hadamard_test_circuits(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    real_part: bool = True,
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
) -> list[QuantumCircuit]:
    circuit = apply_parameters(circuit, parameters)
    circuits = []
    if custom_idx is None:
        custom_idx = [[x] for x in range(0, len(circuit))]
    for x in range(len(custom_idx)):
        hadam_circuit = QuantumCircuit(circuit.num_qubits + 1)
        hadam_circuit.h(0)
        if False == real_part:
            hadam_circuit.sdg(0)
        stuekchen = [g for l in custom_idx[:x] for g in l]
        circuit_stueckchen = [circuit[c] for c in stuekchen]
        if len(circuit_stueckchen):
            c_temp = circuit_instruction_to_circuit(circuit_stueckchen)
            hadam_circuit.append(c_temp, range(1, hadam_circuit.num_qubits))

        for _gate in custom_idx[x]:
            controlled_gate = to_controlled_gate(circuit[_gate], None, handcraft)
            hadam_circuit.append(controlled_gate, range(circuit.num_qubits + 1))

        if real_part != "estimator":
            hadam_circuit.h(0)
            creg = ClassicalRegister(hadam_circuit.num_qubits, name="meas")
            hadam_circuit.add_register(creg)
            # we need to project the whole state back in ket(0)bra(0),
            # so we measure all the qubits of the state
            # because our initial state is ket(0) luckily
            for q in range(hadam_circuit.num_qubits):
                hadam_circuit.measure(q, q)

        circuits.append(hadam_circuit)
    return circuits


def get_overlaps_from_res(
    res: Union[EstimatorResult],
) -> Tuple[float, np.ndarray, np.ndarray]:
    # 1000 lines of code, and 999 of those are dedicated to babysit qiskit incompatibilities
    if isinstance(res, EstimatorResult):
        # this is a v1 primitive, so it depends on the order of your for loops :)
        # I'm going to be a goose farmer i stg
        n_obs = 4
        gs = []
        g_abs = []
        stds = []
        vals = res.values
        variances = [m["variance"] for m in res.metadata]
        for ind in range(0, len(vals), n_obs):
            gs.append(vals[ind + 1] + 1j * vals[ind + 2])
            stds.append(np.sqrt(variances[ind + 1]) + 1j * np.sqrt(variances[ind + 2]))
            g_abs.append(vals[ind] + 1j * vals[ind + 3])
    elif isinstance(res, PrimitiveResult):
        # gs = Re(g_l g*_l-1) + 1j * Im(g_l g*_l-1)
        gs = np.array([x.data.evs[1] + 1j * x.data.evs[2] for x in res])
        stds = np.array([x.data.stds[1] + 1j * x.data.stds[2] for x in res])

        # I-Z = 1/2 (g**2_l-1 + g**2_l) - 1/2 (g**2_l-1 - g**2_l) = g**2_l
        g_abs = np.array([np.sqrt(np.abs(x.data.evs[0] - x.data.evs[3])) for x in res])
        # g_abs is the absolute value of g_l = <psi|U_1...U_l-1 U_l|psi>
        # g_m1 computes g_l-1 so that we get twice the results per shot.
        # I+Z = 1/2 (g**2_l-1 + g**2_l) +  1/2 (g**2_l-1 - g**2_l) = g**2_l-1
        g_m1 = np.array([np.sqrt(np.abs(x.data.evs[0] + x.data.evs[3])) for x in res])

        better_g_abs = gs[:-1] + g_m1[1:] / 2
        # print(f"({y,x} {k} min|g_s|**2={np.min(g_abs**2)}", end="\r")

    if np.any(np.isclose(g_abs, 0)):
        raise ValueError(f"g_abs has null values, algorithm will fail: g_abs={g_abs}")

    if len(gs) == 1:
        final_g = gs[0]
    elif len(gs) % 2 == 0:

        def prodfunc(gs: np.ndarray) -> np.ndarray:
            return np.prod(gs[1::2]) / np.prod(np.conj(gs[0::2]))

    elif len(gs) % 2 == 1:

        def prodfunc(gs: np.ndarray) -> np.ndarray:
            return np.prod(gs[0::2]) / np.prod(np.conj(gs[1::2]))

    final_g = prodfunc(gs)
    _, pmat = unscented_transform(
        vectorize_complex_func(prodfunc),
        complex_to_double_real(gs),
        complex_to_double_real(stds),
    )
    # first order error prop

    assert len(pmat) == 2
    final_std = pmat[0] + 1j * pmat[1]

    return final_g, gs, g_abs, final_std


def iterative_hadamard_test_off_diag(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
    x: str,
    y: str,
    backend: BACKEND_ALIAS = "statevector",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
):

    if x == y:
        jUj, std = iterative_hadamard_test_projecto_estimo(
            circuit, parameters, n_shots, x, y, backend, custom_idx, handcraft
        )
        return (jUj, 0, std)

    # prepare
    # a = ket(x + iy)
    # b = ket(x - iy)
    # u = ket(x + y)
    # v = ket(x - y)

    # then
    # alpha = ia - i(u+v)/2
    # beta  = (u-v)/2
    # alpha+beta = 2 <k|U|j>
    # beta - alpha  = 2<j|U|k>

    d = {"+": 0 + 0j, "-": 0 + 0j, "+i": 0 + 0j, "-i": 0 + 0j}

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
    for k in d:
        prep = prep_circ_superpos(x, y, k)

        out_circs = []

        for circ in circuits:
            prep_dag = prep.inverse()
            out = QuantumCircuit(circ.num_qubits)
            out.compose(prep, range(1, circ.num_qubits), inplace=True)
            out.compose(circ, range(circ.num_qubits), inplace=True)
            out.compose(prep_dag, range(1, circ.num_qubits), inplace=True)
            out_circs.append(out)

        jobs = circuit_estimation_job(
            backend=backend,
            observables=hams,
            circuit=out_circs,
            parameters=None,
            n_shots=n_shots,
        )

        final_g, _, _, std = get_overlaps_from_res(jobs.result())
        d[k] = (final_g, std)

    alpha = 1j * (d["+i"][0] - (d["+"][0] + d["-"][0]) / 2)
    beta = (d["+"][0] - d["-"][0]) / 2

    jUk = beta - alpha
    kUj = alpha + beta

    # I wish I lived in a world without propagation of errors
    std = np.sqrt(0.5 * (d["+"][1] ** 2) + 0.5 * (d["-"][1] ** 2) + d["+i"][1] ** 2)

    return jUk, kUj, std


def iterative_hadamard_test_projecto_estimo(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
    x: str,
    y: str,
    backend: BACKEND_ALIAS = "statevector",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
):

    # this one computes <x|U|y> without doing the x(j) in the iterative hadamard
    # which avoids some of the 0 overlap issue
    # but it's more annyoing
    # well, this was stupid. g0 = 0 so this method cannot in fact work for non-diagonal terms.
    # but it should work for diagonal terms though. Useless. Ack!

    circuits = iterative_hadamard_test_circuits(
        circuit=circuit,
        parameters=parameters,
        real_part="estimator",
        custom_idx=custom_idx,
        handcraft=handcraft,
    )

    ket = y
    bra = x
    init = QuantumCircuit(circuit.num_qubits + 1)
    ket_inds = index_bits(ket + "0", circuit.num_qubits + 1, right_to_left=True)
    for ket_ind in ket_inds:
        init.x(ket_ind)
    for idx in range(len(circuits)):
        circuits[idx] = init.compose(circuits[idx], inplace=False)

    proj = projector_ham(circuit.num_qubits, bitstring=bra)
    hams = [
        proj ^ SparsePauliOp("I"),
        proj ^ SparsePauliOp("X"),
        proj ^ SparsePauliOp("Y"),
        proj ^ SparsePauliOp("Z"),
    ]
    jobs = circuit_estimation_job(
        backend=backend,
        observables=hams,
        circuit=circuits,
        parameters=None,
        n_shots=n_shots,
    )

    final_g, _, _, std = get_overlaps_from_res(jobs.result())

    return final_g, std


def build_stefan_circuit_estimator(
    circuit: QuantumCircuit, parameters: np.ndarray, x: str, y: str
) -> QuantumCircuit:
    n_qubits = circuit.num_qubits + 1
    hadam_circuit = QuantumCircuit(n_qubits)
    hadam_circuit.h(0)

    x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
    y_inds = index_bits(y, circuit.num_qubits, right_to_left=True)

    for y in y_inds:
        hadam_circuit.cx(0, y + 1)
    hadam_circuit.append(apply_parameters(circuit, parameters), range(1, n_qubits))
    for x in x_inds:

        hadam_circuit.cx(0, x + 1)

    return hadam_circuit


def build_stefan_circuits_sample(
    circuit: QuantumCircuit, parameters: np.ndarray, x: str, y: str
) -> QuantumCircuit:
    hadam_circuits = []
    n_qubits = circuit.num_qubits + 1

    x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
    y_inds = index_bits(y, circuit.num_qubits, right_to_left=True)

    for t in ("real", "imag"):
        hadam_circuit = QuantumCircuit(n_qubits)
        hadam_circuit.h(0)
        if t == "imag":
            hadam_circuit.sdg(0)

        for y in y_inds:
            hadam_circuit.cx(0, y + 1)
        hadam_circuit.append(apply_parameters(circuit, parameters), range(1, n_qubits))
        for x in x_inds:
            hadam_circuit.cx(0, x + 1)

        hadam_circuit.h(0)
        hadam_circuit.measure_all()

        hadam_circuits.append(hadam_circuit)

    return hadam_circuits


def stefan_hadamard_test(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
    x: str,
    y: str,
    backend: BACKEND_ALIAS = "statevector",
    counting_method: Literal["estimator", "samples"] = "estimator",
    n_reps: int = 2,
):

    # this is Stefan Wolf's test, a bit modified though
    # because <0|u|0> in our case cancels perfectly, so
    # we ignore it from the start!

    # oUo = circuit_to_matrix(circuit)[0, 0]
    # print(f"Uncomputed overlap is {oUo:.5e}")

    if counting_method == "estimator":
        stef_circuit = build_stefan_circuit_estimator(circuit, parameters, x, y)
        proj = projector_ham(circuit.num_qubits, bitstring="0" * len(x))
        hams = [
            proj ^ SparsePauliOp("X"),
            proj ^ SparsePauliOp("Y"),
        ]
        jobs = circuit_estimation_job(
            backend=backend,
            observables=hams,
            circuit=[
                stef_circuit,
            ]
            * n_reps,
            parameters=None,
            n_shots=n_shots,
        )

        ol, std = get_hadam_estimator(jobs.result(), n_reps)
        return ol, std

    elif counting_method == "samples":
        n_qubits = circuit.num_qubits + 1
        stef_circuits = build_stefan_circuits_sample(circuit, parameters, x, y)
        jobs = circuit_sampling_job(stef_circuits, parameters, backend, n_shots)
        keys = {"p0": "0" * (len(x) + 1), "p1": "0" * (len(x)) + "1"}
        res = jobs.result()
        re, rstd = hadam_diff(get_counts_from_res(res, n_qubits, 0), n_shots, keys=keys)
        im, imstd = hadam_diff(
            get_counts_from_res(res, n_qubits, 1), n_shots, keys=keys
        )

        return re + 1j * im, rstd + 1j * imstd


# computes the real part of overlap between psi and circuit(psi)
def hadamard_test_circuit(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    real_part: bool = True,
    handcraft: bool = True,
) -> QuantumCircuit:

    n_qubits = circuit.num_qubits + 1
    hadam_circuit = QuantumCircuit(n_qubits)
    hadam_circuit.h(0)
    if False == real_part:
        hadam_circuit.sdg(0)

    for ci in circuit:
        controlled_gate = to_controlled_gate(ci, parameters, handcraft)
        hadam_circuit.append(controlled_gate, range(n_qubits))
    if real_part != "estimator":
        hadam_circuit.h(0)
        hadam_circuit.add_register(ClassicalRegister(1, name="meas"))
        hadam_circuit.measure(0, 0)
    return hadam_circuit


def hadamard_test(
    circuit: QuantumCircuit,
    parameters: np.ndarray = None,
    backend: BACKEND_ALIAS = "statevector",
    n_shots: int = 1000,
    real_part: bool = True,
):
    hadam_circuit = hadamard_test_circuit(circuit, parameters, real_part)
    counts = sample_counts_circuit(hadam_circuit, backend, n_shots)
    val, std = hadam_diff(counts, n_shots)
    return val


def get_n0n1_hadam_counts(
    counts: dict[str, int], keys: dict[str, str] = {"p0": "0", "p1": "1"}
):
    # make sure that keys are str and not int!

    if keys["p0"] not in counts.keys():
        n0 = 0
    else:
        n0 = counts[keys["p0"]]
    if keys["p1"] not in counts.keys():
        n1 = 0
    else:
        n1 = counts[keys["p1"]]
    return n0, n1


def hadam_diff(
    counts: dict, n_shots: int, keys: dict[str, str] = {"p0": "0", "p1": "1"}
) -> float:

    n0, n1 = get_n0n1_hadam_counts(counts, keys)
    if n0 == 0 and n1 == 0:
        return 0, 0

    std = std_from_shots(n1, n0, n_shots)

    return (n0 - n1) / n_shots, std


def circuit_with_init(
    circuit: QuantumCircuit,
    x: str,
    y: str,
) -> QuantumCircuit:
    x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
    y_inds = index_bits(y, circuit.num_qubits, right_to_left=True)
    circ_in = QuantumCircuit(circuit.num_qubits)
    for ind_y in y_inds:
        circ_in.x(ind_y)
    circ_in = circ_in.compose(circuit)
    # U1.compose(U2) gives U2 U1 |psi>
    circ_out = QuantumCircuit(circuit.num_qubits)
    # x is left bra and y right ket
    for ind_x in x_inds:
        circ_out.x(ind_x)
    circ_out = circ_in.compose(circ_out, range(circuit.num_qubits))
    return circ_out


def get_hadam_estimator(
    res: EstimatorResult, n_reps: int = 1
) -> tuple[complex, complex]:
    # qiskit developers competing with python developers to see who can make the most breaking changes with a version chanege
    # anyway, here's wonderwall
    if n_reps == 1:
        try:
            return (
                res[0].data.evs[0] + 1j * res[0].data.evs[1],
                res[0].data.stds[0] + 1j * res[0].data.stds[1],
            )

        except Exception:
            return res.values[0] + 1j * res.values[1], np.sqrt(
                res.metadata[0]["variance"]
            ) + 1j * +np.sqrt(res.metadata[1]["variance"])
    else:
        vals_reps = []
        stds_reps = []
        for n in range(n_reps):
            jj = 2 * n
            if isinstance(res, EstimatorResult):
                val = res.values[jj] + 1j * res.values[jj + 1]
                std = np.sqrt(res.metadata[jj]["variance"]) + 1j * +np.sqrt(
                    res.metadata[jj + 1]["variance"]
                )
            else:
                val = res[n].data.evs[0] + 1j * res[n].data.evs[1]
                std = res[n].data.stds[0] + 1j * res[n].data.stds[1]

            vals_reps.append(val)
            stds_reps.append(std)
        return np.mean(vals_reps), mean_std(np.array(stds_reps))


HADAMARD_TESTS = Literal["stefan", "projecto-estimo", "off-diag", "full", "estimator"]


def measure_matrix_element(
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
    which_hadamard_test: HADAMARD_TESTS = "estimator",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
) -> Union[complex, tuple[complex, complex]]:

    # This computes <x|U|y>

    if which_hadamard_test == "projecto-estimo":

        ol, std = iterative_hadamard_test_projecto_estimo(
            circuit,
            parameters,
            n_shots,
            x,
            y,
            backend,
            custom_idx=custom_idx,
            handcraft=handcraft,
        )
        return ol, std

    elif which_hadamard_test == "stefan" or which_hadamard_test == "stefan_samples":
        if which_hadamard_test == "stefan_samples":
            counting_method = "samples"
        else:
            counting_method = "estimator"
        ol, std = stefan_hadamard_test(
            circuit, parameters, n_shots, x, y, backend, counting_method=counting_method
        )
        return ol, std

    elif which_hadamard_test == "off-diag":

        jUk, kUj, std = iterative_hadamard_test_off_diag(
            circuit,
            parameters,
            n_shots,
            x,
            y,
            backend,
            custom_idx=custom_idx,
            handcraft=handcraft,
        )
        return jUk, kUj, std

    elif which_hadamard_test == "full":
        circ_out = circuit_with_init(circuit, x, y)
        re_circ = hadamard_test_circuit(
            circ_out, parameters, real_part=True, handcraft=handcraft
        )
        im_circ = hadamard_test_circuit(
            circ_out, parameters, real_part=False, handcraft=handcraft
        )

        vals = []
        stds = []
        for tmp_circ in (re_circ, im_circ):
            counts = sample_counts_circuit(tmp_circ, parameters, backend, n_shots)
            v, std = hadam_diff(counts, n_shots)
            stds.append(std)
            vals.append(v)

        ol = vals[0] + 1j * vals[1]
        std = stds[0] + 1j * stds[1]
        return ol, std
    elif which_hadamard_test == "estimator":
        circ_out = circuit_with_init(circuit, x, y)
        if custom_idx is None:
            custom_idx = custom_idx_groups_of_n(len([c for c in circ_out]), 1)
        n_qubits = circuit.num_qubits
        hadam_circ = hadamard_test_circuit(
            circ_out, parameters=parameters, real_part="estimator", handcraft=handcraft
        )
        hams = [
            SparsePauliOp(n_qubits * "I" + "X"),
            SparsePauliOp(n_qubits * "I" + "Y"),
        ]

        job = circuit_estimation_job(
            backend=backend,
            observables=hams,
            circuit=hadam_circ,
            parameters=None,
            n_shots=n_shots,
        )

        res = job.result()
        return get_hadam_estimator(res)

    elif which_hadamard_test == "iterative":
        raise ValueError("This area is out of service chief")
        circ_out = circuit_with_init(circuit, x, y)

        ol = iterative_hadamard_test_estimator(
            circ_out,
            parameters,
            n_shots,
            backend,
            custom_idx=custom_idx,
            handcraft=handcraft,
        )
        return ol

    else:
        raise ValueError(
            f"Expected full, estimator, or iterative, got {which_hadamard_test}"
        )


def reorder_matrix_elems(vals: np.ndarray) -> np.ndarray:
    # somehow indices are all messy
    out = np.zeros_like(vals)
    diag = np.diag(vals)

    out = vals[::-1, ::-1]

    out[range(out.shape[0]), range(out.shape[0])] = diag

    return out


def measure_matrix_elements(
    circuit: QuantumCircuit,
    comp_states: list[str],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    which_backend: str,
    n_shots: int,
    which_hadamard_test: str,
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
) -> np.ndarray:

    vals = np.zeros((len(comp_states), len(comp_states)), dtype=complex)
    stds = np.zeros((len(comp_states), len(comp_states)), dtype=complex)

    on_hardware = is_hardware_run(which_backend)

    if on_hardware:
        filename = f"{tmp_filename(circuit,which_backend)}/measure_matrix_elements.json"
        try:

            computed_values = load_tmp_item(filename)
        except FileNotFoundError:
            print(f"Starting from scratch (hadamard {which_hadamard_test} test)")
            computed_values = []

    if which_hadamard_test != "off-diag":
        for ind_x, x in enumerate(comp_states):
            for ind_y, y in enumerate(comp_states):
                if on_hardware and tmp_entry_exists(computed_values, ind_x, ind_y):
                    pass
                    print(f"Already computed {ind_x}, {ind_y}")
                else:
                    jUk, std = measure_matrix_element(
                        circuit,
                        x,
                        y,
                        parameters,
                        backend,
                        n_shots,
                        which_hadamard_test,
                        custom_idx=custom_idx,
                        handcraft=handcraft,
                    )
                    vals[ind_x, ind_y] = jUk
                    stds[ind_x, ind_y] = std
                    if on_hardware:
                        computed_values = add_tmp_entry_and_save(
                            filename, computed_values, jUk, std, ind_x, ind_y
                        )
                    print(f"{ind_x} - {ind_y}: {jUk:.4f} ({std:.4f})")
    elif which_hadamard_test == "off-diag":
        for ind_x, x in enumerate(comp_states):
            for ind_y, y in enumerate(comp_states[ind_x:]):
                ind_y = ind_x + ind_y
                if on_hardware and tmp_entry_exists(computed_values, ind_x, ind_y):
                    pass
                    print(f"Already computed {ind_x}, {ind_y}")
                else:
                    jUk, kUj, std = measure_matrix_element(
                        circuit,
                        x,
                        y,
                        parameters,
                        backend,
                        n_shots,
                        which_hadamard_test,
                        custom_idx=custom_idx,
                        handcraft=handcraft,
                    )
                    vals[ind_x, ind_y] = jUk
                    vals[ind_y, ind_x] = kUj
                    stds[ind_x, ind_y] = std
                    stds[ind_y, ind_x] = std
                    if on_hardware:
                        computed_values = add_tmp_entry_and_save(
                            filename, computed_values, jUk, std, ind_x, ind_y
                        )
                        computed_values = add_tmp_entry_and_save(
                            filename, computed_values, kUj, std, ind_y, ind_x
                        )
                    print(f"{ind_x} - {ind_y}: {jUk:.3f}, {kUj:.3f} ({std:.4f})")
    if on_hardware:
        vals, stds = rebuild_matrices_from_tmp(computed_values, len(comp_states))
    # somehow indices are all messy
    vals = reorder_matrix_elems(vals)
    stds = reorder_matrix_elems(stds)
    return vals, stds


def estimation_from_sample_based(
    res: PrimitiveResult, hamiltonian: SparsePauliOp
) -> tuple[complex, complex]:

    if hamiltonian.__dict__[HAM_NAME_VAR] == "TFIM":
        ratio = get_tfim_ratio(hamiltonian)
        zs = []
        z_stds = []
        for ind_z in range(hamiltonian.num_qubits):
            val, std = trace_out_estimation(res.get_counts()[0], ind_z)
            zs.append(val)
            z_stds.append(std)

        n_even = len(list(res.get_counts()[1].keys())[0].replace(" ", ""))
        n_odd = len(list(res.get_counts()[2].keys())[0].replace(" ", ""))

        xxs_even = []
        xxs_even_stds = []
        xxs_odd = []
        xxs_odd_stds = []
        for ind_x_even in range(n_even):
            val, std = trace_out_estimation(res.get_counts()[1], ind_x_even)
            xxs_even.append(val)
            xxs_even_stds.append(std)
        for ind_x_odd in range(n_odd):
            val, std = trace_out_estimation(res.get_counts()[2], ind_x_odd)
            xxs_odd.append(val)
            xxs_odd_stds.append(std)

        fval = -sum(zs) + -ratio * (sum(xxs_even) + sum(xxs_odd))
        fstd = np.sqrt(
            sum_std(
                [
                    sum_std(z_stds),
                    ratio * sum_std([sum_std(xxs_even_stds), sum_std(xxs_odd_stds)]),
                ]
            )
        )

    return fval, fstd


def measure_hamiltonian_element_manual(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
    n_reps: int = 1,
) -> tuple[complex, complex]:
    if x == y:
        circ_out = QuantumCircuit(circuit.num_qubits)
        x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
        for x_ind in x_inds:
            circ_out.x(x_ind)
        circ_out.compose(circuit, inplace=True)
        job = sample_based_estimation(backend, circ_out, hamiltonian, n_shots)
        res = job.result()
        return estimation_from_sample_based(res, hamiltonian)
    else:
        d = ["+", "-", "+i", "-i"]
        stds = []
        vals = []
        for k in d:
            circ_p = prep_circ_superpos(x, y, k)
            circ_p.compose(circuit, inplace=True)
            job = sample_based_estimation(backend, circ_p, hamiltonian, n_shots)
            res = job.result()
            val, std = estimation_from_sample_based(res, hamiltonian)
            vals.append(val)
            stds.append(std)

        val = (vals[0] - vals[1]) / 2 + 1j * (vals[2] - vals[3]) / 2
        std = sum_std(stds[0:2]) + 1j * sum_std(stds[2:4])
        return val, std


def measure_hamiltonian_element(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
    n_reps: int = 2,
) -> tuple[complex, complex]:
    if x == y:
        circ_out = QuantumCircuit(circuit.num_qubits)
        x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
        for x_ind in x_inds:
            circ_out.x(x_ind)
        circ_out.compose(circuit, inplace=True)
        if n_reps > 1:
            circ_out = [circ_out for _ in range(n_reps)]
        job = circuit_estimation_job(
            circ_out, [hamiltonian], parameters, backend, n_shots
        )

        res = job.result()
        if n_reps == 1:
            if isinstance(res, EstimatorResult):
                return res.values[0], np.sqrt(res.metadata[0]["variance"])
            else:
                return res[0].data.evs, res[0].data.stds
        else:
            if isinstance(res, EstimatorResult):
                return np.mean([res.values[k] for k in range(n_reps)]), mean_std(
                    np.array(
                        [np.sqrt(res.metadata[k]["variance"]) for k in range(n_reps)]
                    )
                )
            else:
                return np.mean([res[k].data.evs for k in range(n_reps)]), mean_std(
                    np.array([res[k].data.stds for k in range(n_reps)])
                )

    else:
        d = ["+", "-", "+i", "-i"]
        circs = []
        for _ in range(n_reps):
            for k in d:
                circ_p = prep_circ_superpos(x, y, k)
                circ_p.compose(circuit, inplace=True)
                circs.append(circ_p)

        job = circuit_estimation_job(circs, [hamiltonian], parameters, backend, n_shots)

        res = job.result()
        if n_reps == 1:
            if isinstance(res, EstimatorResult):
                val = (res.values[0] - res.values[1]) / 2 + 1j * (
                    res.values[2] - res.values[3]
                ) / 2
                stds = [np.sqrt(res.metadata[k]["variance"]) for k in range(4)]
                std = sum_std(stds[0:2]) + 1j * sum_std(stds[2:4])
                return val, std

            else:
                val = (res[0].data.evs - res[1].data.evs) / 2 + 1j * (
                    res[2].data.evs - res[3].data.evs
                ) / 2
                stds = [res[k].data.stds for k in range(4)]
                std = sum_std(stds[0:2]) + 1j * sum_std(stds[2:4])
                return val, std
        else:
            vals_reps = []
            stds_reps = []
            for n in range(n_reps):
                jj = 4 * n
                if isinstance(res, EstimatorResult):

                    val = (res.values[jj] - res.values[jj + 1]) / 2 + 1j * (
                        res.values[jj + 2] - res.values[jj + 3]
                    ) / 2
                    stds = [np.sqrt(res.metadata[jj + k]["variance"]) for k in range(4)]
                    std = sum_std(stds[:2]) + 1j * sum_std(stds[2:4])

                else:
                    val = (res[jj].data.evs - res[jj + 1].data.evs) / 2 + 1j * (
                        res[jj + 2].data.evs - res[jj + 3].data.evs
                    ) / 2
                    stds = [res[jj + k].data.stds for k in range(4)]
                    std = sum_std(stds[0:2]) + 1j * sum_std(stds[2:4])
                vals_reps.append(val)
                stds_reps.append(std)

            return np.mean(vals_reps), mean_std(np.array(stds_reps))


def measure_hamiltonian_elements(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    states: list[str],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    which_backend: str,
    n_shots: int,
):

    vals = np.zeros((len(states), len(states)), dtype=complex)
    stds = np.zeros((len(states), len(states)), dtype=complex)

    on_hardware = is_hardware_run(which_backend)

    if on_hardware:
        filename = (
            f"{tmp_filename(circuit,which_backend)}/measure_hamiltonian_elements.json"
        )
        try:
            computed_values = load_tmp_item(filename)
        except FileNotFoundError:
            print("Starting from scratch (expectations)")
            computed_values = []

    for ind_x, x in enumerate(states):
        for ind_y, y in enumerate(states[ind_x:]):
            ind_y = ind_x + ind_y
            if on_hardware and tmp_entry_exists(computed_values, ind_x, ind_y):
                pass
                print(f"Already computed {ind_x}, {ind_y}")
            else:
                if backend != "statevector" and "IQM" in backend.name:
                    val, std = measure_hamiltonian_element_manual(
                        hamiltonian, circuit, x, y, parameters, backend, n_shots
                    )
                else:
                    val, std = measure_hamiltonian_element(
                        hamiltonian, circuit, x, y, parameters, backend, n_shots
                    )
                vals[ind_x, ind_y] = val.conj()
                vals[ind_y, ind_x] = val
                stds[ind_x, ind_y] = std
                stds[ind_y, ind_x] = std

                print(f"{ind_x} - {ind_y} - {complex(val):.4f} ({complex(std):.4f})")
                if on_hardware:
                    computed_values = add_tmp_entry_and_save(
                        filename, computed_values, val.conj(), std, ind_x, ind_y
                    )
                    computed_values = add_tmp_entry_and_save(
                        filename, computed_values, val, std, ind_y, ind_x
                    )
    # this is correct!
    if on_hardware:
        vals, stds = rebuild_matrices_from_tmp(computed_values, len(states))
    vals = reorder_matrix_elems(vals)[::-1, ::-1]
    stds = reorder_matrix_elems(stds)[::-1, ::-1]
    return vals, stds
