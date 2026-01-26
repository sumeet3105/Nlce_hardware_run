import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace
from scipy.optimize import minimize
from itertools import combinations
from collections.abc import Callable
from qcomp_nlce.utils.circuit import (
    apply_parameters,
    circuit_to_matrix,
    parameters_exist,
)
from qcomp_nlce.utils.hardware import circuits_to_estimator_payload
from qcomp_nlce.utils.hamiltonian import from_bitstring
from typing import Literal


def sv_estimate_observables_jobs(
    circuit: QuantumCircuit | list[QuantumCircuit],
    observables: list[SparsePauliOp],
    parameters: np.ndarray,
    n_shots: int,
):
    estimator = StatevectorEstimator()
    if isinstance(circuit, QuantumCircuit):
        circuit = [circuit]
    ctr = [[c, observables] for c in circuit]
    if parameters_exist(parameters):
        for idx in range(len(ctr)):
            ctr[idx].append(parameters)
    job = estimator.run(ctr, precision=1 / np.sqrt(n_shots))

    return job


def sv_estimate_observables(
    circuit: QuantumCircuit,
    observables: list[SparsePauliOp],
    param_sweep: np.ndarray,
    n_shots: int,
):
    job = sv_estimate_observables_jobs(circuit, observables, param_sweep, n_shots)
    result = job.result()[0]
    return result


def sv_expectation_observables(
    circuit: QuantumCircuit,
    observables: list[SparsePauliOp],
    parameters: np.ndarray,
):
    r = np.zeros((len(observables),), dtype=complex)
    cpar = apply_parameters(circuit, parameters)
    for jj, o in enumerate(observables):
        val = Statevector(cpar).expectation_value(
            o,
            range(0, o.num_qubits),
        )
        r[jj] = val
    return r


def optimize_circuit(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    cost: Callable[[np.ndarray], float] = None,
    initial_params=None,
) -> list:
    energy_values = []
    xks = []

    def callback(xk: np.ndarray, *args):
        e = cost(xk=xk)
        print(e, end="\r")
        energy_values.append(e)
        xks.append(xk)

    if cost is None:

        def cost(xk: np.ndarray):
            result = sv_expectation_observables(
                circuit=circuit,
                observables=[hamiltonian],
                parameters=xk,
            )
            return result[0]

    def jac(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for ind in range(len(x)):
            p = x.copy()
            p[ind] += np.pi / 2
            m = x.copy()
            m[ind] -= np.pi / 2
            res_p = sv_expectation_observables(
                circuit=circuit, observables=[hamiltonian], parameters=p
            )
            res_m = sv_expectation_observables(
                circuit=circuit, observables=[hamiltonian], parameters=m
            )
            out[ind] = 0.5 * (res_m[0] - res_p[0])
        return out

    if initial_params is None:
        initial_params = np.zeros((len(circuit.parameters)))

    #kwargs = {"method": "COBYLA", "bounds": circuit.num_parameters * [(0, np.pi)]}
    # kwargs = {"method": "CG", "options": {"maxiter": 500}}
    kwargs = {"method": "CG"}

    minimize(fun=cost, x0=initial_params, callback=callback, **kwargs)
    return energy_values, xks


def get_circuit_energy(
    hamiltonian: SparsePauliOp, circuit: QuantumCircuit, use_estimator: bool = False
):
    if use_estimator:
        return sv_estimate_observables(circuit, [hamiltonian], None).data.evs[0]
    else:
        return Statevector(circuit).expectation_value(
            hamiltonian,
            range(0, hamiltonian.num_qubits),
        )


def statevector_sample(circuit: QuantumCircuit, n_shots: int):
    sampler = StatevectorSampler(seed=0)
    job = sampler.run([circuit], shots=n_shots)
    counts = job.result()[0].data.meas.get_counts()
    return counts


def statevector_sample_unique(
    circuit: QuantumCircuit,
    n_shots: int,
    qubits: list = None,
    use_sampler: bool = False,
) -> list:
    out_circ = circuit.remove_final_measurements(inplace=False)
    out_circ.measure_all(add_bits=True)
    if qubits is None:
        qubits = list(range(circuit.num_qubits))
    if use_sampler:
        sampler = StatevectorSampler(seed=0)
        job = sampler.run([out_circ], shots=n_shots)
        counts = job.result()[0].data.meas.get_counts()
    else:
        sv = Statevector(circuit)
        if len(qubits) < circuit.num_qubits:
            sv = partial_trace(
                sv, [q for q in range(circuit.num_qubits) if q not in qubits]
            )
        counts = sv.sample_counts(n_shots)
    res = [str(k) for k in counts.keys()]
    return res


def sort_bitstrings(bitstrings: list[str], right_to_left: bool = True) -> list[str]:
    def key(s: str) -> int:
        return from_bitstring(s, len(s), right_to_left)

    return sorted(bitstrings, key=key)


def hamming_weight_states(
    n_qubits: int, hamming_weight: int, sort: Literal["none", "rtl", "ltr"] = "rtl"
) -> list[str]:
    comps = combinations(range(n_qubits), hamming_weight)
    states = []
    for comp in comps:
        state = ["0"] * n_qubits
        for c in comp:
            state[c] = "1"
        states.append("".join(state))
    if sort != "none":
        return sort_bitstrings(states, sort == "rtl")
    return states


def single_x(ind: int, n_qubits: int) -> SparsePauliOp:
    l = ["I" for x in range(n_qubits)]
    l[ind] = "X"
    return SparsePauliOp("".join(l))


def estimate_observable_1qp_initial_state(
    circuit: QuantumCircuit, parameters: np.ndarray, observable: SparsePauliOp
) -> np.ndarray:
    n_qubits = circuit.num_qubits
    circuit_out = apply_parameters(circuit, parameters)

    umat = circuit_to_matrix(circuit_out)

    omat = observable.to_matrix(False)

    uHu = umat.T.conj() @ omat @ umat

    comp_states = hamming_weight_states(n_qubits, 1)
    inds = [from_bitstring(b, n_qubits, True) for b in comp_states]

    return uHu[np.ix_(inds, inds)]


# line 147, circuit needs to be replaced with circ_out
