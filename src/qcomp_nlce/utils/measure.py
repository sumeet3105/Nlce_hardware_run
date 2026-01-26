from qiskit.primitives import StatevectorSampler, PrimitiveJob, StatevectorEstimator
from qcomp_nlce.utils.hardware import backend_sample, backend_estimate, BACKEND_ALIAS
from qcomp_nlce.utils.hamiltonian import to_bitstring
from qcomp_nlce.utils.circuit import parameters_exist
from qcomp_nlce.utils.statevector import (
    sv_estimate_observables,
    sv_estimate_observables_jobs,
    sv_expectation_observables,
)
from qiskit_ibm_runtime import IBMBackend
from qiskit_ibm_runtime.base_runtime_job import BaseRuntimeJob
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from qiskit import ClassicalRegister, QuantumCircuit
from typing import Union
from qiskit.primitives import PrimitiveResult

try:
    from mqp.qiskit_provider.job import MQPJob
except ModuleNotFoundError:
    try:
        from mqss.qiskit_adapter.job import MQSSQiskitJob

        MQPJob = MQSSQiskitJob
    except ModuleNotFoundError as e:
        raise e

    class MQPJob: ...


try:

    from qiskit.result import Result
except Exception:

    class Result: ...


from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit_aer import AerJob
from iqm.qiskit_iqm.iqm_job import IQMJob


def circuit_estimation_job(
    circuit: QuantumCircuit | list[QuantumCircuit],
    observables: Union[SparsePauliOp, list[SparsePauliOp]],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS = "statevector",
    n_shots: int = 1000,
):
    if backend == "statevector":

        job = sv_estimate_observables_jobs(circuit, observables, parameters, n_shots)

    else:
        job = backend_estimate(
            backend=backend,
            observables=observables,
            circuit=circuit,
            parameters=parameters,
            n_shots=n_shots,
        )
    return job


def trace_out_estimation(counts: dict, ind: int) -> tuple[float, float]:
    assert all(ind < len(k) for k in counts.keys())
    count_0 = 0
    count_1 = 0
    for k in counts.keys():
        # random spaces in my bitstrings?
        ks = k.replace(" ", "")
        if ks[ind] == "0":
            count_0 += counts[k]
        elif ks[ind] == "1":
            count_1 += counts[k]
        else:
            raise ValueError(
                f"Expected bitstring to be either zeros or ones, got {k[ind]}"
            )
    return (count_0 - count_1) / (count_0 + count_1), std_from_shots(
        count_1, count_0, count_0 + count_1
    )


def std_from_shots(n_1: int, n_0: int, n_shots: int) -> float:
    if n_1 == 0 and n_0 == 0:
        return 0
    # <(Z x |0X0|)^2> - <(Z x |0X0|)>^2
    # or simply <Z^2> - <Z>^2
    t1 = (n_0 + n_1) / n_shots
    t2 = ((n_0 - n_1) / (n_shots)) ** 2
    return np.sqrt(t1 - t2)


def circuit_sampling_job(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS = "statevector",
    n_shots: int = 1000,
) -> PrimitiveJob:
    if backend == "statevector":
        # statevector always sample the same outcomes if seed is set
        # I guess this is for reproducibility
        # Anyway, don't set seed unless you want to pull your hair out
        # https://github.com/Qiskit/qiskit/issues/13047
        sampler = StatevectorSampler()
        if isinstance(circuit, QuantumCircuit):
            ctr = [circuit]
        else:
            ctr = circuit
        job = sampler.run(ctr, shots=int(n_shots))
    else:
        job = backend_sample(
            backend=backend, circuit=circuit, parameters=parameters, n_shots=n_shots
        )
    return job


def sample_counts_circuit(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    backend: str | FakeBackendV2 | IBMBackend = "statevector",
    n_shots: int = 1000,
):
    job = circuit_sampling_job(circuit, parameters, backend, n_shots)
    counts = get_counts_from_res(job.result(), circuit.num_clbits)
    return counts


def convert_count_keys(counts: dict, n_qubits: int):
    int_keys = any([set(str(k)).issubset(set(("0", "1"))) for k in counts])
    counts_out = {}
    if int_keys:
        for k in counts:
            nk = to_bitstring(k, n_qubits, False)
            counts_out[nk] = counts[k]
    else:
        for k in counts:
            counts_out[str(k)] = counts[k]
    return counts_out


def get_counts_from_res(
    res: Union[PrimitiveResult, Result], n_qubits: int, ind: int = 0
) -> dict:
    # you get them at some point
    try:
        key = list(res[ind].data.keys())[0]
        counts = res[ind].data[key].get_counts()
        return convert_count_keys(counts, n_qubits)
    except Exception:
        pass
    try:
        qd = res.quasi_dists[ind]
        n_shots = res.metadata[ind]["shots"]
        counts = {k: int(qd[k] * n_shots) for k in qd.keys()}
        return convert_count_keys(counts, n_qubits)
    except Exception:
        pass
    try:
        counts = res[ind].join_data().get_counts()
        return convert_count_keys(counts, n_qubits)
    except Exception:
        pass
    try:
        counts = res.get_counts(ind)
        return convert_count_keys(counts, n_qubits)
    except Exception as e:
        raise RuntimeError(f"Unknown result: {res}")


def get_counts_from_job(
    job: Union[BaseRuntimeJob, PrimitiveJob, MQPJob], ind: int = 0
) -> dict:
    if isinstance(job, BaseRuntimeJob):
        key = list(job.result()[ind].data.keys())[0]
        counts = job.result()[ind].data[key].get_counts()
    elif isinstance(job, PrimitiveJob):
        res = job.result()
        try:
            qd = res.quasi_dists[ind]
            n_shots = res.metadata[ind]["shots"]
            counts = {k: int(qd[k] * n_shots) for k in qd.keys()}
        except AttributeError:
            counts = res[ind].join_data().get_counts()
    elif isinstance(job, (MQPJob, AerJob, IQMJob)):
        counts = job.result().get_counts(ind)
    else:
        raise ValueError(f"Unexpected job type: type {type(job)}")
    return counts


def add_meas_layer(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit = circuit.copy()
    meas_register = ClassicalRegister(circuit.num_qubits)
    circuit.add_register(meas_register)
    for q, c in zip(circuit.qubits, meas_register):
        circuit.measure(q, c)
    return circuit


def get_unique_bitstrings(counts: dict) -> list[str]:
    return np.unique(list(counts.keys()))
