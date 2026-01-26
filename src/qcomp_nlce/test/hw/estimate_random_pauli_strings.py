from qiskit_ibm_runtime import Estimator, Sampler
from qiskit.transpiler import generate_preset_pass_manager
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, Statevector
from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMBackendBase
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
import numpy as np
from typing import Literal
from iqm.qiskit_iqm import transpile_to_IQM
from iqm_v2_utils import IQMFakePerfect, IQMBackendEstimator


from qcomp_nlce.utils import measure_pauli_circuit


def random_pauli_op(n_qubits: int, locality: int = 2) -> SparsePauliOp:
    out = ["I" for _ in range(n_qubits)]
    paulis = ["X", "Y", "Z"]
    a = np.random.choice(paulis, locality, replace=True)
    pos = np.random.choice(n_qubits, locality, replace=False)
    for ind, val in enumerate(pos):
        out[val] = a[ind]
    return SparsePauliOp("".join(out))


def reverse_pauli(pauli: SparsePauliOp) -> SparsePauliOp:
    pauli_str = str(pauli.paulis[0])
    return SparsePauliOp("".join(reversed(pauli_str)))


def measure_pauli(
    backend: IQMBackendBase,
    circ: QuantumCircuit,
    pauli: SparsePauliOp,
    n_shots: int,
    version: Literal["v1", "v2"],
) -> float:
    circ2 = measure_pauli_circuit(circ, pauli)

    isa_circuit = transpile_circuit(circ2, backend, "v1")

    if version == "v1":
        job = backend.run(isa_circuit, shots=n_shots)
        res = job.result()
        counts: dict = res.get_counts()
    elif version == "v2":
        sampler = Sampler(backend)
        job = sampler.run([isa_circuit], shots=int(n_shots))
        res = job.result()
        counts: dict = res[0].join_data().get_counts()

    count_0 = counts.get("0", 0)
    count_1 = counts.get("1", 0)
    val = (count_0 - count_1) / n_shots
    return val


def transpile_circuit(
    circuit: QuantumCircuit, backend, version: Literal["v1", "v2"]
) -> QuantumCircuit:
    initial_layout = list(range(circuit.num_qubits))
    if version == "v2":
        pm = generate_preset_pass_manager(
            backend=backend,
            initial_layout=initial_layout,
            coupling_map=backend.coupling_map,
        )
        isa_circuit = pm.run(circuit)
    elif version == "v1":

        isa_circuit = transpile(
            circuit,
            backend=backend,
            initial_layout=initial_layout,
            layout_method="dense",
            # optimization_level=3,
            coupling_map=backend.coupling_map,
            seed_transpiler=42,
        )
    elif version == "iqm":
        isa_circuit = transpile_to_IQM(
            circuit, backend, initial_layout=initial_layout, ignore_barriers=True
        )
    return isa_circuit


if __name__ == "__main__":

    work_ops = ["IZXI", "XIIY", "ZIXI", "XIYI", "ZIIZ", "XIZI", "ZXII", "XIXI", "XXII"]
    dont_work_ops = ["IIYX", "XIIX", "IIYY"]

    do_reps = ["XIXI"] * 10

    n_qubits = 4
    circ = QuantumCircuit(n_qubits)
    circ.h(range(circ.num_qubits))

    circ.ry(1, range(circ.num_qubits))
    for y in range(circ.num_qubits - 1):
        circ.cx(y, y + 1)

    backend = IQMFakePerfect()

    estimator = IQMBackendEstimator(backend=backend)

    for strop in do_reps:

        op = SparsePauliOp(strop)
        isa_circuit: QuantumCircuit = transpile_circuit(circ, backend, "v1")
        # op = random_pauli_op(circ.num_qubits)
        job = estimator.run(
            [[isa_circuit, op.apply_layout(isa_circuit._layout)]],
            precision=1e-3,
        )

        res = job.result()

        ev_sample = measure_pauli(backend, circ, op, n_shots=1e5, version="v1")

        sv = Statevector(circ)
        opmat = op.to_matrix()

        # <0|U* O U|0>
        ev = sv.data.conj().dot(opmat @ sv.data)
        ev_sim = res[0].data.evs

        print(op.paulis)

        print(
            f"true: {np.real(ev):.5f}, estimator: {ev_sim:.5f}, diff: {np.abs(np.real(ev)-ev_sim):.6f}"
        )

        print(
            f"true: {np.real(ev):.5f}, sample: {ev_sample:.5f}, diff: {np.abs(np.real(ev)-ev_sample):.6f}"
        )

        # print(job._args[0][0].circuit.layout)
