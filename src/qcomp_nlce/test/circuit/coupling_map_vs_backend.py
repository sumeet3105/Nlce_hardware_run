from qiskit.transpiler import generate_preset_pass_manager
from qcomp_nlce.utils import get_appropriate_fake_backend, compute_mean_circ_stats
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_clifford_circuit
from qiskit_ibm_runtime import SamplerV2


def test_compilation_schemes(circuit: QuantumCircuit | list):

    if isinstance(circuit, list):
        n_qubits = circuit[0].num_qubits
    else:
        n_qubits = circuit.num_qubits

    backend = get_appropriate_fake_backend(n_qubits, "IBM")

    pm = generate_preset_pass_manager(coupling_map=backend.coupling_map)

    pm2 = generate_preset_pass_manager(backend=backend)

    print("Coupling_map")
    circ1 = pm.run(circuit)
    compute_mean_circ_stats(circ1, True)
    print("backend")
    circ2 = pm2.run(circuit)
    compute_mean_circ_stats(circ2, True)

    sampler = SamplerV2(backend)

    job1 = sampler.run([circ1], shots=1000)
    job2 = sampler.run([circ2], shots=1000)

    res1 = job1.result()
    res2 = job2.result()

    ...


if __name__ == "__main__":
    circ = random_clifford_circuit(5, 40)
    test_compilation_schemes(circ)


# MORAL OF THE STORY: NO COUPLING MAP AT ALL!
