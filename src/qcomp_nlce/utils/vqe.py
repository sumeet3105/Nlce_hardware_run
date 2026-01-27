from collections import defaultdict
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from uuid import uuid4
import numpy as np

from qiskit import __version__ as qiskit_version

if qiskit_version > "1.3.0":
    from qiskit.circuit.library import hamiltonian_variational_ansatz
from qiskit.quantum_info import SparsePauliOp
from qcomp_nlce.utils.hamiltonian import pretty_pauli_str
import itertools


# random ansatz with Ry, Ry and CNOTs as layers
def get_custom_vqe(n_qubits: int, n_layers: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        for x in range(n_qubits):
            param1 = Parameter(name=str(uuid4()))
            param2 = Parameter(name=str(uuid4()))
            circuit.ry(param1, x)
            circuit.rx(param2, x)
        for x, y in zip(range(n_qubits - 1), range(1, n_qubits)):
            circuit.cx(x, y)
    return circuit


def get_optimal_n_layers(n_qubits: int) -> int:
    return int(np.ceil(n_qubits / 2))


# HVA ansatz
def get_hva_vqe(
    hamiltonian: SparsePauliOp, n_layers: int, unique_weights: bool = False
) -> QuantumCircuit:
    """
    Create HVA (Hamiltonian Variational Ansatz) circuit.

    Args:
        hamiltonian: The Hamiltonian whose terms define the ansatz structure
        n_layers: Number of layers
        unique_weights: If True, each term gets a unique parameter
    """
    if unique_weights:
        circuit = QuantumCircuit(hamiltonian.num_qubits)
        for layer in range(n_layers):
            param_counts = defaultdict(int)
            for term in hamiltonian:
                p = term.paulis[0]
                pauli_label = pretty_pauli_str(p) or "I"
                suffix = param_counts[pauli_label]
                param_counts[pauli_label] += 1
                if suffix:
                    param_name = f"l{layer}_{pauli_label}_{suffix}"
                else:
                    param_name = f"l{layer}_{pauli_label}"
                param = Parameter(param_name)
                circuit.append(PauliEvolutionGate(term, time=param), range(len(p)))

    else:
        circuit = hamiltonian_variational_ansatz(hamiltonian, n_layers)
    return circuit


def get_nq_paulistr(n_qubits: int, nq_pauli: str = "XX"):
    combs = itertools.combinations(range(n_qubits), len(nq_pauli))
    paulistr_list = []
    for comb in combs:
        pid = ["I"] * n_qubits
        for ind, val in enumerate(comb):
            pid[val] = nq_pauli[ind]
        paulistr_list.append("".join(pid))
    return paulistr_list


# Givens rotations ansatz: XX + YY terms
def get_givens_vqe(hamiltonian: SparsePauliOp, n_layers: int) -> QuantumCircuit:
    n_qubits = hamiltonian.num_qubits
    circuit = QuantumCircuit(n_qubits)
    xx_list = get_nq_paulistr(n_qubits, nq_pauli="XX")
    yy_list = get_nq_paulistr(n_qubits, nq_pauli="YY")
    for layer in range(n_layers):
        for ind, (xx_str, yy_str) in enumerate(zip(xx_list, yy_list)):
            xx = SparsePauliOp(xx_str)
            yy = SparsePauliOp(yy_str)
            param = Parameter(f"l{layer}_{ind}")
            circuit.append(PauliEvolutionGate(xx + yy, time=param), range(n_qubits))
    return circuit


# Prepares a circuit that first initializes n_qp qubits in the |1> state
def circuit_composed_with_prep(circuit: QuantumCircuit, n_qp: int) -> QuantumCircuit:
    init_circuit = QuantumCircuit(circuit.num_qubits)
    for q in range(n_qp):
        init_circuit.x(q)

    return init_circuit.compose(circuit, qubits=circuit.qubits, inplace=False)
