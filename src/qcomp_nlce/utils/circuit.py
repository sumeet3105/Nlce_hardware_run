from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit import ClassicalRegister
from qiskit.quantum_info import Operator, SparsePauliOp
import numpy as np
from qcomp_nlce.utils.hamiltonian import from_bitstring, hamming_weight, index_bits
from typing import Union, Literal
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.parametertable import ParameterView


def circ_stats(circ: QuantumCircuit):
    depth = circ.depth()
    n_gates = circ.size()
    n_non_local_gates = circ.num_nonlocal_gates()
    return depth, n_gates, n_non_local_gates


def parameters_exist(parameters) -> bool:
    x = (
        (parameters is None)
        or (isinstance(parameters, ParameterView) and parameters == ParameterView([]))
        or (len(parameters) == 0)
    )
    return not x


# despite the warning CircuitInstruction as a type hint is correct.
def circuit_instruction_to_circuit(ci: CircuitInstruction | list, qubits=None):
    if isinstance(ci, list):
        if qubits is None:
            qregs = [c.qubits[0]._register for c in ci]
            m = np.argmax([r.size for r in qregs])
            qubits = qregs[m]
        circ = QuantumCircuit(qubits)
        for ci_inst in ci:
            inds = [q._index for q in ci_inst.qubits]
            circ.append(ci_inst.operation, inds)
    else:
        if qubits is None:
            qubits = ci.qubits[0]._register
        circ = QuantumCircuit(qubits)
        inds = [q._index for q in ci.qubits]
        circ.append(ci.operation, inds)
    return circ


def remove_null_param_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    c_out = QuantumCircuit(circuit.qubits, circuit.clbits)
    for ci in circuit:
        if len(ci.params) == 0 or not np.all([np.isclose(p, 0.0) for p in ci.params]):
            tcirc = circuit_instruction_to_circuit(ci)
            c_out.compose(tcirc, tcirc.qubits, inplace=True)
    return c_out


def apply_parameters(circuit: QuantumCircuit, parameters: np.ndarray) -> QuantumCircuit:
    options = {}
    if parameters_exist(parameters):
        param_dict = parameters_dict(circuit, parameters)
        options["parameter_map"] = param_dict
        return remove_null_param_gates(circuit.assign_parameters(param_dict))
    else:
        return circuit


def rotate_in_basis(
    circuit: QuantumCircuit,
    qubit: int,
    which: Literal["X", "Y", "Z", "I"],
    dag: bool = False,
) -> QuantumCircuit:
    if which == "X":
        circuit.h(qubit)
    elif which == "Y":
        if not dag:
            circuit.s(qubit)
            circuit.h(qubit)
        else:
            circuit.h(qubit)
            circuit.s(qubit)
    elif which == "Z" or which == "I":
        pass
    return circuit


def to_controlled_gate(
    circuit: Union[QuantumCircuit | CircuitInstruction],
    parameters: np.ndarray,
    handcraft: bool = True,
) -> Gate:
    if isinstance(circuit, QuantumCircuit):
        circuit = apply_parameters(circuit, parameters)
        gate = circuit.to_gate().control(1)
    elif isinstance(circuit, CircuitInstruction):
        if circuit.operation.name == "PauliEvolution" and handcraft:
            # homemade artisan controlled pauli evolution
            n_qubits = len(circuit.qubits) + 1
            op = circuit.operation
            time = op.time
            pauli = op.operator.paulis[0]
            coeff = op.operator.coeffs[0]
            cx_inds = [idx for idx in range(len(pauli)) if str(pauli[idx]) != "I"]
            c_out = QuantumCircuit(n_qubits)

            # rotate
            for q in range(1, n_qubits):
                c_out = rotate_in_basis(c_out, q, str(pauli[q - 1]))

            # cxs
            for q in range(len(cx_inds) - 1):
                c_out.cx(1 + cx_inds[q], 1 + cx_inds[q + 1])

            # rz what a waste of time to find that missing 2 factor
            c_out.crz(2 * time * np.real(coeff), 0, max(cx_inds) + 1)

            # cxs
            cx_inds = cx_inds[::-1]
            for q in range(len(cx_inds) - 1):
                c_out.cx(1 + cx_inds[q + 1], 1 + cx_inds[q])

            # rotate back
            for q in range(1, n_qubits):
                c_out = rotate_in_basis(c_out, q, str(pauli[q - 1]), dag=True)
            gate = c_out.to_gate(label=f"c-exp(-it {pauli})")
        else:
            try:
                opdef = circuit.operation._definition.name
            except Exception:
                opdef = circuit.operation.name
            circuit = circuit_instruction_to_circuit(circuit)
            circuit = apply_parameters(circuit, parameters)
            gate = circuit.to_gate(label=opdef).control(1)

    return gate


def parameters_dict(
    circuit: Union[QuantumCircuit, list[QuantumCircuit]], parameters: np.ndarray
) -> dict:
    if not parameters_exist(circuit.parameters) or not parameters_exist(parameters):
        return {}
    return {k: v for k, v in zip(circuit.parameters, parameters)}


def isa_circuit_param_dict(
    new_circuit: QuantumCircuit, old_circuit: QuantumCircuit, parameters: np.ndarray
) -> dict:
    param_dict = parameters_dict(old_circuit, parameters)
    return {k: param_dict[k] for k in new_circuit.parameters}


def circuit_to_matrix(circuit: QuantumCircuit) -> np.ndarray:
    return Operator(circuit).to_matrix()


def circuit_matrix_elements(circuit: QuantumCircuit, states: list[str]) -> np.ndarray:
    mat = circuit_to_matrix(circuit)
    sl = [
        from_bitstring(state, n_qubits=circuit.num_qubits, right_to_left=True)
        for state in states
    ]
    return mat[np.ix_(sl, sl)]


def measure_pauli_circuit(
    circ: QuantumCircuit, pauli: SparsePauliOp, copy: bool = True
) -> QuantumCircuit:
    pauli_str = "".join(reversed(str(pauli.paulis[0])))

    if copy:
        circ2 = circ.copy()
    else:
        circ2 = circ

    inds = []

    for ind, val in enumerate(pauli_str):
        if val != "I":
            if val == "X":
                circ2.h(ind)
            elif val == "Y":
                circ2.sdg(ind)
                circ2.h(ind)
            inds.append(ind)
    for x in range(len(inds) - 1, 0, -1):
        circ2.cx(inds[x], inds[x - 1])
    creg = ClassicalRegister(1, name=f"__c_{pauli_str}")
    circ2.add_register(creg)
    circ2.measure(inds[0], creg[0])
    return circ2


def prep_circ_superpos(x: str, y: str, which: Literal["+", "-", "+i", "-i"]):
    if hamming_weight(x) == 1 and hamming_weight(y) == 1:
        return prep_circ_1qp_superpos(x, y, which)
    else:
        if y == x:
            c = QuantumCircuit(len(x))
            x_inds = index_bits(x, len(x), right_to_left=True)
            for x_ind in x_inds:
                c.x(x_ind)
        else:
            if (
                max(hamming_weight(x), hamming_weight(y)) == len(x)
                and min(hamming_weight(x), hamming_weight(y)) == 0
            ):
                c = QuantumCircuit(len(x))
                if which == "+":
                    c.h(0)
                    for ind in range(1, len(x)):
                        c.cx(0, ind)
                elif which == "-":
                    c.x(0)
                    c.h(0)
                    for ind in range(1, len(x)):
                        c.cx(0, ind)
                elif which == "+i":
                    c.h(0)
                    c.s(0)
                    for ind in range(1, len(x)):
                        c.cx(0, ind)
                elif which == "-i":
                    c.x(0)
                    c.h(0)
                    c.s(0)
                    for ind in range(1, len(x)):
                        c.cx(0, ind)
            else:
                raise NotImplementedError()

    return c


def prep_circ_1qp_superpos(
    x: str, y: str, which: Literal["+", "-", "+i", "-i"]
) -> QuantumCircuit:
    assert hamming_weight(x) == 1
    assert hamming_weight(y) == 1
    assert len(x) == len(y)
    c = QuantumCircuit(len(x))
    x_ind = index_bits(x)[0]
    y_ind = index_bits(y)[0]
    if which == "+":
        c.h(x_ind)
        c.cx(x_ind, y_ind)
        c.x(x_ind)
    elif which == "-":
        c.x(x_ind)
        c.h(x_ind)
        c.cx(x_ind, y_ind)
        c.x(x_ind)
    elif which == "+i":
        c.h(x_ind)
        c.s(x_ind)
        c.cx(x_ind, y_ind)
        c.x(x_ind)
    elif which == "-i":
        c.x(x_ind)
        c.h(x_ind)
        c.s(x_ind)
        c.cx(x_ind, y_ind)
        c.x(x_ind)
    else:
        raise ValueError(f"Incorrect which: {which}")
    return c


def circ_ham_expectation_values(
    hamiltonian: SparsePauliOp, circuit: QuantumCircuit, states: list[str]
) -> np.ndarray:
    # this returns the hamiltonian matrix for the 1 to n+1 first eigenstates
    # this is for the 1qp thing mostly
    out = np.zeros((len(states), len(states)), dtype=complex)
    inds = []
    for state in states:
        inds.append(from_bitstring(state, hamiltonian.num_qubits, right_to_left=True))

    umat = circuit_to_matrix(circuit)
    mat = umat.T.conj() @ hamiltonian.to_matrix() @ umat

    return mat[np.ix_(inds, inds)]


def swap_test_circuit(
    circuit: QuantumCircuit, init_circuit: QuantumCircuit = None
) -> QuantumCircuit:
    n_qubits = circuit.num_qubits
    swap_circuit = QuantumCircuit(circuit.num_qubits * 2 + 1)
    swap_circuit.h(0)
    if init_circuit is not None:
        swap_circuit = swap_circuit.compose(init_circuit, range(1, n_qubits + 1))
        swap_circuit = swap_circuit.compose(
            init_circuit, range(n_qubits + 1, swap_circuit.num_qubits)
        )
    swap_circuit = swap_circuit.compose(circuit, range(1, n_qubits + 1))
    for x in range(1, n_qubits + 1):
        swap_circuit.cswap(0, x, x + n_qubits)

    swap_circuit.h(0)

    return swap_circuit
