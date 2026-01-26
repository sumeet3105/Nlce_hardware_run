from qiskit.quantum_info import (
    SparsePauliOp,
    Pauli,
    state_fidelity,
    Statevector,
)
from scipy.sparse.linalg import eigsh
import numpy as np
from typing import Union, Literal


import itertools

HAM_NAME_VAR = "__ham_name"


def binleftpad(integer: int, left_pad: int):
    b = format(integer, "0{lp}b".format(lp=left_pad))
    return b


def to_bitstring(ind: int, n_qubits: int, right_to_left: bool = False) -> str:
    if isinstance(ind, (int, np.integer)):
        if n_qubits is None:
            b = bin(ind)
        else:
            b = binleftpad(integer=ind, left_pad=n_qubits)
    elif isinstance(ind, str):
        b = ind
    else:
        raise ValueError(f"Expected ind to be int or str, got: {type(ind)}")
    if "b" in b:
        b = b.split("b")[1]
    if right_to_left:
        b = "".join(list(reversed(b)))

    return b


def projector_ham(
    n_qubits: int, normalize: bool = True, bitstring: str = None
) -> SparsePauliOp:
    if bitstring is None:
        bitstring = "0" * n_qubits
    x = SparsePauliOp("I") + (-1) ** (int(bitstring[0])) * SparsePauliOp("Z")
    for n in range(1, n_qubits):
        y = SparsePauliOp("I") + (-1) ** (int(bitstring[n])) * SparsePauliOp("Z")
        x = x ^ y

    cst = 1
    if normalize:
        cst = 2 ** (-n_qubits)
    return x * cst


def index_bits(
    a: Union[str, int], n_qubits: int = None, ones=True, right_to_left: bool = False
) -> list:
    """Takes a binary number and returns a list of indices where the bit is one (or zero)
    Args:
        a (binary number): The binary number whose ones or zeroes will be indexed
        ones (bool): If true, index ones. If false, index zeroes
    Returns:
        list: List of indices where a is one (or zero)
    """
    if isinstance(a, int):
        if n_qubits is None:
            b = bin(a)
        else:
            b = binleftpad(integer=a, left_pad=n_qubits)
    else:
        b = a
    if "b" in b:
        b = b.split("b")[1]
    if right_to_left:
        b = list(reversed(b))
    if ones:
        return [idx for idx, v in enumerate(b) if int(v)]
    elif not ones:
        return [idx for idx, v in enumerate(b) if not int(v)]


def from_bitstring(b: str, n_qubits: int, right_to_left: bool = False) -> int:
    if "b" in b:
        b = b.split("b")[1]
    idx = index_bits(a=b, n_qubits=n_qubits, right_to_left=True)
    if right_to_left:
        return sum([2 ** (n_qubits - 1 - iii) for iii in idx])
    else:
        return sum([2**iii for iii in idx])


def get_op_indices(op: SparsePauliOp) -> list:
    all_indices = []
    for term in op:
        p = term.paulis[0]
        all_indices.append(get_pauli_indices(p))
    return all_indices


def get_pauli_indices(term: Pauli) -> list:
    return [ind for ind in range(len(term)) if str(term)[ind] != "I"]


def get_singlet_ham(n_qubits: int, J: float) -> SparsePauliOp:
    hamiltonian = sum(
        [
            -J * SparsePauliOp(s * "I" + "XX" + (n_qubits - 2 - s) * "I")
            for s in range(0, n_qubits - 1, 2)
        ]
    )
    return hamiltonian


def get_tfim_xx(n_qubits: int, which: Literal["even", "odd"]) -> list[SparsePauliOp]:
    if which == "even":
        start = 0
    elif which == "odd":
        start = 1

    out = []
    for ind in range(start, n_qubits - 1, 2):
        idty = ["I" for _ in range(n_qubits)]
        idty[ind] = "X"
        idty[ind + 1] = "X"
        out.append(SparsePauliOp("".join(idty)))
    return out


def get_tfim_ratio(ham: SparsePauliOp) -> float:
    c = ham.coeffs
    u = np.unique_counts(c)
    idx = np.argsort(u.counts)
    j = u.values[idx[0]]
    h = u.values[idx[-1]]
    return np.real(j / h).astype(float)


def field_pauli_sum(n_qubits: int, pauli: str):
    return sum(
        [
            SparsePauliOp(s * "I" + pauli + (n_qubits - 1 - s) * "I")
            for s in range(n_qubits)
        ]
    )


def TFIM(
    n_qubits: int,
    h: float,
    J: float,
    periodic: bool = False,
    split_parts: bool = False,
    longitudinal_field: float = 0.0,
    ladder: bool = False,
) -> SparsePauliOp:

    hamiltonian_J = sum(
        [
            -J * SparsePauliOp(s * "I" + "XX" + (n_qubits - 2 - s) * "I")
            for s in range(n_qubits - 1)
        ]
    )
    if periodic:
        hamiltonian_J += SparsePauliOp("X" + "I" * (n_qubits - 2) + "X")
    hamiltonian_h = sum(
        [
            -h * SparsePauliOp(s * "I" + "Z" + (n_qubits - 1 - s) * "I")
            for s in range(n_qubits)
        ]
    )
    if longitudinal_field != 0.0:
        hamiltonian_J += sum(
            [
                -longitudinal_field
                * SparsePauliOp(s * "I" + "X" + (n_qubits - 1 - s) * "I")
                for s in range(n_qubits)
            ]
        )

    if ladder:
        # ladder means you have two 1d tfim connected with XX
        # so we just create another tfim
        # and add the ladder rungs
        # and then we tensor the two tfim with identities so that they match
        # and pronto
        hamiltonian2_J, hamiltonian2_h = TFIM(
            n_qubits=n_qubits,
            h=h,
            J=J,
            periodic=periodic,
            split_parts=True,
            longitudinal_field=longitudinal_field,
            ladder=False,
        )
        cross_J = 0
        for q in range(n_qubits):
            idty = ["I" for _ in range(2 * n_qubits)]
            idty[q] = "X"
            idty[q + n_qubits] = "X"
            cross_J += -J * SparsePauliOp("".join(idty))
        idty = SparsePauliOp("I" * n_qubits)

        hamiltonian_h = hamiltonian_h ^ idty + idty ^ hamiltonian2_h
        hamiltonian_J = hamiltonian_J ^ idty + idty ^ hamiltonian2_J + cross_J

    if split_parts:
        # J: XX h: Z
        return hamiltonian_J, hamiltonian_h

    hamiltonian = hamiltonian_J + hamiltonian_h

    hamiltonian.__dict__[HAM_NAME_VAR] = "TFIM"

    return hamiltonian


def hamming_weight(bitstring: str) -> int:
    return sum([1 if x == "1" else 0 for x in bitstring])


def hamming_weight_from_index(idx: int, n_qubits: int):
    bs = to_bitstring(idx, n_qubits, True)
    return hamming_weight(bs)


def n_qp_indices(n_qubits: int, n_qp: int, right_to_left: bool) -> list[int]:
    qp_combs = itertools.combinations(range(n_qubits), n_qp)

    out_idx = []
    for qp_comb in qp_combs:
        if right_to_left:
            iii = sum([2 ** (n_qubits - 1 - iii) for iii in qp_comb])
        else:
            iii = sum([2**iii for iii in qp_comb])
        out_idx.append(iii)
    return out_idx


def n_qp_bitstrings(n_qubits: int, n_qp: int, right_to_left: bool) -> list[str]:
    qp_idx = n_qp_indices(n_qubits, n_qp, right_to_left)
    out_str = []
    for idx in qp_idx:
        out_str.append(to_bitstring(idx, n_qubits, right_to_left))
    return out_str


def get_ground_energy(hamiltonian: SparsePauliOp) -> float:
    mat = hamiltonian.to_matrix(sparse=True)
    ground_energy, _ = eigsh(mat, k=1, which="SA")
    return ground_energy[0]


def get_max_energy(hamiltonian: SparsePauliOp) -> float:
    mat = hamiltonian.to_matrix(sparse=True)
    energy, _ = eigsh(mat, k=1, which="LM")
    return energy[0]


def get_ground_energy_n_qp(hamiltonian: SparsePauliOp, n_qp: int) -> float:
    # this might be wrong: you don't get an eigenstate of the whole Hamiltonian!
    mat = hamiltonian.to_matrix(sparse=True)
    n_qubits = hamiltonian.num_qubits
    idx = n_qp_indices(n_qubits, n_qp, True)
    ground_energy, _ = eigsh(mat[np.ix_(idx, idx)], k=1, which="SA")
    return ground_energy[0]


def get_ground_state(hamiltonian: SparsePauliOp) -> np.ndarray:
    mat = hamiltonian.to_matrix(sparse=True)
    _, ground_state = eigsh(mat, k=1, which="SA")
    return ground_state


def ground_state_fidelity(state: Statevector, hamiltonian: SparsePauliOp) -> float:
    return state_fidelity(state, get_ground_state(hamiltonian))


def ground_state_bitstrings(hamiltonian: SparsePauliOp) -> list[str]:
    ground_state = get_ground_state(hamiltonian)
    idx = np.where(1 - np.isclose(np.abs(ground_state.T[0]), 0))[0]
    return [to_bitstring(jj, hamiltonian.num_qubits, True) for jj in idx]


def get_min_gap(hamiltonian: SparsePauliOp) -> float:
    mat = hamiltonian.to_matrix(sparse=True)
    min_gap = 0
    k = 2
    while np.isclose(min_gap, 0):
        energies, _ = eigsh(mat, k=k, which="SA")
        min_gap = np.abs(energies[-1] - energies[0])
        k += 1
    return min_gap


def pretty_pauli_str(p: Pauli) -> str:
    idx = get_pauli_indices(p)
    return "".join([f"{x}{y}" for x, y in zip(p[::-1][idx], idx)])


def subspace_diagonalize(
    unique_bitstrings: Union[list[int], list[str]], ham: SparsePauliOp
):
    if isinstance(unique_bitstrings[0], str):
        ind_list = [
            from_bitstring(bx, len(bx), right_to_left=True) for bx in unique_bitstrings
        ]
    else:
        ind_list = [int(x) for x in unique_bitstrings]

    sparse_ham = ham.to_matrix(sparse=True)
    n_samples = len(unique_bitstrings)
    sub_ham = np.zeros((n_samples, n_samples), dtype=complex)
    for x in range(n_samples):
        for y in range(x, len(unique_bitstrings)):
            ind_x = ind_list[x]
            ind_y = ind_list[y]
            sub_ham[x, y] = sparse_ham[ind_x, ind_y]
    sub_ham += np.triu(sub_ham, k=1).T
    ground_energy = np.min(np.linalg.eigvalsh(sub_ham))
    return ground_energy


def eigenvector_decomposition(
    state: Statevector, hamiltonian: SparsePauliOp
) -> np.ndarray:
    _, eig_states = np.linalg.eigh(hamiltonian.to_matrix(False))
    fidelities = np.zeros((eig_states.shape[-1],))
    for k in range(eig_states.shape[-1]):
        fidelities[k] = state_fidelity(state, Statevector(eig_states[:, k]))
    return fidelities


def get_1qp_mat(ham: SparsePauliOp, sparse: bool):
    # this function returns the hamiltonian elements indexed
    # by the indices whose binary representation has hamming weight 1
    # i,e, <j|H|k> where |j> = |00010000> for example
    mat = ham.to_matrix(sparse=sparse)
    n = int(np.log2(mat.shape[0]))
    sl = [int(2**x) for x in range(n)]
    return mat[np.ix_(sl, sl)]


def hamiltonian_elems(
    hamiltonian: SparsePauliOp, state_l: np.ndarray, state_r: np.ndarray
) -> np.ndarray:
    return state_l.T.conj() @ hamiltonian.to_matrix() @ state_r


def trace_dist(u1: np.ndarray, u2: np.ndarray) -> float:
    diff = u1 - u2
    out = np.sqrt(diff.T.conj() @ diff)
    return 0.5 * np.real(np.trace(out))
