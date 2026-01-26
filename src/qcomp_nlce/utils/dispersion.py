import numpy as np


def cluster_dispersion(N: int, H_effective: np.ndarray, k: float) -> float:
    mat_el = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):

            # Calculate the phase factor exp(1im*k*(i-j)) and multiply with the matrix element
            # Here k is the momentum, and i, j are the site indices
            func = H_effective[i, j] * np.exp(1j * k * (i - j))
            # Append the result to the "big" array
            mat_el[i, j] = func

    return np.sum(mat_el)


# This is the final function to be evaluated: see Eq. 51 in the paper

# N: number of spins in the largest cluster
# k= momentum, can taken anywhere between 0 to pi ( for 1D, the gap closing occurs at k=0, making it an interesting point to look at)
# H_effective_Nb : H_effective for bigger cluster of two (N spins)
# H_effective_Ns : H_effective for smaller cluster of two (N-1 spins)


def dispersion_1d_thermodynamic(
    k: float, H_eff_Nb: np.ndarray, H_eff_Ns: np.ndarray
) -> float:

    # Compute the difference between the sums of the "big" and "small" arrays
    N = H_eff_Nb.shape[0]
    func_w = np.array(
        cluster_dispersion(N, H_eff_Nb, k) - cluster_dispersion(N - 1, H_eff_Ns, k)
    ).real

    # Return the final computed value
    return func_w


def exact_dispersion_1qp_1d(j: float, k: np.ndarray):
    return 2 * np.sqrt(1 + j**2 - 2 * j * np.cos(k))


def heff_n_expectation_values_1d(n_qubits: int) -> int:

    total = 0

    for q in (n_qubits - 1, n_qubits):

        # 2 per overlap element
        overlap = 2 * q**2
        # 1 per diagonal, 4 per nondiagonal element
        expectations = q + 4 * sum(range(1, q))
        gs = 1
        total += sum([overlap, expectations, gs])
    return total


def heff_n_expectation_values_2d(n_qubits: int) -> int:
    total = 0

    # 1D chains: only two largest
    for m in (n_qubits - 1, n_qubits):
        q = m
        total += 2 * q**2 + q + 4 * sum(range(1, q)) + 1

    # 2D rectangles: m >= n >= 2, boundary only
    for m in range(2, n_qubits + 1):
        for n in range(2, m + 1):
            if m * n <= n_qubits and (m + 2) * (n + 2) > n_qubits:
                q = m * n
                total += 2 * q**2 + q + 4 * sum(range(1, q)) + 1

    return total


# def heff_n_expectation_values_1d(n_qubits: int) -> int:
#     ...
#     return total
