# verify_dispersion.py
# Verification script: loads generated VQE data from res_sim/ and plots
# the 1qp dispersion relation against exact diagonalization.
#
# Uses pairs of clusters (N, N-1) for NLCE thermodynamic limit estimation.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from qcomp_nlce.utils.save import load_optimized_vqe, get_save_directory
from qcomp_nlce.utils.circuit import apply_parameters, circuit_to_matrix
from qcomp_nlce.utils.statevector import hamming_weight_states, estimate_observable_1qp_initial_state
from qcomp_nlce.utils.hamiltonian import from_bitstring, TFIM, get_ground_energy
from qcomp_nlce.utils.effective_hamiltonian import H_effective, H_effective_ed
from qcomp_nlce.utils.dispersion import cluster_dispersion, dispersion_1d_thermodynamic, exact_dispersion_1qp_1d
from qcomp_nlce.utils.vqe import get_optimal_n_layers


def compute_heff_from_vqe(gs_dirname: str, elem_dirname: str):
    """
    Load saved VQE data and compute H_eff in the 1qp subspace.

    Returns H_eff matrix and the Hamiltonian.
    """
    # Load ground state VQE
    ham_gs, gs_circuit, gs_weights, gs_ground_energy = load_optimized_vqe(gs_dirname)
    gs_circuit_bound = apply_parameters(gs_circuit, gs_weights[-1])

    # Load 1qp VQE
    ham_elem, elem_circuit, elem_weights, _ = load_optimized_vqe(elem_dirname)
    elem_circuit_bound = apply_parameters(elem_circuit, elem_weights[-1])

    n_qubits = ham_elem.num_qubits
    hamiltonian = ham_elem

    # Compute ground state energy from the GS circuit
    umat_gs = circuit_to_matrix(gs_circuit_bound)
    hmat = hamiltonian.to_matrix(False)
    gs_energy = np.real((umat_gs.T.conj() @ hmat @ umat_gs)[0, 0])

    # Compute 1qp expectation matrix: <i|U†HU|j> in 1qp subspace
    one_qp_expectation = estimate_observable_1qp_initial_state(
        elem_circuit, elem_weights[-1], hamiltonian
    )

    # Compute overlap matrix: <i|U|j> in 1qp subspace
    n_qubits = elem_circuit.num_qubits
    umat_elem = circuit_to_matrix(elem_circuit_bound)
    comp_states = hamming_weight_states(n_qubits, 1)
    inds = [from_bitstring(b, n_qubits, True) for b in comp_states]
    overlap = umat_elem[np.ix_(inds, inds)]

    # Build H_eff
    heff = H_effective(gs_energy, one_qp_expectation, overlap)

    return heff, hamiltonian


def compute_gs_energy_vqe(gs_dirname: str):
    """Load GS VQE data and compute the ground state energy from the circuit."""
    ham, gs_circuit, gs_weights, _ = load_optimized_vqe(gs_dirname)
    gs_circuit_bound = apply_parameters(gs_circuit, gs_weights[-1])
    umat = circuit_to_matrix(gs_circuit_bound)
    hmat = ham.to_matrix(False)
    return np.real((umat.T.conj() @ hmat @ umat)[0, 0])


def verify_dispersion(
    J: float = 0.5,
    hl: float = 0.1,
    cluster_pairs: list[tuple[int, int]] = None,
):
    """
    Plot dispersion from VQE data vs ED for verification,
    and print/plot ground state energy comparison (VQE vs ED).

    Args:
        J: coupling strength
        hl: longitudinal field
        cluster_pairs: list of (N_big, N_small) pairs for NLCE
    """
    if cluster_pairs is None:
        cluster_pairs = [(6, 5), (5, 4), (4, 3)]

    # Collect all unique cluster sizes
    all_n = sorted(set(n for pair in cluster_pairs for n in pair))

    ks = np.linspace(0, np.pi, 100)

    fig, ax_disp = plt.subplots(figsize=(8, 6))

    # ===== Ground state energy comparison (table only) =====
    print(f"\n{'='*60}")
    print(f"Ground State Energy: VQE vs ED (J={J}, hl={hl})")
    print(f"{'='*60}")
    print(f"{'N':<6} {'E_VQE':<16} {'E_ED':<16} {'Error':<16} {'Rel Error':<12}")
    print(f"{'-'*60}")

    for n in all_n:
        n_layers = get_optimal_n_layers(n)
        gs_dir = get_save_directory(
            f"gs_vqe_{int(J*100)}j_{int(hl*100)}hl/vqe/{n}q_{n_layers}l"
        )

        # ED ground energy
        ham = TFIM(n, h=1, J=J, periodic=False, longitudinal_field=hl)
        e_ed = get_ground_energy(ham)

        # VQE ground energy
        try:
            e_vqe = compute_gs_energy_vqe(str(gs_dir))
            error = e_vqe - e_ed
            rel_error = abs(error / e_ed)
            print(f"{n:<6} {e_vqe:<16.10f} {e_ed:<16.10f} {error:<+16.10f} {rel_error:<12.2e}")
        except Exception as e:
            print(f"{n:<6} {'FAILED':<16} {e_ed:<16.10f} {'---':<16} {'---':<12}  ({e})")

    print(f"{'='*60}\n")

    # ===== Dispersion plot =====
    if np.isclose(hl, 0.0):
        ax_disp.plot(ks, exact_dispersion_1qp_1d(J, ks), 'k-', linewidth=2, label="Exact analytic (hl=0)")

    colors = plt.cm.tab10.colors
    for idx, (n_big, n_small) in enumerate(cluster_pairs):
        n_layers_big = get_optimal_n_layers(n_big)
        n_layers_small = get_optimal_n_layers(n_small)

        gs_dir_big = get_save_directory(f"gs_vqe_{int(J*100)}j_{int(hl*100)}hl/vqe/{n_big}q_{n_layers_big}l")
        elem_dir_big = get_save_directory(f"elem_vqe_{int(J*100)}j_{int(hl*100)}hl/vqe/cost_offdiag/{n_big}q_{n_layers_big}l")
        gs_dir_small = get_save_directory(f"gs_vqe_{int(J*100)}j_{int(hl*100)}hl/vqe/{n_small}q_{n_layers_small}l")
        elem_dir_small = get_save_directory(f"elem_vqe_{int(J*100)}j_{int(hl*100)}hl/vqe/cost_offdiag/{n_small}q_{n_layers_small}l")

        # VQE-based H_eff
        try:
            heff_big, ham_big = compute_heff_from_vqe(str(gs_dir_big), str(elem_dir_big))
            heff_small, ham_small = compute_heff_from_vqe(str(gs_dir_small), str(elem_dir_small))

            disp_vqe = [dispersion_1d_thermodynamic(k, heff_big, heff_small) for k in ks]
            ax_disp.plot(ks, disp_vqe, '-', color=colors[idx],
                    linewidth=1.5, label=f"VQE ({n_big},{n_small})")
        except Exception as e:
            print(f"Skipping VQE ({n_big},{n_small}): {e}")

        # ED-based H_eff (exact reference)
        try:
            ham_big_full = TFIM(n_big, h=1, J=J, periodic=False, longitudinal_field=hl)
            ham_small_full = TFIM(n_small, h=1, J=J, periodic=False, longitudinal_field=hl)

            heff_ed_big = H_effective_ed(ham_big_full)
            heff_ed_small = H_effective_ed(ham_small_full)

            disp_ed = [dispersion_1d_thermodynamic(k, heff_ed_big, heff_ed_small) for k in ks]
            ax_disp.plot(ks, disp_ed, '--', color=colors[idx],
                    linewidth=1.5, label=f"ED ({n_big},{n_small})")
        except Exception as e:
            print(f"Skipping ED ({n_big},{n_small}): {e}")

    ax_disp.set_xlabel("k")
    ax_disp.set_ylabel("ω(k)")
    ax_disp.set_title(f"1qp Dispersion: NLCE (J={J}, hl={hl})")
    ax_disp.legend()
    ax_disp.grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir = Path(__file__).parent.parent.parent / "res_sim" / "figures"
    save_dir.mkdir(exist_ok=True, parents=True)
    filename = save_dir / f"dispersion_verify_j{int(J*100)}_hl{int(hl*100)}.png"
    fig.savefig(filename, dpi=150)
    print(f"Figure saved to: {filename}")

    plt.show()
    return fig


if __name__ == "__main__":
    verify_dispersion(J=0.5, hl=0.1)
