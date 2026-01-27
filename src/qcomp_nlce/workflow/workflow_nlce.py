# workflow_nlce.py
# NLCE calculation for ground state energy in the thermodynamic limit
# E_nlce = E(N=4) - E(N=3) gives the energy contribution per site

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

from qiskit import QuantumCircuit
from qcomp_nlce.utils.vqe import get_optimal_n_layers
from qcomp_nlce.utils.hamiltonian import get_ground_energy, TFIM
from qcomp_nlce.hw.gs.subspace_sampling_hw import measure_ground_state_energy_estimator


# Paths to optimized parameters files
PARAMS_DIR = Path(__file__).parent.parent.parent.parent
OPTIMIZED_PARAMS_N4 = PARAMS_DIR / "n_4_tfim"
OPTIMIZED_PARAMS_N3 = PARAMS_DIR / "n_3_tfim"


def load_optimized_params(n_qubits: int):
    """Load optimized parameters for given cluster size."""
    if n_qubits == 4:
        filepath = OPTIMIZED_PARAMS_N4
    elif n_qubits == 3:
        filepath = OPTIMIZED_PARAMS_N3
    else:
        raise ValueError(f"No optimized parameters for n_qubits={n_qubits}")

    with open(filepath, "r") as f:
        content = f.read()
    all_params = ast.literal_eval(content)
    return all_params


def build_hva_circuit_from_julia_params(n_qubits: int, n_layers: int, params: np.ndarray) -> QuantumCircuit:
    """
    Build HVA circuit from Julia-optimized parameters.

    Julia uses exp(-i * theta * P) convention, while Qiskit's RXX/RZ use exp(-i * theta/2 * P).
    Therefore, we need to multiply Julia params by 2.
    """
    circuit = QuantumCircuit(n_qubits)
    params_per_layer = (n_qubits - 1) + n_qubits  # n_xx + n_z

    for layer in range(n_layers):
        base = layer * params_per_layer
        # XX rotations (scaled by 2 for Qiskit convention)
        for bond in range(n_qubits - 1):
            circuit.rxx(2 * params[base + bond], bond, bond + 1)
        # Z rotations (scaled by 2 for Qiskit convention)
        for qubit in range(n_qubits):
            circuit.rz(2 * params[base + (n_qubits - 1) + qubit], qubit)

    return circuit


def run_cluster(
    n_qubits: int,
    J: float,
    all_optimized_params: list,
    provider_name: str = "AQT",
    n_shots: int = 10000,
):
    """Run a single cluster calculation for given n_qubits and J."""
    h = 1.0
    hamiltonian = TFIM(n_qubits=n_qubits, h=h, J=J)

    n_layers = get_optimal_n_layers(n_qubits)
    j_index = int(round(J * 10))
    params = np.array(all_optimized_params[j_index])

    circuit = build_hva_circuit_from_julia_params(n_qubits, n_layers, params)
    circuit.metadata = circuit.metadata or {}
    circuit.metadata["tmpname"] = f"nlce_{n_qubits}q_J{int(J*100)}_{provider_name}"

    which_backends = ["statevector", "fake_backend"]
    exact_ground_energy = get_ground_energy(hamiltonian)

    results = {"exact": exact_ground_energy}

    for which_backend in which_backends:
        result = measure_ground_state_energy_estimator(
            hamiltonian=hamiltonian,
            target_energy=exact_ground_energy,
            circuit=circuit,
            which_backend=which_backend,
            n_shots=n_shots,
            provider_name=provider_name,
        )
        results[which_backend] = result[which_backend]["energy"]

    return results


def run_nlce(
    J_values: np.ndarray = None,
    provider_name: str = "AQT",
    n_shots: int = 10000,
    n_repetitions: int = 1,
):
    """
    Run NLCE calculation: E_nlce = E(N=4) - E(N=3)
    This gives the ground state energy per site in the thermodynamic limit.
    """
    if J_values is None:
        J_values = np.arange(0.0, 1.0, 0.1)

    # Load optimized params for both cluster sizes
    params_n4 = load_optimized_params(4)
    params_n3 = load_optimized_params(3)
    print(f"Loaded {len(params_n4)} parameter sets for N=4")
    print(f"Loaded {len(params_n3)} parameter sets for N=3")

    # Store results from all repetitions
    all_rep_results = {J: [] for J in J_values}

    for rep in range(n_repetitions):
        print(f"\n{'#'*60}")
        print(f"Repetition {rep + 1}/{n_repetitions}")
        print(f"{'#'*60}")

        for J in J_values:
            print(f"\n{'='*50}")
            print(f"Running for J = {J:.1f}")
            print(f"{'='*50}")

            try:
                # Run N=4 cluster
                print("  Running N=4 cluster...")
                results_n4 = run_cluster(4, J, params_n4, provider_name, n_shots)

                # Run N=3 cluster
                print("  Running N=3 cluster...")
                results_n3 = run_cluster(3, J, params_n3, provider_name, n_shots)

                # Compute NLCE: E(N=4) - E(N=3)
                nlce_result = {
                    "exact": results_n4["exact"] - results_n3["exact"],
                    "statevector": results_n4["statevector"] - results_n3["statevector"],
                    "fake_backend": results_n4["fake_backend"] - results_n3["fake_backend"],
                    "n4": results_n4,
                    "n3": results_n3,
                }

                # Apply 1/(2N) correction for fake backend on each cluster before subtraction
                # Corrected fake_backend: (E4 - 1/8) - (E3 - 1/6)
                fb_n4_corrected = results_n4["fake_backend"] - 1 / (2 * 4)
                fb_n3_corrected = results_n3["fake_backend"] - 1 / (4 * 3)
                nlce_result["fake_backend_corrected"] = fb_n4_corrected - fb_n3_corrected

                all_rep_results[J].append(nlce_result)

            except Exception as e:
                print(f"Skipping J={J:.1f}: {e}")

    # Aggregate results: compute mean and std from repetitions
    nlce_results = {}
    for J in J_values:
        if not all_rep_results[J]:
            continue

        nlce_results[J] = {}

        for key in ["exact", "statevector", "fake_backend", "fake_backend_corrected"]:
            values = [rep[key] for rep in all_rep_results[J]]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
            nlce_results[J][key] = {"mean": mean_val, "std": std_val, "all": values}

    return nlce_results


def plot_nlce_results(nlce_results: dict, n_shots: int, n_repetitions: int, provider_name: str):
    """Plot NLCE results: E_nlce vs J for all backends."""
    fig, ax = plt.subplots(figsize=(10, 6))

    J_plot = sorted(nlce_results.keys())

    # Plot exact NLCE
    exact_energies = [nlce_results[J]["exact"]["mean"] for J in J_plot]
    ax.plot(J_plot, exact_energies, 'k-', marker='s', linewidth=2, markersize=8, label="Exact NLCE")

    # Plot statevector NLCE
    sv_energies = [nlce_results[J]["statevector"]["mean"] for J in J_plot]
    sv_stds = [nlce_results[J]["statevector"]["std"] for J in J_plot]
    ax.errorbar(J_plot, sv_energies, yerr=sv_stds, marker='o', capsize=5, label="Statevector")

    # Plot fake_backend NLCE (uncorrected)
    fb_energies = [nlce_results[J]["fake_backend"]["mean"] for J in J_plot]
    fb_stds = [nlce_results[J]["fake_backend"]["std"] for J in J_plot]
    ax.errorbar(J_plot, fb_energies, yerr=fb_stds, marker='^', capsize=5, label="Fake Backend")

    # Plot fake_backend NLCE (corrected)
    fb_corr_energies = [nlce_results[J]["fake_backend_corrected"]["mean"] for J in J_plot]
    fb_corr_stds = [nlce_results[J]["fake_backend_corrected"]["std"] for J in J_plot]
    ax.errorbar(J_plot, fb_corr_energies, yerr=fb_corr_stds, marker='v', capsize=5, label="Fake Backend (corrected)")

    ax.set_xlabel("J/h")
    ax.set_ylabel("NLCE Energy: E(N=4) - E(N=3)")
    ax.set_title(f"NLCE Ground State Energy ({n_shots} shots, {n_repetitions} reps)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_dir = Path(__file__).parent / "figures"
    save_dir.mkdir(exist_ok=True)
    filename = save_dir / f"nlce_energy_{n_shots}shots_{n_repetitions}reps_{provider_name}.png"
    fig.savefig(filename, dpi=150)
    print(f"\nFigure saved to: {filename}")

    # Print comparison table
    print("\n" + "="*90)
    print("NLCE Results: E(N=4) - E(N=3)")
    print("="*90)
    print(f"{'J':<6} {'Exact':<12} {'Statevector':<12} {'Fake Backend':<14} {'FB Corrected':<14} {'SV Error':<10}")
    print("-"*90)
    for J in J_plot:
        exact = nlce_results[J]["exact"]["mean"]
        sv = nlce_results[J]["statevector"]["mean"]
        fb = nlce_results[J]["fake_backend"]["mean"]
        fb_corr = nlce_results[J]["fake_backend_corrected"]["mean"]
        sv_err = sv - exact
        print(f"{J:<6.1f} {exact:<12.6f} {sv:<12.6f} {fb:<14.6f} {fb_corr:<14.6f} {sv_err:<+10.6f}")
    print("="*90)

    plt.show()

    return fig


if __name__ == "__main__":
    J_values = np.arange(0.0, 1.0, 0.1)

    nlce_results = run_nlce(
        J_values=J_values,
        provider_name="AQT",
        n_shots=2000,
        n_repetitions=10,
    )

    plot_nlce_results(
        nlce_results,
        n_shots=2000,
        n_repetitions=10,
        provider_name="AQT",
    )
