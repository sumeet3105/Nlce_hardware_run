# workflow_gs_hardware.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from qcomp_nlce.utils.folders import res_sim_dir
from qcomp_nlce.utils.vqe import get_optimal_n_layers
from qcomp_nlce.utils.hamiltonian import get_ground_energy
from qcomp_nlce.hw.gs.subspace_sampling_hw import (
    measure_ground_state_energy_estimator,
    load_which,
)
from data_manager import ExperimentDataManager


def workflow_gs_hardware(
    n_qubits: int,
    J: float,
    provider_name: str = "AQT",  # or "AQT", "IBM"
    n_shots: int = int(1e4),
    dry_run: bool = True,
):
    # Load the pre-trained VQE circuit
    vqe_dir = f"{n_qubits}q_{get_optimal_n_layers(n_qubits)}l"
    gs_dirname = Path(res_sim_dir(), f"gs_vqe_{int(J*100)}j/vqe/{vqe_dir}")

    print(f"Loading circuit from: {gs_dirname}")
    hamiltonian, ground_energy, circuit = load_which(gs_dirname, "vqe")

    # Add tmpname to circuit metadata (required for caching)
    circuit.metadata = circuit.metadata or {}
    circuit.metadata["tmpname"] = f"gs_{n_qubits}q_J{int(J*100)}_{provider_name}"

    # Which backends to run
    which_backends = [
        "statevector",
        "fake_backend",
        #"real_backend",
    ]

    # Calculate exact ground energy for comparison
    exact_ground_energy = get_ground_energy(hamiltonian)

    # Measure ground state energy using VQE estimator (not subspace sampling)
    results = {
        "system": {
            "ground_energy": exact_ground_energy,
            "n_qubits": circuit.num_qubits,
            "hamiltonian": hamiltonian,
        }
    }

    for which_backend in which_backends:
        print(f"\nRunning on {which_backend}...")
        result = measure_ground_state_energy_estimator(
            hamiltonian=hamiltonian,
            target_energy=exact_ground_energy,
            circuit=circuit,
            which_backend=which_backend,
            n_shots=n_shots,
            provider_name=provider_name,
        )
        results.update(result)

    return results


def run_and_plot_vs_J(
    n_qubits: int = 5,
    J_values: np.ndarray = None,
    provider_name: str = "AQT",
    n_shots: int = 10000,
    dry_run: bool = True,
):
    if J_values is None:
        J_values = [0.8, 0.9, 1.0]

    all_results = {}

    for J in J_values:
        print(f"\n{'='*50}")
        print(f"Running for J = {J:.1f}")
        print(f"{'='*50}")
        try:
            results = workflow_gs_hardware(
                n_qubits=n_qubits,
                J=J,
                provider_name=provider_name,
                n_shots=n_shots,
                dry_run=dry_run,
            )
            all_results[J] = results
            plt.close()  # Close individual plots
        except FileNotFoundError as e:
            print(f"Skipping J={J:.1f}: {e}")

    # Plot ground state energy vs J
    if all_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        J_plot = sorted(all_results.keys())

        # Get all backends that were run
        backends = [b for b in all_results[J_plot[0]].keys() if b != "system"]

        for backend in backends:
            energies = [all_results[J][backend]["energy"] for J in J_plot]
            stds = [all_results[J][backend].get("std", 0) for J in J_plot]
            ax.errorbar(J_plot, energies, yerr=stds, marker='o', label=backend, capsize=5)

        # Plot exact ground state energy
        exact_energies = [all_results[J]["system"]["ground_energy"] for J in J_plot]
        ax.plot(J_plot, exact_energies, 'r--', marker='s', label="Exact GS")

        ax.set_xlabel("J (coupling strength)")
        ax.set_ylabel("Ground State Energy")
        ax.set_title(f"Ground State Energy vs J ({n_qubits} qubits, {n_shots} shots)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return all_results


if __name__ == "__main__":
    all_results = run_and_plot_vs_J(
        n_qubits=5,
        J_values=[0.8, 0.9, 1.0],
        provider_name="AQT",
        n_shots=2000,  # AQT max is 2000 shots per job
        dry_run=True,
    )
