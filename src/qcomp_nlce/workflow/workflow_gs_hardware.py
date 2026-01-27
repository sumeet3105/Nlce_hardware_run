# workflow_gs_hardware.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

from qiskit import QuantumCircuit
from qcomp_nlce.utils.vqe import get_optimal_n_layers
from qcomp_nlce.utils.hamiltonian import get_ground_energy, TFIM
from qcomp_nlce.hw.gs.subspace_sampling_hw import measure_ground_state_energy_estimator


# Path to optimized parameters file
OPTIMIZED_PARAMS_FILE = Path(__file__).parent.parent.parent.parent / "n_3_tfim"


def load_optimized_params():
    """Load all optimized parameters from n_4_tfim file."""
    with open(OPTIMIZED_PARAMS_FILE, "r") as f:
        content = f.read()
    # Parse the nested list
    all_params = ast.literal_eval(content)
    return all_params


def build_hva_circuit_from_julia_params(n_qubits: int, n_layers: int, params: np.ndarray) -> QuantumCircuit:
    """
    Build HVA circuit from Julia-optimized parameters.

    Julia uses exp(-i * theta * P) convention, while Qiskit's RXX/RZ use exp(-i * theta/2 * P).
    Therefore, we need to multiply Julia params by 2.

    Parameter structure per layer (7 params for 4 qubits):
    - XX rotations: params[0:3] for bonds (0,1), (1,2), (2,3)
    - Z rotations: params[3:7] for qubits 0, 1, 2, 3
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


def workflow_gs_hardware(
    n_qubits: int,
    J: float,
    provider_name: str = "AQT",  # or "AQT", "IBM"
    n_shots: int = int(1e4),
    dry_run: bool = True,
    all_optimized_params: list = None,
):
    # Build hamiltonian
    h = 1.0  # Transverse field strength (fixed)
    hamiltonian = TFIM(n_qubits=n_qubits, h=h, J=J)

    # Get optimal number of layers
    n_layers = get_optimal_n_layers(n_qubits)

    # Get the optimized parameters for this J value
    # J_values are 0.0, 0.1, ..., 1.0 -> indices 0, 1, ..., 10
    j_index = int(round(J * 10))

    if all_optimized_params is None:
        all_optimized_params = load_optimized_params()

    params = np.array(all_optimized_params[j_index])

    # Build circuit with Julia params (includes 2x scaling for Qiskit convention)
    circuit = build_hva_circuit_from_julia_params(n_qubits, n_layers, params)

    print(f"Built circuit for J={J:.1f} with {n_layers} layers, {len(params)} params")

    # Add tmpname to circuit metadata (required for caching)
    circuit.metadata = circuit.metadata or {}
    circuit.metadata["tmpname"] = f"gs_{n_qubits}q_J{int(J*100)}_{provider_name}"

    # Which backends to run
    which_backends = [
        "statevector",
        "fake_backend",
        #"real_backend"
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
    n_qubits: int = 3,
    J_values: np.ndarray = None,
    provider_name: str = "AQT",
    n_shots: int = 10000,
    dry_run: bool = True,
    n_repetitions: int = 1,
):
    if J_values is None:
        J_values = np.arange(0.0, 1.0, 0.1)

    # Load optimized params once
    all_optimized_params = load_optimized_params()
    print(f"Loaded {len(all_optimized_params)} sets of optimized parameters from {OPTIMIZED_PARAMS_FILE}")

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
                results = workflow_gs_hardware(
                    n_qubits=n_qubits,
                    J=J,
                    provider_name=provider_name,
                    n_shots=n_shots,
                    dry_run=dry_run,
                    all_optimized_params=all_optimized_params,
                )
                all_rep_results[J].append(results)
                plt.close()  # Close individual plots
            except Exception as e:
                print(f"Skipping J={J:.1f}: {e}")

    # Aggregate results: compute mean and std from repetitions
    all_results = {}
    for J in J_values:
        if not all_rep_results[J]:
            continue

        # Get system info from first repetition
        all_results[J] = {
            "system": all_rep_results[J][0]["system"]
        }

        # Get backends from first repetition
        backends = [b for b in all_rep_results[J][0].keys() if b != "system"]

        for backend in backends:
            energies = [rep[backend]["energy"] for rep in all_rep_results[J]]
            mean_energy = np.mean(energies)
            std_energy = np.std(energies, ddof=1) / np.sqrt(len(energies))  # SEM

            # Subtract 1/(2N) correction for fake backend (N=4)
            if backend == "fake_backend":
                
                mean_energy = mean_energy - 1 / (4 * 3)

            all_results[J][backend] = {
                "energy": mean_energy,
                "std": std_energy,
                "all_energies": energies,
                "n_shots": n_shots,
                "n_repetitions": len(energies),
            }

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

        # Print comparison between statevector and fake_backend
        if "statevector" in backends and "fake_backend" in backends:
            print("\n" + "="*70)
            print("Comparison: Statevector vs Fake Backend")
            print("="*70)
            print(f"{'J':<6} {'Exact':<12} {'Statevector':<12} {'Fake Backend':<12} {'Diff (FB-SV)':<12} {'FB Error':<12}")
            print("-"*70)
            total_diff = 0
            for J in J_plot:
                exact = all_results[J]["system"]["ground_energy"]
                sv_energy = all_results[J]["statevector"]["energy"]
                fb_energy = all_results[J]["fake_backend"]["energy"]
                fb_std = all_results[J]["fake_backend"]["std"]
                diff = fb_energy - sv_energy
                total_diff += abs(diff)
                print(f"{J:<6.1f} {exact:<12.6f} {sv_energy:<12.6f} {fb_energy:<12.6f} {diff:<+12.6f} {fb_std:<12.6f}")
            print("-"*70)
            print(f"Mean absolute difference: {total_diff / len(J_plot):.6f}")
            print("="*70)

        ax.set_xlabel("J/h")
        ax.set_ylabel("Ground State Energy")
        ax.set_title(f"Ground State Energy ({n_qubits} qubits, {n_shots} shots, {n_repetitions} reps)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        save_dir = Path(__file__).parent / "figures"
        save_dir.mkdir(exist_ok=True)
        filename = save_dir / f"gs_energy_{n_qubits}q_{n_shots}shots_{n_repetitions}reps_{provider_name}.png"
        fig.savefig(filename, dpi=150)
        print(f"\nFigure saved to: {filename}")

        plt.show()

    return all_results


if __name__ == "__main__":
    # J_range = 0.0:0.1:0.9
    J_values = np.arange(0.0, 1.0, 0.1)

    all_results = run_and_plot_vs_J(
        n_qubits=3,
        J_values=J_values,
        provider_name="AQT",
        n_shots=2000,  # AQT max is 2000 shots per job
        dry_run=True,
        n_repetitions=10,
    )

