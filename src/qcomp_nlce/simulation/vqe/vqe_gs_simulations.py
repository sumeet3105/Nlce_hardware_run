import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    get_ground_energy,
    get_hva_vqe,
    TFIM,
    optimize_circuit,
    save_optimized_vqe,
    get_optimal_n_layers,
)


def train_and_save(n_qubits: int, J: float, dirname: str, dry_run: bool):
    periodic = False
    h = 1
    hamiltonian = TFIM(n_qubits, h, J, periodic=periodic, longitudinal_field=0.1)
    n_layers = 2*get_optimal_n_layers(hamiltonian.num_qubits)

    circuit = get_hva_vqe(hamiltonian, n_layers=n_layers, unique_weights=True)
    # circuit = get_custom_vqe(4, n_layers=2)
    ground_energy = get_ground_energy(hamiltonian)

    print(f"The circuit depth is {circuit.depth()}")
    print(f"The number of parameters is {len(circuit.parameters)}")
    energy_values, params = optimize_circuit(
        circuit,
        hamiltonian,
        initial_params=0.1 * np.ones((len(circuit.parameters),)),
    )
    print(
        f"final energy error is {np.abs(ground_energy-energy_values[-1])/np.abs(ground_energy)}"
    )

    save_optimized_vqe(
        f"{dirname}/vqe/{n_qubits}q_{n_layers}l",
        hamiltonian,
        circuit,
        params,
        ground_energy,
        dry_run,
    )


dry_run = False
J_range = ( 0.8,)
hl=0.1
for n_qubits in (5,):
    for J in J_range:
        dirname = f"gs_vqe_{int(J*100)}j_{int(hl*100)}hl"
        train_and_save(n_qubits, J, dirname, dry_run)
