

import re
import sys
import numpy as np
from pathlib import Path

from qiskit import QuantumCircuit
from qcomp_nlce.utils import (
    get_ground_energy,
    TFIM,
    save_optimized_vqe,
    get_optimal_n_layers,
)

JULIA_DATA_DIR = Path(__file__).parent.parent / "julia_data" / "TFIM+LF"  # simulation/julia_data/TFIM+LF


def parse_filename(filename: str) -> dict:
    """Parse a Julia data filename into its components.

    Examples:
        gs_n4_j0p5_h0p1.csv   -> type=gs, n_qubits=4, J=0.5, hl=0.1
        1qp_n3_j0p5_h_0p1.csv -> type=1qp, n_qubits=3, J=0.5, hl=0.1
    """
    stem = Path(filename).stem  # strip .csv

    # Match: (gs|1qp)_n{digits}_j{value}_h[_]{value}
    # where value uses 'p' for decimal point, e.g. 0p5 -> 0.5
    match = re.match(r"^(gs|1qp)_n(\d+)_j([0-9p]+)_h_?([0-9p]+)$", stem)
    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")

    state_type = match.group(1)
    n_qubits = int(match.group(2))
    J = float(match.group(3).replace("p", "."))
    hl = float(match.group(4).replace("p", "."))

    return {"type": state_type, "n_qubits": n_qubits, "J": J, "hl": hl}


def load_params(filepath: Path) -> np.ndarray:
    """Load parameters from a single-row CSV file."""
    return np.loadtxt(filepath, delimiter=",")


def build_hva_circuit_from_julia_params(
    n_qubits: int, n_layers: int, params: np.ndarray, has_longitudinal_field: bool = False
) -> QuantumCircuit:
    """Build HVA circuit from Julia-optimized parameters.

    Julia uses exp(-i * theta * P) convention, while Qiskit's RXX/RX/RZ
    use exp(-i * theta/2 * P). Therefore we multiply Julia params by 2.

    Parameter structure per layer (matching Julia's Gates_tl order):
    - XX rotations: params[0:n-1] for bonds (0,1), (1,2), ..., (n-2,n-1)
    - X rotations:  params[n-1:2n-1] for qubits 0..n-1  (only if has_longitudinal_field)
    - Z rotations:  params[...] for qubits 0..n-1
    """
    circuit = QuantumCircuit(n_qubits)
    n_xx = n_qubits - 1
    n_x = n_qubits if has_longitudinal_field else 0
    n_z = n_qubits
    params_per_layer = n_xx + n_x + n_z

    for layer in range(n_layers):
        base = layer * params_per_layer
        offset = 0
        # XX rotations (scaled by 2 for Qiskit convention)
        for bond in range(n_xx):
            circuit.rxx(2 * params[base + offset + bond], bond, bond + 1)
        offset += n_xx
        # X longitudinal field rotations (scaled by 2)
        if has_longitudinal_field:
            for qubit in range(n_qubits):
                circuit.rx(2 * params[base + offset + qubit], qubit)
            offset += n_x
        # Z transverse field rotations (scaled by 2)
        for qubit in range(n_qubits):
            circuit.rz(2 * params[base + offset + qubit], qubit)

    return circuit


def generate_gs(n_qubits: int, J: float, hl: float, params: np.ndarray):
    """Generate and save ground state VQE data (mirrors vqe_gs_simulations.py)."""
    h = 1
    hamiltonian = TFIM(n_qubits, h, J, periodic=False, longitudinal_field=hl)
    n_layers = get_optimal_n_layers(hamiltonian.num_qubits)
    has_lf = hl != 0.0

    circuit = build_hva_circuit_from_julia_params(n_qubits, n_layers, params, has_longitudinal_field=has_lf)
    ground_energy = get_ground_energy(hamiltonian)

    n_xx = n_qubits - 1
    n_x = n_qubits if has_lf else 0
    n_z = n_qubits
    expected_params = n_layers * (n_xx + n_x + n_z)
    assert len(params) == expected_params, (
        f"Parameter count mismatch: file has {len(params)}, "
        f"circuit expects {expected_params}"
    )

    weights = [params]

    dirname = f"gs_vqe_{int(J*100)}j_{int(hl*100)}hl"
    save_path = f"{dirname}/vqe/{n_qubits}q_{n_layers}l"

    print(f"  Saving to res_sim/{save_path}/")
    save_optimized_vqe(save_path, hamiltonian, circuit, weights, ground_energy, dry_run=False)


def generate_elem(n_qubits: int, J: float, hl: float, params: np.ndarray):
    """Generate and save elementary (1qp) VQE data (mirrors vqe_1qp_simulations.py)."""
    h = 1
    ham_j, ham_h = TFIM(n_qubits, h, J, periodic=False, split_parts=True, longitudinal_field=hl)
    hamiltonian = ham_j + ham_h
    n_layers = get_optimal_n_layers(hamiltonian.num_qubits)
    has_lf = hl != 0.0

    circuit = build_hva_circuit_from_julia_params(n_qubits, n_layers, params, has_longitudinal_field=has_lf)
    ground_energy = get_ground_energy(hamiltonian)

    n_xx = n_qubits - 1
    n_x = n_qubits if has_lf else 0
    n_z = n_qubits
    expected_params = n_layers * (n_xx + n_x + n_z)
    assert len(params) == expected_params, (
        f"Parameter count mismatch: file has {len(params)}, "
        f"circuit expects {expected_params}"
    )

    weights = [params]

    dirname = f"elem_vqe_{int(J*100)}j_{int(hl*100)}hl"
    save_path = f"{dirname}/vqe/cost_offdiag/{n_qubits}q_{n_layers}l"

    print(f"  Saving to res_sim/{save_path}/")
    save_optimized_vqe(save_path, hamiltonian, circuit, weights, ground_energy, dry_run=False)


def main(data_dir: Path = None):
    if data_dir is None:
        data_dir = JULIA_DATA_DIR

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} files in {data_dir}\n")

    for i, filepath in enumerate(csv_files):
        cfg = parse_filename(filepath.name)
        params = load_params(filepath)

        print(f"[{i+1}/{len(csv_files)}] {filepath.name} -> "
              f"type={cfg['type']}, n_qubits={cfg['n_qubits']}, "
              f"J={cfg['J']}, hl={cfg['hl']}, n_params={len(params)}")

        if cfg["type"] == "gs":
            generate_gs(cfg["n_qubits"], cfg["J"], cfg["hl"], params)
        elif cfg["type"] == "1qp":
            generate_elem(cfg["n_qubits"], cfg["J"], cfg["hl"], params)
        else:
            print(f"  WARNING: unknown type '{cfg['type']}', skipping")

    print("\nDone.")


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(data_dir)
