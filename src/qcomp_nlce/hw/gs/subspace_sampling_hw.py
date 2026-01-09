import numpy as np

from qcomp_nlce.utils.hardware import (
    backend_sample,
    get_appropriate_fake_backend,
    get_backend,
    encode_sampler_result,
    backend_estimate,
    mean_std_tuples_from_res,
    is_hardware_run,
)
from qcomp_nlce.utils.statevector import (
    sv_estimate_observables,
    sv_expectation_observables,
)
from qcomp_nlce.utils.measure import (
    get_counts_from_job,
    get_counts_from_res,
    add_meas_layer,
    get_unique_bitstrings,
)
from qcomp_nlce.utils.circuit import (
    apply_parameters,
)
from qcomp_nlce.utils.hamiltonian import (
    get_tfim_ratio,
    get_ground_energy,
    subspace_diagonalize,
)
from qcomp_nlce.utils.save import (
    load_tmp_item,
    save_tmp_item,
    tmp_filename,
    load_optimized_vqe,
    load_sweep,
)

from data_manager import ExperimentDataManager
from qiskit.primitives import StatevectorSampler

from pathlib import Path
import os
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from typing import Literal


def measure_ground_state_energy_estimator(
    hamiltonian: SparsePauliOp,
    target_energy: float,
    circuit: QuantumCircuit,
    which_backend: str,
    n_shots: int = 1e3,
    provider_name: str = "AQT",
):

    def relerr(e):
        return np.abs(e - target_energy) / np.abs(target_energy)

    if which_backend == "fake_backend":
        backend = get_appropriate_fake_backend(circuit.num_qubits, provider_name)
        job = backend_estimate(
            backend=backend,
            observables=hamiltonian,
            parameters=None,
            circuit=circuit,
            n_shots=n_shots,
        )
        energy, std = mean_std_tuples_from_res(job.result())[0]
        bn = backend.name
    elif which_backend == "statevector_shots":
        res0 = sv_estimate_observables(
            circuit=circuit,
            observables=[hamiltonian],
            param_sweep=None,
            n_shots=n_shots,
        )
        energy = res0.data.evs[0]
        std = res0.data.stds[0]
        bn = which_backend

    elif which_backend == "statevector":
        res0 = sv_expectation_observables(
            circuit=circuit,
            observables=[hamiltonian],
            parameters=None,
        )
        energy = res0[0]
        std = 0

        bn = which_backend

    elif which_backend == "real_backend":
        backend = get_backend(circuit.num_qubits, provider_name)

        job = backend_estimate(
            backend=backend,
            observables=hamiltonian,
            parameters=None,
            circuit=circuit,
            n_shots=n_shots,
        )

        energy, std = mean_std_tuples_from_res(job.result())[0]
        bn = backend.name

    print(f"{bn} -- estimated energy: {energy:.5f}, err {relerr(energy):.5f}")

    return {
        which_backend: {
            "name": bn,
            "error": relerr(energy),
            "energy": energy,
            "n_shots": n_shots,
            "std": std,
        }
    }


def loop_measure_ground_state_energy_subspace_sampling_v2(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    which_backends: list[str],
    n_shots: int = 1e3,
    provider_name: str = "AQT",
    solve: bool = False,
) -> dict:

    if solve:
        ground_energy = get_ground_energy(hamiltonian)
    else:
        ground_energy = 1000

    d = {
        "system": {
            "ground_energy": ground_energy,
            "n_qubits": circuit.num_qubits,
            "hamiltonian": hamiltonian,
        }
    }

    for which_backend in which_backends:
        on_hardware = is_hardware_run(which_backend)
        filename = f"{tmp_filename(circuit,which_backend)}/measure_ground_energy.json"
        item = None
        if on_hardware:
            try:
                item = load_tmp_item(filename)
                print(f"Already Computed {circuit.metadata['tmpname']}")
            except FileNotFoundError:
                print("Starting from scratch (ground state)")

        if item is None:
            item = measure_ground_state_energy_subspace_sampling_v2(
                hamiltonian,
                ground_energy,
                circuit,
                which_backend,
                n_shots,
                provider_name,
            )
            if on_hardware:
                save_tmp_item(item, filename)
        d[which_backend] = item[which_backend]
    return d


def measure_ground_state_energy_subspace_sampling_v2(
    hamiltonian: SparsePauliOp,
    ground_energy: float,
    circuit: QuantumCircuit,
    which_backend: str,
    n_shots: int = 1e3,
    provider_name: str = "AQT",
) -> dict:

    def relerr(e):
        return np.abs(e - ground_energy) / np.abs(ground_energy)

    circuit_to_sample = add_meas_layer(circuit)
    # assume that the std is sample dependent ish.
    # TODO: derive a better expression!!
    std = 1 / np.sqrt(n_shots)
    if which_backend == "statevector_shots" or which_backend == "statevector":
        sv_energy = sv_estimate_observables(
            circuit=circuit, observables=[hamiltonian], param_sweep=[], n_shots=n_shots
        ).data.evs[0]
        print(f"Statevector {relerr(sv_energy):.5e}")

        bn = which_backend

        sv = StatevectorSampler()

        job = sv.run([circuit_to_sample], shots=n_shots)

        counts = job.result()[0].join_data().get_counts()

        bn = which_backend

    elif which_backend == "fake_backend" or which_backend == "real_backend":

        if which_backend == "fake_backend":
            backend = get_appropriate_fake_backend(circuit.num_qubits, provider_name)

        elif which_backend == "real_backend":
            backend = get_backend(circuit.num_qubits, provider_name)

        job = backend_sample(
            backend=backend,
            circuit=circuit_to_sample,
            parameters=[],
            n_shots=n_shots,
        )
        counts = get_counts_from_res(job.result(), hamiltonian.num_qubits, 0)

        bn = backend.name

    unique_bitstring = get_unique_bitstrings(counts)
    bitstring_prop = len(unique_bitstring) / (2**circuit.num_qubits)
    comp_energy = subspace_diagonalize(
        unique_bitstrings=unique_bitstring, ham=hamiltonian
    )

    print(
        f"{which_backend} ({bn}) {relerr(comp_energy):.5e} sub. mat.: {len(unique_bitstring)} × {len(unique_bitstring)}: {bitstring_prop*100:.3f}%"
    )

    return {
        which_backend: {
            "name": bn,
            "error": relerr(comp_energy),
            "energy": comp_energy,
            "unique_bitstrings": unique_bitstring,
            "bitstring_prop": bitstring_prop,
            "n_shots": n_shots,
            "std": std,
            "counts": counts,
        }
    }


def measure_ground_state_energy_subspace_sampling(
    hamiltonian: SparsePauliOp,
    ground_energy: float,
    circuit: QuantumCircuit,
    n_shots: int = 1e3,
    run_on_hardware: bool = False,
    backend_name: str = "AQT",
) -> dict:
    fake_backend = get_appropriate_fake_backend(circuit.num_qubits, backend_name)
    ratio = get_tfim_ratio(hamiltonian)

    print(f"n_qubits: {hamiltonian.num_qubits}, j/h {ratio}")

    circuit_to_sample = add_meas_layer(circuit)

    def relerr(e):
        return np.abs(e - ground_energy) / np.abs(ground_energy)

    fake_job = backend_sample(
        backend=fake_backend,
        circuit=circuit_to_sample,
        parameters=[],
        n_shots=n_shots,
    )
    fake_counts = get_counts_from_job(fake_job)
    fake_unique_bitstring = get_unique_bitstrings(fake_counts)
    fake_backend_bitstring_prop = len(fake_unique_bitstring) / (2**circuit.num_qubits)
    fake_backend_energy = subspace_diagonalize(
        unique_bitstrings=fake_unique_bitstring, ham=hamiltonian
    )

    statevector_energy = sv_estimate_observables(
        circuit=circuit, observables=[hamiltonian], param_sweep=[], n_shots=n_shots
    ).data.evs[0]

    if run_on_hardware:
        real_backend = get_backend(circuit.num_qubits, backend_name)
        real_job = backend_sample(
            backend=real_backend,
            circuit=circuit_to_sample,
            parameters=[],
            n_shots=n_shots,
        )
        real_counts = get_counts_from_job(real_job)
        real_unique_bitstring = get_unique_bitstrings(real_counts)
        real_bitstring_prop = len(real_unique_bitstring) / (2**circuit.num_qubits)
        real_backend_energy = subspace_diagonalize(
            unique_bitstrings=real_unique_bitstring, ham=hamiltonian
        )

        print(
            f"RealBackend ({real_backend.name}) {relerr(real_backend_energy):.5e} sub. mat.: {len(real_unique_bitstring)} × {len(real_unique_bitstring)}: {real_bitstring_prop*100:.3f}%"
        )

    print(f"Statevector {relerr(statevector_energy):.5e}")

    print(
        f"FakeBackend ({fake_backend.name}) {relerr(fake_backend_energy):.5e} sub. mat.: {len(fake_unique_bitstring)} × {len(fake_unique_bitstring)}: {fake_backend_bitstring_prop*100:.3f}%"
    )

    var_dict = {
        "fake_backend": {
            "name": fake_backend.name,
            "error": relerr(fake_backend_energy),
            "energy": fake_backend_energy,
            "unique_bitstrings": fake_unique_bitstring,
            "bitstring_prop": fake_backend_bitstring_prop,
            "n_shots": n_shots,
            "job": encode_sampler_result(fake_job),
        },
        "statevector": {
            "error": relerr(statevector_energy),
            "energy": statevector_energy,
        },
        "system": {
            "ground_energy": ground_energy,
            "n_qubits": circuit.num_qubits,
            "hamiltonian": hamiltonian,
            "ratio": ratio,
        },
    }
    if run_on_hardware:
        var_dict["real_backend"] = {
            "name": real_backend.name,
            "error": relerr(real_backend_energy),
            "energy": real_backend_energy,
            "unique_bitstrings": real_unique_bitstring,
            "bitstring_prop": real_bitstring_prop,
            "n_shots": n_shots,
            "job": encode_sampler_result(real_job),
        }

    return var_dict


def save_ground_state_energy_run(
    edm: ExperimentDataManager,
    dirname: str,
    n_shots: int = 1e3,
    run_on_hardware: bool = False,
    which: str = "vqe",
    backend_name: str = "AQT",
):

    hamiltonian, ground_energy, circuit = load_which(dirname, which)

    var_dict = measure_ground_state_energy_subspace_sampling(
        hamiltonian,
        ground_energy,
        circuit,
        n_shots,
        run_on_hardware,
        which,
        backend_name,
    )

    var_dict["system"]["dirname"] = dirname
    var_dict["system"]["which"] = which

    edm.save_dict(var_dict)

    return var_dict


def run_all(
    dirname: str,
    edm: ExperimentDataManager,
    run_on_hardware: bool,
    which: str,
    n_shots: int,
    backend_name: str,
):
    HOME = Path(__file__).parent
    dirname = Path(HOME, f"../../res_sim/{dirname}/{which}")
    experiments = os.listdir(dirname)
    for x in experiments:
        print(f"Loading {x}")
        save_ground_state_energy_run(
            edm, dirname, n_shots, run_on_hardware, which, backend_name
        )
        edm.new_run()


def load_which(dirname: Path, which: Literal["sweep", "vqe"]):
    if which == "vqe":
        hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(dirname)
        parameters = weights[-1]
        circuit = apply_parameters(circuit, parameters)
        
    elif which == "sweep":
        hamiltonian, circuit, ground_energy = load_sweep(dirname)

    return hamiltonian, ground_energy, circuit


if __name__ == "__main__":
    dry_run = False
    which = "sweep"
    edm = ExperimentDataManager(f"nlce_hardware_run_{which}_gs", dry_run=dry_run)
    run_on_hardware = True
    backend_name = "AQT"
    n_shots = 1e3
    # if which == "sweep":
    #     run_all(
    #         "5q_3q_gs_aqt_midway",
    #         edm,
    #         run_on_hardware,
    #         which=which,
    #         backend_name=backend_name,
    #         n_shots=n_shots,
    #     )
    if which == "vqe":
        run_all(
            "5q_3q_gs",
            edm,
            run_on_hardware,
            which=which,
            backend_name=backend_name,
            n_shots=n_shots,
        )
