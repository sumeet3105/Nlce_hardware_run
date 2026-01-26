from qcomp_nlce.hw.gs.subspace_sampling_hw import (
    loop_measure_ground_state_energy_subspace_sampling_v2,
)
from qcomp_nlce.hw.quasiparticle.hamiltonian_1qp_expectation import (
    loop_measure_1qp_expectations_v2,
)
from qcomp_nlce.hw.quasiparticle.unitary_1qp_sector import (
    loop_measure_1qp_elements_v2,
)
from qcomp_nlce.utils import (
    TFIM,
    apply_parameters,
    get_min_gap,
    get_tfim_sweep_circuit,
    load_optimized_vqe,
    get_tfim_ratio,
    sweep_time_from_steps,
    BACKEND_TYPES,
    baptize_circuit,
    HAM_NAME_VAR,
)
from qcomp_nlce.utils.dispersion import (
    dispersion_1d_thermodynamic,
    exact_dispersion_1qp_1d,
)
import io
import json
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qcomp_nlce.utils.effective_hamiltonian import H_effective, H_effective_ed


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def dispersion_all_backends(dict_l: dict, dict_s: dict, ks: np.ndarray):
    disp = {}
    for which_backend in dict_l["heff"].keys():
        H_eff_Nb = dict_l["heff"][which_backend]
        H_eff_Ns = dict_s["heff"][which_backend]
        disp[which_backend] = [
            dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns) for k in ks
        ]
    return disp


def vqe_dispersion(
    gs_dirname_l: Path,
    elem_dirname_l: Path,
    gs_dirname_s: Path,
    elem_dirname_s: Path,
    which_backends: list[str],
    provider_name: str,
    n_shots: int,
    ks: np.ndarray,
    group_size: int,
    which_hadamard_test: str,
):
    dict_l = vqe_heff(
        gs_dirname_l,
        elem_dirname_l,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
    )
    dict_s = vqe_heff(
        gs_dirname_s,
        elem_dirname_s,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
    )

    return dict_l, dict_s, dispersion_all_backends(dict_l, dict_s, ks)


def sweep_dispersion(
    n_qubits_max: int,
    J: float,
    n_steps: int,
    which_backends: list[str],
    provider_name: str,
    n_shots: int,
    ks: np.ndarray,
    group_size: int,
    which_hadamard_test: str,
    which_model: str = "TFIM",
):
    dict_l = sweep_heff(
        n_qubits_max,
        J,
        n_steps,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
        which_model=which_model,
    )
    dict_s = sweep_heff(
        n_qubits_max - 1,
        J,
        n_steps,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
        which_model=which_model,
    )

    return dict_l, dict_s, dispersion_all_backends(dict_l, dict_s, ks)


def vqe_heff(
    gs_dirname: Path,
    elem_dirname: Path,
    which_backends: list[str],
    provider_name: str,
    n_shots: int,
    group_size: int,
    which_hadamard_test: str,
) -> dict:

    hamiltonian1, elem_circuit, elem_weights, ground_energy = load_optimized_vqe(
        elem_dirname
    )
    n_qubits = hamiltonian1.num_qubits
    elem_circuit = apply_parameters(elem_circuit, elem_weights[-1])
    ratio = get_tfim_ratio(hamiltonian1)
    name_part = baptize_circuit(
        "dim1d", "vqe", n_qubits, ratio, provider_name, n_shots, which_hadamard_test
    )
    elem_circuit.metadata["tmpname"] = name_part
    hamiltonian2, gs_circuit, gs_weights, ground_energy = load_optimized_vqe(gs_dirname)
    gs_circuit = apply_parameters(gs_circuit, gs_weights[-1])
    gs_circuit.metadata["tmpname"] = name_part

    assert hamiltonian1 == hamiltonian2
    hamiltonian = hamiltonian1

    hamiltonian.__dict__[HAM_NAME_VAR] = "TFIM"

    return get_heff_values(
        gs_circuit,
        elem_circuit,
        hamiltonian,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
    )


def pick_model(
    n_qubits: int,
    J: float,
    which_model: str,
):
    if "ladder" in which_model:
        ladder = True
    else:
        ladder = False

    if "HL" in which_model:
        if len(which_model.split("_")) == 3:
            parts = which_model.split("_")
            hl = int(parts[-1]) / 100
        else:
            hl = 0.1
    else:
        hl = 0.0

    ham_j, ham_h = TFIM(
        n_qubits,
        1,
        J,
        periodic=False,
        split_parts=True,
        ladder=ladder,
        longitudinal_field=hl,
    )
    return ham_j, ham_h


def sweep_heff(
    n_qubits: int,
    J: float,
    n_steps: int,
    which_backends: list[str],
    provider_name: str,
    n_shots: int,
    group_size: int,
    which_hadamard_test: str,
    which_model: str = "TFIM",
):
    ham_j, ham_h = pick_model(n_qubits, J, which_model)
    hamiltonian = ham_j + ham_h
    min_gap = get_min_gap(hamiltonian)
    # the most important thing is to have enough steps!!!
    # the total time can be low-ish

    total_time = sweep_time_from_steps(ham_h, hamiltonian, n_steps)
    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=total_time,
        n_steps=n_steps,
    )
    ratio = get_tfim_ratio(hamiltonian)
    name_part = baptize_circuit(
        "dim1d", "sweep", n_qubits, ratio, provider_name, n_shots, which_hadamard_test
    )
    circuit.metadata["tmpname"] = f"sweep_circuit_{name_part}"
    return get_heff_values(
        circuit,
        circuit,
        hamiltonian,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
    )


def load_and_plot(fpath: Path):
    with io.open(fpath, "r", encoding="utf8") as f:
        d = json.loads(f.read())
    plot_all_backends_dispersions(d)


def heff_backend_type(eld: dict, gsd: dict, exd: dict, which_backend: BACKEND_TYPES):
    overlap = eld[which_backend]["values"]
    one_qp_expectation = exd[which_backend]["values"]
    if which_backend == "system":
        gsk = "ground_energy"
    else:
        gsk = "energy"
    gs_expectation = gsd[which_backend][gsk]

    heff = H_effective(gs_expectation, one_qp_expectation, overlap)

    return heff


def get_heff_values(
    gs_circuit: QuantumCircuit,
    elem_circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    which_backends: list[str],
    provider_name: str = "IBM",
    n_shots: int = 1000,
    group_size: int = None,
    which_hadamard_test: str = "stefan",
    solve: bool = True,
):

    print(f"Using {which_hadamard_test} hadamard test")
    print("-- Computing <j|U|k> --")
    elements_dict = loop_measure_1qp_elements_v2(
        circuit=elem_circuit,
        which_backends=which_backends,
        solve=solve,
        provider_name=provider_name,
        n_shots=n_shots,
        group_size=group_size,
        which_hadamard_test=which_hadamard_test,
    )

    print("-- Computing ground state energy --")
    ground_state_dict = loop_measure_ground_state_energy_subspace_sampling_v2(
        hamiltonian=hamiltonian,
        circuit=gs_circuit,
        which_backends=which_backends,
        n_shots=n_shots,
        provider_name=provider_name,
        solve=solve,
    )

    print("-- Computing <j|U* H U|k> --")
    expectations_dict = loop_measure_1qp_expectations_v2(
        hamiltonian=hamiltonian,
        circuit=elem_circuit,
        which_backends=which_backends,
        provider_name=provider_name,
        n_shots=n_shots,
        solve=solve,
    )

    heff = {}

    for which_backend in which_backends:
        heff[which_backend] = heff_backend_type(
            elements_dict,
            ground_state_dict,
            expectations_dict,
            which_backend,
        )

    return {
        "eld": elements_dict,
        "gsd": ground_state_dict,
        "exd": expectations_dict,
        "heff": heff,
    }


def tfim_ed_dispersion(n_qubits_max: int, J: float, ks: np.ndarray):
    hamiltonian_l = TFIM(n_qubits_max, 1, J)
    hamiltonian_s = TFIM(n_qubits_max - 1, 1, J)
    return ed_dispersion(hamiltonian_l, hamiltonian_s, ks)


def ed_dispersion(
    hamiltonian_l: SparsePauliOp, hamiltonian_s: SparsePauliOp, ks: np.ndarray
) -> list[float]:
    heff_l = H_effective_ed(hamiltonian_l)
    heff_s = H_effective_ed(hamiltonian_s)
    return [dispersion_1d_thermodynamic(k, heff_l, heff_s) for k in ks]


def plot_all_backends_dispersions(_d: dict) -> plt.Figure:
    ks = _d["inputs"]["ks"]
    j = _d["inputs"]["J"]
    fig, ax = plt.subplots()
    res = _d["results"]
    for k in res:
        disp = res[k]["disp"]
        for be in disp:
            ax.plot(ks, disp[be], label=f"{k} - {be}")
    ax.plot(ks, exact_dispersion_1qp_1d(j, ks), label="analytic")
    ax.legend()
    return fig


def plot_multiple_dispersions_1d(_l: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots()
    for ind, _d in enumerate(_l):
        ks = _d["inputs"]["ks"]
        j = _d["inputs"]["J"]
        res = _d["results"]
        for k in res:
            disp = res[k]["disp"]
            for be in disp:
                ax.plot(ks, disp[be], label=f"{k} - {be} ({ind})")
    ax.plot(ks, exact_dispersion_1qp_1d(j, ks), label="analytic")
    ax.legend()
    return fig


def plot_all_backends_dispersions_montecarlo(_d: dict) -> plt.Figure:

    ks = _d["inputs"]["ks"]
    j = _d["inputs"]["J"]

    fig, ax = plt.subplots()
    res = _d["results"]

    for circ in res:
        if circ == "ed":
            disp = res[circ]["disp"]
            for be in disp:
                ax.plot(ks, disp[be], label=f"{circ} - {be}")
        else:
            disp = res[circ]["disp"]
            for be in res[circ]["mc_disp"]:
                ax.errorbar(
                    ks,
                    res[circ]["mc_disp"][be]["mean"],
                    res[circ]["mc_disp"][be]["std"],
                    label=f"{circ} montecarlo - {be}",
                )
                ax.plot(
                    ks,
                    disp[be],
                    "x-",
                    label=f"{circ} - {be}",
                )
    if "which_model" not in _d["inputs"].keys():
        which_model = "TFIM"
    else:
        which_model = _d["inputs"]["which_model"]
    if which_model == "TFIM":
        ax.plot(ks, exact_dispersion_1qp_1d(j, ks), label="analytic")
    ax.legend()
    return fig
