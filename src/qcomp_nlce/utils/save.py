from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import SparsePauliOp

import os
from pathlib import Path
import numpy as np
import json
from data_manager import ExtendedJSONDecoder, ExtendedJSONEncoder
import io


from qcomp_nlce.utils.constants import ALGORITHMS_ALIAS


def get_save_directory(dirname: str):
    home = Path(__file__).parent.parent
    save_dir = Path(home, f"res_sim/{dirname}/")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def save_sweep(
    dirname: str,
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    ground_energy: float,
    dry_run: bool,
):
    if not dry_run:
        save_dir = get_save_directory(dirname)
        with open(Path(save_dir, "circuit.qpy"), "wb+") as fd:
            qpy.dump(circuit, fd)
        with open(Path(save_dir, "ground_energy.npy"), "wb+") as fd:
            np.save(fd, np.array(ground_energy))
        with open(Path(save_dir, "hamiltonian.npy"), "wb+") as fd:
            np.save(fd, hamiltonian.to_list(True))


def load_sweep(dirname: str):
    with open(Path(dirname, "circuit.qpy"), "rb") as fd:
        circuit = qpy.load(fd)[0]
    with open(Path(dirname, "ground_energy.npy"), "rb") as fd:
        ground_energy = np.load(fd)
    with open(Path(dirname, "hamiltonian.npy"), "rb") as fd:
        hamiltonian = SparsePauliOp.from_list(np.load(fd))

    return hamiltonian, circuit, ground_energy


def save_optimized_vqe(
    dirname: str,
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    weights: np.ndarray,
    ground_energy: float,
    dry_run: bool,
):
    if not dry_run:
        save_dir = get_save_directory(dirname)
        with open(Path(save_dir, "circuit.qpy"), "wb+") as fd:
            qpy.dump(circuit, fd)
        with open(Path(save_dir, "weights.npy"), "wb+") as fd:
            np.save(fd, weights)
        with open(Path(save_dir, "ground_energy.npy"), "wb+") as fd:
            np.save(fd, np.array(ground_energy))
        with open(Path(save_dir, "hamiltonian.npy"), "wb+") as fd:
            np.save(fd, hamiltonian.to_list(True))


def load_optimized_vqe(dirname: str):
    with open(Path(dirname, "circuit.qpy"), "rb") as fd:
        circuit = qpy.load(fd)[0]
    with open(Path(dirname, "weights.npy"), "rb") as fd:
        weights = np.load(fd)
    with open(Path(dirname, "ground_energy.npy"), "rb") as fd:
        ground_energy = np.load(fd)
    with open(Path(dirname, "hamiltonian.npy"), "rb") as fd:
        hamiltonian = SparsePauliOp.from_list(np.load(fd))

    return hamiltonian, circuit, weights, ground_energy


TempListType = list[tuple[tuple[int, int], tuple[complex, complex]]]
# l[ii] = ((ind_x,ind_y),(val,std))


def save_tmp_item(d, filename: str):
    fpath = Path(Path(__file__).parent, f"../tmp/{filename}")

    if not fpath.parent.exists():
        os.makedirs(fpath.parent)

    with io.open(fpath, "w+", encoding="utf8") as fd:
        fd.write(json.dumps(d, cls=ExtendedJSONEncoder, ensure_ascii=False, indent=4))


def load_tmp_item(filename: str) -> TempListType:
    fpath = Path(Path(__file__).parent, f"../tmp/{filename}")

    with io.open(fpath, "r", encoding="utf8") as fd:
        jobj = json.loads(fd.read(), cls=ExtendedJSONDecoder)
    return jobj


def add_tmp_entry_and_save(
    filename: str, l: TempListType, val: complex, std: complex, ind_x: int, ind_y: int
) -> TempListType:
    l.append(((ind_x, ind_y), (val, std)))
    save_tmp_item(l, filename)
    return l


def tmp_entry_exists(l: TempListType, ind_x: int, ind_y: int):
    for k in l:
        if ind_x == k[0][0] and ind_y == k[0][1]:
            return True
    return False


def rebuild_matrices_from_tmp(l: TempListType, n: int) -> tuple[np.array, np.array]:
    vals = np.zeros((n, n), dtype=complex)
    stds = np.zeros((n, n), dtype=complex)
    for k in l:
        ind_x = k[0][0]
        ind_y = k[0][1]
        vals[ind_x, ind_y] = k[1][0]
        stds[ind_x, ind_y] = k[1][1]
    return vals, stds


def tmp_filename(circuit: QuantumCircuit, which_backend: str):
    if "tmpname" not in circuit.metadata.keys():
        raise ValueError(f"You did not give a tmpname to the circuit: {circuit.name}")

    return f"{circuit.metadata['tmpname']}_{which_backend}"


def load_result_matrices(fpath: Path = None):

    if fpath is None:
        fpath = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_10\workflow_tfim_hadam_stefan_AQT_2000shots_00004\run_00000\data\green_00000.json"
        n_shots = 2000
        fac = np.sqrt(n_shots)
    with open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read(), cls=ExtendedJSONDecoder)

    be = "fake_backend"

    data_l = jobj["results"]["vqe"]["dict_l"]
    data_s = jobj["results"]["vqe"]["dict_s"]

    expect_l = data_l["exd"][be]["values"]
    expect_std_l = data_l["exd"][be]["stds"] / fac
    overlap_l = data_l["eld"][be]["values"]
    overlap_std_l = data_l["eld"][be]["stds"] / fac
    gs_l = data_l["gsd"][be]["energy"]
    gs_std_l = data_l["gsd"][be]["std"] / fac

    expect_s = data_s["exd"][be]["values"]
    expect_std_s = data_s["exd"][be]["stds"] / fac
    overlap_s = data_s["eld"][be]["values"]
    overlap_std_s = data_s["eld"][be]["stds"] / fac
    gs_s = data_s["gsd"][be]["energy"]
    gs_std_s = data_s["gsd"][be]["std"] / fac

    return (
        expect_l,
        expect_std_l,
        overlap_l,
        overlap_std_l,
        gs_l,
        gs_std_l,
        expect_s,
        expect_std_s,
        overlap_s,
        overlap_std_s,
        gs_s,
        gs_std_s,
    )


def baptize_circuit(
    prefix: str,
    method: ALGORITHMS_ALIAS,
    n_qubits: int,
    ratio: float,
    provider_name: str,
    n_shots: int,
    which_hadamard_test: str,
) -> str:

    return f"{prefix}_{method}_{n_qubits}_{int(ratio*100)}j_{provider_name}_{n_shots}shots_{which_hadamard_test}_test"
