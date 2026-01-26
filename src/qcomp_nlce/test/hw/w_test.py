from qcomp_nlce.utils import get_backend, backend_sample, get_counts_from_res
from qiskit import QuantumCircuit
from data_manager import ExperimentDataManager
from qiskit.visualization import plot_distribution
import os, json
from pathlib import Path
import matplotlib.pyplot as plt


def w_circ(n_qubits: int):
    c = QuantumCircuit(n_qubits)
    c.h(0)
    for n in range(n_qubits - 1):
        c.cx(n, n + 1)
    c.measure_all()
    return c


def hardware_run():
    n_qubits = 8
    circ = w_circ(n_qubits)
    edm = ExperimentDataManager("w_test_all_backends", project="nlce", dry_run=False)

    for provider_name, n_shots in (("IQM", int(2e4)),):
        backend = get_backend(n_qubits, provider_name)
        job = backend_sample(backend, circ, None, n_shots)
        counts = get_counts_from_res(job.result(), n_qubits, 0)
        edm.save_dict({"counts": counts})
        edm.var_dump(provider_name=provider_name, n_shots=n_shots)

        fig = plot_distribution(
            counts,
            title=f"{provider_name}, {n_qubits} qubits, {n_shots} shots",
            number_to_keep=5,
        )

        edm.save_figure(fig)


def load_and_plot(fpath: Path = None, title: str = None):
    edm = ExperimentDataManager("w_test_path", project="nlce", dry_run=False)
    if fpath is None:
        title = "QExa20, 2000 shots"
        fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_12\w_test_all_backends_00004\run_00000\data\grapefruit_00000.json"
    with open(fpath, "r", encoding="utf8") as f:
        counts = json.loads(f.read())
    if "__timestamp" in counts.keys():
        del counts["__timestamp"]
    if "counts" in counts.keys():
        counts = counts["counts"]
    fig = plot_distribution(counts, title=title, number_to_keep=2)
    edm.save_figure(fig, fig_shape="double-size")

    plt.show()


if __name__ == "__main__":
    hardware_run()
