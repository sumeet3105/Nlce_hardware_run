import copy
from Nlce_hardware_run.src.qcomp_nlce.utils.workflow_tfim_1d import (
    heff_backend_type,
    dispersion_all_backends,
)

from qcomp_nlce.utils.stats import (
    mean_std,
    mc_disp_from_res_dict,
)
from qcomp_nlce.utils.dispersion import dispersion_1d_thermodynamic
from qcomp_nlce.utils.hardware import BACKEND_TYPES

import matplotlib.pyplot as plt
import numpy as np


def average_result_matrices(dicts: list[dict]) -> dict:

    n_dicts = len(dicts)

    out_d = copy.deepcopy(dicts[0])
    ks = out_d["inputs"]["ks"]
    out_d["inputs"]["n_shots"] = 200 * n_dicts

    for si in ("dict_l", "dict_s"):
        for backend in out_d["results"]["vqe"][si]["eld"]:
            if backend != "system":
                for mnd in ("eld", "exd"):
                    out_d["results"]["vqe"][si][mnd][backend]["values"] = np.mean(
                        list(
                            d["results"]["vqe"][si][mnd][backend]["values"]
                            for d in dicts
                        ),
                        axis=0,
                    )
                    out_d["results"]["vqe"][si][mnd][backend]["stds"] = mean_std(
                        list(
                            d["results"]["vqe"][si][mnd][backend]["stds"] for d in dicts
                        ),
                        axis=0,
                    )

                out_d["results"]["vqe"][si]["gsd"][backend]["energy"] = np.mean(
                    np.array(
                        list(
                            d["results"]["vqe"][si]["gsd"][backend]["energy"]
                            for d in dicts
                        )
                    )
                )
                out_d["results"]["vqe"][si]["gsd"][backend]["std"] = mean_std(
                    np.array(
                        list(
                            d["results"]["vqe"][si]["gsd"][backend]["std"]
                            for d in dicts
                        )
                    )
                )

    for si in ("dict_l", "dict_s"):
        for backend in out_d["results"]["vqe"][si]["eld"]:
            if backend != "system":
                eld = out_d["results"]["vqe"][si]["eld"]
                gsd = out_d["results"]["vqe"][si]["gsd"]
                exd = out_d["results"]["vqe"][si]["exd"]
                heff = heff_backend_type(eld, gsd, exd, backend)
                out_d["results"]["vqe"][si]["heff"][backend] = heff
        disp = dispersion_all_backends(
            out_d["results"]["vqe"]["dict_l"], out_d["results"]["vqe"]["dict_s"], ks
        )
    out_d["results"]["vqe"]["disp"] = disp
    n_mc_samples = int(1e4)
    mc_disp = mc_disp_from_res_dict(ks, out_d, "vqe", n_mc_samples, to_eotm=True)
    out_d["results"]["vqe"]["mc_disp"] = mc_disp
    return out_d


def get_available_backends(d: dict) -> list[str]:
    backends = []
    for si in ("dict_l", "dict_s"):
        eld = d["results"]["vqe"][si]["eld"]
        gsd = d["results"]["vqe"][si]["gsd"]
        exd = d["results"]["vqe"][si]["exd"]
        backends.append(set(eld.keys()) & set(gsd.keys()) & set(exd.keys()))
    return list(backends[0] & backends[1])


def plot_disps(ds: list[dict], labels: list[str] = None):
    all_disps = {}
    if labels is None:
        labels = list(range(len(ds)))
    for ind, d in enumerate(ds):
        disps = {}
        ks = d["inputs"]["ks"]

        for backend in get_available_backends(d):
            heffs = []
            for si in ("dict_l", "dict_s"):
                eld = d["results"]["vqe"][si]["eld"]
                gsd = d["results"]["vqe"][si]["gsd"]
                exd = d["results"]["vqe"][si]["exd"]
                heff = heff_backend_type(eld, gsd, exd, backend)
                heffs.append(heff)
            disp = [dispersion_1d_thermodynamic(k, heffs[0], heffs[1]) for k in ks]
            disps[backend] = disp
        all_disps[labels[ind]] = disps
    fig, ax = plt.subplots()
    for label in all_disps:
        for k in all_disps[label].keys():
            ax.plot(ks, all_disps[label][k], label=k + f" ({label})")
    ax.legend()
    plt.show()


def pretty_print_diff(a, b):
    pretty_print(np.abs(a - b))


def pretty_print(a):
    print(np.array_str(a, precision=3, suppress_small=True))


def show_diffs(d, which_backend: BACKEND_TYPES):
    for which in ("eld", "exd"):
        print(which)
        vals_be = d["results"]["vqe"]["dict_l"][which][which_backend]["values"]
        vals_true = d["results"]["vqe"]["dict_l"][which]["system"]["values"]

        print(which_backend)
        pretty_print(vals_be)
        print("TRUE")
        pretty_print(vals_true)
        print("DIFF")
        pretty_print_diff(vals_be, vals_true)
