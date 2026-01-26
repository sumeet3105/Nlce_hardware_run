from data_manager import ExperimentDataManager
import numpy as np
from qcomp_nlce.utils.workflow_1d import heff_backend_type
from qcomp_nlce.utils.results import plot_disps, average_result_matrices, show_diffs
import matplotlib.pyplot as plt
import copy
from pathlib import Path


def replace_eld():
    fpaths_aqt = (
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_11\workflow_tfim_hadam_estimator_AQT_200shots",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_12\workflow_tfim_hadam_estimator_AQT_200shots",
    )
    fpaths_qexa = (
        r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_14\vqe_gsonly_workflow_tfim_hadam_stefan_QEXA_100000shots_00001",
    )

    mean_res_aqt = averaged_results_from_paths(fpaths_aqt)
    mean_res_qexa = averaged_results_from_paths(fpaths_qexa)

    out_d2 = copy.deepcopy(mean_res_aqt)

    for si in ("dict_l", "dict_s"):
        mean_res_aqt["results"]["vqe"][si]["eld"] = mean_res_qexa["results"]["vqe"][si][
            "eld"
        ]

    plot_disps([mean_res_aqt, out_d2], ["replaced", "not replaced"])


def averaged_results_from_paths(fpaths: list[Path]):
    if len(fpaths) == 1:
        edm = ExperimentDataManager.load(fpaths[0])
        out = edm.load_last_saved_data_file
        return out
    outs = []
    for fpath in fpaths:
        edm = ExperimentDataManager.load(fpath)
        # pray that your last saved file is the result
        out = edm.load_last_saved_data_file
        outs.append(out)
    return average_result_matrices(outs)


def analyze_fpath():

    fpath_aqt = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_12\workflow_tfim_hadam_estimator_AQT_200shots_00001"
    fpath_qexa = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_12\workflow_tfim_hadam_stefan_QEXA_10000shots"

    edm = ExperimentDataManager.load(fpath_qexa)
    d = edm.load_last_saved_data_file

    show_diffs(d)
    plot_disps(d)


if __name__ == "__main__":
    replace_eld()
