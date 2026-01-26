from Nlce_hardware_run.src.qcomp_nlce.workflow.workflow_tfim_1qp_dispersion import (
    workflow_nq_aqt,
    workflow_nq_qexa,
)

from time import sleep


def count_to_awakening(secs: int):
    print("I sleep but I will awaken")
    for s in range(secs):
        print(f"in {secs-s:0>6} seconds", end="\r")
        sleep(1)
    print("I have awoken")


def keep_trying(func: callable):
    success = False
    while not success:
        try:
            func(True)
            success = True
        except Exception as e:
            print(f"Encoutered exception: {e}")
            count_to_awakening(300)

    func(False)
    print("🚀🪐🌕🌑☄️🛸")


if __name__ == "__main__":
    keep_trying(workflow_nq_aqt)
