from qcomp_nlce.utils import (
    hamming_weight_states,
    measure_hamiltonian_elements,
    res_sim_dir,
    get_optimal_n_layers,
    load_optimized_vqe,
    get_appropriate_fake_backend,
    circ_ham_expectation_values,
    apply_parameters,
    HAM_NAME_VAR,
)
from qcomp_nlce.utils.results import pretty_print_diff, pretty_print
from pathlib import Path

n_qubits_max = 5
n_qubits = 5
J = 1
vqe_l_dir = f"{n_qubits_max}q_{get_optimal_n_layers(n_qubits_max)}l"
elem_dirname_l = Path(
    res_sim_dir(),
    f"elem_vqe_{int(J*100)}j/vqe/cost_offdiag/{vqe_l_dir}",
)
hamiltonian, circuit, elem_weights, ground_energy = load_optimized_vqe(elem_dirname_l)

hamiltonian.__dict__[HAM_NAME_VAR] = "TFIM"
circuit = apply_parameters(circuit, elem_weights[-1])
which_backend = "IQM"
backend = get_appropriate_fake_backend(n_qubits, which_backend)
n_shots = int(1e5)

comp_states = hamming_weight_states(n_qubits, 1)
true_comp_vals = circ_ham_expectation_values(hamiltonian, circuit, comp_states)
comp_vals, stds = measure_hamiltonian_elements(
    hamiltonian,
    circuit,
    comp_states,
    None,
    backend,
    which_backend,
    n_shots,
)


print("True")
pretty_print(true_comp_vals)
print(f"fake_backend {which_backend}")
pretty_print(comp_vals)
print("Diff")
pretty_print_diff(true_comp_vals, comp_vals)
