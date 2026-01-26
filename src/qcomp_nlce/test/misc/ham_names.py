from qiskit.quantum_info import SparsePauliOp


ham = SparsePauliOp("XXIXIX") + SparsePauliOp("YZYZIX")

ham.name = "TFIM"


print
