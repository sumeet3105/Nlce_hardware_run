from qiskit.transpiler import Layout
from qiskit.circuit import QuantumRegister

# the layout that doesn't work:
# Layout({
# 10: <Qubit register=(5, "q"), index=0>,
# 9: <Qubit register=(5, "q"), index=1>,
# 4: <Qubit register=(5, "q"), index=2>,
# 3: <Qubit register=(5, "q"), index=3>,
# 8: <Qubit register=(5, "q"), index=4>,
# 0: <Qubit register=(15, "ancilla"), index=0>,
# 1: <Qubit register=(15, "ancilla"), index=1>,
# 2: <Qubit register=(15, "ancilla"), index=2>,
# 5: <Qubit register=(15, "ancilla"), index=3>,
# 6: <Qubit register=(15, "ancilla"), index=4>,
# 7: <Qubit register=(15, "ancilla"), index=5>,
# 11: <Qubit register=(15, "ancilla"), index=6>,
# 12: <Qubit register=(15, "ancilla"), index=7>,
# 13: <Qubit register=(15, "ancilla"), index=8>,
# 14: <Qubit register=(15, "ancilla"), index=9>,
# 15: <Qubit register=(15, "ancilla"), index=10>,
# 16: <Qubit register=(15, "ancilla"), index=11>,
# 17: <Qubit register=(15, "ancilla"), index=12>,
# 18: <Qubit register=(15, "ancilla"), index=13>,
# 19: <Qubit register=(15, "ancilla"), index=14>
# })
# layout.get_registers()

q1 = QuantumRegister(3, "q")
q2 = QuantumRegister(4, "ancilla")
q3 = QuantumRegister(4, "bancilla")


input_dict = {j: q for j, q in enumerate([*q1, *q2, *q3])}
layout = Layout(input_dict)

regs = layout.get_registers()

layout2 = Layout()
layout2.add_register(q1)
layout2.add_register(q2)
layout2.add_register(q3)

regs2 = layout2.get_registers()

# True
assert layout == layout2


layout.add_register(q1)
layout.add_register(q2)
layout.add_register(q3)

# False
assert regs == regs2
