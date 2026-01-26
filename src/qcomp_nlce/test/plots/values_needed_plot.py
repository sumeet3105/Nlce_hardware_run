from qcomp_nlce.utils import heff_n_expectation_values_1d
from qcomp_nlce.utils import heff_n_expectation_values_2d

import matplotlib.pyplot as plt


x = range(3, 100)
y = []
z = []

for n in x:
    y.append(heff_n_expectation_values_1d(n))
    z.append(heff_n_expectation_values_2d(n))

fig, ax = plt.subplots()


ax.plot(x, y, label="1D")
ax.plot(x, z, label="2D")

ax.set_xlabel("Number of qubits")
ax.set_ylabel("Number of expectation values for pCAT")
ax.set_yscale("log")
ax.legend()
plt.show()
