from qiskit_ibm_runtime import SamplerV2
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

n_shots = 1000

circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
circ.measure_all()

backend = IQMFakeGarnet()
pm = generate_preset_pass_manager(backend=backend)

out = pm.run(circ)

sampler = SamplerV2(mode=backend)
ress = sampler.run([out], shots=n_shots).result()


resb = backend.run([out], shots=n_shots).result()


...
