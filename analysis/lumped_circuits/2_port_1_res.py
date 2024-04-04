# %%
import numpy as np
import matplotlib.pyplot as plt
from tools.circuits.lumped_circuits import CascadeCL

# 2 port network with each port capacitively coupled to 1 resonator

Cq1 = 70e-15
Cq2 = 72e-15
C1c = 4e-15
C2c = 4.2e-15
Cc = 200e-15
C12 = 1e-15

C = np.array([
    [Cq1, C12, C1c],
    [C12, Cq2, C2c],
    [C1c, C2c,  Cc]
])

L = np.array([3.38682e-9])

circuit = CascadeCL(C, L, 'spice')
f = np.linspace(0.5e9, 20.5e9, 25000)
fig, ax = circuit.plot_Z_compare_amp_phase(f)
plt.show()