# %%
import numpy as np
import matplotlib.pyplot as plt
from tools.circuits.lumped_circuits import CascadeCL

Cq1 = 68e-15
Cq2 = 70e-15
Cq3 = 72e-15
C1c1 = 4e-15
C1c2 = 0e-15
C2c1 = 4.2e-15
C2c2 = 4.1e-15
C3c1 = 0e-15
C3c2 = 4.3e-15
Cc1 = 200e-15
Cc2 = 225e-15
C12 = 0e-15 
C23 = 0e-15 
C13 = 0e-15 
Cc1cc2 = 0e-15

C = np.array([
    [Cq1, C12, C13, C1c1, C1c2],
    [C12, Cq2, C23, C2c1, C2c2],
    [C13, C23, Cq3, C3c1, C3c2],
    [C1c1, C2c1, C3c1, Cc1, Cc1cc2],
    [C1c2, C2c2, C3c2, Cc1cc2, Cc2]
])

L = np.array([4e-9, 5e-9])

circuit = CascadeCL(C, L, 'spice')
f = np.linspace(0.5e9, 20.5e9, 25000)
fig, ax = circuit.plot_Z_compare_amp_phase(f)
plt.show()