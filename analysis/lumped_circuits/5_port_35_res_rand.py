# %% 
import numpy as np
import matplotlib.pyplot as plt
from tools.circuits.lumped_circuits import CascadeCL

N = 40
M = 35

C = 1e-15*np.random.uniform(0, 10, (N, N))
C = (C + C.T)/2
C_diag = 1e-15*np.random.uniform(30, 200, (N))
C[range(N), range(N)] = C_diag

L = 1e-9*np.random.uniform(0, 2.5, (M))

circuit = CascadeCL(C, L, 'spice')
f = np.linspace(0.5e9, 20.5e9, 25000)
fig, ax = circuit.plot_Z_compare_amp_phase(f)
plt.show()
