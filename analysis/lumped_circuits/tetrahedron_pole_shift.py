# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tools.circuits.lumped_circuits import CascadeCL

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])

# %%
# Construct the symmetric tetrahedron circuit and plot the resonant frequencies while shifting one of the coupling capacitances

C_main = 70e-15
C_c = 4e-15
C = np.array([
    [C_main, 0, 0, C_c, C_c, 0],
    [0, C_main, 0, 0, C_c, C_c],
    [0, 0, C_main, C_c, 0, C_c],
    [C_c, 0, C_c, C_main, 0, 0],
    [C_c, C_c, 0, 0, C_main, 0],
    [0, C_c, C_c, 0, 0, C_main]
])

L = np.array([4e-9, 4e-9, 4e-9])

C_shifts = np.linspace(-1e-15, 1e-15, 100)
pole_positions = []
for shift in C_shifts:
    C_new = C.copy()
    C_new[0,3] += shift
    C_new[3,0] += shift
    circuit = CascadeCL(C_new, L, 'spice')
    pole_positions.append(circuit.rational_Z.poles[1:])
pole_positions = np.array(pole_positions)

fig, ax = plt.subplots(1, 1, figsize=(5,4.5))
# fig.tight_layout(pad=4)

ax.plot((C_c + C_shifts)/1e-15, pole_positions[:,:]/(2*np.pi)/1e9)
ax.set_xlabel(r'$C_{c}$ (fF)')
ax.set_ylabel(r'Resonant Pole Positions $f_r$ (GHz)')
# plt.savefig('../../thesis/figures/tetra_poles.pdf')