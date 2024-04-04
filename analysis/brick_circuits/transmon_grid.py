# %%
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import argrelmax
import pickle

from tools.circuits.builder.brick import Brick
from tools.circuits.builder.chip import ChipGrid
from tools.analysis.transmon_network import TransmonNetwork

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])

# %%
qubit_port_names = ['JJ', 'WP1', 'WP2', 'WP3', 'WP4']
qubit_port_geo = ['WP1', 'WP2', 'WP3', 'WP4']

cpw_port_names = ['WP1', 'WP2']
cpw_port_geo = ['WP1', None, 'WP2', None]

# %%
transmon2 = pickle.load(open('../fitting/transmon2.obj', 'rb'))
transmon3 = pickle.load(open('../fitting/transmon3.obj', 'rb'))
cpw9 = pickle.load(open('../fitting/cpw_9mm.obj', 'rb'))
cpw9_5 = pickle.load(open('../fitting/cpw_9.5mm.obj', 'rb'))

# %%
transmon = Brick('Q', qubit_port_names, qubit_port_geo, rational_z=transmon2, img_file="../../models/ansys/transmon2.png")
cpw9 = Brick('9mm', cpw_port_names, cpw_port_geo, rational_z=cpw9, img_file="../../models/ansys/cpw_9mm.png")
cpw9_5 = Brick('9.5mm', cpw_port_names, cpw_port_geo, rational_z=cpw9_5, img_file="../../models/ansys/cpw_9.5mm.png")

# %% Build 2 x 6 transmon lattice
n_col = 6
grid = [ [], [], [] ]

cpw_9_num = 1
cpw_9_5_num = 1

for i in range(n_col):
    grid[0].append(transmon.copy(transmon.name + f'_{i+1}', orientation=i%2))
    grid[2].append(transmon.copy(transmon.name + f'_{(2*n_col)-i}', orientation=(i+1)%2))

    # if i >= n_col - 1:
    #     break
    
    if i % 2 == 0:
        grid[1].append(cpw9_5.copy(cpw9_5.name + f'_{cpw_9_5_num}', orientation=1))
        cpw_9_5_num += 1

        if i < n_col - 1:
            grid[0].append(cpw9_5.copy(cpw9_5.name + f'_{cpw_9_5_num}'))
            cpw_9_5_num += 1
            grid[2].append(cpw9.copy(cpw9.name + f'_{cpw_9_num}'))
            cpw_9_num += 1
    else:
        grid[1].append(cpw9.copy(cpw9.name + f'_{cpw_9_num}', orientation=1))
        cpw_9_num += 1

        if i < n_col - 1:
            grid[0].append(cpw9.copy(cpw9.name + f'_{cpw_9_num}'))
            cpw_9_num += 1
            grid[2].append(cpw9_5.copy(cpw9_5.name + f'_{cpw_9_5_num}'))
            cpw_9_5_num += 1

    grid[1].append(None)

chip = ChipGrid(grid)
chip.chip_image(label_names=True)

# %% bring qubit indices to the front of the impedance function
qubit_idxs = [i for i, name in enumerate(chip.network.port_names) if 'JJ' in name]
q_idx_sort = np.array([chip.network.port_names[i].split('_')[1] for i in qubit_idxs]).astype(int).argsort()
qubit_idxs = [qubit_idxs[i] for i in q_idx_sort]
print([chip.network.port_names[i] for i in qubit_idxs])
chip_Z = chip.network.bring_to_front(qubit_idxs)

# %% Setup transmon network Hamiltonian
transmon_net = TransmonNetwork(chip_Z, 2*n_col*[4.25e9])

# %% Compute the effective coupling between qubit 1 and other others
g1_eff = transmon_net.g_eff_coupling(2*n_col, frequency_cutoff=60)[0,1:]/(2*np.pi*1e6)
print(g1_eff)

fig, ax = plt.subplots(1, 1, figsize=(5,3))

ax.plot(list(range(2,2*n_col+1)), np.abs(g1_eff), marker='o', markersize=4.5)
ax.set_yscale('log')
ax.set_xlabel(r'Second Transmon Index $j$')
ax.set_ylabel(r'$\tilde{g}_{1j}/2\pi$ (MHz)')

# plt.savefig('../../thesis/figures/2x6_coupling.pdf')



# %%
