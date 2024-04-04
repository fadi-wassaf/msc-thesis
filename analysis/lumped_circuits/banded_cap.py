# %%
import numpy as np
import numpy as num_poles
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import bandwidth

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science'])

# %% Create test banded capacitance corresponding to a qubit grid and show how the off diagonals decay in the inverse

Cs = 100
Cc = 10

n = 10
m = 10

qubit_grid = nx.grid_2d_graph(m, n)
pos = {(x,y):(y,-x) for x,y in qubit_grid.nodes()}

# Make capacitance matrix out of the adjacency matrix for the qubit grid
C = np.diag(Cs*np.ones(m*n))
adj = nx.adjacency_matrix(qubit_grid).todense()
for i in range(0, n*m):
    for j in range(i, n*m):
        if adj[i,j]:
            C[i,j] = -Cc
            C[j,i] = -Cc
            C[i,i] += Cc

C_inv = np.linalg.inv(C)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
fig.tight_layout(pad=4)

ax_c = ax[0].imshow(np.abs(C), vmin=0, vmax=10)
fig.colorbar(ax_c, ax=ax[0], fraction=0.046, pad=0.04, label='Capacitance (fF)')
ax[0].set_title(r'$C$')

ax_cinv = ax[1].imshow((np.abs(C_inv)), vmax=0.0008)
fig.colorbar(ax_cinv, ax=ax[1], fraction=0.046, pad=0.04)
ax[1].set_title(r'$C^{-1}$')

ax_cinv = ax[2].imshow(np.log(np.abs(C_inv)))
fig.colorbar(ax_cinv, ax=ax[2], fraction=0.046, pad=0.04)
ax[2].set_title(r'$\log_{10}(|C^{-1}|)$')

# plt.savefig('../../thesis/figures/10x10_grid_cap.pdf')

# %%
