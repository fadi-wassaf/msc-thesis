# %%
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
from qutip import (sesolve, basis, tensor, ket2dm)
from scipy.optimize import curve_fit
import pickle

from tools.analysis.transmon_network import TransmonNetwork

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])

# %% Setup the transmon network containing the fit of the ideal TL coupler
ideal_TL_coupler = pickle.load(open('../fitting/ideal_TL_coupler.obj', 'rb'))
network = TransmonNetwork(ideal_TL_coupler, ft=[4e9, 4e9])
print(network.resonator_frequencies/(2*np.pi*1e9))
print(network.g_matrix[:2,:]/(2*np.pi*1e6))
print(network.g_eff_coupling(2)/(2*np.pi*1e6))

# %%
network.g_matrix[0,:]/(2*np.pi*1e6)

# %%
t_list = np.linspace(0, 200e-9, int(100e3))
dim = 3

# %% Simulate time evolution of full model
exp_pop = []
psi0 = tensor([basis(dim, 1)] + 5*[basis(dim, 0)])
exp_pop.append(ket2dm(tensor([basis(dim, 1)] + 5*[basis(dim, 0)])))
exp_pop.append(ket2dm(tensor([basis(dim, 0)] + [basis(dim, 1)] + 4*[basis(dim, 0)])))
a, ad, H = network.full_hamiltonian(dim)
data_full = sesolve(H, psi0, t_list, e_ops=exp_pop, progress_bar=True)
np.save('ideal_TL_full_transfer.npy', data_full.expect)

# %%
data_full_expect = np.load('ideal_TL_full_transfer.npy')

# %% Simulate time evolution of effective model
exp_pop = []
psi0 = tensor([basis(dim, 1), basis(dim, 0)])
exp_pop.append(ket2dm(tensor([basis(dim, 1), basis(dim, 0)])))
exp_pop.append(ket2dm(tensor([basis(dim, 0), basis(dim, 1)])))
a, ad, H = network.effective_hamiltonian(2, dim)
data_eff = sesolve(H, psi0, t_list, e_ops=exp_pop, progress_bar=True)

# %% Compare full to effective time evolution
fig, ax = plt.subplots(1, 1, figsize=(5,3))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
ax.plot(t_list/1e-9, data_full_expect[1], linewidth=2, label='Full')
ax.plot(t_list/1e-9, data_eff.expect[1], color=colors[3], linewidth=2, label='Effective', linestyle='dashed', dashes=(5, 5))
ax.set_xlabel('Time (ns)')
ax.set_ylabel(r'$p_{\left|01\right>}$')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('../../thesis/figures/ideal_TL_sim.pdf')
