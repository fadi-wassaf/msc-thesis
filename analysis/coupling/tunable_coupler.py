# Tunable coupler example - two transmons coupled using a tunable transmon/coupler

# %%
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
from qutip import (sesolve, basis, tensor, ket2dm)
from scipy.optimize import curve_fit

from tools.analysis.transmon_network import TransmonNetwork

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])

# %% Setup circuit
Cq1 = 70e-15
Cq2 = 72e-15
C1c = 4e-15
C2c = 4.2e-15
Cc = 200e-15
C12 = 0.1e-15

C = np.array([
    [Cq1, C12, C1c],
    [C12, Cq2, C2c],
    [C1c, C2c,  Cc]
])

network = TransmonNetwork.from_CL(C, [], ft=[4e9, 4e9, 5e9], C_type='mutual')

# %% Setup state transfer simulation for multiple coupler frequencies
fc_list = np.linspace(4.2e9, 6.75e9, 200)
t_list = np.linspace(0, 600e-9, int(100e3))
dim = 3
psi0 = tensor([basis(dim, 1), basis(dim, 0), basis(dim, 0)])
transfer_pop = ket2dm(tensor([basis(dim, 0), basis(dim, 1), basis(dim, 0)]))

transfer_sim = []

# %% Run the simulation for all the coupler frequencies
for fc in tqdm(fc_list):
    network.update_transmon_frequency(2, fc)
    a, ad, H = network.full_hamiltonian(dim)
    data = sesolve(H, psi0, t_list, e_ops=[transfer_pop])
    transfer_sim.append((data.expect[0]))
transfer_sim = np.array(transfer_sim)

# %% Save the transfer simulation data
# np.save('transfer_sim_data.npy', transfer_sim)

# %% Load the transfer simulation data
transfer_sim = np.load('transfer_sim_data.npy')

# %% Fitting of the oscillations 
# Use the first few oscillations to fit, otherwise fitting can have trouble
# Use g_eff as initial guess, and to determine where to cut off the data in time
def transfer_func(t, a, b):
    return b*(np.sin(a*t))**2

g12_eff_th = []
g12_eff_sim = []
g12_mag = []
g1c_delta1c = []

for i, fc in tqdm(enumerate(fc_list)):
    # Estimate the effective coupling between transmon 1 and 2
    network.update_transmon_frequency(2, fc)
    g12_eff = np.abs(float(network.g_eff_coupling(2)[0,1]))
    g12_eff_th.append(g12_eff)
    g1c_delta1c.append(network.g_coupling(0,2)/(network.transmon_frequency(0) - network.transmon_frequency(2)))

    # Determine cutoff time for fit 1 (3 periods) and apply first fit
    t_end = 3*2*np.pi/g12_eff
    t_end_idx = np.argmax(t_list > t_end) if t_end < t_list[-1] else -1
    params, _ = curve_fit(transfer_func, t_list[:t_end_idx], transfer_sim[i][:t_end_idx], p0=[g12_eff, 1], maxfev=1000)

    # Determine cutoff time for fit 2 and apply second fit
    t_end = 1.5*2*np.pi/params[0]
    t_end_idx = np.argmax(t_list > t_end) if t_end < t_list[-1] else -1
    params, _ = curve_fit(transfer_func, t_list[:t_end_idx], transfer_sim[i][:t_end_idx], p0=[params[0], params[1]], maxfev=1000)

    g12_eff_sim.append(params[0])
    g12_mag.append(params[1])

g12_eff_th = np.array(g12_eff_th)
g12_eff_sim = np.array(g12_eff_sim)
g1c_delta1c = np.array(g1c_delta1c)

# %% Test plot to view fit result
idx = 60
pts = -1
print(fc_list[idx]/1e9)
plt.plot(t_list[:pts], transfer_sim[idx][:pts])
plt.plot(t_list[:pts], transfer_func(t_list, g12_eff_sim[idx], g12_mag[idx])[:pts])
plt.plot(t_list[:pts], transfer_func(t_list, g12_eff_th[idx], g12_mag[idx])[:pts])
plt.ylim([0,1])
plt.show()

# %% Theory vs Simulation effective coupling rate plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.tight_layout(pad=4)
ax[0].scatter(fc_list/1e9, g12_eff_sim/(2*np.pi*1e6), label=r'Simulation $|\tilde{g}_{12}^S|/2\pi$', s=26, marker='x')
ax[0].plot(fc_list/1e9, g12_eff_th/(2*np.pi*1e6), label=r'Theory $|\tilde{g}_{12}^T|/2\pi$', color='#FF2C00', linewidth=2)
ax[0].legend()

ax[0].set_xlabel(r'$\omega_c/2\pi$ (GHz)')
ax[0].set_ylabel(r'$|\tilde{g}_{12}|/2\pi$ (MHz)')

ax[1].plot(fc_list/1e9, np.abs(g12_eff_sim - g12_eff_th)/(2*np.pi*1e6), label=r"$|\tilde{g}_{12,Simulation} - \tilde{g}_{12,th}|$")
ax[1].set_yscale('log')
ax[1].set_xlabel(r'$\omega_c/2\pi$ (GHz)')
ax[1].set_ylabel(r'$|\tilde{g}_{12}^S - \tilde{g}_{12}^T|/2\pi$ (MHz)')

# plt.savefig('../../thesis/figures/TC_sim_th_eff_coupling.pdf')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 5.5))
extent=[fc_list[0]/1e9, fc_list[-1]/1e9, t_list[0]/1e-9, t_list[-1]/1e-9]
evolve = ax.imshow(np.flip(transfer_sim.T, 0), aspect='auto', extent=extent, cmap='Blues', interpolation='none', vmin=0, vmax=1, rasterized=True)
fig.colorbar(evolve, ax=ax, label=r'$p_{\left|001\right>}$')
ax.set_xlabel(r'$\omega_c/2\pi$ (GHz)')
ax.set_ylabel(r'Time (ns)')
plt.savefig('../../thesis/figures/TC_time_evolution.pdf', dpi=300)
# plt.savefig('../../thesis/figures/TC_time_evolution.png', dpi=300)
# plt.show()

# %% Separate tunable coupler simulation to compare coupler at 4.2 GHz vs 4.6 GHz

t_list = np.linspace(0, 60e-9, int(100e3))
dim = 3
exp_pop = []
exp_pop.append(ket2dm(tensor([basis(dim, 1), basis(dim, 0), basis(dim, 0)])))
exp_pop.append(ket2dm(tensor([basis(dim, 0), basis(dim, 1), basis(dim, 0)])))
exp_pop.append(ket2dm(tensor([basis(dim, 0), basis(dim, 0), basis(dim, 1)])))

psi0 = tensor([basis(dim, 1), basis(dim, 0), basis(dim, 0)])

network.update_transmon_frequencies([4e9, 4e9, 4.6e9]) 
print(f'g_1c/Delta_1c = {network.g_coupling(0,2)/(network.transmon_frequency(0) - network.transmon_frequency(2))}')
a, ad, H = network.full_hamiltonian(dim)
data1 = sesolve(H, psi0, t_list, e_ops=exp_pop, progress_bar=True)
g_eff_1 = network.g_eff_coupling(2)[0,1]

network.update_transmon_frequencies([4e9, 4e9, 4.2e9]) 
print(f'g_1c/Delta_1c = {network.g_coupling(0,2)/(network.transmon_frequency(0) - network.transmon_frequency(2))}')
a, ad, H = network.full_hamiltonian(dim)
data2 = sesolve(H, psi0, t_list, e_ops=exp_pop, progress_bar=True)
g_eff_2 = network.g_eff_coupling(2)[0,1]

# %% Plots comparing the populations for tunable coupler at 4.2 GHz vs 4.6 GHz

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.tight_layout(pad=3)

ax[0].plot(t_list/1e-9, data1.expect[0])
ax[0].plot(t_list/1e-9, data1.expect[1])
ax[0].plot(t_list/1e-9, data1.expect[2])

ax[1].plot(t_list/1e-9, data2.expect[0], label=r'$p_{\left|100\right>}$')
ax[1].plot(t_list/1e-9, data2.expect[1], label=r'$p_{\left|001\right>}$')
ax[1].plot(t_list/1e-9, data2.expect[2], label=r'$p_{\left|010\right>}$')

ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax[0].set_xlabel('Time (ns)')
ax[1].set_xlabel('Time (ns)')
ax[0].set_ylabel('Population')
ax[1].set_ylabel('Population')

ax[0].set_title(r'$\omega_c/2\pi = $ 4.6 GHz')
ax[1].set_title(r'$\omega_c/2\pi = $ 4.2 GHz')
# plt.savefig('../../thesis/figures/TC_time_evolution_examples.pdf')