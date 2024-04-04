# %%
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import argrelmin
import pickle

from tools.circuits.impedance import ImpedanceLR
from tools.analysis.transmon_network import TransmonNetwork
from tools.circuits.multiports import compare_multiport

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])

# %% Load S/Z parameters
xmon_1x3_discrete = rf.Network('../../models/ansys/xmon_1x3.s3p')
xmon_1x4_discrete = rf.Network('../../models/ansys/xmon_1x4.s4p')

xmon_1x3 = pickle.load(open('../fitting/xmon_1x3.obj', 'rb'))
xmon_1x4 = pickle.load(open('../fitting/xmon_1x4.obj', 'rb'))

xmon_1x3_cap = ImpedanceLR(xmon_1x3.R0)
xmon_1x4_cap = ImpedanceLR(xmon_1x4.R0)

# %% Display attempted fitting results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=4)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], colors[3], colors[1]]

f1 = xmon_1x3_discrete.f
f2 = xmon_1x4_discrete.f

compare_multiport(ax[0], f1, xmon_1x3_discrete.z, xmon_1x3(f1), colors=colors)
compare_multiport(ax[1], f2, xmon_1x4_discrete.z, xmon_1x4(f2), colors=colors)

ax[0].set_xlabel('Frequency (GHz)')
ax[1].set_xlabel('Frequency (GHz)')
ax[0].set_ylabel(r'$\log_{10}(|Z_{ij}|)$')
ax[1].set_ylabel(r'$\log_{10}(|Z_{ij}|)$')

ax[1].lines[0].set_label('Simulation')
ax[1].lines[1].set_label('Fit')
ax[1].lines[2].set_label('Difference')
ax[1].legend(fontsize = 'x-small', loc='lower right')

ax[0].set_title('Three Xmon Chain')
ax[1].set_title('Four Xmon Chain')

# plt.savefig('../../thesis/figures/xmon_chain_fitting.pdf')

# %% Plot resonance frequencies against xmon number for the various eigenmode simulations
num_xmons = [3, 4, 5, 6, 10]
res_freqs = [52.5322, 46.8515, 43.5313, 41.4314, 37.8159]

fig, ax = plt.subplots(1, 1, figsize=(5, 3.75))
ax.plot(num_xmons, res_freqs, marker='o')

ax.set_ylabel('Parasitic Mode Frequency (GHz)')
ax.set_xlabel('Number of Xmons in the Chain')

# plt.savefig('../../thesis/figures/xmon_chain_res.pdf')

# %%
# 1x3 coupling
xmon_net = TransmonNetwork(xmon_1x3, [4e9, 4e9, 4e9])
g = xmon_net.g_matrix/(2*np.pi*1e6)
print('3-Xmon Chain')
for i in range(3):
    print(f'g{i}3 = {g[i,3]}')
print()

# 1x4 coupling
xmon_net = TransmonNetwork(xmon_1x4, [4e9, 4e9, 4e9, 4e9])
g = xmon_net.g_matrix/(2*np.pi*1e6)
print('4-Xmon Chain')
for i in range(4):
    print(f'g{i}4 = {g[i,4]}')

# %% S-parameters at dips of lowest parasitic resonance
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=4)

dip = argrelmin(np.abs(xmon_1x3_discrete.s[:,1,1]))[0][1]
num_pts = 250
for i in range(3):
    ax[0].plot(
        xmon_1x3_discrete.f[dip-num_pts:dip+num_pts]/1e9,
        np.log10(np.abs(xmon_1x3_discrete.s[dip-num_pts:dip+num_pts,i,i])),
        label=r'$S_{{{}{}}}$'.format(i+1, i+1)
    )
ax[0].legend()

dip = argrelmin(np.abs(xmon_1x4_discrete.s[:,1,1]))[0][1]
num_pts = 250
for i in range(4):
    ax[1].plot(
        xmon_1x4_discrete.f[dip-num_pts:dip+num_pts]/1e9,
        np.log10(np.abs(xmon_1x4_discrete.s[dip-num_pts:dip+num_pts,i,i])),
        label=r'$S_{{{}{}}}$'.format(i+1, i+1)
    )
ax[1].legend()

ax[0].set_xlabel('Frequency (GHz)')
ax[1].set_xlabel('Frequency (GHz)')
ax[0].set_ylabel(r'$\log_{10}(|S_{ii}|)$')
ax[1].set_ylabel(r'$\log_{10}(|S_{ii}|)$')

ax[0].set_title('Three Xmon Chain')
ax[1].set_title('Four Xmon Chain')

# plt.savefig('../../thesis/figures/xmon_chain_S.pdf')

# %%
xmon_1x5_discrete = rf.Network('../../models/ansys/xmon_1x5.s5p')
xmon_1x6_discrete = rf.Network('../../models/ansys/xmon_1x6.s6p')

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=4)

dip = argrelmin(np.abs(xmon_1x5_discrete.s[:,1,1]))[0][1]
num_pts = 250
for i in range(5):
    ax[0].plot(
        xmon_1x5_discrete.f[dip-num_pts:dip+num_pts]/1e9,
        np.log10(np.abs(xmon_1x5_discrete.s[dip-num_pts:dip+num_pts,i,i])),
        label=r'$S_{{{}{}}}$'.format(i+1, i+1)
    )
ax[0].legend()

dip = argrelmin(np.abs(xmon_1x6_discrete.s[:,1,1]))[0][1]
num_pts = 250
for i in range(6):
    ax[1].plot(
        xmon_1x6_discrete.f[dip-num_pts:dip+num_pts]/1e9,
        np.log10(np.abs(xmon_1x6_discrete.s[dip-num_pts:dip+num_pts,i,i])),
        label=r'$S_{{{}{}}}$'.format(i+1, i+1)
    )
ax[1].legend()

ax[0].set_xlabel('Frequency (GHz)')
ax[1].set_xlabel('Frequency (GHz)')
ax[0].set_ylabel(r'$\log_{10}(|S_{ii}|)$')
ax[1].set_ylabel(r'$\log_{10}(|S_{ii}|)$')

ax[0].set_title('Five Xmon Chain')
ax[1].set_title('Six Xmon Chain')

# plt.savefig('../../thesis/figures/xmon_chain_S_5_6.pdf')