# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skrf as rf
from scipy.signal import argrelmax
import pickle

from tools.circuits.interconnects import interconnect_z, ZConnect
from tools.circuits.multiports import compare_multiport, compare_singleport

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])

# %% Load s-parameters of the original models
cap1 = rf.Network('../../models/ansys/finger_cap.s2p')
cap2 = rf.Network('../../models/ansys/finger_cap.s2p')
cpw_6mm = rf.Network('../../models/ansys/cpw_6mm_CRC.s2p')
cap_res_cap = rf.Network('../../models/ansys/cap_res_cap.s2p')

# %% Interconnect the original s-parameter bricks
split_model = rf.connect(cap1, 1, cpw_6mm, 0)
split_model = rf.connect(split_model, 1, cap2, 0)

# %% Load the fitted impedance functions
finger_cap_rational = pickle.load(open('../fitting/finger_cap.obj', 'rb'))
finger_cap_rational.port_names = ['1', '2']
cpw_6mm_rational = pickle.load(open('../fitting/cpw_6mm_CRC.obj', 'rb'))
cpw_6mm_rational.port_names = ['1', '2']

# %% Compare rational bricks to simulation
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=4)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], colors[3], colors[1]]
compare_multiport(ax[0], cap_res_cap.f, cap1.s, rf.z2s(finger_cap_rational(cap_res_cap.f)), colors=colors)
compare_multiport(ax[1], cap_res_cap.f, cpw_6mm.s, rf.z2s(cpw_6mm_rational(cap_res_cap.f)), colors=colors)

ax[0].set_xlabel('Frequency (GHz)')
ax[1].set_xlabel('Frequency (GHz)')
ax[0].set_ylabel(r'$\log_{10}(|S_{ij}|)$')
ax[1].set_ylabel(r'$\log_{10}(|S_{ij}|)$')

ax[0].set_title('Finger Capacitor Brick')
ax[1].set_title('6 mm CPW Brick')

ax[1].lines[0].set_label('Simulation')
ax[1].lines[1].set_label('Fit')
ax[1].lines[2].set_label('Difference')
ax[1].legend(fontsize = 'x-small')

# plt.savefig('../../thesis/figures/cap_res_fit.pdf')

# %% Connect the rational impedances of the bricks together

rational_split = interconnect_z(
    [finger_cap_rational, cpw_6mm_rational, finger_cap_rational],
    [ZConnect(0,1,1,0), ZConnect(1,1,2,0)]
)
rational_split_z = rational_split(split_model.f)
rational_split_s = rf.z2s(rational_split_z)


# %% Compare full model S-param to brick model
fig, ax = plt.subplots(1, 1, figsize=(6.5,5))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], colors[3], colors[1]]

compare_singleport(ax, cap_res_cap.f, cap_res_cap.s[:,0,1], rational_split_s[:,0,1], colors=colors)

ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel(r'$\log_{10}(|S_{12}|)$')
ax.lines[0].set_label('Full Model')
ax.lines[1].set_label('Brick Model')
ax.lines[2].set_label('Difference')
ax.legend(fontsize = 'x-small', loc='lower right')

# plt.savefig('../../thesis/figures/full_vs_brick.pdf')

# %% Resonance difference estimate
idx = argrelmax(np.abs(cap_res_cap.s[:,0,1]))
full_peaks = np.array(cap_res_cap.f[idx]/(1e9))

idx = argrelmax(np.abs(split_model.s[:,0,1]))
split_peaks = split_model.f[idx]/(1e9)

idx = argrelmax(np.abs(rational_split_s[:,0,1]))
split_fit_peaks = split_model.f[idx]/(1e9)

print(f'Full peaks = {full_peaks}')
print(f'Split peaks = {split_peaks}')
print(f'Split fit peaks = {split_fit_peaks}')

print(f'Full peaks - Split peaks = {full_peaks - split_peaks} -> {100*(full_peaks - split_peaks)/full_peaks} %')
print(f'Full peaks - Split fit peaks = {full_peaks - split_fit_peaks} -> {100*(full_peaks - split_fit_peaks)/full_peaks} %')
print(f'Split peaks - Split fit peaks = {split_peaks - split_fit_peaks} -> {100*(split_peaks - split_fit_peaks)/full_peaks} %')

# %%
