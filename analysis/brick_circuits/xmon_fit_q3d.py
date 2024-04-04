# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skrf as rf
import pickle
from tqdm import tqdm

# from tools.fitting.vector_fitting import lrvf
from tools.circuits.capacitance import maxwell_to_mutual
from tools.circuits.multiports import compare_multiport, compare_singleport, rearrange_ports, truncate_ports
from tools.analysis.transmon_network import TransmonNetwork
from tools.circuits.impedance import ImpedanceLR
from tools.circuits.interconnects import interconnect_z, ZConnect

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])
              
# Original simulation results
xmon_og = rf.Network('../../models/ansys/xmon.s6p')
z = xmon_og.z
z_no_flux = rearrange_ports(z, [0,1,3,4,5,2])
z_no_flux = truncate_ports(z_no_flux, 5)

# Fit with flux line
fit = pickle.load(open('../fitting/xmon.obj', 'rb'))
fit_no_flux = pickle.load(open('../fitting/xmon_no_flux.obj', 'rb'))

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=4)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], colors[3], colors[1]]

f = xmon_og.f
compare_multiport(ax[0], f, z, fit(f, show_progress=False), colors=colors)
compare_multiport(ax[1], f, z_no_flux, fit_no_flux(f, show_progress=False), colors=colors)

ax[0].set_xlabel('Frequency (GHz)')
ax[1].set_xlabel('Frequency (GHz)')
ax[0].set_ylabel(r'$\log_{10}(|Z_{ij}|)$')
ax[1].set_ylabel(r'$\log_{10}(|Z_{ij}|)$')

ax[0].lines[0].set_label('Simulation')
ax[0].lines[1].set_label('Fit')
ax[0].lines[2].set_label('Difference')
ax[0].legend(fontsize = 'x-small')

ax[0].set_title('Including Flux Line Port')
ax[1].set_title('Flux Line Port Left Open')

# plt.savefig('../../thesis/figures/xmon_fit.pdf')

# %% Capacitance matrix comparison
q3d_cap = 1e-15 * np.genfromtxt('../../models/ansys/xmon_cap.csv', delimiter=',', skip_header=2)

# Impedance order = [junction, ctrl, cpw1, cpw2, cpw3]
cap_to_z_idxs = [3, 4, 0, 1, 2]
q3d_cap = q3d_cap[cap_to_z_idxs, :]
q3d_cap = q3d_cap[:, cap_to_z_idxs]

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
print(q3d_cap/1e-15)

z_cap = np.linalg.inv(fit_no_flux.R0)
print(z_cap/1e-15)
print('[junction, ctrl, cpw1, cpw2, cpw3]')
print((z_cap-q3d_cap)/q3d_cap)

port_order = 'junction, ctrl, cpw1, cpw2, cpw3'

np.savetxt('xmon_cap_hfss.csv', z_cap/1e-15, delimiter=',', fmt='%0.3f', header=port_order)
np.savetxt('xmon_cap_q3d.csv', q3d_cap/1e-15, delimiter=',', fmt='%0.3f', header=port_order)
np.savetxt('xmon_cap_diff.csv', 100*(z_cap-q3d_cap)/q3d_cap, delimiter=',', fmt='%0.3f', header=port_order)

