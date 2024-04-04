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

# %% Port name/geometry setup - Hamiltonian analysis
qubit_port_names = ['JJ1', 'CTRL', 'WP1', 'WP2', 'WP3']
qubit_port_geo = ['WP1', 'WP2', 'WP3', None]

cpw_port_names = ['WP1', 'WP2']
cpw_port_geo = ['WP1', None, 'WP2', None]

t_port_names = ['WP1', 'WP2', 'WP3']
t_port_geo = ['WP1', None, 'WP2', 'WP3']

# %% Get rational impedance functions corresponding to the different bricks
xmon = pickle.load(open('../fitting/xmon_no_flux.obj', 'rb'))
tcouple = pickle.load(open('../fitting/tcouple.obj', 'rb'))
cpw2 = pickle.load(open('../fitting/cpw_2mm.obj', 'rb'))
cpw5 = pickle.load(open('../fitting/cpw_5mm.obj', 'rb'))
cpw5_5 = pickle.load(open('../fitting/cpw_5.5mm.obj', 'rb'))
cpw6 = pickle.load(open('../fitting/cpw_6mm.obj', 'rb'))
cpw9 = pickle.load(open('../fitting/cpw_9mm.obj', 'rb'))
cpw9_5 = pickle.load(open('../fitting/cpw_9.5mm.obj', 'rb'))

# %% Setup bricks and interconnect grid

x1 = Brick('x1', qubit_port_names, qubit_port_geo, rational_z=xmon)
x2 = x1.copy('x2')
x3 = x1.copy('x3')

t1 = Brick('t1', t_port_names, t_port_geo, rational_z=tcouple)
t2 = t1.copy('t2')
t3 = t1.copy('t3')

s1 = Brick('s1', cpw_port_names, cpw_port_geo, rational_z=cpw2)
s2 = s1.copy('s2')

cpw5 = Brick('cpw5', cpw_port_names, cpw_port_geo, rational_z=cpw5)
cpw5_5 = Brick('cpw5.5', cpw_port_names, cpw_port_geo, rational_z=cpw5_5)
cpw6 = Brick('cpw6', cpw_port_names, cpw_port_geo, rational_z=cpw6)
cpw9 = Brick('cpw9', cpw_port_names, cpw_port_geo, rational_z=cpw9)
cpw9_5 = Brick('cpw9.5', cpw_port_names, cpw_port_geo, rational_z=cpw9_5)

cpw5.rotate_ports_90()
cpw5_5.rotate_ports_90()
cpw6.rotate_ports_90()

grid = [
    [t1, s1, t2, s2, t3],
    [cpw5, None, cpw5_5, None, cpw6],
    [x1, cpw9, x2, cpw9_5, x3]
]

chip = ChipGrid(grid)

# %% Move qubit ports to the front
chip_Z = chip.network.bring_to_front([2, 5, 7])
f = np.linspace(0.5e9, 20.5e9, 20000)
chip_S = rf.z2s(chip_Z(f))

# %% S-parameter plot showing main resonances
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=4)

ax[0].plot(f/1e9, np.log10(np.abs(chip_S[:,0,1])), label=r'$S_{12}$')
ax[0].plot(f/1e9, np.log10(np.abs(chip_S[:,0,2])), label=r'$S_{13}$')
ax[0].plot(f/1e9, np.log10(np.abs(chip_S[:,1,2])), label=r'$S_{23}$')
ax[0].legend(fontsize='small', loc='lower right')
ax[0].set_xlabel('Frequency (GHz)')
ax[0].set_ylabel(r'$\log_{10}(|S_{ij}|)$')

ax[1].plot(f/1e9, np.log10(np.abs(chip_S[:,3,0])), label=r'$S_{41}$')
ax[1].plot(f/1e9, np.log10(np.abs(chip_S[:,3,1])), label=r'$S_{42}$')
ax[1].plot(f/1e9, np.log10(np.abs(chip_S[:,3,2])), label=r'$S_{43}$')
ax[1].legend(fontsize='small', loc='lower right')
ax[1].set_xlabel('Frequency (GHz)')
ax[1].set_ylabel(r'$\log_{10}(|S_{ij}|)$')

# plt.savefig('../../thesis/figures/three_qubit_S.pdf')

# %% Setup transmon Hamiltonian
transmon_net = TransmonNetwork(chip_Z, [4.5e9, 4.5e9, 4.5e9])

# %% Effective coupling example
f_cutoff = 21.5 # Above this frequency, weird poles are included that are not accurate

num_f = 100
f_list_1 = np.linspace(4e9, 4.6e9, num_f)
g_23 = []
for fq in f_list_1:
    transmon_net.update_transmon_frequencies([4e9, fq, fq])
    g_eff = transmon_net.g_eff_coupling(3, frequency_cutoff=f_cutoff)/(2*np.pi*1e6)
    g_23.append(g_eff[1,2])

f_list_2 = np.linspace(4e9, 4.8e9, num_f)
g_12 = []
for fq in f_list_2:
    transmon_net.update_transmon_frequencies([fq, fq, 4e9])
    g_eff = transmon_net.g_eff_coupling(3, frequency_cutoff=f_cutoff)/(2*np.pi*1e6)
    g_12.append(g_eff[0,1])


fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.tight_layout(pad=4)

ax[0].plot(f_list_2/1e9, g_12, label=r'$\tilde{g}_{12}$')
ax[0].plot(f_list_1/1e9, g_23, label=r'$\tilde{g}_{23}$')

ax[0].set_xlabel(r'$\omega_{i/j}/2\pi$ (GHz)')
ax[0].set_ylabel(r'$\tilde{g}_{ij}/2\pi$ (MHz)')
ax[0].legend(fontsize='small')

# Readout resonator dispersive shifts from its directly connected qubit
num_f = 100
f_list = np.linspace(4e9, 4.8e9, num_f)

chi1 = []
chi2 = []
chi3 = []

for fq in f_list:
    transmon_net.update_transmon_frequencies([fq, fq, fq])
    chi1.append(transmon_net.resonator_dispersive_shift(0, 5)/(2*np.pi*1e3))
    chi2.append(transmon_net.resonator_dispersive_shift(1, 4)/(2*np.pi*1e3))
    chi3.append(transmon_net.resonator_dispersive_shift(2, 3)/(2*np.pi*1e3))

ax[1].plot(f_list/1e9, chi1, label='Readout Q1')
ax[1].plot(f_list/1e9, chi2, label='Readout Q2')
ax[1].plot(f_list/1e9, chi3, label='Readout Q3')
ax[1].set_xlabel(r'$\omega_i/2\pi$ (GHz)')
ax[1].set_ylabel(r'$\chi_{i,R_k}$ (kHz)')
ax[1].legend(fontsize='small',)

# plt.savefig('../../thesis/figures/three_qubit_geff_chi.pdf')

# %% Setup bricks for decay rate example

qubit_port_names = ['JJ1', 'CTRL', 'FLUX', 'WP1', 'WP2', 'WP3']
qubit_port_geo = ['WP1', 'WP2', 'WP3', None]

xmon = pickle.load(open('../fitting/xmon.obj', 'rb'))
tcouple = pickle.load(open('../fitting/tcouple.obj', 'rb'))
cpw2 = pickle.load(open('../fitting/cpw_2mm.obj', 'rb'))
cpw5 = pickle.load(open('../fitting/cpw_5mm.obj', 'rb'))
cpw5_5 = pickle.load(open('../fitting/cpw_5.5mm.obj', 'rb'))
cpw6 = pickle.load(open('../fitting/cpw_6mm.obj', 'rb'))
cpw9 = pickle.load(open('../fitting/cpw_9mm.obj', 'rb'))
cpw9_5 = pickle.load(open('../fitting/cpw_9.5mm.obj', 'rb'))

xmon1 = xmon.rearrange_ports([0,1,2,4,5,3])
xmon1 = xmon1.truncate_ports(5)
xmon3 = xmon.truncate_ports(5)

x1_names = ['JJ1', 'CTRL', 'FLUX', 'WP2', 'WP3']
x1_geo = [None, 'WP2', 'WP3', None]

x2_names = ['JJ1', 'CTRL', 'FLUX', 'WP1', 'WP2', 'WP3']
x2_geo = ['WP1', 'WP2', 'WP3', None]

x3_names = ['JJ1', 'CTRL', 'FLUX', 'WP1', 'WP2']
x3_geo = ['WP1', 'WP2', None, None]

# %% Setup grid

x1 = Brick('x1', x1_names, x1_geo, rational_z=xmon1)
x2 = Brick('x2', x2_names, x2_geo, rational_z=xmon)
x3 = Brick('x1', x3_names, x3_geo, rational_z=xmon3)

t1 = Brick('t1', t_port_names, t_port_geo, rational_z=tcouple)
t2 = t1.copy('t2')
t3 = t1.copy('t3')

s1 = Brick('s1', cpw_port_names, cpw_port_geo, rational_z=cpw2)
s2 = s1.copy('s2')

cpw5 = Brick('cpw5', cpw_port_names, cpw_port_geo, rational_z=cpw5)
cpw5_5 = Brick('cpw5.5', cpw_port_names, cpw_port_geo, rational_z=cpw5_5)
cpw6 = Brick('cpw6', cpw_port_names, cpw_port_geo, rational_z=cpw6)
cpw9 = Brick('cpw9', cpw_port_names, cpw_port_geo, rational_z=cpw9)
cpw9_5 = Brick('cpw9.5', cpw_port_names, cpw_port_geo, rational_z=cpw9_5)

cpw5.rotate_ports_90()
cpw5_5.rotate_ports_90()
cpw6.rotate_ports_90()

grid = [
    [t1, s1, t2, s2, t3],
    [cpw5, None, cpw5_5, None, cpw6],
    [x1, cpw9, x2, cpw9_5, x3]
]

chip = ChipGrid(grid)

# %%
chip_Z = chip.network.bring_to_front([2, 5, 8])
transmon_net = TransmonNetwork(chip_Z, [4e9, 4e9, 4e9])

# %% Compute the decay rates using the poles while varying qubit inductance
# Qubit inductance here is very "non-physical" due to the weird behaviour of fitting the flux line port
L1_list = np.logspace(-10, -7, 400)
poles = []
L1 = []
for L in L1_list:
    transmon_net.L[0] = L
    L1.append(transmon_net.L[0])
    poles.append(transmon_net.compute_lossy_poles())


# %% Compute decay rate using only the lossy admittance of the network, no transmon shunt inductors, original shunt capacitance
f = np.linspace(2.5e9, 20e9, 2000)
decay = transmon_net.transmon_decay_rate(f, use_eff_C=False)

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=4)

decay = np.array(decay)
ax[0].plot(f/1e9, 1/decay[:,0]/1e-6)
for i in range(len(poles)):
    p1 = poles[i]
    ax[0].scatter(np.abs(np.imag(p1))/(2*np.pi)/1e9, 1/np.abs(2*np.real(p1))/1e-6, s=2, color='r')

ax[0].set_yscale('log')
ax[0].set_xlim(2, 20)
ax[0].set_ylim(1e-4, 1e4)
ax[0].set_xlabel('Frequency (GHz)')
ax[0].set_ylabel(r'Transmon 1 $T_1$ ($\mu$s)')

ax[1].plot(f/1e9, 1/decay[:,0]/1e-6, label='Transmon 1')
ax[1].plot(f/1e9, 1/decay[:,1]/1e-6, label='Transmon 2')
ax[1].plot(f/1e9, 1/decay[:,2]/1e-6, label='Transmon 3')
ax[1].set_yscale('log')
ax[1].set_xlabel('Frequency (GHz)')
ax[1].set_ylabel(r'$T_1$ ($\mu$s)')
ax[1].legend(fontsize='x-small')

# plt.savefig('../../thesis/figures/three_qubit_T1.pdf')

# %%
