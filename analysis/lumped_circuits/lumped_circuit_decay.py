# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
from qutip import (sesolve, basis, tensor, ket2dm)

from tools.analysis.transmon_network import TransmonNetwork

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])

# %% Setup circuit
# branches:
# 0,1 - qubit 1 and 2
# 2,4 - qubit cap drive
# 3,5 - qubit readout port
# 6 - qubit resonant coupler
# 7,8 - qubit readout resonators

# port and resonator capacitance
C = np.zeros((9,9))
C[0,0] = 70e-15
C[1,1] = 75e-15
C[(2,3,4,5),(2,3,4,5)] = 100e-15
C[(6,7,8),(6,7,8)] = 300e-15

# qubit to port couplings
C[0,2] = 0.15e-15
C[1,4] = 0.15e-15

# qubit to resonator coupling
C[0,6] = 10e-15
C[0,7] = 10e-15
C[1,6] = 10e-15
C[1,8] = 10e-15

# resonator port coupling
C[3,7] = 10e-15
C[5,8] = 10e-15

L = [3.25e-9, 2.1e-9, 1.6e-9]

# add other small random coupling in the upper diagonal of C that is currently zero
# for i in range(9):
#     for j in range(i,9):
#         if C[i,j] == 0:
#             C[i,j] = 1e-15#*np.random.uniform(0, 5)

C = C + np.triu(C, 1).T
network = TransmonNetwork.from_CL(C, L, ft=[3.25e9, 3.75e9], C_type='mutual')
print(network.resonator_frequencies/(2*np.pi))

# %% Pole location test
network.L[0] = 18e-9
network.L[1] = 15.5e-9
poles = np.array(network.compute_lossy_poles())

pole_idx = 4
pole_real = np.real(poles[pole_idx])/(2*np.pi)
pole_imag = np.imag(poles[pole_idx])/(2*np.pi)
points = 201
pole_range = 8e9
f_re = np.linspace(2*pole_real, 0, points)
f_im = np.linspace(pole_imag - np.abs(pole_real), pole_imag + np.abs(pole_real), points)

Z_cmplx = network.fully_shunted_impedance(f_re, f_im)

# %% Plotting the impedance in the complex plane to see where the polesare
fig, (ax_mag, ax_ph) = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=4)
ax_mag.imshow(np.log10(np.abs(Z_cmplx[0,0])), extent=np.array([f_im[0], f_im[-1], f_re[0], f_re[-1]])/1e9, aspect='auto', origin="lower")
ax_mag.scatter(pole_imag/1e9, pole_real/1e9, color='r', s=5)
ax_ph.imshow(np.angle(Z_cmplx[0,0]), extent=np.array([f_im[0], f_im[-1], f_re[0], f_re[-1]])/1e9, aspect='auto', origin="lower")
ax_ph.scatter(pole_imag/1e9, pole_real/1e9, color='r', s=5)
ax_mag.set_title('$\log_{10}|Z_{11}(\sigma + i\omega)|$')
ax_mag.set_xlabel('$\omega/2\pi$ (GHz)', labelpad=20)
ax_mag.set_ylabel('$\sigma/2\pi$ (GHz)')
ax_ph.set_title('arg$(Z_{11}(\sigma + i\omega)$)')
ax_ph.set_xlabel('$\omega/2\pi$ (GHz)', labelpad=20)
ax_ph.set_ylabel('$\sigma/2\pi$ (GHz)')
ax_mag.grid()
ax_ph.grid()
# plt.savefig('../../thesis/figures/fully_shunted_impedance.pdf')
# plt.show()

# %%
network.update_transmon_frequencies([3.8e9, 4e9])
poles1 = []
L1 = []
print(network.L)
f_list = np.linspace(2.5e9, 10e9, 100)
for f in f_list:
    network.update_transmon_frequency(0, f)
    L1.append(network.L[0])
    poles1.append(network.compute_lossy_poles())

network.update_transmon_frequencies([3.8e9, 4e9])
poles2 = []
for f in f_list:
    network.update_transmon_frequency(1, f)
    poles2.append(network.compute_lossy_poles())

# %% poles plotting real and imaginary part
fig, (pole_real, pole_imag) = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=4)
L1 = np.array(L1)
poles1 = np.array(poles1)
poles2 = np.array(poles2)
for i in range(int(poles1.shape[1]/2)):
    p1 = poles1[:, i]
    pole_imag.plot(L1/1e-9, np.abs(np.imag(p1))/(2*np.pi)/1e9)
    pole_real.plot(L1/1e-9, np.abs(2*np.real(p1)))

pole_imag.set_ylim([2,11])
pole_real.set_ylim([2e3, 5e7])
pole_real.set_yscale('log')
pole_imag.set_xlabel(r'$L_1$ (nH)')
pole_real.set_xlabel(r'$L_1$ (nH)')
pole_imag.set_ylabel(r'Resonant Frequency $\omega_i/2\pi$ (GHz)')
pole_real.set_ylabel(r'Decay Rate $\kappa_i$ (sec$^{-1}$)')
# plt.savefig('../../thesis/figures/poles_real_imag.pdf')

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5.5))
fig.tight_layout(pad=4)
network.update_transmon_frequencies([3.8e9, 4e9])
f = np.linspace(2.5e9, 10e9, 25000)

# Plot admittance T1 estimate with qubit inductances
decay1 = network.transmon_decay_rate(f, shunt_transmons=True, transmon_idx=0)
decay2 = network.transmon_decay_rate(f, shunt_transmons=True, transmon_idx=1)
# plt.figure(figsize=(5,5))
ax[0].plot(f/1e9, 1/decay1/1e-6)
ax[1].plot(f/1e9, 1/decay2/1e-6)

# Plot poles
for i in range(len(poles1)):
    p1 = poles1[i]
    ax[0].scatter(np.abs(np.imag(p1))/(2*np.pi)/1e9, 1/np.abs(2*np.real(p1))/1e-6, s=2, color='r')
    p2 = poles2[i]
    ax[1].scatter(np.abs(np.imag(p2))/(2*np.pi)/1e9, 1/np.abs(2*np.real(p2))/1e-6, s=2, color='r')

# Plot admittance T1 estimate without qubit inductances
decay = network.transmon_decay_rate(f)
ax[0].plot(f/1e9, 1/decay[:, 0]/1e-6)
ax[1].plot(f/1e9, 1/decay[:, 1]/1e-6)

ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].set_xlim(2.5,10)
ax[1].set_xlim(2.5,10)

ax[0].set_xlabel('Transmon 1 Oscillator Frequency (GHz)')
ax[1].set_xlabel('Transmon 2 Oscillator Frequency (GHz)')
ax[0].set_ylabel(r'Transmon 1 $T_1$ ($\mu$s)')
ax[1].set_ylabel(r'Transmon 2 $T_1$ ($\mu$s)')

# plt.savefig('../../thesis/figures/lumped_transmons_T1.pdf')
# plt.savefig('../../thesis/figures/lumped_transmons_T1_all_to_all.pdf')
