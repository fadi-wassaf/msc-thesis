# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import skrf as rf
from scipy.signal import argrelmax

from tools.fitting.vector_fitting import lrvf
from tools.circuits.transmission_line import cpw_parameters
from tools.circuits.multiports import compare_singleport
from tools.analysis.transmon_network import TransmonNetwork
from qutip import (sesolve, basis, tensor, ket2dm)
from scipy.optimize import curve_fit

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])

# %% Obtain the impedance function for the ideal TL coupler using ABCD matrices
Cq1 = 70e-15
Cq2 = 72e-15
C1c = 6.5e-15
C2c = 6.5e-15
L, C = cpw_parameters(10e-6, 6e-6, 11.5)
print(L, C)
l = 12e-03
Z0 = np.sqrt(L/C)
Y0 = 1/Z0

def ideal_TL_coupler_circuit_ABCD(f):
    w = 2*np.pi*f
    beta = w * np.sqrt(L * C)
    bl = beta * l
    ones = np.ones(len(w))
    zeros = np.zeros(len(w))

    ABCD_Q1 = np.transpose(np.array([[ones, zeros], [1j*w*Cq1, ones]]), (2, 0, 1))
    ABCD_Q2 = np.transpose(np.array([[ones, zeros], [1j*w*Cq2, ones]]), (2, 0, 1))
    ABCD_C1c = np.transpose(np.array([[ones, -1j/(w*C1c)], [zeros, ones]]), (2, 0, 1))
    ABCD_C2c = np.transpose(np.array([[ones, -1j/(w*C2c)], [zeros, ones]]), (2, 0, 1))
    ABCD_TL = np.transpose(np.array([
        [np.cos(bl), 1j*Z0*np.sin(bl)],
        [1j*Y0*np.sin(bl), np.cos(bl)]
    ]), (2, 0, 1))

    ABCD = np.einsum("aij, ajk -> aik", ABCD_Q1, ABCD_C1c)
    ABCD = np.einsum("aij, ajk -> aik", ABCD, ABCD_TL)
    ABCD = np.einsum("aij, ajk -> aik", ABCD, ABCD_C2c)
    ABCD = np.einsum("aij, ajk -> aik", ABCD, ABCD_Q2)
    return ABCD

def ideal_TL_coupler_circuit_Z(f):
    ABCD = ideal_TL_coupler_circuit_ABCD(f)
    A, B, C, D = ABCD[:,0,0], ABCD[:,0,1], ABCD[:,1,0], ABCD[:,1,1]
    Z = np.zeros((len(f), 2, 2), dtype=np.complex128)
    Z[:,0,0] = A/C
    Z[:,0,1] = 1/C
    Z[:,1,0] = Z[:,0,1]
    Z[:,1,1] = D/C
    return Z


# %% Compute the impedance and fit
f = np.linspace(1e9, 22.5e9, 10001)
Z = ideal_TL_coupler_circuit_Z(f)

# %%
vf_opts = {
    'Niter1' : 30,
    'Niter2' : 60,
    'asymp' : 1,
    'plot': 0,
    'weightparam' : 1,
    'stable': 1,
    'poletype': 'lincmplx'
}   
fit_opts = {
    'only_vf': True,
    'rad_dc': 2*np.pi*1e9,
    'rad_dc_real': 2*np.pi*10e9,
    'res_eval_thresh': 1e6,
    'shift_poles' : True,
    'maxfev': 1
}

Z_fit_rational = lrvf(f, Z, 4, vf_opts=vf_opts, fit_opts=fit_opts)

# %%
file_z = open('ideal_TL_coupler.obj', 'wb')
pickle.dump(Z_fit_rational, file_z)

# %% Compare the rational fit to the original
Z_fit_rational = pickle.load(open('ideal_TL_coupler.obj', 'rb'))

fig, ax = plt.subplots(1, 2, figsize=(10,5))
fig.tight_layout(pad=4)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], colors[3], colors[1]]
compare_singleport(ax[0], f, Z[:,0,1], Z_fit_rational(f, False)[:,0,1], colors=colors)

S = rf.z2s(Z)
S_r = rf.z2s(Z_fit_rational(f, False))
compare_singleport(ax[1], f, S[:,0,1], S_r[:,0,1], colors=colors)

ax[0].set_xlabel('Frequency (GHz)')
ax[1].set_xlabel('Frequency (GHz)')
ax[0].set_ylabel(r'$\log_{10}(|Z_{12}|)$')
ax[1].set_ylabel(r'$\log_{10}(|S_{12}|)$')

ax[1].lines[0].set_label('Original')
ax[1].lines[1].set_label('Fit')
ax[1].lines[2].set_label('Difference')
ax[1].legend(fontsize = 'x-small')

plt.savefig('../../thesis/figures/ideal_TL_fit.pdf')

# %% Fitting for multiple different frequency ranges to show difference in coupling rates when including higher modes
f_end = 1e9*(5*np.arange(1,41,1) + 2.5)
num_res = np.arange(1,41,1)

for i, fe in enumerate(f_end):
    f = np.linspace(1e9, fe, 25001)
    Z = ideal_TL_coupler_circuit_Z(f)

    vf_opts = {
        'Niter1' : 30,
        'Niter2' : 30,
        'asymp' : 1,
        'plot': 0,
        'weightparam' : 1,
        'stable': 1,
        'poletype': 'lincmplx'
    }   
    fit_opts = {
        'only_vf': True,
        'rad_dc': 2*np.pi*1e9,
        'rad_dc_real': 2*np.pi*10e9,
        'res_eval_thresh': 1e6,
        'shift_poles' : True,
        'maxfev': 1
    }

    Z_fit_rational = lrvf(f, Z, num_res[i], vf_opts=vf_opts, fit_opts=fit_opts)
    file_z = open(f'./ideal_TL_multiple/ideal_TL_coupler_fe_{fe/1e9:.1f}.obj', 'wb')
    pickle.dump(Z_fit_rational, file_z)
    file_z = open(f'./ideal_TL_multiple/ideal_TL_coupler_fe_{fe/1e9:.1f}.obj', 'wb')
    pickle.dump(Z_fit_rational, file_z)
    file_z = open(f'./ideal_TL_multiple/ideal_TL_coupler_fe_{fe/1e9:.1f}.obj', 'wb')
    pickle.dump(Z_fit_rational, file_z)

# %% Fit result 40 resonant modes
TL_res = pickle.load(open(f'./ideal_TL_multiple/ideal_TL_coupler_fe_202.5.obj', 'rb'))
f1 = np.linspace(1e9, 202.5e9, 25001)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.tight_layout(pad=4)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], colors[3], colors[1]]

compare_singleport(ax, f1, rf.z2s(ideal_TL_coupler_circuit_Z(f1))[:,0,1], rf.z2s(TL_res(f1))[:,0,1], colors=colors)

ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel(r'$\log_{10}(|S_{12}|)$')

ax.lines[0].set_label('Original')
ax.lines[1].set_label('Fit')
ax.lines[2].set_label('Difference')
ax.legend(fontsize = 'x-small')

# plt.savefig('../../thesis/figures/ideal_TL_fit_40_res.pdf')

# %% Obtain the coupling rates for the networks as the number of poles and range increases
f_end = 1e9*(5*np.arange(1,41,1) + 2.5)
num_res = np.arange(1,41,1)
fq = 4e9

g_eff = []

for i, fe in enumerate(f_end):
    ideal_TL_coupler = pickle.load(open(f'./ideal_TL_multiple/ideal_TL_coupler_fe_{fe/1e9:.1f}.obj', 'rb'))
    network = TransmonNetwork(ideal_TL_coupler, ft=[fq, fq])
    g_eff.append(network.g_eff_coupling(2)[0,1]/(2*np.pi*1e6))

# %% Effective coupling change for more included resonance modes
fig, ax = plt.subplots(1, 1, figsize=(5,3.5))

ax.plot(num_res, g_eff, marker='o', markersize=4)
ax.set_xticks(np.arange(1,41,4)+1)

ax.set_xlabel('Number of Resonances Included in Fit')
ax.set_ylabel(r'$\tilde{g}_{12}/2\pi$ (MHz)')
# plt.savefig('../../thesis/figures/ideal_TL_geff_change.pdf')

# %% qubit-res coupling and effective coupling for different numbers of resonant modes taken into account
fig, ax = plt.subplots(1, 2, figsize=(10,4.5))
fig.tight_layout(pad=4)

ax[0].plot(network.resonator_frequencies[network.g_matrix[0,2:] < -5e7]/(2*np.pi*1e9), -network.g_matrix[0,2:][network.g_matrix[0,2:] < -5e7]/(2*np.pi*1e6), marker='o', markersize=4)
ax[0].set_xlabel(r'Frequency of Resonance Mode $R_k$ (GHz)')
ax[0].set_ylabel(r'$-g_{1,R_k}/2\pi$ (MHz)')

ax[1].plot(num_res, g_eff, marker='o', markersize=4)
ax[1].set_xticks([0, 10, 20, 30, 40])

ax[1].set_xlabel('Number of Resonances Included in Fit')
ax[1].set_ylabel(r'$\tilde{g}_{12}/2\pi$ (MHz)')
# plt.savefig('../../thesis/figures/ideal_TL_coupling_change.pdf')
