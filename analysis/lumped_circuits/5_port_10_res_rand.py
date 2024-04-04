# %% 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tools.circuits.lumped_circuits import CascadeCL
import scienceplots

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])

# Create random circuit with 5 ports and 10 poles
# Plot theoretical impedance vs. the one solved for using cascade analysis

N = 15
M = 10

np.random.seed(13)

C = 1e-15*np.random.uniform(0, 10, (N, N))
C = (C + C.T)/2

C1_bound = 100e-15
C2_bound = 200e-15
C_diag = 1/(np.random.uniform(1/np.sqrt(C2_bound), 1/np.sqrt(C1_bound), (N)))**2
C[range(N), range(N)] = C_diag

L1_bound = .4e-9
L2_bound = 5e-9
L = 1/(np.random.uniform(1/np.sqrt(L2_bound), 1/np.sqrt(L1_bound), (M)))**2

circuit = CascadeCL(C, L, 'mutual')
f = np.linspace(0.5e9, 20.5e9, 5000)
Z_rational = circuit.rational_Z_discrete(f)
Z_cascade = circuit.cascade_Z_discrete(f)
f = f/1e9
np.seterr(divide='ignore')

fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
fig.tight_layout(pad=4)
N = N - M
for i in range(N):
    for j in range(i, N):
        ax[0].plot(f, np.log10(np.abs(Z_rational[:,i,j])), label=f'Z{i+1}{j+1}')
        ax[0].set_title('Rational Impedance - Prediction')
        ax[0].set_ylabel(r'$\log_{10}(|Z_{ij}^R|)$')
        
        ax[1].plot(f, np.log10(np.abs(Z_rational[:,i,j]-Z_cascade[:,i,j])), label=f'Z{i+1}{j+1}')
        ax[1].set_title('Difference - Rational vs. Cascade')
        ax[1].set_ylabel(r'$\log_{10}(|Z_{ij}^R - Z_{ij}^C|)$')

ax[0].set_xlabel('Frequency (GHz)')
ax[1].set_xlabel('Frequency (GHz)')


# plt.savefig('../../thesis/figures/5_port_10_res_rand.pdf')
# %%
