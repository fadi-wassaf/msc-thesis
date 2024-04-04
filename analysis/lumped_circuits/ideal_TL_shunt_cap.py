# %%
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import argrelmin
import pickle

from tools.circuits.transmission_line import LosslessTL

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])

# %% Show how the shunt capacitances of the cascade representation for an ideal TL will diverge

# Transmission line parameters
L_TL = 0.438e-6 # H/m
C_TL = 0.159e-9 # F/m
l_TL = 12e-3 # m
num_k = 10

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=4)

C_port_1 = 1e-15
C_port_2 = C_port_1
C1 = []
C2 = []
for k in range(0, num_k):
    TL = LosslessTL(k, L_TL, C_TL, l_TL, C_port_1, C_port_2)
    C = TL.rational_Z.mutual_capacitance
    C1.append(C[0,0]/1e-9)
    C2.append(C[1,1]/1e-9)
ax[0].plot(C1, label=r'$C_1$')
ax[0].plot(C2, label=r'$C_2$')
ax[0].set_title(f"$C_1=C_2=$ {C_port_1/1e-15} fF")
ax[0].set_xlabel('Number of Poles')
ax[0].set_ylabel('Capacitance (nF)')
ax[0].legend()

C_port_1 = 1e-8
C_port_2 = C_port_1
C1 = []
C2 = []
for k in range(0, num_k):
    TL = LosslessTL(k, L_TL, C_TL, l_TL, C_port_1, C_port_2)
    C = TL.rational_Z.mutual_capacitance
    C1.append(C[0,0])
    C2.append(C[1,1])
ax[1].plot(C1, label=r'$C_1$')
ax[1].plot(C2, label=r'$C_2$')
ax[1].set_title(f"$C_1=C_2=$ {C_port_1/1e-9} nF")
ax[1].set_xlabel('Number of Poles')
ax[1].set_ylabel('Capacitance (F)')
ax[1].legend(loc='lower left')

# plt.savefig('../../thesis/figures/ideal_TL_shunt_cap.pdf')