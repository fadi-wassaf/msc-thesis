# %%
import numpy as np
from tools.circuits.lumped_circuits import CascadeCL
from tools.circuits.capacitance import maxwell_to_mutual

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use(['science', 'grid'])

C0_1 = 100
C0_2 = 1
C0 = np.diag([C0_1, C0_2])

C1 = []
C2 = []
C12 = []

theta = np.linspace(0, np.pi, 100)

# While varying the parameter t, find the capacitance values for the cascade representation
for t in theta:
    U = np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])
    Cm = U @ C0 @ U.T
    Cs = maxwell_to_mutual(Cm)
    C1.append(Cs[0,0])
    C2.append(Cs[1,1])
    C12.append(Cs[0,1])

C1 = np.array(C1)
C2 = np.array(C2)
C12 = np.array(C12)

plt.plot(theta, C1/C0_1, label=r'$C_1$', color='r')
plt.plot(theta, C2/C0_1, label=r'$C_2$', color='b')
plt.plot(theta, C12/C0_1, label=r'$C_{12}$', color='g')
plt.axhline(0, color='k', linestyle='--')

plt.xlabel(r'$\theta$')
plt.ylabel(r'$C_i/\tilde{C}_1$')

plt.legend(loc='upper right')
plt.tight_layout()
# plt.savefig(f'../../thesis/figures/DC_residue_eig_ratio_{C0_2/C0_1}.pdf')
# plt.show()
