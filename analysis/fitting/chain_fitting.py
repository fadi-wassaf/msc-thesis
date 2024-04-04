# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skrf as rf
import pickle

from tools.fitting.vector_fitting import lrvf
from tools.circuits.multiports import compare_multiport

mpl.rcParams.update({'font.size': 15})
plt.style.use(['science', 'grid'])

# File contains various fitting parameters for some of the xmon chains

# %%
# circuit = rf.Network('../../models/ansys/xmon_1x3.s3p')

# vf_opts = {
#     'Niter1' : 200,
#     'Niter2' : 200,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 1,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*20e9,
#     'rad_dc_real': 2*np.pi*50e9,
#     'res_eval_thresh': 1e11,
#     'shift_poles' : False,
#     'maxfev': 600
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 3, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('xmon_1x3.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %%
circuit = rf.Network('../../models/ansys/xmon_1x4.s4p')

vf_opts = {
    'Niter1' : 200,
    'Niter2' : 200,
    'asymp' : 1,
    'plot': 0,
    'weightparam' : 2,
    'stable': 1,
    'poletype': 'lincmplx'
}   
fit_opts = {
    'only_vf': False,
    'rad_dc': 2*np.pi*20e9,
    'rad_dc_real': 2*np.pi*50e9,
    'res_eval_thresh': 1e11,
    'shift_poles' : False,
    'maxfev': 400
}

Z_fit_rational = lrvf(circuit.f, circuit.z, 3, vf_opts=vf_opts, fit_opts=fit_opts)
file_z = open('xmon_1x4.obj', 'wb')
pickle.dump(Z_fit_rational, file_z)
