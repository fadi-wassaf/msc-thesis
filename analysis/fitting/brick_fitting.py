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

# File contains various fitting parameters for the various bricks

# %% DC residue estimation to compare to Q3D model
# circuit = rf.Network('../../models/ansys/xmon.s6p')

# vf_opts = {
#     'Niter1' : 100,
#     'Niter2' : 100,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 1,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*30e9,
#     'res_eval_thresh': 1e13,
#     'shift_poles' : False,
#     'maxfev': 200
# }

# # No flux port wanted for the comparison to Q3D (flux port is part of ground net)
# Z = circuit.z
# Z = np.delete(Z, 2, 1)
# Z = np.delete(Z, 2, 2)

# Z_fit_rational = lrvf(circuit.f, Z, 0, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('xmon_no_flux_test.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %% xmon full fitting test
# circuit = rf.Network('../../models/ansys/xmon.s6p')

# vf_opts = {
#     'Niter1' : 100,
#     'Niter2' : 100,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 2,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': True,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*30e9,
#     'res_eval_thresh': 1e12,
#     'shift_poles' : False,
#     'maxfev': 200
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 3, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('xmon.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %%
# circuit = rf.Network('../../models/ansys/xmon_low.s6p')

# vf_opts = {
#     'Niter1' : 100,
#     'Niter2' : 100,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 1,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*30e9,
#     'res_eval_thresh': 1e10,
#     'shift_poles' : False,
#     'maxfev': 300
# }

# # No flux port wanted for the comparison to Q3D (flux port is part of ground net)
# Z = circuit.z
# Z = np.delete(Z, 2, 1)
# Z = np.delete(Z, 2, 2)

# Z_fit_rational = lrvf(circuit.f, Z, 1, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('xmon_low.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %% 6mm CPW fitting
# circuit = rf.Network('../../models/ansys/cpw_6mm.s2p')

# vf_opts = {
#     'Niter1' : 100,
#     'Niter2' : 100,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 1,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   

# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*20e9,
#     'res_eval_thresh': 1e5,
#     'shift_poles' : False,
#     'maxfev': 50
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 4, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('cpw_6mm.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %% Finger cap fitting
# circuit = rf.Network('../../models/ansys/finger_cap.s2p')

# vf_opts = {
#     'Niter1' : 30,
#     'Niter2' : 60,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 1,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   

# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*10e9,
#     'res_eval_thresh': 1e5,
#     'shift_poles' : False,
#     'maxfev': 300
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 1, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('finger_cap.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %%
# circuit = rf.Network('../../models/ansys/tcouple.s3p')

# vf_opts = {
#     'Niter1' : 60,
#     'Niter2' : 30,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 1,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*10e9,
#     'res_eval_thresh': 1e5,
#     'shift_poles' : False,
#     'maxfev': 200
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 2, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('tcouple.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %%
# circuit = rf.Network('../../models/ansys/cpw_5mm.s2p')

# vf_opts = {
#     'Niter1' : 60,
#     'Niter2' : 30,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 2,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*20e9,
#     'res_eval_thresh': 1e11,
#     'shift_poles' : True,
#     'maxfev': 200
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 3, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('cpw_5mm.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %%
# circuit = rf.Network('../../models/ansys/cpw_5.5mm.s2p')

# vf_opts = {
#     'Niter1' : 60,
#     'Niter2' : 30,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 2,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*20e9,
#     'res_eval_thresh': 1e11,
#     'shift_poles' : True,
#     'maxfev': 200
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 3, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('cpw_5.5mm.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %%
# circuit = rf.Network('../../models/ansys/cpw_6mm.s2p')

# vf_opts = {
#     'Niter1' : 60,
#     'Niter2' : 30,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 2,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*20e9,
#     'res_eval_thresh': 1e11,
#     'shift_poles' : True,
#     'maxfev': 200
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 4, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('cpw_6mm.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %%
# circuit = rf.Network('../../models/ansys/cpw_9.5mm.s2p')

# vf_opts = {
#     'Niter1' : 60,
#     'Niter2' : 30,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 2,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*30e9,
#     'res_eval_thresh': 2e12,
#     'shift_poles' : False,
#     'maxfev': 600
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 5, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('cpw_9.5mm.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)
# %%
# circuit = rf.Network('../../models/ansys/cpw_9mm.s2p')

# vf_opts = {
#     'Niter1' : 60,
#     'Niter2' : 30,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 4,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*30e9,
#     'res_eval_thresh': 2e12,
#     'shift_poles' : True,
#     'maxfev': 600
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 5, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('cpw_9mm.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %%
# circuit = rf.Network('../../models/ansys/cpw_2mm.s2p')

# vf_opts = {
#     'Niter1' : 60,
#     'Niter2' : 30,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 2,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*30e9,
#     'res_eval_thresh': 1e12,
#     'shift_poles' : True,
#     'maxfev': 300
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 1, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('cpw_2mm.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %%
# circuit = rf.Network('../../models/ansys/transmon2.s5p')

# vf_opts = {
#     'Niter1' : 60,
#     'Niter2' : 30,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 2,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*10e9,
#     'res_eval_thresh': 3e12,
#     'shift_poles' : False,
#     'maxfev': 400
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 1, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('transmon2.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)

# %%
# circuit = rf.Network('../../models/ansys/transmon3.s5p')

# vf_opts = {
#     'Niter1' : 60,
#     'Niter2' : 30,
#     'asymp' : 1,
#     'plot': 0,
#     'weightparam' : 2,
#     'stable': 1,
#     'poletype': 'lincmplx'
# }   
# fit_opts = {
#     'only_vf': False,
#     'rad_dc': 2*np.pi*1e9,
#     'rad_dc_real': 2*np.pi*10e9,
#     'res_eval_thresh': 3e12,
#     'shift_poles' : False,
#     'maxfev': 400
# }

# Z_fit_rational = lrvf(circuit.f, circuit.z, 1, vf_opts=vf_opts, fit_opts=fit_opts)
# file_z = open('transmon3.obj', 'wb')
# pickle.dump(Z_fit_rational, file_z)