from typing import List
import io, time
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import matplotlib as mpl
import skrf as rf
from scipy.optimize import curve_fit

from tools.circuits.impedance import ImpedanceLR
from tools.circuits.multiports import compare_multiport

# Options passed to the MATLAB function that runs the vector fitting routine - for more, see mtrxfit documentation
vf_opts_def = {
    'Niter1' : 10,
    'Niter2' : 10,
    'asymp' : 1,
    'plot': 0,
    'weightparam' : 2,
    'stable': 1,
    'poletype': 'lincmplx',
}

"""
Options that are used within the lossless enforcement secondary fitting process
    only_vf : Used if you only want to run the vector fitting without secondary fitting after lossless extraction
    rad_dc : Radius (in Hz) around the DC point where residues from VF will be combined into one DC residue
    rad_dc_real : Distance (in Hz) away from the DC point on the real frequency axis where real poles will be combined like above
    res_eval_thresh : Threshold above which rank-1 eigenvector/value of a residue will be kept
    shift_poles : Allows the secondary fitting routine to shift the input poles
    maxfev : Maximum function evaluations for scipy.optimize.curve_fit
"""
fit_opts_def = {
    'only_vf': False,
    'rad_dc': 2*np.pi*1e9,
    'rad_dc_real': 2*np.pi*10e9,
    'res_eval_thresh': 1,
    'shift_poles' : True,
    'maxfev': 100
}

def squish_log10_mag_ZS(
    num_ports : int, 
    discrete_Z : np.ndarray, 
    discrete_S : np.ndarray
) -> List:
    """Returns a 1D List of the log of unwrapped Z and S parameters/matrices.

    Parameters
    ----------
    num_ports : int
        Number of ports in the network - needed in some cases if you want to truncate
    discrete_Z : np.ndarray
        The discretized impedance with shape (number_frequency_points, num_ports, num_ports)
    discrete_S : np.ndarray
        The discretized scattering parameter. Same shape as impedance

    Returns
    -------
    np.ndarray
        The "squished" list containing Z and S
    """
    Z_log10_mag_squished = []
    S_log10_mag_squished = []
    for i in range(num_ports):
        for j in range(i, num_ports):
            Z_log10_mag_squished = Z_log10_mag_squished + np.log10(np.abs(discrete_Z[:,i,j])).tolist()
            S_log10_mag_squished = S_log10_mag_squished + np.log10(np.abs(discrete_S[:,i,j])).tolist()
    return Z_log10_mag_squished + S_log10_mag_squished

def lrvf(f : np.ndarray, 
         discrete_Z : np.ndarray,
         num_res : int, 
         poles : List = [],
         vf_opts : dict = {},
         fit_opts : dict = {}
) -> ImpedanceLR:
    """Lossless Reciprocal Vector Fitting (LRVF) routine that uses mtrxfit MATLAB package (see https://www.sintef.no/en/software/vector-fitting/downloads/matrix-fitting-toolbox/)
    Takes in a discretized impedance function and then obtains a lossless reciprocal rational function approximation .

    Parameters
    ----------
    f : np.ndarray
        Frequency point list corresponding to discrete_Z
    discrete_Z : np.ndarray
        The discretized multiport impedance function
    num_res : int
        The number of resonances to fit to (other than the DC pole)
    poles : List, optional
        User set starting poles for vector fitting, by default []
        Leaving this empty tells the program to automatically generate pole locations.
    vf_opts : dict, optional
        Contains parameters for the mtrxfit vector fitting, by default {} and gets replaced with vf_opts_dict
    fit_opts : dict, optional
        Contains parameters for the secondary lossless enforcement fitting, by default {}

    Returns
    -------
    ImpedanceLR
        The final lossless reciprocal impedance function obtained from the fitting process.
    """

    # Set any fitting parameters not passed in to their default value
    vf_opts.update({k:v for k, v in vf_opts_def.items() if not k in vf_opts.keys()})
    if poles == []:
        vf_opts['N'] = 1 + 2*num_res
    fit_opts.update({k:v for k, v in fit_opts_def.items() if not k in fit_opts.keys()})
    

    # --------------------------------- MATLAB VF -------------------------------- #
        
    # Setup matlab engine to use the mtrxfit package (see https://www.sintef.no/en/software/vector-fitting/downloads/matrix-fitting-toolbox/)
    # print(pkg_resources.)
    eng = matlab.engine.start_matlab()
    out = io.StringIO()
    err = io.StringIO()
    
    # Apply general vector fitting using mtrxfit MATLAB package and VFWrapper.m
    discrete_Z_ML = np.ascontiguousarray(np.transpose(discrete_Z, (1, 2, 0)))
    s = 2*np.pi*1j*f
    poles = np.array(poles)
    vf_start = time.time()
    try:
        poles, vf_residues, vf_fit, rmserr = eng.VFWrapper(discrete_Z_ML, s, poles, vf_opts, nargout=4, stdout=out, stderr=err)
    except Exception as e:
        print(e)
        print(out.getvalue())
        print(err.getvalue())
        eng.quit()
        exit()
    eng.quit()

    # Convert MATLAB data types to Python types
    poles = np.array(poles)
    vf_residues = np.array(vf_residues)
    vf_fit = np.transpose(np.array(vf_fit), (2, 0, 1))

    # Print VF Results
    print(f"VF Fitting Time : {time.time()- vf_start : .2f} seconds")
    print(f"rmserr : {rmserr}")
    print("VF Poles (GHz):")
    for p in poles/(2*np.pi)/1e9:
        print(p)    

    # --------------------- Extract Lossless part of the  VF result -------------------- #

    # Combine the residues corresponding to complex poles within rad_dc and real poles within rad_dc_real
    # * This process defines the "zero" poles that come from the vector fitting
    # * Also enforces that the final DC residue is symmetric
    N = discrete_Z.shape[1]
    vf_R0 = np.zeros((N, N))
    zero_pole_idxs = []
    for i, x in enumerate(poles):
        if np.abs(x) < fit_opts['rad_dc'] or (np.isreal(x) and np.abs(x) < fit_opts['rad_dc_real']):
            vf_R0 += np.real(vf_residues[:,:,i])
            zero_pole_idxs.append(i)
    vf_R0 = np.triu(vf_R0) + np.triu(vf_R0).T - np.diag(vf_R0.diagonal())

    print(f'DC Residue Eigenvalues = {np.linalg.eigvals(vf_R0)}')

    if not np.all(np.linalg.eigvals(vf_R0) > 0):
        raise ValueError('Extracted DC residue not positive definite. Try to adjust the vector fitting parameters.')

    # Save the real part of the residues that correspond to nonzero poles
    nonzero_poles = []
    residues = []
    for i in range(vf_residues.shape[2]):
        if (i not in zero_pole_idxs) and np.imag(poles[i]) > 0:
            nonzero_poles.append(np.imag(poles[i])[0])
            residues.append(2*np.real(vf_residues[:, :, i]))

    # For each of the residues - extract its approximate contribution to the turns ratios.
    # If residues are not rank-1, allow multiple rank-1 residues for a single frequency.
    N = vf_R0.shape[0]
    M = len(nonzero_poles)
    turns_ratios = []
    nonzero_poles_deg = []
    print('Residue Eigenvalues:')
    for i, r in enumerate(residues):
        # Diagonalize the residue so that the eigenvalue "weights" can be checked
        r = np.triu(r) + np.triu(r).T - np.diag(r.diagonal())
        evals, evecs = np.linalg.eigh(r)
        
        print(f'pole {nonzero_poles[i]/(2*np.pi)/1e9} : {evals}')

        # For positive eigenvalues above a threshold, create a corresponding residue
        for j, e in enumerate(evals):
            if e > fit_opts['res_eval_thresh']:
                nonzero_poles_deg.append(nonzero_poles[i])
                r = (np.sqrt(e) * evecs[:, j]).T
                turns_ratios.append(r)

    # Create the resulting turns ratio matrix R
    nonzero_poles = nonzero_poles_deg
    M = len(nonzero_poles)
    R = np.zeros((M, N))
    for i, r in enumerate(turns_ratios):
        R[i, :] = r

    # rational_impedance = ImpedanceLR(nonzero_poles, vf_R0, R)
    rational_impedance = ImpedanceLR(vf_R0, nonzero_poles, R)

    # -------------------------- Secondary Curve Fitting ------------------------- #

    if not fit_opts['only_vf']:

        # Setup the list of arguments that's used for the secondary fitting process
        # * Order of args is [nonzero_poles, vf_R0 eigenvalues, raveled vf_R0 eigenvectors, raveled R turns ratio matrix]
        # * For more details, see tools.circuits.impedance.ImpedanceLR.args_list_to_ImpedanceLR.
        # * nonzero_poles is left out if the user chooses to keep the VF poles fixed
        args = []
        if fit_opts['shift_poles']:
            args = nonzero_poles
        C0, U = np.linalg.eig(vf_R0)
        args.extend(C0)
        args.extend(np.ravel(U))
        args.extend(np.ravel(R))

        class Args:
            def __init__(self, args = None):
                self.args = args # []
            def set(self, args):
                self.args = args
        last_args = Args(args)

        # Saves the fixed poles if the user specifies
        fixed_poles = None
        if not fit_opts['shift_poles']:
            fixed_poles = nonzero_poles
        
        # Setup plot for secondary fitting updates
        fig, ax = plt.subplots(2, 1, figsize=(6, 12))
        plt.show(block=False)
        discrete_S = rf.z2s(discrete_Z)

        def wrapper_ZS_func(f, num_ports, num_res, args, fixed_poles=None):
            # Used as the model function in the curve_fit process.
            # Returns the log of the magnitude of the Z and S parameters.
            # Also plots the updated functions against the original Z and S parameters.
            Z = ImpedanceLR.args_list_to_ImpedanceLR(num_ports, num_res, args, fixed_poles=fixed_poles).Z(f, show_progress=False)
            S = rf.z2s(Z)

            last_args.set(args)

            for a in ax:
                a.cla()
                a.grid()

            compare_multiport(ax[0], f, discrete_Z, Z)
            compare_multiport(ax[1], f, discrete_S, S)

            ax[0].set_ylabel(r'$\log_{10}(|Z_{ij}|)$')
            ax[1].set_ylabel(r'$\log_{10}(|S_{ij}|)$')
            ax[0].set_title('Lossless Fitting Progress')
            ax[1].set_xlabel('f (GHz)')
            
            plt.pause(0.001)

            return squish_log10_mag_ZS(num_ports, Z, S)

        ZS_squished = squish_log10_mag_ZS(N, discrete_Z, discrete_S)
        
        try:
            popt, pcov, infodict, mesg, ier = curve_fit(
                lambda f, *params_0 : wrapper_ZS_func(f, N, M, params_0, fixed_poles=fixed_poles),
                f, ZS_squished, p0=args, full_output=True, nan_policy='omit',
                maxfev = fit_opts['maxfev']
            )
            last_args.set(popt)
        except RuntimeError as e: 
            print(str(e))
            print('Since fitting has gone past maxfev, using the last arguments from the fit')

        rational_impedance = ImpedanceLR.args_list_to_ImpedanceLR(N, M, last_args.args, fixed_poles=fixed_poles)

    if not np.all(np.linalg.eigvals(rational_impedance.R0) > 0):
        print('NOTE: The final fit does not have a positive definite DC residue.')

    print(f"LRZ Poles (GHz): {', '.join(map(str, (np.array(rational_impedance.poles)/(2*np.pi)/1e9).tolist()))}")

    # Plot the vector fit impedance and LRZ function against the original input
    # Also plot against the converted S-parameters
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    N = discrete_Z.shape[1]
    rational_impedance_f = rational_impedance(f, show_progress=False)

    # Convert original, VF fit Z and rational Z to S parameters
    discrete_S = rf.z2s(discrete_Z)
    vf_fit_S = rf.z2s(vf_fit)
    rational_S = rf.z2s(rational_impedance_f)

    np_err = np.geterr()
    np.seterr(divide='ignore')

    compare_multiport(ax[0,0], f, discrete_Z, vf_fit)
    compare_multiport(ax[0,1], f, discrete_Z, rational_impedance_f)
    compare_multiport(ax[1,0], f, discrete_S, vf_fit_S)
    compare_multiport(ax[1,1], f, discrete_S, rational_S)

    ax[0,0].set_ylabel(r'$\log_{10}(|Z_{ij}|)$')
    ax[0,0].set_title('VF Result')
    ax[1,0].set_ylabel(r'$\log_{10}(|S_{ij}|)$')
    ax[0,1].set_title('Lossless')
    ax[1,0].set_xlabel('f (GHz)')
    ax[1,1].set_xlabel('f (GHz)')

    np.seterr(**np_err)
    plt.show()

    return rational_impedance