from typing import List

import numpy as np
import scipy.constants as const
from itertools import product
from qutip import (destroy, tensor, identity, Qobj)
import skrf as rf
from tqdm import tqdm

from tools.circuits.impedance import ImpedanceLR
from tools.circuits.capacitance import mutual_to_maxwell
from tools.circuits.lumped_circuits import CascadeCL

PHI0 = const.physical_constants['mag. flux quantum'][0] # Wb

def decay(f):
    def wrapper(*args, **kwargs):
        if args[0].Ne == 0:
            raise ValueError('Cannot use decay analysis on transmon network with no external ports.')
        return f(*args, **kwargs)
    return wrapper

def effective(f):
    def wrapper(*args, **kwargs):
        Nt = args[1]
        if Nt > args[0].N:
            raise ValueError(f'Number of computational qubits should be less than or equal to the total number of qubits ({args[0].N})')
        return f(*args, **kwargs)
    return wrapper

class TransmonNetwork(object):
    """
    Provides analysis and simulation methods for a transmon network. The network is represented by a rational
    impedance function. For more information, see tools.circuits.impedance. The number of ports of the impedance
    function corresponds to the number of transmons plus the number of external ports.
    
    First N ports - transmons, should be ordered as (computation transmons, coupler transmons)
    Next Ne ports - external ports (e.g. drive, flux, readout)
    Final M ports - internal resonant modes

    Attributes
    ----------
    N : int
        Number of transmons (computational + coupler) in the network
    Ne : int
        Number of external ports in the full network
    M : int
        Number of resonant modes in the network
    Z_full : ImpedanceLR
        The rational impedance function of the full network including the external ports.
    C_full : np.ndarray
        The Maxwell capacitance matrix for the full network including the external ports.
    Z : ImpedanceLR
        Truncated rational impedance function excluding the external ports.
    C : np.ndarray
        The Maxwell capacitance matrix for the truncated network excluding the external ports.
    C_inv : np.ndarray
        The inverse of C
    L : np.ndarray
        A list of the qubit and resonant mode inductances (should be of length N + M)
    """

    def __init__(
        self, 
        Z_full : ImpedanceLR,
        ft : np.ndarray = None,
        Lj : np.ndarray = None
    ):
        """Used to setup the Transmon network using a rational impedance function

        Parameters
        ----------
        Z_full : ImpedanceLR
            The rational impedance function of the full network including the external ports.
        ft : np.ndarray, optional
            The desired qubit frequencies for the Hamiltonian that will set the qubit inductances, by default None
        Lj : np.ndarray, optional
            If already known, the qubit inductances can be provided directly, by default None

        Raises
        ------
        ValueError
            If neither ft or Lj are supplied as arguments.
        ValueError
            If the number of qubit frequencies/inductances are more than the number of ports in the impedance.
        """

        # Generate the circuit parameters for the full impedance function
        self.Z_full = Z_full
        self.C_full = self.Z_full.maxwell_capacitance
        self.L = 1/(np.array(self.Z_full.poles[1:])**2)

        if Lj == None and ft == None:
            raise ValueError('Need to supply either ft or Lj.')
        Nt = len(ft) if ft != None else len(Lj)
        if Nt > self.Z_full.N:
            raise ValueError('Number of supplied qubit frequencies or inductances must be less than or equal to the number of ports of the impedance function.')

        # Generate the circuit parameters for the impedance function with any external ports left open
        # Keeps the first Nt ports of the impedance
        self.Z = self.Z_full.truncate_ports(Nt)
        self.C = self.Z.maxwell_capacitance
        self.C_inv = np.linalg.inv(self.C)

        self.N = self.Z.N
        self.Ne = self.Z_full.N - self.N 
        self.M = self.Z_full.M

        # Only initialize Lj using the qubit frequencies Lj is not given.
        if Lj == None:
            Lj = self.transmon_inductances(ft)
        self.L = np.concatenate((Lj, np.array(self.L)))

    @classmethod
    def from_CL(
        cls,
        C : np.ndarray,
        L : np.ndarray, 
        ft : np.ndarray = None,
        Lj : np.ndarray = None,
        C_type : str = "maxwell"
    ) -> 'TransmonNetwork':
        """Creates the transmon network object by generating the impedance of a CL cascade network first.
        For more on the CL cascade see tools.circuits.lumped_circuits.CascadeCL.

        Parameters
        ----------
        C : np.ndarray
            Maxwell capacitance matrix for a (N + Ne + M)-port purely capacitive network
        L : np.ndarray
            M shunt inductances for the M resonant modes of the CL cascade
        ft : np.ndarray, optional
            The desired qubit frequencies for the Hamiltonian that will set the qubit inductances, by default None
        Lj : np.ndarray, optional
            If already known, the qubit inductances can be provided directly, by default None            
        C_type : str, optional
            Specifies the form of the capacitance matrix passed in. Can be 'maxwell', 'mutual', or 'spice'.
            By default "maxwell"

        Returns
        -------
        TransmonNetwork
            The transmon network that corresponds to the CL cascade network.
        """
        network_CL = CascadeCL(C, L, C_type)
        return cls(network_CL.rational_Z, ft, Lj)

    def eff_C(self, i : int) -> float:
        """Compute the effective capacitance that appears in the circuit Hamiltonian.

        Parameters
        ----------
        i : int
            Transmon or resonant mode index corresponding to the capacitance matrix C.

        Returns
        -------
        float
            Effective capacitance of the transmon or resonant mode in Farads.
        """
        return 1/self.C_inv[i,i]

    def eff_EC(self, i : int) -> float:
        """Returns the effective charging energy for a given branch of the circuit.

        Parameters
        ----------
        i : int
            Transmon or resonant mode index corresponding to the capacitance matrix C.

        Returns
        -------
        float
            Effective charging energy of the transmon or resonant mode in Joules.
        """
        return const.e**2 / (2 * self.eff_C(i))

    def EL(self, i : int) -> float:
        """Computes the inductive energy of the specified transmon or resonator.

        Parameters
        ----------
        i : int
            Transmon or resonant mode index corresponding to the capacitance matrix C.

        Returns
        -------
        float
            Inductive energy of the transmon or resonant mode in Joules.
        """
        return PHI0**2 / ( 4 * const.pi**2 * self.L[i] )

    def resonator_frequency(self, k : int) -> float:
        """Computes the resonator frequency for a given branch using the capacitance matrix and a resonator inductance.

        Parameters
        ----------
        k : int
            Resonator index from 0 to M.

        Returns
        -------
        float
            Resonator k frequency in 2*pi*Hz
        """
        return np.sqrt(8 * self.EL(self.N + k) * self.eff_EC(self.N + k)) / const.hbar

    @property
    def resonator_frequencies(self) -> np.ndarray:
        """Computes the resonance frequencies for all the resonator branches.

        Returns
        -------
        np.ndarray
            1d array of the resonator frequencies in 2*pi*Hz
        """
        return np.array([ self.resonator_frequency(k) for k in range(self.M) ])

    def transmon_frequency(self, i : int) -> float:
        """Computes the transmon frequencies using the capacitance matrix and a transmon inductance.

        Parameters
        ----------
        i : int
            Transmon index from 0 to N.

        Returns
        -------
        float
            Transmon i frequency in 2*pi*Hz
        """
        return (np.sqrt(8 * self.EL(i) * self.eff_EC(i)) - self.eff_EC(i)) / const.hbar

    @property
    def transmon_frequencies(self) -> np.ndarray:
        """Compute the transmon frequencies for all the transmon branches.

        Returns
        -------
        np.ndarray
            Transmon frequencies in 2*pi*Hz
        """
        return np.array([ self.transmon_frequency(i) for i in range(self.N) ])

    def transmon_anharmonicity(self, i : int) -> float:
        """Compute the transmon anharmonicity using the capacitance matrix.

        Parameters
        ----------
        i : int
            Transmon index from 0 to N.

        Returns
        -------
        float
            Transmon i anharmonicity in 2*pi*Hz
        """
        return -self.eff_EC(i)/const.hbar

    def transmon_inductance(self, i : int, ft : float) -> float:
        """Compute the inductance of a transmon branch given a desired frequency.

        Parameters
        ----------
        i : int
            Transmon index from 0 to N.
        ft : float
            Desired transmon frequency in Hz.

        Returns
        -------
        float
            Transmon inductance in Henries.
        """
        return 1/(self.eff_C(i) * (2*const.pi*ft + self.eff_EC(i)/const.hbar)**2)
    
    def transmon_inductances(self, ft : np.ndarray) -> np.ndarray:
        """Computes all the transmon inductances given desired frequencies for all the branches.

        Parameters
        ----------
        ft : np.ndarray
            List of transmon frequencies.

        Returns
        -------
        np.ndarray
            Transmon inductances in Henries.
        """
        return np.array([self.transmon_inductance(i, ft[i]) for i in range(self.N)])

    def update_transmon_frequency(self, i : int, ft : float):
        """Update the inductance of a chosen transmon branch with a chosen transmon frequency.

        Parameters
        ----------
        i : int
            Transmon index from 0 to N.
        ft : float
            New desired transmon frequency.
        """
        if i >= self.N:
            raise ValueError(f"Transmon index i should be less than number of transmon ports ({self.N})")
        self.L[i] = self.transmon_inductance(i, ft)

    def update_transmon_frequencies(self, ft : np.ndarray):
        """Update all the transmon inductances at once for a list of desired transmon frequencies.

        Parameters
        ----------
        ft : np.ndarray
            List of new desired transmon frequencies.
        """
        if len(ft) != self.N:
            raise ValueError(f"Length of array ft should be the number of transmon ports ({self.N})")
            
        for i, f in enumerate(ft):
            self.L[i] = self.transmon_inductance(i, f)

    def g_coupling(self, i : int, j : int) -> float:
        """Compute the coupling rate between branch i and j in the transmon network.
        This coupling could be between any of the transmon or resonator branches.

        Parameters
        ----------
        i : int
            Branch 1 index
        j : int
            Branch 2 index

        Returns
        -------
        float
            Coupling rate in 2*pi*Hz
        """
        ELi = self.EL(i)
        ELj = self.EL(j)
        ECi = self.eff_EC(i)
        ECj = self.eff_EC(j)
        return const.e**2 * self.C_inv[i,j] * (ELi * ELj / (4 * ECi * ECj)) ** (0.25) / const.hbar

    @property
    def g_matrix(self) -> np.ndarray:
        """Computes the full coupling matrix between all the branches in the network

        Returns
        -------
        np.ndarray
            (N+M)x(N+M) array containing the coupling rates between the branches in 2*pi*Hz
        """
        g = np.zeros((self.N + self.M, self.N + self.M), dtype=np.complex128)
        for i in range(self.N + self.M):
            for j in range(i+1, self.N + self.M):
                g[i,j] = self.g_coupling(i,j)
        return (g + g.T)

    def resonator_dispersive_shift(self, i : int, k : int) -> float:
        """Computes the dispersive shift contribution to a resonator due to a specific transmon.

        Parameters
        ----------
        i : int
            Transmon index
        k : int
            Resonator Index

        Returns
        -------
        float
            The dispersive shift of resonator k due to transmon i in 2*pi*Hz
        """
        wq = self.transmon_frequency(i)
        wr = self.resonator_frequency(k)
        delta_ik = wq - wr
        sigma_ik = wq + wr
        g_ik = self.g_coupling(i, self.N + k)
        return 2 * g_ik**2 * self.transmon_anharmonicity(i) * ( (1/delta_ik)**2 + (1/sigma_ik)**2 )

    @effective
    def g_eff_coupling(self, Nt : int, frequency_cutoff : float = None) -> np.ndarray:
        """Computes the effective coupling matrix between all the computational qubits.
        Allows for a selection of Nt computational qubits, the rest will be used as couplers alongside the resonant modes.

        Parameters
        ----------
        Nt : int
            Number of transmons to use as "computational" qubits. The rest will be used as couplers
        frequency_cutoff : float, optional
            Frequency in (GHz) above which resonant modes are left out of the effective coupling calculation
            
        Returns
        -------
        np.ndarray
            Effective coupling matrix between computational qubits in 2*pi*Hz
        """
        # if Nt > self.N:
        #     raise ValueError(f'Number of computational qubits should be less than or equal to the total number of qubits ({self.N})')
        
        # Compute the normal branch coupling portion of the matrix
        g = np.triu(self.g_matrix)
        g_eff = np.zeros((Nt, Nt), dtype=np.complex128)
        g_eff = g[:Nt,:Nt]

        transmon_freqs = self.transmon_frequencies
        resonator_freqs = self.resonator_frequencies

        coupler_freqs = np.concatenate((transmon_freqs[Nt:], resonator_freqs))
        transmon_freqs = transmon_freqs[:Nt]
        N = len(transmon_freqs)
        M = len(coupler_freqs)

        # Compute the effective couplings with perturbations from each of the coupler modes linking two qubits.
        for i, j, k in product((range(N)), (range(N)), (range(M))):
            if j > i and (frequency_cutoff is None or coupler_freqs[k] < (2*np.pi*1e9)*frequency_cutoff):
                delta_ik = transmon_freqs[i] - coupler_freqs[k]
                sigma_ik = transmon_freqs[i] + coupler_freqs[k]
                delta_jk = transmon_freqs[j] - coupler_freqs[k]
                sigma_jk = transmon_freqs[j] + coupler_freqs[k]

                g_eff[i,j] += 0.5 * g[i,N + k] * g[j,N + k] * (1/delta_ik + 1/delta_jk - 1/sigma_ik - 1/sigma_jk)

        return (g_eff + g_eff.T)        

    @effective
    def transmon_eff_frequencies(self, Nt : int) -> np.ndarray:
        """Obtains the effective frequencies of the transmon given some number of computational transmons.

        Parameters
        ----------
        Nt : int
            Number of transmons to use as "computational" qubits. The rest will be used as couplers.

        Returns
        -------
        np.ndarray
            The effective transmon frequencies.
        """
        # if Nt > self.N:
        #     raise ValueError(f'Number of computational qubits should be less than or equal to the total number of qubits ({self.N})')
        
        transmon_freqs = self.transmon_frequencies
        resonator_freqs = self.resonator_frequencies
        
        coupler_freqs = np.concatenate((transmon_freqs[Nt:], resonator_freqs))
        transmon_freqs = transmon_freqs[:Nt]
        
        g = self.g_matrix

        N = len(transmon_freqs)
        M = len(coupler_freqs)

        # Compute the effective transmon frequencies with the perturbations due to each coupler mode.
        transmon_eff_freqs = transmon_freqs
        for i in range(N):
            for k in range(M):
                delta_ik = transmon_freqs[i] - coupler_freqs[k]
                sigma_ik = transmon_freqs[i] + coupler_freqs[k]
                transmon_eff_freqs[i] += (g[i,k]**2 * (1/delta_ik - 1/sigma_ik))

        return transmon_eff_freqs

    def full_hamiltonian(self, dim : int | List) -> Qobj:
        """Generates a QuTiP Qobj operator that is the Hamiltonian of the transmon and resonator network.
        The annihilation/creation operators are truncated based on the input dimension.

        Parameters
        ----------
        dim : int | List
            Hilbert dimension for each branch mode. Specified either for all the branches, or individually.

        Returns
        -------
        Qobj
            Full hamiltonian of the transmon and resonator network.
        """
        # Specify the Hilbert dimension for all the branches
        if isinstance(dim, int):
            dim = (self.N + self.M)*[dim]

        if len(dim) != (self.N + self.M):
            raise ValueError(f"Number of specified Hilbert space dimensions should be equal to N ({self.N})")
        
        id = [identity(dim[i]) for i in range(self.N + self.M)]

        # Setup the annihilation and creation operators for each branch mode
        a = []
        ad = []
        for i in range(self.N + self.M):
            a_i = tensor( [id[j] for j in range(0,i)] + [destroy(dim[i])] + [id[j] for j in range(i+1,self.N+self.M)] )
            a.append(a_i)
            ad.append(a_i.dag())

        H = 0

        # Transmon resonance frequency terms
        for i, w in enumerate(self.transmon_frequencies):
            H += w * ad[i] * a[i]

        # Transmon anharmonicity terms
        for i in range(self.N):
            H += 0.5 * self.transmon_anharmonicity(i) * ad[i] * ad[i] * a[i] * a[i]

        # Resonator frequency terms
        for k, w in enumerate(self.resonator_frequencies):
            H += w * ad[self.N + k] * a[self.N + k]

        # Coupling terms
        g = self.g_matrix
        for i in range(self.N + self.M):
            for j in range(i+1, self.N + self.M):
                H += g[i,j] * (ad[i]*a[j] + a[i]*ad[j] - ad[i]*ad[j] - a[i]*a[j])

        return a, ad, H

    @effective
    def effective_hamiltonian(self, Nt :int, dim : int | List) -> Qobj:
        """Generates a QuTiP Qobj operator that is the Hamiltonian of the effective transmon network.
        The annihilation/creation operators are truncated based on the input dimension.

        Parameters
        ----------
        Nt : int
            Number of transmons to use as "computational" qubits. The rest will be used as couplers.
        dim : int | List
            Hilbert dimension for each transmon mode. Specified either for all the branches, or individually.

        Returns
        -------
        Qobj
            Full hamiltonian of the effective transmon network.
        """
        # Specify the Hilbert dimension for all the branches
        if isinstance(dim, int):
            dim = (Nt)*[dim]

        if len(dim) != (Nt):
            raise ValueError(f"Number of specified Hilbert space dimensions should be equal to Nt ({Nt})")
        
        id = [identity(dim[i]) for i in range(Nt)]

        # Setup the annihilation and creation operators for each branch mode
        a = []
        ad = []
        for i in range(Nt):
            a_i = tensor( [id[j] for j in range(0,i)] + [destroy(dim[i])] + [id[j] for j in range(i+1,Nt)] )
            a.append(a_i)
            ad.append(a_i.dag())

        H = 0

        # Transmon resonance frequency terms
        for i, w in enumerate(self.transmon_eff_frequencies(Nt)):
            H += w * ad[i] * a[i]

        # Transmon anharmonicity terms
        for i in range(Nt):
            H += 0.5 * self.transmon_anharmonicity(i) * ad[i] * ad[i] * a[i] * a[i]

        # Coupling terms
        g = self.g_eff_coupling(Nt)
        for i in range(Nt):
            for j in range(i+1, Nt):
                H += g[i,j] * (ad[i] * a[j] + a[i]*ad[j] - ad[i]*ad[j] - a[i]*a[j])

        return a, ad, H

    @decay
    def compute_lossy_poles(self, Ze : int | np.ndarray = 50) -> np.ndarray:
        """Compute the poles of the current network for a given set of impedances placed at the external ports.

        Parameters
        ----------
        Ze : np.ndarray
            Impedances shunting the external ports of the network. Should correspond to the characteristic impedances of the ports.

        Returns
        -------
        np.ndarray
            The poles of the lossy network
        """

        if isinstance(Ze, int):
            Ze = [Ze]*self.Ne
        Ze = np.array(Ze)

        if len(Ze) != self.Ne:
            raise ValueError(f'Number of port characteristic impedances should match the number of external ports ({self.Ne})')

        N = self.N
        Ne = self.Ne
        M = self.M
        K = N + Ne + M
    
        # Initialize the inverse inductance matrix using the qubit/resonator inductances with no inductances on the external ports
        M_ind = np.diag(np.concatenate((1/self.L[:N], [0]*Ne, 1/self.L[N:])))
        
        # Initialize the inverse impedance matrix for the resistors at the external ports
        Z_inv = np.zeros((K, K))
        Z_inv[N:N+Ne, N:N+Ne] = np.diag(1/Ze)

        # Construct the matrix that we diagonalize to find the poles
        A = np.zeros((2*K, 2*K))
        zero = np.zeros((K, K))
        eye = np.eye((K))

        A[:K, :K] = zero
        A[:K, K:] = eye
        C_inv = np.linalg.inv(self.C_full)
        A[K:, :K] = -C_inv @ M_ind
        A[K:, K:] = -C_inv @ Z_inv
        poles = np.linalg.eigvals(A)
        
        sort_idx = np.argsort(poles.imag)[::-1]
        poles = poles[sort_idx]

        return poles

    @decay
    def fully_shunted_impedance(
        self, 
        f_re : np.ndarray, 
        f_im : np.ndarray, 
        Ze : int | np.ndarray = 50
    ) -> np.ndarray:
        """Computes the impedance of the CL cascade network for a 2D-grid of complex frequencies.
        Qubits are shunted with their corresponding inductances. External ports are shunted with their 
        corresponding characteristic impedances. Returns 

        Parameters
        ----------
        f_re : np.ndarray
            Real part of the complex frequencies to compute the impedance at.
        f_im : np.ndarray
            Imaginary part of the complex frequencies to compute the impedance at.
        Ze : int | np.ndarray
            Impedances shunting the external ports of the network. Should correspond to the characteristic impedances of the ports.

        Returns
        -------
        np.ndarray
            The multi-port impedance parameter for the fully shunted network.
        """
        # ! NOTE : this needs to be done this way since we need to be able to *compute* at an arbitrary complex frequency

        if isinstance(Ze, int):
            Ze = [Ze]*self.Ne

        N = self.N
        Ne = self.Ne
        N_tot = N + Ne
        M = self.Z_full.M
        Z_full = np.zeros((N_tot, N_tot, len(f_re), len(f_im)), dtype=np.complex128)

        def abcd_to_s(a,b,c,d):
            # For a reciprocal 2-port network
            denom = a + (b/50) + c*50 + d
            S = np.array([
                [a + (b/50) - c*50 - d, 2],
                [2, -a + (b/50) - c*50 + d]
            ])/denom
            return S

        def parallel_Y_to_S(Y):
            # Computes the S parameter for a 2 port with admittance shunt (single frequency)
            return abcd_to_s(1, 0, Y, 1)

        # Setup the shunt scattering parameter and add the external port impedances
        S_shunt = np.zeros((2*N_tot, 2*N_tot), dtype=np.complex128)
        for i in range(len(Ze)):
            S_res = parallel_Y_to_S(1/Ze[i])
            S_shunt[N + i, N + i] = S_res[0,0]
            S_shunt[N + i + N_tot, N + i] = S_res[1,0]
            S_shunt[N + i, N + i + N_tot] = S_res[0,1]
            S_shunt[N + i + N_tot, N + i + N_tot] = S_res[1,1]

        eye = np.eye(N_tot)

        # Add the shunt inductances for the given complex frequency and compute the impedance for each contribution
        for i in tqdm(range(len(f_re))):
            for j in range(len(f_im)):
                
                # Compute the impedance parameter of the unshunted network at a complex frequency
                f = np.array([-1j*f_re[i] + f_im[j]])
                s = 2j * const.pi * f[0]
                Z = self.Z_full(f, show_progress=False)
                S = rf.z2s(Z)[0,:,:]

                # Add the qubit inductances to S_shunt for the given complex frequency s
                for k in range(N):
                    S_L = parallel_Y_to_S(1/(s * self.L[k]))
                    S_shunt[k,k] = S_L[0,0]
                    S_shunt[N_tot + k,k] = S_L[1,0]
                    S_shunt[k, k + N_tot] = S_L[0,1]
                    S_shunt[k + N_tot, k + N_tot] = S_L[1,1]

                S_full = S_shunt[:N_tot, :N_tot] + \
                    S_shunt[:N_tot, N_tot:] @ S @ np.linalg.inv(eye - S_shunt[N_tot:, N_tot:] @ S) @ S_shunt[N_tot:, :N_tot]
                Z_full[:,:,i,j] = rf.s2z(np.array([S_full]))
        
        return Z_full

    @staticmethod
    def lossy_admittance(
        f : np.ndarray, 
        Z : np.ndarray, 
        Nt : int, 
        Ze : int | np.ndarray = 50, 
        Lt : np.ndarray = None, 
        transmon_idx = None
    ) -> np.ndarray:
        """Computes the admittance for a given discretized impedance function. If the transmon inductances are provided,
        computes the one-port admittance for a selected transmon index where that transmon port is not shunted with an inductor.
        If the transmon inductances are not provided, the only the external ports are shunted with characteristic impedances.

        Parameters
        ----------
        f : np.ndarray
            Frequency range corresponding to the passed in discretized impedance function.
        Z : np.ndarray
            Discretized multi-port impedance function.
        Nt : int
            Number of ports corresponding to the transmons in the network. The first Nt ports, should correspond to transmons.
        Ze : int | np.ndarray, optional
            Impedances shunting the external ports of the network. 
            Should correspond to the characteristic impedances of the ports, by default 50
        Lt : np.ndarray, optional
            Transmon inductances. if not provided, will not shunt the transmon ports, by default None
        transmon_idx : _type_, optional
            If Lt is provided, this index will leave one transmon port not shunted by an inductance.
            Then a single-port admittance is returned, by default None

        Returns
        -------
        np.ndarray
            The multi-port or single-port lossy admittance parameter for the shunted network.
            Multi-port if transmons are not shunted with inductors.
            Single-port if all but one of the transmon ports are shunted with an inductance.
        """
        
        if (transmon_idx is not None) != (Lt is not None):
            raise ValueError('If including transmon inductances, both Lt and transmon_idx need tot be supplied.')

        if (Lt is not None) and (len(Lt) != Nt):
            raise ValueError('Number of transmon inductances in Lt should match the number of passed in transmons Nt.')

        N_tot = Z.shape[1]
        N = Nt
        Ne = N_tot - N

        if isinstance(Ze, int):
            Ze = [Ze]*Ne
        
        # Initialize Z-parameter for the external port shunts (with impedances in Ze)
        Z_ext = np.zeros((len(f), Ne, Ne))
        for i in range((len(Ze))):
            Z_ext[:,i,i] = Ze[i]*np.ones(len(f))

        # Shunt the external ports with the impedances Ze
        S = rf.z2s(Z)
        S_ext = rf.z2s(Z_ext)
        S_full = np.zeros((len(f), N_tot + Ne, N_tot + Ne), dtype=np.complex128)
        S_full[:, :N_tot, :N_tot] = S.copy()
        S_full[:, N_tot:, N_tot:] = S_ext.copy()
        S_full = rf.Network(frequency=f, s=S_full)
        S_full = rf.network.innerconnect(S_full, N, N_tot, Ne)

        # Close off the remaining transmon ports with inductances if required.
        # NOTE: This leaves the transmon_idx port open, no inductance should be across it when sweeping frequency.
        num_ind_connected = 0
        if transmon_idx is not None:
            for i, L in enumerate(Lt):
                if i == transmon_idx:
                    continue
                
                Z_ind = np.array(2j*const.pi*f*L)[:, None, None]
                S_ind = rf.z2s(Z_ind)
                net_ind = rf.Network(frequency=f, s=S_ind)
                S_full = rf.connect(S_full, i - num_ind_connected, net_ind, 0)
                num_ind_connected += 1

        Y_full = rf.s2y(S_full.s)
        return Y_full

    @decay
    def transmon_decay_rate(
        self,
        f : np.ndarray, 
        Ze : int | np.ndarray = 50,
        shunt_transmons = False,
        transmon_idx = None,
        use_eff_C = True
    ) -> np.ndarray:
        """Computes the decay rates of one or all the transmons for a given frequency range.
        If shunt_transmons is True, a single transmon is selected for decay rate computation using transmon_idx.
        Otherwise, all the transmon decay rates are estimated without shunting the transmon ports with inductances.

        Parameters
        ----------
        f : np.ndarray
            Frequency range over which to compute the decay rate.
        Ze : int | np.ndarray, optional
            Impedances shunting the external ports of the network. 
            Should correspond to the characteristic impedances of the ports, by default 50
        shunt_transmons : bool, optional
            Indicates whether to shunt transmons and only compute the decay rate of a single transmon, by default False
        transmon_idx : _type_, optional
            The transmon index for which to compute the decay rate if the other transmons are shunted by an inductance, by default None
        use_eff_C : bool, optional
            Indicates whether to use effective C in decay computation, by default True
            
        Returns
        -------
        np.ndarray
            Decay rate estimate over the chosen frequency range for one or all of the transmons.
        """
        
        if shunt_transmons and (transmon_idx is None):
            raise ValueError('If transmons are to be shunted by inductors, one must be selected to be left open.')
        
        lossy_Y = TransmonNetwork.lossy_admittance(
            f, self.Z_full(f, show_progress=False), self.N, Ze, 
            Lt = self.L[:self.N] if shunt_transmons else None, transmon_idx = transmon_idx
        )

        decay = None

        if shunt_transmons:
            C = self.eff_C(transmon_idx) if use_eff_C else self.C_full[transmon_idx,transmon_idx]
            decay = np.real(lossy_Y) / C
            return decay[:,0,0]
        
        decay = np.zeros((len(f), self.N))
        for i in range(self.N):
            C = self.eff_C(i) if use_eff_C else self.C_full[i,i]
            decay[:,i] = np.real(lossy_Y[:,i,i]) / C
        return decay