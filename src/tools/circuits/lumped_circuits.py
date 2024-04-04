from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm import tqdm

from tools.circuits.impedance import ImpedanceLR
from tools.circuits.capacitance import mutual_to_maxwell

class CascadeCL(object):
    """Constructs the impedance function for a lossless lumped element circuit of the following form:

              _________           _________ 
        o----|         |----o----|         |
             |         |         |         |
          N  |   N_C   |  M   M  |   N_L   |
             |         |         |         |
        o----|         |----o----|         |
              ‾‾‾‾‾‾‾‾‾           ‾‾‾‾‾‾‾‾‾

    where N_C is a purely capacitive (N+M)-port network and N_L is a purely inductive M-port
    network with one inductor across each port. NOTE: M should be less than N.
    
    The circuit is described by the following:
        - A (N+M)x(N+M) matrix corresponding to the capacitance matrix of the network N_C
        - A list of M inductances corresponding to the inductances of the network N_L
    
    Attributes
    ----------
    N : int
        The number of ports of the fully cascaded network.
    M : int
        The number of ports in the network N_L (also the number of inductors in the circuit).
    C : np.ndarray
        The capacitance matrix in Maxwell form for the network N_C.
        First N ports (first N rows of C) correspond to the ports of the fully cascaded network, and the last
        M ports should be the ports of the inductors.
    L : np.ndarray
        The inductances across the ports in N_L.
    rational_Z : ImpedanceLR
        Holds the poles, residues and turns ratios for the rational impedance function
    """ 

    def __init__(self, C : np.ndarray, L : np.ndarray = [], C_type : str = 'maxwell') -> None:
        """Saves the maxwell capacitance matrix and generates the poles and residues of the N-port impedance function.

        Parameters
        ----------
        C : np.ndarray
            The capacitance matrix in Maxwell or Mutual/SPICE form for the network N_C. Should be a 2d square matrix.
            First N ports (first N rows of C) correspond to the ports of the fully cascaded network, and the last
            M ports should be the ports of the inductors.
        L : np.ndarray
            The inductances across the ports in N_L. Should be a 1d array.
        C_type : str
            Specifies the form of the capacitance matrix passed in. Can be 'maxwell', 'mutual', or 'spice'.
            By default 'maxwell'
        """

        C_type = C_type.lower()
        if C_type not in ['maxwell', 'spice', 'mutual']:
            raise ValueError('C_type must be either "maxwell", "mutual", or "spice".')
        
        C = np.array(C)
        if C.shape[0] != C.shape[1]:    
            raise ValueError('C must be a square matrix')

        if not np.allclose(C, C.T):
            raise ValueError('C must be symmetric')

        L = np.array(L)
        if len(L.shape) != 1:
            raise ValueError('L must be a 1d array')

        # Make sure N_C has more ports than N_L.
        self.M = L.shape[0]
        self.N = C.shape[0] - self.M
        if self.N <= 0:
            raise ValueError('N_C should have more ports than N_L.')

        C_type = C_type.lower()
        if C_type == 'spice' or C_type == 'mutual':
            C = mutual_to_maxwell(C)
    
        # Save the initial circuit parameters
        self.C = C
        self.L = L

        # Generate the parameters for the rational impedance function
        # NOTE: numerical errors can come up in the following process due to the fact that numerically inverting the
        # symmetric C0 can give a slightly non-symmetric C0_inv.
        C0 = self.C
        C0_inv = np.linalg.inv(C0)
        M = np.diag(np.concatenate((np.zeros(self.N), 1/self.L)))

        C0_inv_R = C0_inv[self.N:, self.N:]
        M_R = M[self.N:, self.N:]

        # Diagonalize the pair (C0_inv_R, M_R) such that the diagonal matrix W2 contains the squares of the poles
        # First diagonalize the resonator block of the inverse capacitance matrix
        D, Oc = np.linalg.eigh(C0_inv_R)
        T = Oc @ np.sqrt(np.diag(D))

        # Diagonalize the matrix T.T @ M_R @ T such that W2 contains the squares of the poles
        W2, Om = np.linalg.eigh(T.T @ M_R @ T)

        # Construct the transformation that converts M_R to the diagonal matrix W2
        S = T @ Om
        S_full = np.eye(self.N + self.M)
        S_full[self.N:, self.N:] = np.real(S)

        # Construct the capacitance matrix in the new basis and extract the non-zero poles
        C1 = S_full.T @ C0 @ S_full
        poles = list(np.concatenate(([0], np.real(np.sqrt(W2)))))
        # poles = list(np.real(np.sqrt(W2)))

        # Extract DC residue from the leftover purely capacitive portion
        Cj = C1[:self.N, :self.N]
        R0 = np.linalg.inv(Cj)
        
        # Extract the turns ratios in the Cauer network corresponding to our cascaded network
        R = (-np.linalg.inv(Cj) @ C1[:self.N, self.N:]).T

        self.rational_Z = ImpedanceLR(R0, poles, R)

    def rational_Z_discrete(self, f : np.ndarray, show_progress : bool = True) -> np.ndarray:
        """Evaluates the rational impedance function at the given frequencies using the 
        computed poles and residues.

        Parameters
        ----------
        f : np.ndarray
            The frequencies to evaluate the impedance at
        show_progress : bool
            Use to toggle progress bar

        Returns
        -------
        Z : np.ndarray
            The rational impedance function evaluated at the given frequencies
        """
        return self.rational_Z(f, show_progress)
    
    def cascade_Z_discrete(self, f : np.ndarray, show_progress : bool = True) -> np.ndarray:
        """Evaluates the impedance of the fully cascaded network at the given frequencies
        using the cascade method (see Newcomb, Linear Multiport Synthesis, 1996 Eq. 3-20(a-c))

        Parameters
        ----------
        f : np.ndarray
            The frequencies to evaluate the impedance at
        show_progress : bool
            Use to toggle progress bar

        Returns
        -------
        Z : np.ndarray
            The impedance of the fully cascaded network at the given frequencies
        """
        s = 2j * np.pi * f
        Z = np.zeros((len(f), self.N, self.N), dtype=np.complex128)

        C_inv = np.linalg.inv(self.C)

        pbar = tqdm(total=len(f), ncols=100, disable = not show_progress)
        pbar.set_description('Computing Cascaded Z')
        for i in range(len(f)):
            # Compute the Z parameter N_C and N_L
            Z_c = C_inv/s[i]
            Z_l = s[i]*np.diag(self.L)
            
            # Compute the S parameters of N_C and N_L from the Z parameters
            S_c = np.linalg.inv(Z_c + np.eye(self.N + self.M)) @ (Z_c - np.eye(self.N + self.M))
            S_l = np.linalg.inv(Z_l + np.eye(self.M)) @ (Z_l - np.eye(self.M))

            # Cascade-load N_C with N_L and conver the resulting S paramter back to Z
            S11 = S_c[:self.N, :self.N]
            S12 = S_c[:self.N, self.N:]
            S21 = S_c[self.N:, :self.N]
            S22 = S_c[self.N:, self.N:]
            S_final = S11 + S12 @ S_l @ np.linalg.inv(np.eye(self.M) - S22 @ S_l) @ S21
            Z_final = (np.eye(self.N) + S_final) @ np.linalg.inv(np.eye(self.N) - S_final)
            Z[i,:,:] = Z_final

            pbar.update(1)
        pbar.close()

        return Z

    def print_poles_residues(self) -> None:
        self.rational_Z.print_poles_residues()
        
    def plot_Z_compare_amp_phase(self, f : np.array, show_progress : bool = True, show_legend : bool = True) -> Tuple[Figure, np.ndarray]:
        Z_rational = self.rational_Z_discrete(f, show_progress)
        Z_cascade = self.cascade_Z_discrete(f, show_progress)
        f = f/1e9
        np_err = np.geterr()
        np.seterr(divide='ignore')

        fig, ax = plt.subplots(2, 3, figsize=(17, 11))
        fig.tight_layout(pad=4)
        for i in range(self.N):
            for j in range(i, self.N):
                ax[0,0].plot(f, np.log10(np.abs(Z_rational[:,i,j])), label=f'Z{i+1}{j+1}')
                ax[0,0].set_title('Rational Impedance - Prediction')
                ax[0,0].set_ylabel(r'$\log_{10}(|Z_{ij}^R|)$')

                ax[0,1].plot(f, np.log10(np.abs(Z_cascade[:,i,j])), label=f'Z{i+1}{j+1}')
                ax[0,1].set_title('Cascaded Impedance')
                ax[0,1].set_ylabel(r'$\log_{10}(|Z_{ij}^C|)$')
                
                ax[0,2].plot(f, np.log10(np.abs(Z_rational[:,i,j]-Z_cascade[:,i,j])), label=f'Z{i+1}{j+1}')
                ax[0,2].set_title('Difference')
                ax[0,2].set_ylabel(r'$\log_{10}(|Z_{ij}^R - Z_{ij}^C|)$')

                ax[1,0].plot(f, np.angle(Z_rational[:,i,j]), label=f'Z{i+1}{j+1}')
                ax[1,0].set_ylabel(r'$\arg Z_{ij}^R$')

                ax[1,1].plot(f, np.angle(Z_cascade[:,i,j]), label=f'Z{i+1}{j+1}')
                ax[1,1].set_ylabel(r'$\arg Z_{ij}^C$')

                ax[1,2].plot(f, np.log10(np.abs(np.angle(Z_rational[:,i,j])-np.angle(Z_cascade[:,i,j]))), label=f'Z{i+1}{j+1}')
                ax[1,2].set_ylabel(r'$\log_{10}(|\arg Z_{ij}^R - \arg Z_{ij}^C|)$')

        for i,j in np.ndindex(ax.shape): 
            if show_legend: 
                ax[i,j].legend(loc='upper right')
            ax[i,j].grid()
            
        np.seterr(**np_err)
        return fig, ax

    def plot_Z_compare_real_imag(self, f : np.array, show_progress : bool = True, show_legend : bool = True) -> Tuple[Figure, np.ndarray]:
        Z_rational = self.rational_Z_discrete(f, show_progress)
        Z_cascade = self.cascade_Z_discrete(f, show_progress)
        f = f/1e9
        np_err = np.geterr()
        np.seterr(divide='ignore')
        
        fig, ax = plt.subplots(2, 3, figsize=(17, 11))
        fig.tight_layout(pad=4)
        for i in range(self.N):
            for j in range(i, self.N):
                ax[0,0].plot(f, np.log10(np.abs(np.real((Z_rational[:,i,j])))), label=f'Z{i+1}{j+1}')
                ax[0,0].set_title('Rational Impedance - Prediction')
                ax[0,0].set_ylabel(r'$\log_{10}(|\operatorname{Re}({Z_{ij}^R})|)$')

                ax[0,1].plot(f, np.log10(np.abs(np.real((Z_cascade[:,i,j])))), label=f'Z{i+1}{j+1}')
                ax[0,1].set_title('Cascaded Impedance')
                ax[0,1].set_ylabel(r'$\log_{10}(|\operatorname{Re}(Z_{ij}^C)|)$')
                
                ax[0,2].plot(f, np.log10(np.abs(np.real(Z_rational[:,i,j])-np.real(Z_cascade[:,i,j]))), label=f'Z{i+1}{j+1}')
                ax[0,2].set_title('Difference')
                ax[0,2].set_ylabel(r'$\log_{10}(|\operatorname{Re}(Z_{ij}^R - Z_{ij}^C)|)$')

                ax[1,0].plot(f, np.log10(np.abs(np.imag(Z_rational[:,i,j]))), label=f'Z{i+1}{j+1}')
                ax[1,0].set_ylabel(r'$\log_{10}(|\operatorname{Im}({Z_{ij}^R})|)$')

                ax[1,1].plot(f, np.log10(np.abs(np.imag(Z_cascade[:,i,j]))), label=f'Z{i+1}{j+1}')
                ax[1,1].set_ylabel(r'$\log_{10}(|\operatorname{Im}({Z_{ij}^R})|)$')

                ax[1,2].plot(f, np.log10(np.abs(np.imag(Z_rational[:,i,j])-np.imag(Z_cascade[:,i,j]))), label=f'Z{i+1}{j+1}')
                ax[1,2].set_ylabel(r'$\log_{10}(|\operatorname{Im}(Z_{ij}^R - Z_{ij}^C)|)$')

        for i,j in np.ndindex(ax.shape): 
            if show_legend: 
                ax[i,j].legend(loc='upper right')
            ax[i,j].grid()
            
        np.seterr(**np_err)
        return fig, ax