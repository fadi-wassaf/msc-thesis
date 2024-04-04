# elliptic integrals?
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from scipy.special import ellipk
import scipy.constants as const

from tools.circuits.impedance import ImpedanceLR
from tools.circuits.lumped_circuits import CascadeCL

def cpw_parameters(
    S : float, 
    W : float, 
    eps_sub: float
) -> Tuple[float, float]:
    """Obtain the L and C per unit length parameters for a CPW

    Parameters
    ----------
    S : float
        CPW central line width
    W : float
        CPW slit/gap width
    eps_sub : float
        Substrate relative dielectric constant

    Returns
    -------
    Tuple[float, float]
        The inductance (L) and capacitance (C) per unit length for the CPW
    """
    k = S/(S+2*W)
    kp = np.sqrt(1 - k**2)
    K_k = ellipk(k)
    K_kp = ellipk(kp)
    eps_eff = (1 + eps_sub)/2

    # Compute L and C per unit length
    L = const.mu_0 * K_kp / (4 * K_k)
    C = 4 * const.epsilon_0 * eps_eff * K_k / K_kp
    return L, C

class LosslessTL(object):

    def __init__(
        self, 
        k : int, 
        L : float, 
        C : float, 
        l : float, 
        C1 : float, 
        C2 : float
    ):
        """Initializes the rational impedance function for an ideal transmission line with two in series
        capacitive elements for fixed number of poles.

                   C1                               C2
            o------||------o-----------------o------||------o
                                 Ideal TL
            o--------------o-----------------o--------------o
                            ^---------------^
                            beta = w*sqrt(LC)
                               Z0=sqrt(L/C)
                                 length=l  
        
        Parameters
        ----------
        k : int
            Number of poles to include in the rational model
        L : float
            Inductance per unit length of the TL
        C : float
            Capacitance per unit length of the TL
        l : float
            Length of the transmission line
        C1 : float
            In series capacitance on the first port
        C2 : float
            In series capacitance on the second port
        """
        self.k = k
        self.L = L
        self.C = C
        self.l = l
        self.C1 = C1
        self.C2 = C2

        self.Z0 = np.sqrt(self.L/self.C)
        self.Y0 = 1/self.Z0
        self.Ct = self.l * np.sqrt(self.L * self.C) / self.Z0

        R0 = np.array([
            [(1/self.C1) + (1/self.Ct), 1/self.Ct],
            [1/self.Ct, (1/self.C2) + (1/self.Ct)]
        ])

        poles = [np.pi * i / (np.sqrt(self.L * self.C) * l) for i in range(1,k+1)]
            
        R = []
        for i in range(1, k+1):
            R.append(np.sqrt(2/self.Ct)*np.array([1, (-1)**i]))
        R = np.array(R)
        
        self.rational_Z = ImpedanceLR(R0, poles, R)

    def discrete_Z_from_rational(self, f : np.ndarray) -> np.ndarray:
        # Computes the impedance from the rational finite pole approximation
        return self.rational_Z(f)

    def discrete_ABCD(self, f):
        # Computes the ABCD matrices of the ideal transmission line with the series capacitors on the end
        w = 2*np.pi*f
        beta = w * np.sqrt(self.L * self.C)
        bl = beta * self.l
        ones = np.ones(len(w))
        zeros = np.zeros(len(w))

        ABCD_C1 = np.transpose(np.array([[ones, -1j/(w*self.C1)], [zeros, ones]]), (2, 0, 1))
        ABCD_C2 = np.transpose(np.array([[ones, -1j/(w*self.C2)], [zeros, ones]]), (2, 0, 1))
        ABCD_TL = np.transpose(np.array([
            [np.cos(bl), 1j*self.Z0*np.sin(bl)],
            [1j*self.Y0*np.sin(bl), np.cos(bl)]
        ]), (2, 0, 1))
        
        ABCD = np.einsum("aij, ajk -> aik", ABCD_TL, ABCD_C2)
        ABCD = np.einsum("aij, ajk -> aik", ABCD_C1, ABCD)
        return ABCD

    def discrete_Z_from_ABCD(self, f):
        # Computes the impedance using the ABCD matrices of the network
        ABCD = self.discrete_ABCD(f)
        A, B, C, D = ABCD[:,0,0], ABCD[:,0,1], ABCD[:,1,0], ABCD[:,1,1]
        Z = np.zeros((len(f), 2, 2), dtype=np.complex128)
        Z[:,0,0] = A/C
        Z[:,0,1] = 1/C
        Z[:,1,0] = Z[:,0,1]
        Z[:,1,1] = D/C
        return Z
    
    def discrete_S_from_ABCD(self, f):
        # Computes the S parameter using the ABCD matrices of the network
        Z0 = 50
        ABCD = self.discrete_ABCD(f)
        A, B, C, D = ABCD[:,0,0], ABCD[:,0,1], ABCD[:,1,0], ABCD[:,1,1]
        S = np.zeros((len(f), 2, 2), dtype=np.complex128)
        denom = A + (B/Z0) + C*Z0 + D
        S[:,0,0] = (A + (B/Z0) - C*Z0 - D)/denom
        S[:,0,1] = 2/denom
        S[:,1,0] = S[:,0,1]
        S[:,1,1] = (-A + (B/Z0) - C*Z0 + D)/denom
        return S

    @classmethod
    def from_cpw_geometry(
        cls, 
        num_poles : int, 
        S : float, 
        W : float, 
        eps_sub: float, 
        l : float,
        C1 : float, 
        C2 : float
    ) -> 'LosslessTL':
        """Obtains the rational TL model approximation for a CPW.

        Parameters
        ----------
        num_poles : int
            Number of poles to use for the finite pole approximation
        S : float
            CPW central line width
        W : float
            CPW slit/gap width
        eps : float
            Substrate relative dielectric constant
        l : float
            Length of the transmission line
        C1 : float
            In series capacitance on the first port
        C2 : float
            In series capacitance on the second port
        """
        L, C = cpw_parameters(S, W, eps_sub)
        return cls(num_poles, L, C, l, C1, C2)
