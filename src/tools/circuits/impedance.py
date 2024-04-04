from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm import tqdm

from tools.circuits.capacitance import maxwell_to_mutual

class ImpedanceLR(object):
    """Holds the information for a rational lossless reciprocal impedance function with no pole at infinity:

        Z(s) = (R0/s) + sum( s*Rk/(s^2 + w_k^2) )

    Attributes
    ----------
    N : int
        Number of ports in the circuit
    M : int
        Number of resonant modes in the circuit
    poles : List[float]
        List of the nonzero poles of the impedance function w_k.
        NOTE: First pole is the DC pole w=0.
    residues : List[np.ndarray]
        List of the residues corresponding to the poles.
        NOTE: First residue corresponds to the DC pole.
    R0 : np.ndarray
        The DC residue. Also the first entry in the residues list. This is extra.
    R : np.ndarray
        MxN matrix containing turns ratios of the Cauer network representation of the impedance
    """

    def __init__(
        self, 
        R0 : np.ndarray, 
        poles : List[float] = [], 
        R : np.ndarray = [],
        name : str = "Z",
        port_names : List[str] = None
    ):
        """Geneters the residues from the turns ratio matrix R.

        Parameters
        ----------
        R0 : np.ndarray
            The NxN DC residue of the impedance function.
        poles : List[float]
            Length of the of poles should be one less or equal to the number of resonant modes M.
            If one less, the zero pole is added at the beginning.
        R : np.ndarray
            The MxN matrix containing the turns ratios of the resonant stage in the Cauer network 
            representation of the impedance. Will be used to generate the residues.
        name : str, optional
            Can be useful in port interconnection, by default "Z"
        port_names : List[str], optional
            Can be useful in keeping track of ports, will default to numbered ports
        """
        if R0.shape[0] != R0.shape[1]:
            raise ValueError("R0 should be a square matrix")
        
        if not np.allclose(R0, R0.T):
            raise ValueError("R0 should be symmetric.")

        self.N = R0.shape[0]
        self.R0 = R0
        self.residues = [R0]
        
        if np.asarray(poles).size != 0:
            if R.shape[1] != self.N:
                raise ValueError(f"R should have {self.N} columns. Instead has {R.shape[1]}.")

            if poles[0] != 0:
                poles = [0] + poles

            self.M = len(poles) - 1
            self.poles = poles

            if R.shape[0] != len(poles) - 1:
                raise ValueError(f"R should have {self.M} rows. Instead has {R.shape[0]}.")

            self.R = R
            
            # Generate the residues of the impedance function
            for i in range(self.M):
                self.residues.append(np.outer(self.R[i, :], self.R[i, :]))

        else:
            self.M = 0
            self.poles = [0]
            self.R = np.zeros((self.M, self.N))

        self.name = name
        self.port_names = port_names if port_names is not None else [str(i) for i in range(self.N)]

    def copy(self):
        return ImpedanceLR(self.R0, self.poles, self.R, self.name, self.port_names)

    def Z(self, f : np.ndarray, show_progress : bool = True) -> np.ndarray:
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
        s = 2j * np.pi * f
        s_full = np.transpose(np.tile(s,(self.N, self.N, 1)), (2,0,1))
        Z = np.zeros((len(f), self.N, self.N), dtype=np.complex128)
        pbar = tqdm(total=len(self.poles), ncols=100, disable = not show_progress)
        pbar.set_description('Computing Rational Z')
        for j in range(len(self.poles)):
            Z += s_full * self.residues[j] / np.transpose(np.tile((s**2 + self.poles[j]**2),(self.N, self.N, 1)), (2,0,1))
            pbar.update(1)
        pbar.close()

        return Z
        
    def __call__(self, f : np.ndarray, show_progress : bool = True) -> np.ndarray:
        return self.Z(f, show_progress)

    def print_poles_residues(self) -> None:
        for i in range(len(self.poles)):
            print(f'Pole {i+1}: {self.poles[i]/(2*np.pi)/1e9} GHz')
            print(f'Residue {i+1} w/ rank {np.linalg.matrix_rank(self.residues[i])}:\n{self.residues[i]}\n')

    @property
    def number_of_ports(self) -> int:
        return self.N

    @property
    def maxwell_capacitance(self) -> np.ndarray:
        """Constructs and returns the capacitance matrix for the CL cascade synthesis representation 
        of the rational impedance function.

        Returns
        -------
        np.ndarray
            (N+M)x(N+M) Maxwell form capacitance matrix corresponding to the rational impedance.
        """
        R0_inv = np.linalg.inv(self.R0)
        C = np.vstack((
            np.hstack((R0_inv, -R0_inv @ self.R.T)),
            np.hstack((-self.R @ R0_inv, np.eye((self.M)) + self.R @ R0_inv @ self.R.T))
        ))
        return C

    @property
    def mutual_capacitance(self) -> np.ndarray:
        return maxwell_to_mutual(self.maxwell_capacitance)

    def shunt_inductors(self) -> np.ndarray:
        """Returns a list containing the shunt inductances for the CL cascade synthesis representation 
        of the rational impedance function.

        Returns
        -------
        np.ndarray
            Array containing the shunt inductances
        """
        L = 1/(self.poles[1:])**2

    def rearrange_ports(self, idxs : List) -> 'ImpedanceLR':
        """Rearrange the ports of the rational impedance function given a new index ordering.

        Parameters
        ----------
        idxs : List
            New port index ordering.

        Returns
        -------
        ImpedanceLR
            The rational impedance function with the rearrange ports
        """
        if len(idxs) != len(set(idxs)):
            raise ValueError(f'Make sure no port indices are repeated in {idxs}')

        if set(idxs) != set(range(self.N)):
            raise ValueError(f'Make sure all ports {set(range(self.N))} are included in the shuffle.')
        
        # For the DC residue, rows and ports must be shuffled.
        R0 = self.R0[idxs, :]
        R0 = R0[:, idxs]

        # For the turns ratio matrix, just the columns need to be shuffled.
        R = self.R[:, idxs]
        
        return ImpedanceLR(R0, self.poles, R)

    def bring_to_front(self, idxs : List) -> 'ImpedanceLR':
        """Rearrange the ports such that the passed in indices are moved to the beginning.

        Parameters
        ----------
        idxs : List
            Indices of ports to move to the front

        Returns
        -------
        ImpedanceLR
            Rational impedance with the rearranged ports
        """
        new_idxs = list(range(self.N))
        new_idxs = idxs + [i for i in new_idxs if i not in idxs]
        return self.rearrange_ports(new_idxs)

    def truncate_ports(self, N_keep : int) -> 'ImpedanceLR':
        """Truncates the rational impedance by keeping all but the first N_keep ports open

        Parameters
        ----------
        N_keep : int
            Number of ports to keep

        Returns
        -------
        ImpedanceLR
            The rational impedance function only N_keep ports left, other ports left open
        """
        R0_new = self.R0[:N_keep, :N_keep]
        R_new = self.R[:, :N_keep] 
        return ImpedanceLR(R0_new, self.poles, R_new)


    @classmethod
    def args_list_to_ImpedanceLR(
        cls, 
        num_ports : int, 
        num_res : int, 
        args : List[float], 
        fixed_poles : List[float] = None
    ) -> 'ImpedanceLR':
        """This function is used as a wrapper to convert from the format of a list into an ImpedanceLR object.

        Parameters
        ----------
        num_ports : int
            Number of ports in the circuit
        num_res : int
            Number of residues of the impedance function 
        args : List[float]
            List containing all the information about the poles and residues of the impedance function.
            Order of args is [nonzero poles, R0 eigenvalues, raveled R0 eigenvectors, raveled R turns ratio matrix].
            nonzero_poles is left out if the user chooses to keep the VF poles fixed.
        fixed_poles : List[float], optional
            Alternative method of passing in the poles of the circuit, by default None

        Returns
        -------
        ImpedanceLR
        """
        poles = list(args[:num_res]) if fixed_poles is None else fixed_poles

        # DC Residue extraction
        k = num_res if fixed_poles is None else 0
        C0 = np.diag(list(args[k:k+num_ports]))
        U = np.zeros((num_ports, num_ports))
        k += num_ports 
        for i in range(0, num_ports):
            for j in range(0, num_ports):
                U[i, j] = args[k]
                k += 1
        R0 = U @ C0 @ U.T

        if not np.all(np.diag(C0) > 0):
            print('Warning: DC residue is not positive definite!')

        # Turns ratios extraction
        R = np.zeros((num_res, num_ports))
        for i in range(0, num_res):
            for j in range(0, num_ports):
                R[i,j] = args[k]
                k += 1

        return cls(R0, poles, R)