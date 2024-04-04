from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import pickle
import skrf as rf

from tools.circuits.impedance import ImpedanceLR
from tools.circuits.lumped_circuits import CascadeCL

@dataclass
class ZConnect:
    z1 : int
    p1 : int
    z2 : int
    p2 : int

def interconnect_z(impedances : List[ImpedanceLR], connections : List[ZConnect]) -> ImpedanceLR:
    """Interconnects the rational impedance functions based on the passed in connection list

    Parameters
    ----------
    impedances : List[ImpedanceLR]
        List of rational impedance functions to interconnect
    connections : List[ZConnect]
        List of connections with each connection specified by:
            ZConnect(Idx of impedance 1, Idx of port impedance 1, Idx of impedance 2, Idx of port on impedance 2)

    Returns
    -------
    ImpedanceLR
        Fully interconnected rational impedance function
    """
    # Check that no port is referenced twice in the connections list
    # Also at the same time check that the ports can exist on the corresponding impedance function
    check_connect = set()
    for i, c in enumerate(connections):
        # Check if the impedance indices are valid
        if c.z1 >= len(impedances):
            raise ValueError('')
        if c.z2 >= len(impedances):
            raise ValueError('')
        
        # Check if the port indices are valid
        if c.p1 >= impedances[c.z1].N:
            raise ValueError('')
        if c.p2 >= impedances[c.z2].N:
            raise ValueError('')

        # Make sure that each port can only have one connection
        if (c.z1, c.p1) in check_connect:
            raise ValueError(f'Port (z1, p1) = ({c.z1}, {c.p1}) is already used. Make sure each port is only used once!')
        else:
            check_connect.add((c.z1, c.p1))

        if (c.z2, c.p2) in check_connect:
            raise ValueError(f'Port (z2, p2) = ({c.z2}, {c.p2}) is already used. Make sure each port is only used once!')
        else:
            check_connect.add((c.z2, c.p2))

    port_names = []
    N = np.sum([Z.N for Z in impedances])
    M = np.sum([Z.M for Z in impedances])
    C = np.zeros((N + M, N + M))
    L = np.zeros(M)
    i = 0
    j = N

    # Construct the matrices needed for the CL cascade representation of the disjoint networks
    for Z in impedances:
        # Construct the blocks of the capacitance matrix for each individual impedance function
        R0_inv = np.real(np.linalg.inv(Z.R0))
        C[i:(i+Z.N), i:(i+Z.N)] = R0_inv
        C[i:(i+Z.N), j:(j+Z.M)] = -R0_inv @ Z.R.T
        C[j:(j+Z.M), i:(i+Z.N)] = -Z.R @ R0_inv
        C[j:(j+Z.M), j:(j+Z.M)] = np.eye(Z.M) + Z.R @ R0_inv @ Z.R.T

        # Collect the "inductances" corresponding to the resonant modes
        L[(j-N):(j-N+Z.M)] = 1./np.array(Z.poles[1:])**2

        # Save the port names
        port_names += Z.port_names

        i += Z.N
        j += Z.M    

    C_old = C.copy()

    # Setup the lists to keep track of port indices for the full disjoint network
    port_connections = []
    z_n = [Z.N for Z in impedances]
    for c in connections:
        port_connections.append((
            c.p1 if c.z1 == 0 else np.sum(z_n[:(c.z1)]) + c.p1,
            c.p2 if c.z2 == 0 else np.sum(z_n[:(c.z2)]) + c.p2
        ))

    # Make list of number of ports for each impedance which will be updated after each connection
    z_n = [Z.N for Z in impedances]
    z_sub = [0]*len(impedances)
    leftover_port_idxs = []
    for i in range(len(port_connections)):

        p1_idx, p2_idx = port_connections[i]

        # Add row p2 to row p1
        C[p1_idx, :] += C[p2_idx, :]
        # Add column p2 to column p1
        C[:, p1_idx] += C[:,p2_idx]

        # Delete the row and column p2 from C
        C = np.delete(C, p2_idx, 0)
        C = np.delete(C, p2_idx, 1)
        z_n[c.z2] -= 1
        z_sub[c.z2] -= 1

        # Delete the port 2 name from the list
        del port_names[p2_idx]

        # Update "leftover" port indices since deleted row/column will shift the leftover ports
        # down one index. Then afterwards add the new leftover port.
        leftover_port_idxs.append(p1_idx)
        for i in range(len(leftover_port_idxs)):
            if leftover_port_idxs[i] > p2_idx: leftover_port_idxs[i] -= 1

        for j in range(i, len(port_connections)):
            p1_new, p2_new = port_connections[j]
            if p1_new > p2_idx: p1_new -= 1
            if p2_new > p2_idx: p2_new -= 1
            port_connections[j] = (p1_new, p2_new)

    # Plot C matrix before and after connection
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.abs(C_old) > 0)
    ax[1].imshow(np.abs(C) > 0)
    plt.show()

    # Construct cascaded network to get the rational impedance function
    connected_circuit = CascadeCL(C, L)
    R0 = connected_circuit.rational_Z.R0
    R = connected_circuit.rational_Z.R

    # Leave the leftover ports used for connection "open" and remove them from the impedance function
    for i in range(len(leftover_port_idxs)):
        lp = leftover_port_idxs[i]
        del port_names[lp]

        R0 = np.delete(R0, lp, 0)
        R0 = np.delete(R0, lp, 1)
        R = np.delete(R, lp, 1)

        for j in range(len(leftover_port_idxs)):
            if leftover_port_idxs[j] > lp: leftover_port_idxs[j] -= 1

    connected_Z = ImpedanceLR(R0, connected_circuit.rational_Z.poles, R, port_names=port_names)
    return connected_Z