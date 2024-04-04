from typing import List
import numpy as np
import matplotlib.pyplot as plt

def compare_singleport(
    ax : plt.Axes,
    f : np.ndarray, 
    H1 : np.ndarray,
    H2 : np.ndarray,
    colors : List = ['blue', 'red', 'limegreen']
):
    """Plots log10 magnitude of two single port immittance or scattering parameters together with a difference on a given axis.

    Parameters
    ----------
    ax : plt.Axes
        Axis on which to plot
    f : np.ndarray
        Frequency list corresponding to the H-parameters
    H1 : np.ndarray
        First immittance/scattering parameter
    H2 : np.ndarray
        Second immittance/scattering parameter
    colors : List
        Colors for the individual lines in order: H1, H2, H1-H2
    """
    if H1.shape != H2.shape:
        raise ValueError('Immittance/scattering parameters should have the same number of ports.')
    
    ax.plot(f/1e9, np.log10(np.abs(H1)), color=colors[0], linewidth=1.5)
    ax.plot(f/1e9, np.log10(np.abs(H2)), color=colors[1], linewidth=1.5, linestyle='dashed', dashes=(5, 5))
    ax.plot(f/1e9, np.log10(np.abs(H1 - H2)), color=colors[2], linewidth=0.5, zorder=-1)

def compare_multiport(
    ax : plt.Axes,
    f : np.ndarray, 
    H1 : np.ndarray,
    H2 : np.ndarray,
    colors : List = ['blue', 'red', 'limegreen']
):
    """Plots log10 magnitude of two multiport immittance or scattering parameters together with a difference on a given axis.

    Parameters
    ----------
    ax : plt.Axes
        Axis on which to plot
    f : np.ndarray
        Frequency list corresponding to the H-parameters
    H1 : np.ndarray
        First immittance/scattering parameter
    H2 : np.ndarray
        Second immittance/scattering parameter
    colors : List
        Colors for the individual lines in order: H1, H2, H1-H2
    """
    if H1.shape != H2.shape:
        raise ValueError('Immittance/scattering parameters should have the same number of ports.')

    if H1.shape[1] != H1.shape[2]:
        raise ValueError('Immittance/scattering parameters should be square.')

    for i in range(H1.shape[1]):
        for j in range(i, H2.shape[1]):
            compare_singleport(ax, f, H1[:,i,j], H2[:,i,j], colors=colors)
            # ax.plot(f/1e9, np.log10(np.abs(H1[:, i, j])), color='blue', linewidth=1)
            # ax.plot(f/1e9, np.log10(np.abs(H2[:, i, j])), color='red', linewidth=1, linestyle='dashed')
            # ax.plot(f/1e9, np.log10(np.abs(H1[:, i, j] - H2[:, i, j])), color='limegreen', linewidth=0.5)

def rearrange_ports(
    H : np.ndarray,
    idxs : List
) -> np.ndarray:
    """Rearranges the ports for a given multiport parameter

    Parameters
    ----------
    H : np.ndarray
        Multiport of shape (N_freq, N_ports, N_ports).
    idxs : List
        New port index ordering.

    Returns
    -------
    np.ndarray
        Multiport parameter of shape (Nf, N, N) with the ports rearranged as specified
    """
    if len(idxs) != len(set(idxs)):
        raise ValueError(f'Make sure no port indices are repeated in {idxs}')

    N = H.shape[1]
    if set(idxs) != set(range(N)):
        raise ValueError(f'Make sure all ports {set(range(N))} are included in the shuffle.')
        
    H_new = H[:, idxs, :]
    H_new = H_new[:, :, idxs]

    return H_new

def bring_to_front(
    H : np.ndarray,
    idxs : List
) -> np.ndarray:
    """Rearranges the ports such that the passed in indices are moved to the front of the array

    Parameters
    ----------
    H : np.ndarray
        Multiport of shape (N_freq, N_ports, N_ports).
    idxs : List
        Indices of ports to move to front

    Returns
    -------
    np.ndarray
        Rearranged multiport parameter
    """
    new_idxs = list(range(H.shape[1]))
    new_idxs = idxs + [i for i in new_idxs if i not in idxs]
    print(new_idxs)
    return rearrange_ports(H, new_idxs)

def truncate_ports(
    Z : np.ndarray,
    N_keep : int
) -> np.ndarray:
    """Keeps only the first N_keep ports, leaves the rest open. NOTE: Should only be used for impedance parameters.

    Parameters
    ----------
    Z : np.ndarray
        Multiport of shape (N_freq, N_ports, N_ports).
    N_keep : int
        Number of ports to keep (first N_keep ports, so order matters)

    Returns
    -------
    np.ndarray
        Multiport parameter of shape (N_freq, N_keep, N_keep)
    """
    return Z[:, :N_keep, :N_keep]