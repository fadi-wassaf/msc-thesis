import numpy as np

def mutual_to_maxwell(C : np.ndarray) -> np.ndarray:
    """Converts a mutual/SPICE form capacitance matrix to a Maxwell form capacitance matrix

    Parameters
    ----------
    C : np.ndarray
        The capacitance matrix in SPICE form

    Returns
    -------
    C_maxwell : np.ndarray
        the capacitance matrix in Maxwell form
    """
    # Get the number of ports
    N = C.shape[0]

    # Create the Maxwell form capacitance matrix
    C_maxwell = -C.copy()

    # Fill in the diagonal elements (sum of all elements in the row)
    for i in range(N):
        C_maxwell[i, i] = np.sum(C[i, :])

    return C_maxwell

def maxwell_to_mutual(C : np.ndarray) -> np.ndarray:
    """Converts a Maxwell form capacitance matrix to mutual/SPICE form.
    Uses mutual_to_maxwell since the conversion is reciprocal.

    Parameters
    ----------
    C : np.ndarray
        The capacitance matrix in Maxwell form

    Returns
    -------
    np.ndarray
        The Capacitance matrix in Mutual/SPICE form
    """
    return mutual_to_maxwell(C)