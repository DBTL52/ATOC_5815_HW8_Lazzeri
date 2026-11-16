"""
Based on MATLAB function made by:
    Ben K. Bradley 
"""

import numpy as np


def mean2eccentric(M, ecc):
    """
    Solve Kepler's equation for eccentric anomaly given mean anomaly.

    Parameters
    ----------
    M : float or array_like
        Mean anomaly (single value or vector), rad.
    ecc : float or array_like
        Eccentricity of orbit (single value or vector; same shape as M
        if not scalar).

    Returns
    -------
    E : float or np.ndarray
        Eccentric anomaly (same shape as input M), rad.
    """
    scalar_input = np.isscalar(M) and np.isscalar(ecc)

    M = np.asarray(M, dtype=float)
    ecc = np.asarray(ecc, dtype=float)

    if np.any(ecc > 0.999) or np.any(ecc < 0):
        raise ValueError(
            'Not an elliptic orbit.'
        )

    # Set tolerance
    twopi = 2 * np.pi
    tol = 1.0e-12
    maxiter = 20

    # Make sure mean anomaly is positive and within 2*pi
    M = np.remainder(M, twopi)
    M = (M < 0) * twopi + M

    # Make an initial guess for E, eccentric anomaly
    sinM = np.sin(M)
    E = M + (ecc * sinM) / (1 - np.sin(M + ecc) + sinM)

    # Initialize iteration count and error
    iter_count = 1
    err = 1

    # Iterate to find E, eccentric anomaly
    while np.any(np.abs(err) > tol) and (iter_count <= maxiter):
        err = (E - ecc * np.sin(E) - M) / (1 - ecc * np.cos(E))
        E = E - err
        iter_count += 1

        if iter_count > maxiter:
            import warnings
            warnings.warn('Iterations maxed out.')

    if scalar_input:
        return float(E)
    return E