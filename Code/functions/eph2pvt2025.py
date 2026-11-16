
import os
import numpy as np
from functions.mean2eccentric import mean2eccentric

"""
Based on MATLAB function made by:
    Ben K. Bradley 
    07/19/2009
"""

def eph2pvt(ephemeris, t_input, prn):
    """
    Calculates GPS satellite position, velocity, and clock correction
    from a broadcast ephemeris array.

    INPUT:
      ephemeris - ndarray (n x 25) of GPS satellite orbit parameters,
                  see read_clean_GPSbroadcast() for column definitions
      t_input   - ndarray (n x 2) of GPS times to compute values at,
                  columns are [WN, TOW]
      prn       - PRN to compute values for (single satellite)      int

    OUTPUT:
      health     - ndarray (n,)   satellite health (0 = good)
      satPos     - ndarray (n,3)  ECEF satellite position [x y z], m
      satVel     - ndarray (n,3)  ECEF satellite velocity [vx vy vz], m/s
      satClkCorr - ndarray (n,)   satellite clock correction, m
      junk       - float          relativistic correction placeholder
                                   (always 0; kept for API compatibility
                                   with the MATLAB source)
      tgd        - ndarray (n,)   group delay, m
    """

    # WGS-84 constants
    muE = 3.986005e14      # m^3/s^2
    wE = 7.2921151467e-5   # rad/s
    c = 2.99792458e8       # m/s

    sz = t_input.shape[0]
    satPos = np.full((sz, 3), np.nan)
    satVel = np.full((sz, 3), np.nan)
    satClkCorr = np.full(sz, np.nan)
    relsv = np.full(sz, np.nan)
    tgd = np.full(sz, np.nan)
    health = np.full(sz, np.nan)

    ephemeris = np.asarray(ephemeris, dtype=float)
    kk = np.where(ephemeris[:, 0] == prn)[0]

    # No matching PRN found -> return all-NaN outputs
    if kk.size == 0:
        return health, satPos, satVel, satClkCorr, relsv, tgd

    sat_eph0 = ephemeris[kk, :]

    for tt in range(sz):

        dt_search = ((t_input[tt, 0] - sat_eph0[:, 18]) * 604800 + (t_input[tt, 1] - sat_eph0[:, 16]))
        dt_search = dt_search[dt_search >= -10]

        if dt_search.size == 0:
            return health, satPos, satVel, satClkCorr, relsv, tgd

        dt_min_index = int(np.argmin(np.abs(dt_search)))
        temp = np.abs(dt_search[dt_min_index])

        if temp > 2 * 3600:
            print('Ephemeris expired!')

        sat_eph = sat_eph0[dt_min_index, :]

        # Pull out variables from the ephemeris row
        # ------------------------------------------------------------
        Toe = sat_eph[16]
        gps_wk = sat_eph[18]
        dt = (t_input[tt, 0] - gps_wk) * 604800 + (t_input[tt, 1] - Toe)
        a = sat_eph[4] ** 2          # semimajor axis (m); sqrt(a) = col4
        ecc = sat_eph[3]             # eccentricity
        n0 = np.sqrt(muE / a ** 3)   # nominal mean motion (rad/s)
        n = n0 + sat_eph[2]          # corrected mean motion
        M = sat_eph[1] + n * dt      # mean anomaly
        inc = sat_eph[6]             # inclination
        perigee = sat_eph[7]         # argument of perigee

        # Compute true and eccentric anomaly
        # ------------------------------------------------------------
        E = mean2eccentric(M, ecc)
        cosE = np.cos(E)
        sinE = np.sin(E)
        nu = np.arctan2(np.sqrt(1 - ecc * ecc) * sinE, cosE - ecc)
        u = nu + perigee  # argument of latitude

        Cuc = sat_eph[10]
        Cus = sat_eph[11]
        Crc = sat_eph[12]
        Crs = sat_eph[13]
        Cic = sat_eph[14]
        Cis = sat_eph[15]

        du = Cus * np.sin(2 * u) + Cuc * np.cos(2 * u)
        dr = Crs * np.sin(2 * u) + Crc * np.cos(2 * u)
        di = Cis * np.sin(2 * u) + Cic * np.cos(2 * u)

        r = a * (1 - ecc * cosE)  # radius before correction

        u = u + du
        r = r + dr
        inc = inc + di + dt * sat_eph[9]

        cosu = np.cos(u)
        sinu = np.sin(u)
        cos2u = np.cos(2 * u)
        sin2u = np.sin(2 * u)

        # Satellite position in orbital plane
        xo = r * cosu
        yo = r * sinu

        # Corrected longitude of ascending node
        node = sat_eph[5] + (sat_eph[8] - wE) * dt - (wE * Toe)

        # Satellite position in ECEF (m)
        cosi = np.cos(inc)
        sini = np.sin(inc)
        coso = np.cos(node)
        sino = np.sin(node)

        satPos[tt, 0] = xo * coso - yo * cosi * sino
        satPos[tt, 1] = xo * sino + yo * cosi * coso
        satPos[tt, 2] = yo * sini

        # Rates of change
        E_dot = n / (1 - ecc * cosE)
        nu_dot = E_dot * np.sqrt(1 - ecc ** 2) / (1 - ecc * cosE)
        i_dot = sat_eph[9] + 2 * nu_dot * (Cis * cos2u + Cic * sin2u)
        u_dot = nu_dot + 2 * nu_dot * (Cus * cos2u - Cuc * sin2u)
        r_dot = (a * ecc * sinE * n / (1 - ecc * cosE)
                 + 2 * nu_dot * (Crs * cos2u - Crc * sin2u))
        node_dot = sat_eph[8] - wE

        # Satellite velocity in orbital plane
        xo_dot = r_dot * cosu - yo * u_dot
        yo_dot = r_dot * sinu + xo * u_dot

        # Satellite velocity in ECEF (m/s)
        satVel[tt, 0] = ((xo_dot - yo * cosi * node_dot) * coso - (xo * node_dot + yo_dot * cosi - yo * sini * i_dot) * sino)
        satVel[tt, 1] = ((xo_dot - yo * cosi * node_dot) * sino + (xo * node_dot + yo_dot * cosi - yo * sini * i_dot) * coso)
        satVel[tt, 2] = yo_dot * sini + yo * cosi * i_dot

        # Satellite clock bias (m)
        satClkCorr[tt] = c * ((sat_eph[22] * dt + sat_eph[21]) * dt + sat_eph[20])

        # Satellite health (0.00 = usable)
        health[tt] = sat_eph[24]

        # Compute relativistic correction in seconds
        relsv[tt] = -2 * np.sqrt(muE * a) * ecc * sinE / (c**2)
        
        tgd[tt] = c * sat_eph[23]
    return health, satPos, satVel, satClkCorr, relsv, tgd