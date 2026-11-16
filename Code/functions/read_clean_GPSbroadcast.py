import os
import numpy as np

def parse_float(field):
    """Parse a fixed-width RINEX numeric field the way MATLAB's str2num
    would, but also tolerating Fortran 'D' exponents. Returns NaN for a
    blank field."""
    s = field.strip()
    if s == '':
        return np.nan
    s = s.replace('D', 'E').replace('d', 'e')
    return float(s)

def read_clean_GPSbroadcast(navfilename):
    """
    Reads an IGS GPS Broadcast Ephemeris file for all GPS satellites and
    constructs an array of all ephemeris values. Each row is a new entry
    in the broadcast file (most likely multiple entries per PRN).

    INPUT:
      navfilename - name of IGS broadcast ephemeris file to read in  (str)
      clean       - optional flag to clean bad ephemeris entries out
                    (default False)

    OUTPUT:
      gps_ephem   - ndarray (n x 25) of GPS satellite orbit parameters

                    col0:  prn,      PRN number of satellite
                    col1:  M0,       mean anomaly at reference time, rad
                    col2:  delta_n,  mean motion difference from computed
                                     value, rad/s
                    col3:  ecc,      eccentricity of orbit
                    col4:  sqrt_a,   square root of semi-major axis, m^0.5
                    col5:  Loa,      longitude of ascending node of orbit
                                     plane at weekly epoch, rad
                    col6:  incl,     inclination angle at reference time, rad
                    col7:  perigee,  argument of perigee, rad
                    col8:  ra_rate,  rate of change of right ascension, rad/s
                    col9:  i_rate,   rate of change of inclination angle, rad/s
                    col10: Cuc,      cosine harmonic correction to arg. of
                                     latitude
                    col11: Cus,      sine harmonic correction to arg. of
                                     latitude
                    col12: Crc,      cosine harmonic correction to orbit
                                     radius
                    col13: Crs,      sine harmonic correction to orbit radius
                    col14: Cic,      cosine harmonic correction to
                                     inclination
                    col15: Cis,      sine harmonic correction to inclination
                    col16: Toe,      reference time ephemeris (s into GPS week)
                    col17: IODE,     issue of data (ephemeris)
                    col18: GPS_week, GPS week number (wrt 06jan80, no rollover)
                    col19: Toc,      time of clock
                    col20: Af0,      satellite clock bias (s)
                    col21: Af1,      satellite clock drift (s/s)
                    col22: Af2,      satellite clock drift rate (s/s/s)
                    col23: TGD,      group delay for single-frequency use
                    col24: health,   satellite health (0 = good/usable)

      ionoparams  - ndarray, Klobuchar ionospheric model parameters
                    [A0 A1 A2 A3 B0 B1 B2 B3]
    BASED ON MATLAB CODE BY: Ben K. Bradley ON: 07/19/2029
    """

    if not os.path.isfile(navfilename):
        raise IOError('Unable to find broadcast file: {}'.format(navfilename))

    with open(navfilename, 'r') as fid:
        lines = fid.readlines()

    # ---- Step through header, pull out iono parameters ----------------
    ALPHA = []
    BETA = []
    body_start = len(lines)

    for idx, tline in enumerate(lines):
        if 'ION ALPHA' in tline:
            parts = tline.split()
            ALPHA = [parse_float(p) for p in parts[:4]]

        if 'ION BETA' in tline:
            parts = tline.split()
            BETA = [parse_float(p) for p in parts[:4]]

        if 'END OF HEADER' in tline:
            body_start = idx + 1
            break

    ionoparams = np.array(ALPHA + BETA)

    # ---- Main loop: read the rest of the ephemeris file ----------------
    rows = []
    i = body_start
    n = len(lines)

    while i < n:
        tline = lines[i]

        # Blank / trailing lines
        if tline.strip() == '':
            i += 1
            continue

        # Need a full 8-line record remaining
        if i + 7 >= n:
            break

        # -----------------------------------------------------------
        # FIRST LINE
        # -----------------------------------------------------------
        line1 = tline
        prn = int(parse_float(line1[0:2]))
        Af0 = parse_float(line1[22:41])
        Af1 = parse_float(line1[41:60])
        Af2 = parse_float(line1[60:79])

        # -----------------------------------------------------------
        # SECOND LINE
        # -----------------------------------------------------------
        line2 = lines[i + 1]
        IODE = parse_float(line2[3:22])
        Crs = parse_float(line2[22:41])
        delta_n = parse_float(line2[41:60])
        M0 = parse_float(line2[60:79])

        # -----------------------------------------------------------
        # THIRD LINE
        # -----------------------------------------------------------
        line3 = lines[i + 2]
        Cuc = parse_float(line3[3:22])
        ecc = parse_float(line3[22:41])
        Cus = parse_float(line3[41:60])
        sqrt_a = parse_float(line3[60:79])

        # -----------------------------------------------------------
        # FOURTH LINE
        # -----------------------------------------------------------
        line4 = lines[i + 3]
        Toe = parse_float(line4[3:22])
        Toc = Toe
        Cic = parse_float(line4[22:41])
        Loa = parse_float(line4[41:60])
        Cis = parse_float(line4[60:79])

        # -----------------------------------------------------------
        # FIFTH LINE
        # -----------------------------------------------------------
        line5 = lines[i + 4]
        incl = parse_float(line5[3:22])
        Crc = parse_float(line5[22:41])
        perigee = parse_float(line5[41:60])
        ra_rate = parse_float(line5[60:79])

        # -----------------------------------------------------------
        # SIXTH LINE
        # -----------------------------------------------------------
        line6 = lines[i + 5]
        i_rate = parse_float(line6[3:22])
        GPS_week = parse_float(line6[41:60])
        if GPS_week < 1024:
            GPS_week = GPS_week + 2048  # avoid GPS week rollover/modulo 1024

        # -----------------------------------------------------------
        # SEVENTH LINE: SV accuracy, SV health, TGD, IODC
        # -----------------------------------------------------------
        line7 = lines[i + 6]
        health = parse_float(line7[22:41])
        TGD = parse_float(line7[41:60])

        # -----------------------------------------------------------
        # EIGHTH LINE: time of transmission of message (unused)
        # -----------------------------------------------------------
        # line8 = lines[i + 7]

        rows.append([
            prn, M0, delta_n, ecc, sqrt_a, Loa, incl, perigee, ra_rate,
            i_rate, Cuc, Cus, Crc, Crs, Cic, Cis, Toe, IODE, GPS_week,
            Toc, Af0, Af1, Af2, TGD, health
        ])

        i += 8

    gps_ephem = np.array(rows, dtype=float) if rows else np.empty((0, 25))

    # ---- Cleaning of bad/duplicate ephemeris entries ----------
    
    prn_all = np.unique(gps_ephem[:, 0])
    badrows = []

    for prn_val in prn_all:
        kk = np.where(gps_ephem[:, 0] == prn_val)[0]
        sat_ephem = gps_ephem[kk, :]

        if kk.size == 0:
            continue

        # If a new upload is within 4 minutes of the following
        # entry, drop that following (usual 00:00) entry and keep
        # the new upload.
        newuploads = np.where(np.mod(sat_ephem[:, 16], 3600) != 0)[0]

        if newuploads.size > 0:
            for zz in newuploads:
                if zz == sat_ephem.shape[0] - 1:
                    # new upload is the last entry for this PRN
                    continue
                if (sat_ephem[zz + 1, 16] - sat_ephem[zz, 16]) < 240:
                    badrows.append(kk[zz + 1])

    if badrows:
        gps_ephem = np.delete(gps_ephem, badrows, axis=0)

    return gps_ephem, ionoparams