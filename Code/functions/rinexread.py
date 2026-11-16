"""
rinexread.py is a moduel that has functions that is used to read in RINEX 
    files.
"""
import numpy as np
import pandas as pd

SYSTEM_NAMES = {"G": "GPS", "R": "GLONASS", "E": "Galileo", "C": "BeiDou",
                "J": "QZSS", "I": "NavIC", "S": "SBAS"}


def read_header(lines):
    """
    This reads in the header of the given RINEX file
    """
    # Preallocation
    hdr = {"obs_types": {}, "comments": []} # header dictionary
    cur_sys = None
    
    # Take in the header "lines" and make header dictionary
    for i, line in enumerate(lines):
        label, body = line[60:].strip(), line[:60]
        
        # if we have reached the end of the header, return hdr
        if label == "END OF HEADER":
            return hdr, i + 1
        
        # Save site information
        if label == "RINEX VERSION / TYPE":
            hdr["version"] = float(body[:9])
            hdr["file_type"] = body[20:40]
            hdr["system"] = body[40:59]
        # Save sat sys code, observation types
        elif label == "SYS / # / OBS TYPES":
            if body[0] != " ":             # first line of a system
                cur_sys = body[0]
                hdr["obs_types"][cur_sys] = []
            hdr["obs_types"][cur_sys] += body[7:].split()
        # Geocentric approximate marker position
        elif label == "APPROX POSITION XYZ":
            hdr["approx_xyz"] = [float(x) for x in body.split()]
        # Observation interval (Delta t) in seconds
        elif label == "INTERVAL":
            hdr["interval"] = float(body[:10])
        # Self Explanitory, 4-digit year (year,month,day,hour,min,sec)
        elif label == "TIME OF FIRST OBS":
            hdr["first_obs"] = _parse_time(body)
        # Self Explanitory, 4-digit year (year,month,day,hour,min,sec)
        elif label == "TIME OF LAST OBS":
            hdr["last_obs"] = _parse_time(body)
        # Miscilanious information
        elif label in ("MARKER NAME", "REC # / TYPE / VERS", "ANT # / TYPE"):
            hdr[label] = body.strip()
        # Comments given by station
        elif label == "COMMENT":
            hdr["comments"].append(body.strip())
    # If no END OF HEADER found. Panic
    raise ValueError("END OF HEADER not found")


def _parse_time(s):
    """
        This parses time into a pandas timestamp. Used for time of first and last observation
    """
    y, mo, d, h, mi = (int(x) for x in s[:30].split()[:5])
    sec = float(s[:43].split()[5])
    return pd.Timestamp(y, mo, d, h, mi) + pd.to_timedelta(sec, unit="s")


def _field(chunk):
    """One 16-char observation field: F14.3 value + LLI + SSI. Blank -> NaN."""
    chunk = chunk.ljust(16)
    val = float(chunk[:14]) if chunk[:14].strip() else np.nan
    lli = int(chunk[14]) if chunk[14].strip() else 0
    ssi = int(chunk[15]) if chunk[15].strip() else 0
    return val, lli, ssi


def read_obs(path):
    with open(path) as f:
        lines = f.read().splitlines()

    # Get Header Information
    hdr, start = read_header(lines) # Start - first line after the header

    # RINEX version protection
    if int(hdr["version"]) != 3:
        raise NotImplementedError("Only RINEX 3.x handled in this starter")

    # Make a dictionary of Systems (G/R/E/..) with the observation types seen (C1C/D1C/..)
    rows = {s: [] for s in hdr["obs_types"]}

    # Preallocation
    t = None
    for line in lines[start:]:
        if line.startswith(">"): # Epoch record identifier >
            f = line[1:].split()
            # Epoch time
            sec = float(f[5])
            t = pd.Timestamp(int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4])) + pd.to_timedelta(sec, unit="s")
            # Epoch Flag: 0 -> okay , 1 -> power failure , >1: Special event
            flag = int(f[6])
            if flag > 1:
                raise NotImplementedError(f"Epoch flag {flag} (event records) not handled")
            # Once we have recorded Epoch data, skip to the next line
            continue
        # Record the system id (G/R/E/..) and the PRN number
        sysid, prn = line[0], line[:3]
        # Pull out the types of obervations based on system id
        types = hdr["obs_types"].get(sysid)
        # if we do not have observation types, then skip to the next like
        if types is None:
            continue
        # Record the time observed with PRN without system id
        rec = {"Time": t, "SatelliteID": int(prn[1:])}
        # Save observation data
        for k, name in enumerate(types):
            val, lli, ssi = _field(line[3 + 16 * k: 3 + 16 * (k + 1)])
            rec[name] = val
            if name[0] == "L":
                rec[name + "_LLI"] = lli
            if name[0] in "LC":
                rec[name + "_SSI"] = ssi
        rows[sysid].append(rec)

    # Convert system id to their actual names (G->GPS,...)
    out = {SYSTEM_NAMES[s]: pd.DataFrame(r) for s, r in rows.items() if r}
    # retun data and header info
    return out, hdr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data, hdr = read_obs("dat/NIST00USA_R_20252230000_01D_30S_MO.rnx")
    for name, df in data.items():
        print(name, df.shape)
    gps = data["GPS"]
    print(gps)
    c1c_g13 = gps.loc[gps["SatelliteID"] == 13, ["Time", "C1C"]]

    plt.close("all")
    plt.figure()
    plt.plot(c1c_g13["Time"],c1c_g13["C1C"])
    plt.xlabel("Time")
    plt.ylabel("C1C Psudorange [m]")
    plt.title("PRN 13")
    plt.show()
    