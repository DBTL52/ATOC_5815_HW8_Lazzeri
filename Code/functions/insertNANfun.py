import numpy as np
import pandas as pd
"""
Based on MATLAB function made by me! Decker Lazzeri
"""

def insertNANfun(oldArray,baseArray):
    """
    insertNaNFun is a function made for GNSS data that is wanted to be broken
       up by the user through nans/nats. This takes the BaseArray (normally
       time) and finds where a discontiunity exsist and inserts NaN/NaT where
       the dicontiuity happens.
    -----------------------inputs---------------------------------------------
       oldArray - 1xn array that is wanted to be broken up.
       BaseArray - 1xn array that shows the discontiuity based on element difference 
    -----------------------outputs--------------------------------------------
      newArray - 1xn array with nan/nat at spots of discontiunity. 
    
    Author: Decker Lazzeri
    """
    base = pd.DatetimeIndex(baseArray)
    old = np.asarray(oldArray)
    if len(base) < 2:
        return old

    dt = np.diff(base.to_numpy())          # step between consecutive epochs
    gap_pos = np.where(dt > dt.min())[0] + 1   # index of the first sample after each gap

    fill = np.datetime64("NaT") if np.issubdtype(old.dtype, np.datetime64) else np.nan
    return np.insert(old, gap_pos, fill)   # inserts the fill just before each of those samples


if __name__ == '__main__':
    from rinexread import read_obs
    import matplotlib.pyplot as plt
    # Use my function to read RINEX file
    data, hdr = read_obs("dat/NIST00USA_R_20252230000_01D_30S_MO.rnx")
    # Define GPS sat type
    gps = data["GPS"]
    # Define that we want PRN 13's C1C data
    c1c_g13 = gps.loc[gps["SatelliteID"] == 13, ["Time", "C1C"]]
    # Add in NANs into C1C data where PRN 13 is not visible
    baseArray = c1c_g13["Time"]; oldArray = c1c_g13["C1C"]
    plot = insertNANfun(oldArray,baseArray)
    # Plot C1C Pseudorange
    plt.close("all")
    plt.figure()
    plt.plot(c1c_g13["Time"],plot)
    plt.xlabel("Time")
    plt.ylabel("C1C Psudorange [m]")
    plt.title("PRN 13")
    plt.show()