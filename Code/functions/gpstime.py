import pandas as pd
import numpy as np

# Constants required
GPS_START = pd.Timestamp(1980, 1, 6)   # GPS start date (no leading zeros in Python)
WEEK = pd.Timedelta(weeks=1) # Week unit

def gpstime(times):
    """
    Converts pandas time into GPS week number and Time of Week [TOW]
    INPUT:
          date  -   pandas timestamp that the user wants to convert

    OUTPUT:
        wn      -   GPS week number since January 6, 1980
        tow     -   time of week in seconds
    """

    # How long it has been since the start of GPS
    delta = pd.DatetimeIndex(times) - GPS_START
    # Whole week number
    wn = np.asarray(delta // WEEK)
    # The time of the week              
    tow = np.asarray((delta % WEEK).total_seconds())

    return wn, tow

if __name__ == "__main__":
    from rinexread import read_obs
    print("Reading RINEX")
    data, hdr = read_obs("dat/NIST00USA_R_20252230000_01D_30S_MO.rnx")
    print("Finished Reading RINEX")

    gps = data["GPS"]
    time_g13 = gps.loc[gps["SatelliteID"] == 13, "Time"]
    print(time_g13)
    wn, tow = gpstime(time_g13)
    print(wn)
    print(tow)