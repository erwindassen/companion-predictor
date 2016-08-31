"""
Convert HDF5 Data into a Pandas Dataframe.
Takes ~5 Minutes.
"""

import pandas as pd
import numpy as np
import h5py
import argparse


# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fname_in', required=True, \
                    help='Input Filename.')
args = parser.parse_args()

# Debug/Info
print "// Loading Stations From %s" % args.fname_in

# Base Dataframe
with h5py.File(args.fname_in, 'r') as fx:
    
    # Get Station Names
    stations = fx.keys()

    # Remove Wtf?
    if '__DATA_TYPES__' in stations:
        stations.remove('__DATA_TYPES__')

    # Loop Only 5 Stations
    # stations = stations[:10]

    # Build Dataframes
    for istation, station in enumerate(stations):
        if istation % 100 == 0: print "   Station %i" % istation

        tlo = fx["%s/temperature" % station][:,0]
        thi = fx["%s/temperature" % station][:,1]
        tmp = fx["%s/temperature" % station][:,2]
        flw = fx["%s/trafficflow_lores" % station][:,2]
        spd = fx["%s/trafficspeed_lores" % station][:,2]
        pre = fx["%s/precipitation" % station][:,2]
        wnd = fx["%s/windspeed" % station][:,2]

        data = { 'timestamp_start': pd.to_datetime(tlo, unit='s'), \
                 'timestamp_end': pd.to_datetime(thi, unit='s'), \
                 'temperature': tmp, \
                 'precipitation': pre, \
                 'windspeed': wnd, \
                 'trafficspeed': spd, \
                 'trafficflow': flw, \
                 'station': [ station ] * len(tlo) }

        cols = [ 'station', \
                 'timestamp_start', \
                 'timestamp_end', \
                 'trafficspeed', \
                 'trafficflow', \
                 'temperature', \
                 'precipitation', \
                 'windspeed' ]

        df_tmp = pd.DataFrame(data, columns=cols)

        if istation == 0:
            df = df_tmp
        else:
            df = pd.concat([df,df_tmp])


# Save
fname_out = "%s.pddf.hdf5" % args.fname_in[:-5]
print "// Saving Pandas Dataframe to %s" % fname_out
with pd.HDFStore(fname_out, 'w') as store:
    store['df'] = df
