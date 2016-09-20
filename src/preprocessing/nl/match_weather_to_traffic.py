"""
Matches Weather to Traffic Measurements.

@todo: Document pipeline.

Call Signature:
$ python match_weather_to_traffic.py \
    --fname_traffic_stations \
        traffic_stations_linked_to_weather_stations.pddf.hdf5 \
    --fname_traffic \
        NL_2016_01.pddf.hdf5 \
    --fname_weather_observations \
        weather_observations.pddf.hdf5
"""

import time
import pandas as pd
import numpy as np
import argparse


###############################################################################
###############################################################################
# FUNCTIONS
###############################################################################
###############################################################################

def find_index_of_closest_weather_observation(timestamp_seconds, \
                                              station_id, \
                                              df_observations):
    """
    For some station, finds the closest weather measurement to some time.
    Requires scalars for timestamp and station ID.
    If a station is absent, return NaN for the index.

    @param: timestamp_seconds -- Timestamp to Scan For [Unix Epoch in Seconds]
    @param: station_id        -- Which Weather Station
    @param: df_observations   -- Dataframe with Weather Observations
    @returns: idx             -- Index in Dataframe of Nearest Time Observation
    """

    try:
        idx = np.abs(df_observations[df_observations.station == \
                                     station_id].timestamp_end_seconds - \
                                                 timestamp_seconds).idxmin()
    except ValueError:
        idx = np.nan
    return idx


###############################################################################
###############################################################################
# MAIN CODE
###############################################################################
###############################################################################

#
# Parse Arguments
#
parser = argparse.ArgumentParser()
parser.add_argument('--fname_traffic', required=True, \
                    help='Traffic Data')
parser.add_argument('--fname_traffic_stations', required=True, \
                    help='Traffic Station Data')
parser.add_argument('--fname_weather_observations', required=True, \
                    help='Weather Observations')
args = parser.parse_args()

#
# Load Weather Observations
#

print '// Loading Weather Data'
# basedir = '/snufkin/projects/companion/reduced_netherlands_weather'
# fname = "%s/weather_observations.pddf.hdf5" % basedir
df_weather_obs = pd.read_hdf(args.fname_weather_observations, 'df')
print "   Found %i Weather Observations" % len(df_weather_obs.index)

# Cause...
df_weather_obs['timestamp_end_seconds'] = \
    df_weather_obs.date.astype(np.int64)/1000/1000/1000 + \
    df_weather_obs.hour*3600
df_weather_obs['timestamp_start_seconds'] = \
    df_weather_obs.timestamp_end_seconds - 3600
    
df_weather_obs['timestamp_start'] = pd.to_datetime(df_weather_obs.timestamp_start_seconds, unit='s')
df_weather_obs['timestamp_end'] = pd.to_datetime(df_weather_obs.timestamp_end_seconds, unit='s')

# Drop Drop
del df_weather_obs['date']
del df_weather_obs['hour']

#
# Load Traffic
#

print '// Loading Traffic Data'
# basedir = '/snufkin/companion/Outputs_from_Java_Tool/2016_01'
# fname = "%s/NL_2016_01.pddf.hdf5" % basedir
df_traffic = pd.read_hdf(args.fname_traffic, 'df')

# De-Unicode Traffic Station ID in Traffic Data
df_traffic['station'] = \
    df_traffic.station.apply(lambda x: x.encode('utf-8').strip())

# Chopchop
print "   Found %i Traffic Measurements" % len(df_traffic.index)
df_traffic = df_traffic.sample(frac=0.01)
df_traffic.reset_index(drop=True, inplace=True)
# df_traffic = df_traffic.head(5).reset_index(drop=True)
print "   Processing %i Traffic Measurements" % len(df_traffic.index)


#
# Load Weather Traffic Station Linkage
#

print '// Loading Matched Traffic & Weather Stations'
# basedir = '/snufkin/projects/companion/reduced_netherlands_traffic'
# fname = "%s/traffic_stations_linked_to_weather_stations.pddf.hdf5" % basedir
df_traffic_stations = pd.read_hdf(args.fname_traffic_stations, 'df')
print "   Found %i Traffic Stations w/ Linked Weather Stations" % \
    len(df_traffic_stations.index)

# Merge Weather Station into Traffic
df_traffic = df_traffic.merge(df_traffic_stations, on='station', how='left')
del df_traffic['latitude']
del df_traffic['longitude']

# Superior Time Unit...
df_traffic['timestamp_start_seconds'] = \
    df_traffic.timestamp_start.astype(np.int64)/1000/1000/1000
df_traffic['timestamp_end_seconds'] = \
    df_traffic.timestamp_end.astype(np.int64)/1000/1000/1000

#
# Merge Weather Data
#

indices_weather_observation = []
print '// Finding Closest Weather Observations'
tstart = time.time()
for iix in df_traffic.index:
    indices_weather_observation.append(\
        find_index_of_closest_weather_observation(\
            df_traffic.ix[iix,'timestamp_end_seconds'], \
            df_traffic.ix[iix,'closest_weather_station_id'], \
            df_weather_obs))
tend = time.time()
print "   Took %.2f Seconds" % (tend - tstart)

# Build List of Weather Keys
weather_keys = []
print '// Building List of Weather Keys'
for key in list(df_weather_obs.keys()):
    if not key in [ 'station', \
                    'timestamp_start', 'timestamp_end', \
                    'timestamp_start_seconds', 'timestamp_end_seconds ']:
        weather_keys.append(key)

# Copy Columns
print '// Copy Weather into Traffic Dataframe'
for key in weather_keys:
    df_traffic[key] = \
        np.asarray(df_weather_obs.ix[indices_weather_observation,key])
    
# Weather Time Difference
print '// Compute Time Deltas'
df_traffic['weather_obs_dt_seconds'] = \
    np.asarray(df_weather_obs.ix[indices_weather_observation, \
                                 'timestamp_end_seconds']) - \
    np.asarray(df_traffic['timestamp_end_seconds'])

# Store
fname = 'traffic_with_weather.pddf.hdf5'
print "// Storing Dataframe to %s" % fname
df_traffic.to_hdf(fname, 'df', mode='w', format='table')
