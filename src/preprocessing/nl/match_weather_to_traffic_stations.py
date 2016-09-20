"""
Link Weather and Traffic Stations.

Call Signature:
$ python match_weather_to_traffic_stations.py \
    --fname_weather_stations weather_stations.pddf.hdf5 \
    --fname_traffic_stations traffic_stations.pddf.hdf5
"""

import pandas as pd
import argparse
import numpy as np
import time


###############################################################################
###############################################################################
# FUNCTIONS
###############################################################################
###############################################################################

# Convenience
r2d = 180.0 / np.pi
d2r = np.pi / 180.0

def geo_distance(lat_01_deg, lon_01_deg, lat_02_deg, lon_02_deg):
    """
    Computes great-circle distance. Angles in degree decimal.
    Accepts vectors and scalars.

    @param: lat_01 -- Latitude  of First Point  [Deg]
    @param: lon_01 -- Longitude of First Point  [Deg]
    @param: lat_02 -- Latitude  of Second Point [Deg]
    @param: lon_02 -- Longitude of Second Point [Deg]
    @return: d     -- Distance                  [km]
    """

    # Convert
    lat_01 = lat_01_deg * d2r
    lon_01 = lon_01_deg * d2r
    lat_02 = lat_02_deg * d2r
    lon_02 = lon_02_deg * d2r

    # https://en.wikipedia.org/wiki/Haversine_formula
    # 6371.0 = Earth Radius
    h = np.sin((lat_02-lat_01)/2.0)**2.0 + \
        np.cos(lat_01) * np.cos(lat_02) * np.sin((lon_02-lon_01)/2.0)**2.0
    d = 2.0 * 6371.0 * np.arcsin(np.sqrt(h))
    
    # Return
    return d


def find_closest_weather_station(lat_event, lon_event, df_stations):
    """
    Finds the closest weather station to where a situation occurs.
    Requires scalars for latitude & longitude.

    @param: lat_event   -- Latitude  of Event                   [Deg]
    @param: lon_event   -- Longitude of Event                   [Deg]
    @param: df_stations -- Dataframe w/ Weather Stations
    @return: station_id -- ID of Closest Weather Station
    @return: dxdy       -- Distance to Closest Weather Station  [km]
    """

    lat_stations = np.asarray(df_stations.latitude_deg)
    lon_stations = np.asarray(df_stations.longitude_deg)
    df_stations['dxdy'] = \
        geo_distance(lat_event * np.ones_like(lat_stations), \
                     lon_event * np.ones_like(lat_stations), \
                     lat_stations, \
                     lon_stations)
    df_stations_head = df_stations.sort_values(by='dxdy').head(1).iloc[0]
    return df_stations_head.ix['station_id'], df_stations_head.ix['dxdy']


###############################################################################
###############################################################################
# MAIN CODE
###############################################################################
###############################################################################

###############################################################################
# Preamble
###############################################################################

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fname_traffic_stations', required=True, \
                    help='HDF5 w/ Traffic Stations in Pandas Dataframe.')
parser.add_argument('--fname_weather_stations', required=True, \
                    help='HDF5 w/ Weather Stations in Pandas Dataframe.')
args = parser.parse_args()

###############################################################################
# Do Stuff
###############################################################################

# Load Traffic Stations
print '// Loading Traffic Stations'
# basedir = '/snufkin/projects/companion/reduced_netherlands_traffic'
# fname = "%s/traffic_stations.pddf.hdf5" % basedir
df_traffic_stations = pd.read_hdf(args.fname_traffic_stations, 'df')

# Load Weather Station
print '// Loading Weather Stations'
# basedir = '/snufkin/projects/companion/reduced_netherlands_weather'
# fname = "%s/weather_stations.pddf.hdf5" % basedir
df_weather_stations = pd.read_hdf(args.fname_weather_stations, 'df')

# Loop Traffic Stations
closest_weather_station_id = []
closest_weather_station_dxdy = []
print '// Linking Traffic and Weather Stations'
tstart = time.time()
for iix in df_traffic_stations.index:
    tmp = \
        find_closest_weather_station(df_traffic_stations.ix[iix,'latitude'], \
                                     df_traffic_stations.ix[iix,'longitude'], \
                                     df_weather_stations)
    closest_weather_station_id.append(tmp[0])
    closest_weather_station_dxdy.append(tmp[1])
tend = time.time()
print "   Took %.2f Seconds" % (tend - tstart)

df_traffic_stations['closest_weather_station_id'] = \
    closest_weather_station_id
df_traffic_stations['closest_weather_station_dxdy'] = \
    closest_weather_station_dxdy

# Store
fname = 'traffic_stations_linked_to_weather_stations.pddf.hdf5'
print "// Storing Dataframe to %s" % fname
df_traffic_stations.to_hdf(fname, 'df', mode='w', format='table')
