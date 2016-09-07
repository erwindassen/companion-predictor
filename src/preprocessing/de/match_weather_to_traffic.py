"""
Match Weather Data to Traffic Events.
Pretty Slow (~40 Minutes for 45k Events).
Only Operates on 2015 Data for Weather Stations (Lines 184 to 197).

@todo: Speed Up
"""

import numpy as np
import pandas as pd
import time
import argparse
import sys


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

    lat_stations = np.asarray(df_stations.latitude)
    lon_stations = np.asarray(df_stations.longitude)
    df_stations['dxdy'] = \
        geo_distance(lat_event * np.ones_like(lat_stations), \
                     lon_event * np.ones_like(lat_stations), \
                     np.asarray(df_stations.latitude), \
                     np.asarray(df_stations.longitude))
    df_stations_head = df_stations.sort_values(by='dxdy').head(1).iloc[0]
    return df_stations_head.ix['station_id'], df_stations_head.ix['dxdy']


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
        idx = np.abs(df_observations[df_observations.station_id == \
                                     station_id].timestamp_seconds - \
                                                 timestamp_seconds).idxmin()
    except ValueError:
        idx = np.nan
    return idx


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
parser.add_argument('--basedir_weather', required=True, \
                    help='Directory w/ Reduced DWD Weather Data and Stations.')
args = parser.parse_args()

# List of Files
if sys.stdin.isatty():
    print "!! No File List (Use Stdin)."
    sys.exit()
else:
    fnames_traffic = sys.stdin.read().rstrip("\n").split("\n")
    print ''
    print "** Reading %i Traffic Event Files" % len(fnames_traffic)
    print ''

#
# Load Weather Data
#

fname_observations_precipitation = \
    "%s/weather_obs_precipitation.pddf.hdf5" % args.basedir_weather
fname_observations_temperature = \
    "%s/weather_obs_temperature.pddf.hdf5" % args.basedir_weather
fname_observations_wind = \
    "%s/weather_obs_wind.pddf.hdf5" % args.basedir_weather

fname_stations_precipitation = \
    "%s/weather_stations_precipitation.pddf.hdf5" % args.basedir_weather
fname_stations_temperature = \
    "%s/weather_stations_temperature.pddf.hdf5" % args.basedir_weather
fname_stations_wind = \
    "%s/weather_stations_wind.pddf.hdf5" % args.basedir_weather

print "// Loading Precipitation Observation Stations"
df_stations_precipitation = pd.read_hdf(fname_stations_precipitation, 'df')
print "   %i Records" % len(df_stations_precipitation.index)

print "// Loading Temperature Observation Stations"
df_stations_temperature = pd.read_hdf(fname_stations_temperature, 'df')
print "   %i Records" % len(df_stations_temperature.index)

print "// Loading Wind Observation Stations"
df_stations_wind = pd.read_hdf(fname_stations_wind, 'df')
print "   %i Records" % len(df_stations_wind.index)

print "// Loading Precipitation Observations"
df_observations_precipitation = \
    pd.read_hdf(fname_observations_precipitation, 'df')
print "   %i Records" % len(df_observations_precipitation.index)

print "// Loading Temperature Observations"
df_observations_temperature = \
    pd.read_hdf(fname_observations_temperature, 'df')
print "   %i Records" % len(df_observations_temperature.index)

print "// Loading Wind Observations"
df_observations_wind = \
    pd.read_hdf(fname_observations_wind, 'df')
print "   %i Records" % len(df_observations_wind.index)

# Silly...
df_observations_precipitation.reset_index(inplace=True, drop=True)
df_observations_temperature.reset_index(inplace=True, drop=True)
df_observations_wind.reset_index(inplace=True, drop=True)
df_stations_precipitation.reset_index(inplace=True, drop=True)
df_stations_temperature.reset_index(inplace=True, drop=True)
df_stations_wind.reset_index(inplace=True, drop=True)

# This removes a lot of headache
# By default, this yields nanoseconds since epoch, so we convert
df_observations_precipitation['timestamp_seconds'] = \
    df_observations_precipitation.timestamp.astype(np.int64)/1000/1000/1000
df_observations_temperature['timestamp_seconds'] = \
    df_observations_temperature.timestamp.astype(np.int64)/1000/1000/1000
df_observations_wind['timestamp_seconds'] = \
    df_observations_wind.timestamp.astype(np.int64)/1000/1000/1000

# Filter Active Weather Stations
df_stations_precipitation = \
    df_stations_precipitation[\
        (df_stations_precipitation.operational_from <= \
         pd.to_datetime("2015010100", format="%Y%m%d%H")) & \
        (df_stations_precipitation.operational_to >= \
         pd.to_datetime("2015123123", format="%Y%m%d%H"))]

df_stations_temperature = \
    df_stations_temperature[\
        (df_stations_temperature.operational_from <= \
         pd.to_datetime("2015010100", format="%Y%m%d%H")) & \
        (df_stations_temperature.operational_to >= \
         pd.to_datetime("2015123123", format="%Y%m%d%H"))]

#
# The Master Loop
#
for ift, fname_traffic in enumerate(fnames_traffic):
    print ''
    print "** Processing Traffic File %i/%i (%s)." % \
        (ift+1, len(fnames_traffic), fname_traffic)
    print ''

    #
    # Load Traffic Data
    #

    print "// Loading Traffic Data"
    df_traffic = pd.read_hdf(fname_traffic, 'df')
    print "   %i Records" % len(df_traffic.index)

    # Chop Chop for Development
    # df_traffic = df_traffic.head(10000)

    # Silly...
    df_traffic.reset_index(inplace=True, drop=True)

    # This removes a lot of headache
    # By default, this yields nanoseconds since epoch, so we convert
    df_traffic['timestamp_seconds'] = \
        df_traffic.timestamp.astype(np.int64)/1000/1000/1000

    ###########################################################################
    # Find Closest Weather Stations
    ###########################################################################

    #
    # Precipitation
    #

    precipitation_closest_weather_station_id = []
    precipitation_closest_weather_station_dxdy = []
    print "// Finding Closest Weather Stations (Precipitation)"
    tstart = time.time()
    for iix in df_traffic.index:
        tmp = \
            find_closest_weather_station(\
                df_traffic.ix[iix, 'location_lat_decimal'], \
                df_traffic.ix[iix, 'location_lon_decimal'], \
                df_stations_precipitation)
        precipitation_closest_weather_station_id.append(tmp[0])
        precipitation_closest_weather_station_dxdy.append(tmp[1])
    tend = time.time()
    print "   Took %.2f Seconds" % (tend - tstart)

    df_traffic['precipitation_closest_weather_station_id'] = \
        precipitation_closest_weather_station_id
    df_traffic['precipitation_closest_weather_station_dxdy'] = \
        precipitation_closest_weather_station_dxdy

    #
    # Temperature
    #

    temperature_closest_weather_station_id = []
    temperature_closest_weather_station_dxdy = []
    print "// Finding Closest Weather Stations (Temperature)"
    tstart = time.time()
    for iix in df_traffic.index:
        tmp = \
            find_closest_weather_station(\
                df_traffic.ix[iix, 'location_lat_decimal'], \
                df_traffic.ix[iix, 'location_lon_decimal'], \
                df_stations_temperature)
        temperature_closest_weather_station_id.append(tmp[0])
        temperature_closest_weather_station_dxdy.append(tmp[1])
    tend = time.time()
    print "   Took %.2f Seconds" % (tend - tstart)

    df_traffic['temperature_closest_weather_station_id'] = \
        temperature_closest_weather_station_id
    df_traffic['temperature_closest_weather_station_dxdy'] = \
        temperature_closest_weather_station_dxdy

    #
    # Wind
    #

    wind_closest_weather_station_id = []
    wind_closest_weather_station_dxdy = []
    print "// Finding Closest Weather Stations (Wind)"
    tstart = time.time()
    for iix in df_traffic.index:
        tmp = \
            find_closest_weather_station(\
                df_traffic.ix[iix, 'location_lat_decimal'], \
                df_traffic.ix[iix, 'location_lon_decimal'], \
                df_stations_wind)
        wind_closest_weather_station_id.append(tmp[0])
        wind_closest_weather_station_dxdy.append(tmp[1])
    tend = time.time()
    print "   Took %.2f Seconds" % (tend - tstart)

    df_traffic['wind_closest_weather_station_id'] = \
        wind_closest_weather_station_id
    df_traffic['wind_closest_weather_station_dxdy'] = \
        wind_closest_weather_station_dxdy

    ###########################################################################
    # Find Closest Weather Observations
    ###########################################################################

    #
    # Precipitation
    #

    precipitation_indices_weather_observations = []
    print "// Finding Closest Weather Observations (Precipitation)"
    tstart = time.time()
    for iix in df_traffic.index:
        precipitation_indices_weather_observations.append(\
                find_index_of_closest_weather_observation(\
                    df_traffic.ix[iix,'timestamp_seconds'], \
                    df_traffic.ix[iix,\
                                  'precipitation_closest_weather_station_id'], \
                    df_observations_precipitation))
    tend = time.time()
    print "   Took %.2f Seconds" % (tend - tstart)

    #
    # Temperature
    #

    temperature_indices_weather_observations = []
    print "// Finding Closest Weather Observations (Temperature)"
    tstart = time.time()
    for iix in df_traffic.index:
        temperature_indices_weather_observations.append(\
                find_index_of_closest_weather_observation(\
                    df_traffic.ix[iix,'timestamp_seconds'], \
                    df_traffic.ix[iix,\
                                  'temperature_closest_weather_station_id'], \
                    df_observations_temperature))
    tend = time.time()
    print "   Took %.2f Seconds" % (tend - tstart)

    #
    # Wind
    #

    wind_indices_weather_observations = []
    print "// Finding Closest Weather Observations (Wind)"
    tstart = time.time()
    for iix in df_traffic.index:
        wind_indices_weather_observations.append(\
                find_index_of_closest_weather_observation(\
                    df_traffic.ix[iix,'timestamp_seconds'], \
                    df_traffic.ix[iix,\
                                  'wind_closest_weather_station_id'], \
                    df_observations_wind))
    tend = time.time()
    print "   Took %.2f Seconds" % (tend - tstart)

    ###########################################################################
    # Write Weather Observations into Traffic Dataframe
    ###########################################################################
    print "// Writing Weather Observations into Traffic Dataframe"
    tstart = time.time()

    #
    # Precipitation
    #

    df_traffic['precipitation_dt'] = \
        np.asarray(df_observations_precipitation.\
            ix[precipitation_indices_weather_observations, \
               'timestamp_seconds']) - \
        np.asarray(df_traffic.ix[:,'timestamp_seconds'])

    df_traffic['precipitation'] = \
        np.asarray(df_observations_precipitation.\
            ix[precipitation_indices_weather_observations, \
               'precipitation'])

    df_traffic['precipitation_amount'] = \
        np.asarray(df_observations_precipitation.\
            ix[precipitation_indices_weather_observations, \
               'precipitation_amount'])

    df_traffic['precipitation_kind'] = \
        np.asarray(df_observations_precipitation.\
            ix[precipitation_indices_weather_observations, \
               'precipitation_kind'])

    #
    # Temperature
    #

    df_traffic['temperature_dt'] = \
        np.asarray(df_observations_temperature.\
            ix[temperature_indices_weather_observations, \
               'timestamp_seconds']) - \
        np.asarray(df_traffic.ix[:,'timestamp_seconds'])

    df_traffic['temperature'] = \
        np.asarray(df_observations_temperature.\
            ix[temperature_indices_weather_observations, \
               'temperature'])

    df_traffic['relative_humidity'] = \
        np.asarray(df_observations_temperature.\
            ix[temperature_indices_weather_observations, \
               'relative_humidity'])

    #
    # Wind
    #

    df_traffic['wind_dt'] = \
        np.asarray(df_observations_wind.\
            ix[wind_indices_weather_observations, \
               'timestamp_seconds']) - \
        np.asarray(df_traffic.ix[:,'timestamp_seconds'])

    df_traffic['wind_speed'] = \
        np.asarray(df_observations_wind.\
            ix[wind_indices_weather_observations, \
               'wind_speed'])

    df_traffic['wind_direction'] = \
        np.asarray(df_observations_wind.\
            ix[temperature_indices_weather_observations, \
               'wind_direction'])

    tend = time.time()
    print "   Took %.2f Seconds" % (tend-tstart)

    ###########################################################################
    # Write Output
    ###########################################################################
    # Save
    fout = "TrafficWeather_%s" % fname_traffic[-20:]
    print "// Saving To %s" % fout
    with pd.HDFStore(fout, 'w') as store:
        store['df'] = df_traffic
