#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Given a circular area, produce a time series of weather features + incident
count for each hour. Output file is written to
<basedir>/reduced_germany_2015_timeseries.pddf.hdf5

Basedir for weather+traffic data has the following directory structure:
reduced_germany_2015_merged
├── 2015_01
│   ├── TrafficWeather_2015_01_02.pddf.hdf5
│   ├── TrafficWeather_2015_01_03.pddf.hdf5
│   ├── TrafficWeather_2015_01_04.pddf.hdf5
│   ├── TrafficWeather_2015_01_05.pddf.hdf5
│   ├── TrafficWeather_2015_01_07.pddf.hdf5
│   ├── TrafficWeather_2015_01_08.pddf.hdf5
│   ├── TrafficWeather_2015_01_09.pddf.hdf5
│   ├── TrafficWeather_2015_01_10.pddf.hdf5
├── 2015_02
....
├── 2015_11
├── 2015_12
reduced_germany_2015_traffic
├── 2015_01
...
└── 2015_12
reduced_germany_2015_weather
├── weather_obs_precipitation.pddf.hdf5
├── weather_obs_temperature.pddf.hdf5
├── weather_obs_wind.pddf.hdf5
├── weather_stations_precipitation.pddf.hdf5
├── weather_stations_temperature.pddf.hdf5
└── weather_stations_wind.pddf.hdf5
"""

import sys
import argparse
import os
import datetime as dt
import fnmatch
from tqdm import tqdm # Progress bar

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DT_START = dt.datetime.strptime('2015-01-01 00:00', '%Y-%m-%d %H:%M')
DT_STOP = dt.datetime.strptime('2015-12-31 23:00', '%Y-%m-%d %H:%M')
DATERANGE = pd.date_range(DT_START, DT_STOP, freq='H')

WEATHER_BASE = "reduced_germany_2015_weather"
TRAFFIC_BASE = "reduced_germany_2015_traffic"
PRECIPITATION_OBS = "weather_obs_precipitation.pddf.hdf5"
PRECIPITATION_ST = "weather_stations_precipitation.pddf.hdf5"
TEMPERATURE_OBS = "weather_obs_temperature.pddf.hdf5"
TEMPERATURE_ST = "weather_stations_temperature.pddf.hdf5"
WIND_OBS = "weather_obs_wind.pddf.hdf5"
WIND_ST = "weather_stations_wind.pddf.hdf5"


def _deg(rad):
    return rad * 180.0 / np.pi


def _rad(deg):
    return deg * np.pi / 180


def haversine(c1, c2):
    """Calculate the distance between (lat1, lon1) and (lat2, lon2) using
       the haversine formula. Approximates earth to a sphere.
    """
    R = 6371e3
    lat1, lon1 = c1
    lat2, lon2 = c2
    dlambda = _rad((lon2 - lon1))
    dphi = _rad(lat2 - lat1)

    a = np.sin(dphi/2) * np.sin(dphi/2) + \
        np.cos(_rad(lat1)) * np.cos(_rad(lat2)) * \
        np.sin(dlambda/2) * np.sin(dlambda/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R*c
    return d


def get_filelist(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, 'Traffic*.pddf.hdf5'):
            filelist.append(os.path.join(root, filename))
    return filelist


def _trafficfile_sort(filename):
    basename = os.path.basename(filename)
    parts = basename.split('_')
    return int(parts[1] + parts[2] + parts[3].split('.')[0]) # Fuck regex


def area_obs(stfile, obsfile, lat, lon, radius):
    print "Reading stations file %s ..." % stfile
    df_st = pd.read_hdf(stfile, 'df')

    print "%d stations" % df_st.size
    print "Apply spatial filter ..."
    df_st = df_st[lambda df: haversine((df.latitude, df.longitude), (lat, lon)) < radius]
    print "%d stations within area" % df_st.size

    print "Reading observations file %s ..." % obsfile
    df_obs = pd.read_hdf(obsfile, 'df')
    print "Observations: %d" % df_obs.size
    df_obs = pd.merge(df_obs, df_st, how='inner', on=u'station_id')
    print "Observations within area: %d" % df_obs.size
    print "Group by timestamp and calculate mean ... "
    return df_obs.groupby(['timestamp']).mean()


def incident_count(f, lat, lon, radius):
    #print "Count incidents in file %s ..." % f
    df_traffic = pd.read_hdf(f, 'df')
    #print "%d incidents" % df_traffic.size
    #print "Apply spatial filter ..."
    df_traffic = df_traffic[lambda df: haversine((df.location_lat_decimal,
                                                  df.location_lon_decimal),
                                                  (lat, lon)) < radius]
    #print "%d incidents within area" % df_traffic.size
    #print "Remove duplicates ..."
    df_traffic.drop_duplicates(inplace=True, subset=['timestamp',
                                                     'location_openlr_base64',
                                                     'event_type'])
    #print "%d incidents within area (after sanitation)" % df_traffic.size
    
    df_traffic['count'] = 1
    count = df_traffic.groupby([pd.Grouper(freq='1H',key='timestamp'),
                                'event_type']).sum()
    count.reset_index(inplace=True)
    return count


def fill_in_precipitation_kind(temperature):
    """
    Generates a precipitation type based on temperature T.
    Conventions from DWD.
    - 4C < T      -- Rain  (Type 6)
    - 0C < T < 4C -- Slush (Type 8)
    -      T < 0C -- Snow  (Type 7)
    @param: temperature         [Deg C]
    @return: precipitation_type [Integer]
    """
    if temperature >= 4: return 6
    elif temperature < 0: return 7
    else: return 8


def main(basedir, lat, lon, radius):

    df_out = pd.DataFrame(index=DATERANGE)

    # Filter and aggregate precipitation
    df_obs = area_obs(os.path.join(basedir, WEATHER_BASE, PRECIPITATION_ST),
                      os.path.join(basedir, WEATHER_BASE, PRECIPITATION_OBS),
                      lat, lon, radius)
    df_out[u'precipitation'] = df_obs[u'precipitation']
    df_out[u'precipitation_amount'] = df_obs[u'precipitation_amount']
    df_out[u'precipitation_kind'] = df_obs[u'precipitation_kind']

    # Filter and aggregate temp
    df_obs = area_obs(os.path.join(basedir, WEATHER_BASE, TEMPERATURE_ST),
                      os.path.join(basedir, WEATHER_BASE, TEMPERATURE_OBS),
                      lat, lon, radius)
    df_out[u'temperature'] = df_obs[u'temperature']
    df_out[u'relative_humidity'] = df_obs[u'relative_humidity']


    # Filter and aggregate precipitation
    df_obs = area_obs(os.path.join(basedir, WEATHER_BASE, WIND_ST),
                      os.path.join(basedir, WEATHER_BASE, WIND_OBS),
                      lat, lon, radius)
    df_out[u'wind_speed'] = df_obs[u'wind_speed']

    # Get list of traffic files
    filelist = get_filelist(os.path.join(basedir, TRAFFIC_BASE))
    filelist.sort(key=_trafficfile_sort)
    df_out['incidents_queueing'] = np.nan
    df_out['incidents_slow'] = np.nan
    df_out['incidents_stationary'] = np.nan

    # Iterate over traffic data
    print "Adding incident counts ..."
    for f in tqdm(filelist):
        count = incident_count(f, lat, lon, radius)
        for i,row in count.iterrows():
            if row['event_type'] == 'queueingTraffic':
                df_out.ix[row.timestamp,'incidents_queueing'] = row['count']
            elif row['event_type'] == 'slowTraffic':
                df_out.ix[row.timestamp,'incidents_slow'] = row['count']
            elif row['event_type'] == 'stationaryTraffic':
                df_out.ix[row.timestamp,'incidents_stationary'] = row['count']
            else:
                pass

    print "Classify Unclassified Precipitation"
    idx = (df_out.precipitation_kind==-999) & (df_out.precipitation_amount>0.0)
    df_out.ix[idx,'precipitation_kind'] = \
                df_out[idx].temperature.apply(fill_in_precipitation_kind)

    print df_out.head(30)
    print df_out.tail(5)
    assert len(df_out.index) == 8760

    outfile = os.path.join(basedir, 'reduced_germany_2015_timeseries.pddf.hdf5')
    print "Writing to %s ..." % outfile
    df_out.to_hdf(outfile, 'df', mode='w')
    print "Done."


if __name__ == '__main__':
    _help = \
"""
Given a circular area, produce a time series of weather features + incident
count for each hour. Output file is written to
<basedir>/reduced_germany_2015_timeseries.pddf.hdf5
"""


    parser = argparse.ArgumentParser(description="")
    parser.add_argument('basedir',
                        help="Basedir for weather+traffic data. \
                              See docstring for directory structure")
    parser.add_argument('--lat',
                        required=True,
                        help="Latitude of area center")
    parser.add_argument('--lon',
                        required=True,
                        help="Longitude of area center")
    parser.add_argument('--radius',
                        required=True,
                        help="Area radius [km]")
    args = parser.parse_args()

    try:
        sys.exit(main(args.basedir, float(args.lat), float(args.lon), int(args.radius)*1e3))
    except Exception as ex:
        print("Something went wrong: %s" % ex)
        sys.exit(1)
    

