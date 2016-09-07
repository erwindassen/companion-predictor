"""
Ingest Weather Observations into Pandas/HDF5.
Cf. ftp://ftp-cdc.dwd.de/pub/CDC/observations_germany/climate/hourly/
We keep only the year 2015 at the moment. See line 83.

You must pass a list of produkt_XXXX_*.zip files, where
XXXX = wind  - Wind Data
XXXX = synop - Precipitation Data
XXXX = temp  - Air Temperature Data

Call Signatures (Precipitation, Wind, Temperature):
$ python ingest_weather_stations.py --precipitation < fnames
$ python ingest_weather_stations.py --wind < fnames
$ python ingest_weather_stations.py --temperature < fnames

$ cat fnames
produkt_wind_Terminwerte_19370101_20110331_00003.txt
produkt_wind_Terminwerte_19690101_19951130_00044.txt
...

To generate the file list, use
$ ls -1 produkt_wind*.txt > fnames

NB: You may want to use unpack_weather_obs.sh to unpack weather data.
"""

import pandas as pd
import sys
import numpy as np
import argparse


# Parse Arguments
parser = argparse.ArgumentParser()
group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument('--temperature', action='store_true', \
                    help='Ingest Temperature Data.')
group1.add_argument('--precipitation', action='store_true', \
                    help='ingest Precipitation Data.')
group1.add_argument('--wind', action='store_true', \
                     help='Ingest Wind Data.')
args = parser.parse_args()

# List of Files
if sys.stdin.isatty():
    print "!! No File List (Use Stdin)."
    sys.exit()
else:
    fnames = sys.stdin.read().rstrip("\n").split("\n")
    print "// Reading %i Stations" % len(fnames)

# Read Weather Data
first = True
for ifname, fname in enumerate(fnames):
    print "// Reading %s" % fname

    # Type of Records & Output File
    if args.precipitation:
        names = [ 'station_id', 'timestamp', \
                  'quality', 'precipitation', \
                  'precipitation_amount', \
                  'precipitation_kind', 'X' ]
        cols = [ 0, 1, 2, 3, 4, 5 ]
        fout = 'weather_obs_precipitation.pddf.hdf5'
    elif args.temperature:
        names = [ 'station_id', 'timestamp', \
                  'quality', 'structure_version', 
                  'temperature', 'relative_humidity', 'X' ]
        cols = [ 0, 1, 2 , 3, 4, 5 ]
        fout = 'weather_obs_temperature.pddf.hdf5'
    elif args.wind:
        names = [ 'station_id', 'timestamp', \
                  'quality', 'structure_version', \
                  'wind_speed', 'wind_direction', 'X' ]
        cols = [ 0, 1, 2 , 3, 4, 5 ]
        fout = 'weather_obs_wind.pddf.hdf5'

    # Load Data
    df = pd.read_csv(fname, sep=';', skipinitialspace=True, names=names, \
                     skiprows=1, usecols=cols, skipfooter=1, engine='python')

    # We Work On 2015 Traffic.
    df = df[df.timestamp > 2015000000]

    # We may have just removed all useful data
    if len(df) > 0:

        # Typecast
        df['timestamp'] = pd.to_datetime(df.timestamp, format='%Y%m%d%H')
        if args.precipitation:
            df['precipitation'] = df['precipitation'].astype(np.bool)

        # Drop useless columns
        if args.precipitation:
            df = df[['station_id', 'timestamp', \
                     'precipitation', \
                     'precipitation_amount', \
                     'precipitation_kind']]
        elif args.temperature:
            df = df[['station_id', 'timestamp', \
                     'temperature', 'relative_humidity']]
        elif args.wind:
            df = df[[ 'station_id', 'timestamp', \
                      'wind_speed', 'wind_direction' ]]

        # Append to Master Frame
        if first:
            df_master = df.copy()
            first = False
        else:
            df_master = pd.concat([df_master,df])

# Reset Indices
df_master.reset_index(drop=True, inplace=True)

# # Save
print "// Storing %i Records as %s" % (len(df_master.index), fout)
with pd.HDFStore(fout, 'w') as store:
    store['df'] = df_master

