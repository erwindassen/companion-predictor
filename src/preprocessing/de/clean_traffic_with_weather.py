"""
Clean up table of incidents (with weather).

In Particular:
1/ Throw away wind direction
2/ Throw away measurements from stations that are too-far-away(TM)
3/ Throw away -999 (missing observatons)
4/ Fix -999 (missing observation) for Precipitation (fill based on temperature)
5/ Throw away duplicates (same location, same time, same event type)

This greatly reduces data volume!
Expect to keep at most a few thousand rows per month.

Call Signature:
$ python clean_traffic_with_weather.py --fname_out out_clean.pddf.hdf5 < fnames

$ cat fnames
TrafficWeather_2015_01_02.pddf.hdf5
TrafficWeather_2015_01_03.pddf.hdf5

To generate the file list, use
$ ls -1 TrafficWeather_*.pddf.hdf5 > fnames
"""

import pandas as pd
import numpy as np
import sys
import argparse


###############################################################################
###############################################################################
# FUNCTIONS
###############################################################################
###############################################################################

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


###############################################################################
###############################################################################
# MAIN CODE
###############################################################################
###############################################################################

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fname_out', required=True, \
                    help='Output Filename.')
args = parser.parse_args()

# List of Files
if sys.stdin.isatty():
    print "!! No File List (Use Stdin)."
    sys.exit()
else:
    fnames_in = sys.stdin.read().rstrip("\n").split("\n")
    print ''
    print "** Reading %i Traffic Event Files" % len(fnames_in)
    print ''

# Loop Inputs
for ifile, fname_in in enumerate(fnames_in):

    # Read Input
    print "// Reading %s (%i/%i)" % (fname_in, ifile+1, len(fnames_in))
    df_in = pd.read_hdf(fname_in, 'df')
    print "   %i Rows Pre-Cleaning" % len(df_in.index)

    # Clean
    df_in[df_in.precipitation_amount<0.0] = np.nan
    df_in[df_in.temperature==-999.0] = np.nan
    df_in[df_in.relative_humidity==-999.0] = np.nan
    df_in[df_in.wind_speed==-999.0] = np.nan

    # Drop Wind Direction
    if 'wind_direction' in df_in.keys():
        df_in.drop('wind_direction', axis=1, inplace=True)
        
    # Drop Wind Speed
    # if 'wind_speed' in df.keys():
    #     df.drop('wind_speed', axis=1, inplace=True)

    # Drop NaN
    df_in.dropna(how='any', inplace=True)

    # Remove Too Far Away Stations
    df_in = df_in[df_in.precipitation_closest_weather_station_dxdy <= 2.0]
    df_in = df_in[df_in.temperature_closest_weather_station_dxdy <= 0.5]
    df_in = df_in[df_in.wind_closest_weather_station_dxdy <= 2.0]

    # Info
    print "   %i Rows Post-Cleaning" % len(df_in.index)

    # Classify Unclassified Precipitation
    idx = (df_in.precipitation_kind==-999) & (df_in.precipitation_amount>0.0)
    df_in.ix[idx,'precipitation_kind'] = \
        df_in[idx].temperature.apply(fill_in_precipitation_kind)

    # Classify Unclassified Non-Precipitation
    idx = (df_in.precipitation_kind==-999) & (df_in.precipitation_amount==0.0)
    df_in.ix[idx,'precipitation_kind'] = np.zeros(np.sum(idx))

    # Reclassify Misclassified Precipitation
    # Data that has 0.0 precipitation but is marked as rain/snow/slush
    # will be reassigned precipitation_kind 0
    idx = (df_in.precipitation_kind>1) & (df_in.precipitation_amount==0.0)
    df_in.ix[idx,'precipitation_kind'] = np.zeros(np.sum(idx))

    # Drop Duplicates
    df_in.drop_duplicates(inplace=True, subset=['timestamp', \
                                                'location_openlr_base64', \
                                                'event_type'])
    
    # Info
    print "   %i Rows After Removing Duplicates" % len(df_in.index)

    # Copy to Output
    if ifile == 0:
        df_out = df_in
    else:
        df_out = pd.concat([df_out, df_in])

    # Close
    del df_in

# Reset Index
df_out.reset_index(inplace=True, drop=True)

# Save
print "// Saving to %s" % args.fname_out
with pd.HDFStore("%s" % args.fname_out) as store:
    store['df'] = df_out
