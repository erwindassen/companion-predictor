"""
Ingest Weather Stations into Pandas/HDF5.
Note that the lists are different for each observable.
Cf. ftp://ftp-cdc.dwd.de/pub/CDC/observations_germany/climate/hourly/

Call Signatures (Precipitation, Wind, Temperature):
$ python ingest_weather_stations.py --precipitation \
    RR_Stundenwerte_Beschreibung_Stationen.txt
$ python ingest_weather_stations.py --wind \
    FF_Stundenwerte_Beschreibung_Stationen.txt 
$ python ingest_weather_stations.py --temperature \
    TU_Stundenwerte_Beschreibung_Stationen.txt 
"""

import pandas as pd
import argparse


###############################################################################
###############################################################################
# FUNCTIONS
###############################################################################
###############################################################################

def remove_umlauts(string):
    """
    Removes (ae,oe,ue,sz) Umlauts.
    @param: string -- Input String
    @return: string -- Cleaned Up String
    """
    string = str.replace(string, '\xe4', 'ae')
    string = str.replace(string, '\xfc', 'ue')
    string = str.replace(string, '\xf6', 'oe')
    string = str.replace(string, '\xdf', 'ss')
    return string


###############################################################################
###############################################################################
# MAIN CODE
###############################################################################
###############################################################################

# Parse Arguments
parser = argparse.ArgumentParser()
group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument('--temperature', \
                    help='DWD List of Stations w/ Temperature Data.')
group1.add_argument('--precipitation', \
                    help='DWD List of Stations w/ Precipitation Data.')
group1.add_argument('--wind',\
                     help='DWD List of Stations w/ Wind Data.')
args = parser.parse_args()

# Info & Output Filename
if args.temperature:
    print '// Parsing Stations w/ Temperature Data'
    fout = 'weather_stations_temperature.pddf.hdf5'
    fin = args.temperature
elif args.wind:
    print '// Parsing Stations w/ Wind Data'
    fout = 'weather_stations_wind.pddf.hdf5'
    fin = args.wind
elif args.precipitation:
    print '// Parsing Stations w/ Precipitation Data'
    fout = 'weather_stations_precipitation.pddf.hdf5'
    fin = args.precipitation
else:
    raise Exception('Something Went Wrong.')

# DWD Weather Data Station Lists Are Fixed-Width-Files
# Wind & Temperature Have Same Spacings, Precipitation Is Different
if args.temperature or args.wind:
    colspecs = [ (0,11), (12,20), (21,29), \
                 (30,44), (45,56), (57,66), \
                (67,107), (108,140) ]
elif args.precipitation:
    colspecs = [ (0,5), (6,14), (15,23), \
                 (24,38), (39,50), (51,60), \
                 (61,101), (102,130) ]
names = [ 'station_id', 'operational_from', 'operational_to', \
          'elevation', 'latitude', 'longitude', \
          'station_name', 'station_bundesland' ]

# Load Data
df = pd.read_fwf(fin, skiprows=2, colspecs=colspecs, \
                 names=names, \
                 header=None, skipfooter=1, engine='python')

# Fix Dates
df['operational_from'] = pd.to_datetime(df.operational_from, format='%Y%m%d')
df['operational_to'] = pd.to_datetime(df.operational_to, format='%Y%m%d')

# Fix Umlauts
df['station_name'] = df.station_name.apply(lambda x: remove_umlauts(x))
df['station_bundesland'] = \
    df.station_bundesland.apply(lambda x: remove_umlauts(x))

# Save
print "// Saving To %s" % fout
with pd.HDFStore(fout, 'w') as store:
    store['df'] = df
