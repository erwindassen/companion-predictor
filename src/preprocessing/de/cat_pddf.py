"""
Concatenates Pandas Dataframes encapsulated in HDF5 Files.
Dataframe must be called "df" within the HDF5.

Call Signature:
$ python cat_pddf.py --fname_out out_merged.pddf.hdf5 < fnames

$ cat fnames
TrafficWeather_2015_01_02.pddf.hdf5
TrafficWeather_2015_01_03.pddf.hdf5

To generate the file list, use
$ ls -1 TrafficWeather_*.txt > fnames
"""

import sys
import pandas as pd
import argparse


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
    print "** Reading %i HDF5 Wrapped Dataframes" % len(fnames_in)
    print ''

# Loop
for ifile, fname_in in enumerate(fnames_in):
    print "// Reading %s (%i/%i)" % (fname_in, ifile+1, len(fnames_in))
    df_in = pd.read_hdf(fname_in, 'df')
    if ifile == 0:
        df_out = df_in
    else:
        df_out = pd.concat([df_out, df_in])

# Reindex
df_out.reset_index(drop=True, inplace=True)

# Save
print "// Saving %s" % args.fname_out
with pd.HDFStore(args.fname_out, 'w') as store:
    store['df'] = df_out
