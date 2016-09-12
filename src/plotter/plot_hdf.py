"""
Plots the input data to the predictor (unprocessed HDF5 files).
"""

import sys
import argparse

import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt



#DATEFMT = '%d/%m %H:%M:%S'
DATEFMT = '%A %H:%M:%S'

FEATURES = [
    'precipitation',
    'temperature',
    'windspeed'
]
TARGET = [
    'trafficflow',
    'trafficspeed'
]
FEATURES_DF = [
    'precipitation mm/h',
    'temperature C',
    'windspeed m/s'
]
TARGET_DF = [
    'trafficflow counts/h',
    'trafficspeed km/h'
]

def plot_site_df(df, fields=TARGET_DF):
    """Plot pandas DataFrame"""
 
    for i, f in enumerate(fields):
        # Set up subplot for field f
        plt.subplot(len(fields), 1, i+1)
        dates = [dt.datetime.fromtimestamp(ts) for ts in df['timestamp_start']]
        datenums = md.date2num(dates)
        plt.xticks(rotation=25)
        ax = plt.gca()
        xfmt = md.DateFormatter(DATEFMT)
        ax.xaxis.set_major_formatter(xfmt)
        plt.ylabel(f)

        # Plot data
        plt.plot(datenums, df[f], 'r.-')


def plot_site(hdf_filename, site_hash, fields=TARGET):
    """Show the time series for a particular site in a subplot"""

    # Open file and select group/site
    h5file = h5py.File(hdf_filename, 'r')
    ds = h5file[site_hash]

    for i, f in enumerate(fields):
        # Set up subplot for field f
        plt.subplot(len(fields), 1, i+1)
        dates = [dt.datetime.fromtimestamp(ts) for ts in ds[f][:,0]]
        datenums = md.date2num(dates)
        plt.xticks(rotation=25)
        ax = plt.gca()
        xfmt = md.DateFormatter(DATEFMT)
        ax.xaxis.set_major_formatter(xfmt)
        plt.ylabel(f)

        # Plot data
        plt.plot(datenums, ds[f][:,2], 'r.-')


def random_site_hash(hdf_filename):
    """Returns a random site identifier"""
    h5file = h5py.File(hdf_filename, 'r')
    sites = [k for k in h5file.keys() if k.startswith('rws')]
    return random.choice(sites)


def plot_df(df):
    """Plot pandas DataFrame"""

    # Plot target measurements
    plot_site_df(df, fields=TARGET_DF)
    plt.suptitle("Site: '%s'" % df['site'][0])
    plt.subplots_adjust(hspace=0.6)

    # Plot features
    plt.figure()
    plot_site_df(df, fields=FEATURES_DF)
    plt.suptitle("Site: '%s'" % df['site'][0])
    plt.subplots_adjust(hspace=0.6)

    # Show plots
    plt.show()


def plot_hdf(hdf_filenames, site_hash=None):
    """Plot HDF's"""
    if site_hash is None:
        site_hash = random_site_hash(random.choice(hdf_filenames))

    # Plot target measurements
    for hdf in hdf_filenames:
        try:
            plot_site(hdf, site_hash)
        except Exception as ex:
            print("Could not plot site '%s' in file '%s': %s" % (site_hash, hdf, ex))
    plt.suptitle("Site: '%s'" % site_hash)
    plt.subplots_adjust(hspace=0.6)

    # Plot features
    plt.figure()
    plot_site(random.choice(hdf_filenames), site_hash, fields=FEATURES)
    plt.suptitle("Site: '%s'" % site_hash)
    plt.subplots_adjust(hspace=0.6)

    # Show plots
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file',
                        nargs='*',
                        help="HDF5 input file")
    parser.add_argument('--site',
                        required=False,
                        help="Site hash. If omitted, a random site is chosen.")
    args = parser.parse_args()
    try:
        sys.exit(plot_hdf(args.input_file, site_hash=args.site))
    except Exception as ex:
        print("Something went wrong: %s" % ex)
        sys.exit(1)
    
