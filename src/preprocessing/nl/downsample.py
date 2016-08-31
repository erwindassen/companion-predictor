"""
Downsample Data. Takes ~1 Hour per Week for ~12k Stations.

Do note that there is a large amount of redundancy in the data. For example,
timestamps are stored for each station/group and each dataset. After this
downsampler is done, all timestamps should be aligned, so we can remove a lot
of this redundant data.

Call Signature:
$ python cat_pddf.py --fname_out out_merged.pddf.hdf5 < fnames

$ cat fnames
TS_2016-05-24-00_2016-06-01-00_0-200_20160624101922.hdf
TS_2016-05-24-00_2016-06-01-00_1000-1200_20160624101922.hdf
TS_2016-05-24-00_2016-06-01-00_10000-10200_20160624101922.hdf

To generate the file list, use
$ ls -1 TS_2016_*.hdf > fnames

@todo: Use more appropriate datatypes (int, etc). There's no need to encode 
       integer data as floats! This isn't so much to conserve storage space,
       but make filtering operations (<>=) more robust.
@todo: Copy over attributes where units are described.
@todo: Remove redundant timestamp encodings.
"""

import h5py
import numpy as np
import time
import sys
import argparse


###############################################################################
###############################################################################
# FUNCTIONS
###############################################################################
###############################################################################

def get_indices_for_hour_markers(timestamps_start):
    """
    This gets the indices in the timestamp array for when a new hour starts.
    E.g., from [ 12:48, 12:58, 13:08 ] the index to the second item is given.

    @param: timestamps_start           -- Array of Unix Timestamps [Seconds]
    @return: indices_where_hours_start -- Array of first index in an hour
    @return: indices_where_hours_end   -- Array of last index in an hour
    """
    
    timestamps_start = np.asarray(timestamps_start, dtype=np.int64)
    timestamps_start_mod_tszz = timestamps_start %  3600
    
    indices_where_hours_start = \
        np.where(np.diff(timestamps_start_mod_tszz)<0)[0]+1
    indices_where_hours_end = \
        indices_where_hours_start[1:]-1
    indices_where_hours_end = \
        np.concatenate([indices_where_hours_end, \
                        np.atleast_1d(timestamps_start.shape[0])-1])

    # Does the first hour not start at the zero-th index?
    # Then we have a partial hour before...
    if indices_where_hours_start[0] > 0:
        indices_where_hours_start = \
            np.concatenate([np.atleast_1d(0), indices_where_hours_start])
        indices_where_hours_end = \
            np.concatenate([np.atleast_1d(indices_where_hours_start[1]-1), \
                           indices_where_hours_end])

    # Debug
    # print indices_where_hours_start
    # print indices_where_hours_end

    # Return
    return indices_where_hours_start, indices_where_hours_end


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
    fnames = sys.stdin.read().rstrip("\n").split("\n")
    print "// Loading from %i Files" % len(fnames)

# Load First File, Build Station list, Remove Non-Station Entry
with h5py.File(fnames[0], 'r') as f0:
    stations = f0.keys()
    print "// %i Total Stations" % len(stations)
    if '__DATA_TYPES__' in stations:
        stations.remove('__DATA_TYPES__')

# DEBUG, FEWER STATIONS
# stations = stations[:100]
print "// Working On %i Stations" % len(stations)

# Remove Bad Weather Data
weather_fields = [ 'precipitation', 'temperature', 'windspeed' ]
print "// Checking for Bad Stations"
with h5py.File(fnames[0], 'r') as f0:
    bad_stations = []
    for istation, station in enumerate(stations):
        if istation % 100 == 0: print "   Station %i" % istation
        for weather_field in weather_fields:
            if np.isnan(f0["%s/%s" % (station, weather_field)][:,:]).any():
                bad_stations.append(station)
                break

print "// %i Bad Stations (Missing Weather Timestamps)" % len(bad_stations)
for bad_station in bad_stations:
    if bad_station in stations:
        stations.remove(bad_station)

# Open New File, Create Groups, Copy in Weather Data & Data Types
with h5py.File(args.fname_out, 'w') as fx:
    with h5py.File(fnames[0], 'r') as f0:
        for station in stations:
            fx.create_group(station)
            f0.copy("%s/precipitation" % station, fx["/%s" % station])
            f0.copy("%s/temperature" % station, fx["/%s" % station])
            f0.copy("%s/windspeed" % station, fx["/%s" % station])
        f0.copy('__DATA_TYPES__', fx)

# Make Sure Weather Timestamps Are Aligned
print "// Verifying Timestamp Alignment"
with h5py.File(fnames[0], 'r') as f0:
    for istation, station in enumerate(stations):
        if istation % 100 == 0: print "   Station %i" % istation
        for ifield in range(len(weather_fields)-1):
            assert \
                np.array_equal(\
                        f0["%s/%s" % (station, weather_fields[ifield])][:,0], \
                        f0["%s/%s" % (station, weather_fields[ifield+1])][:,0])
            assert \
                np.array_equal(\
                        f0["%s/%s" % (station, weather_fields[ifield])][:,1], \
                        f0["%s/%s" % (station, weather_fields[ifield+1])][:,1])

# Datasets to Copy & Downsample
dset_names = [ 'trafficflow', 'trafficspeed' ]

# Load Entire Timeseries For A Station
print "// Downsampling Time Series"
with h5py.File(args.fname_out, 'a') as fx:
    for istation, station in enumerate(stations):
        if istation % 100 == 0: print "   Station %i" % istation

        total_rows = [ 0 ] * len(dset_names)
        where_to_append = [ 0 ] * len(dset_names)
        dset_hires = [ None ] * len(dset_names)
        dset_lores = [ None ] * len(dset_names)
        
        for ifile, fname in enumerate(fnames):
            for dset_id, dset_name in enumerate(dset_names):
                try:
                    with h5py.File(fname, 'r') as ff:
                        total_rows[dset_id] += \
                            ff["%s/%s" % (station, dset_name)].shape[0]

                        # Copy Over All Data
                        if ifile == 0:
                            dset_hires[dset_id] = \
                                fx.create_dataset("%s/%s_hires" % \
                                                  (station, dset_name), \
                                                  (total_rows[dset_id], 3), \
                                                  maxshape=(None, 3), \
                                                  dtype=np.float64)
                            dset_hires[dset_id][:,:] = \
                                ff["%s/%s" % (station, dset_name)]
                            where_to_append[dset_id] = total_rows[dset_id]

                        else:
                            dset_hires[dset_id].resize(total_rows[dset_id], \
                                                       axis=0)
                            dset_hires[dset_id][where_to_append[dset_id]:total_rows[dset_id]:] = \
                                ff["%s/%s" % (station, dset_name)]
                            where_to_append[dset_id] = total_rows[dset_id]
                except:
                    pass

        # Reduce Data
        dset_lores = [ None ] * len(dset_names)
        for dset_id, dset_name in enumerate(dset_names):

            # Extract Hour Indices
            indices_where_hours_start, indices_where_hours_end = \
                get_indices_for_hour_markers(dset_hires[dset_id][:,0])
            timestamps_start = dset_hires[dset_id][indices_where_hours_start,0]
            timestamps_end = dset_hires[dset_id][indices_where_hours_end,1]

            # Average Over Hours
            averaged_data = np.nan * \
                np.zeros(len(indices_where_hours_start), dtype=np.float64)
            for ihour in range(len(indices_where_hours_start)):
                set_to_average = \
                    dset_hires[dset_id][indices_where_hours_start[ihour]:\
                                        indices_where_hours_end[ihour]+1,2]
                # Trafficspeed is zero if no cars pass by. We don't want
                # to average over zero, so we remove those points
                # Of course, this is also a float, so don't compare on 0.0
                if dset_name == 'trafficspeed':
                    set_to_average = set_to_average[set_to_average>0.01]
                averaged_data[ihour] = np.mean(set_to_average)

            # Write Back
            dset_lores[dset_id] = \
                fx.create_dataset("%s/%s_lores" % \
                                  (station, dset_name), \
                                  (len(averaged_data), 3), \
                                  dtype=np.float64)
            dset_lores[dset_id][:,0] = timestamps_start
            dset_lores[dset_id][:,1] = timestamps_end
            dset_lores[dset_id][:,2] = averaged_data

        # Remove Hires Data
        for dset_id, dset_name in enumerate(dset_names):
            del fx["%s/%s_hires" % (station, dset_name)]

# Debug
# with h5py.File(OUTPUT_FILE, 'r') as fx:
#     for istation, station in enumerate(stations):
#         for dset_id, dset_name in enumerate(dset_names):
#             print fx["%s/%s_lores" % (station, dset_name)].shape

# Syncing Time Stamps
print "// Synchronising Time Stamps"
with h5py.File(args.fname_out, 'a') as fx:
    for istation, station in enumerate(stations):
        if istation % 100 == 0: print "   Station %i" % istation

        # Compute Intersections
        t_weather = np.array(fx["%s/temperature" % station][:,0], \
                             dtype=np.int64)
        for dset_id, dset_name in enumerate(dset_names):
            t_traffic = \
                np.array(fx["%s/%s_lores" % (station, dset_name)][:,0], \
                         dtype=np.int64)
            t_intrsct = np.intersect1d(t_weather, t_traffic)

        # Sync Traffic Fields
        for dset_id, dset_name in enumerate(dset_names):
            ds_tmp = \
                fx["%s/%s_lores" % \
                   (station, dset_name)][np.in1d(t_traffic, t_intrsct),:]
            del fx["%s/%s_lores" % (station, dset_name)]
            fx.create_dataset("%s/%s_lores" % (station, dset_name), \
                              data=ds_tmp, \
                              dtype=np.float64)

        # Sync Weather Fields
        for weather_field in weather_fields:
            ds_tmp = \
                fx["%s/%s" % \
                   (station, weather_field)][np.in1d(t_weather, t_intrsct),:]
            del fx["%s/%s" % (station, weather_field)]
            fx.create_dataset("%s/%s" % (station, weather_field), \
                              data=ds_tmp, \
                              dtype=np.float64)

# Done
print "// Done"
