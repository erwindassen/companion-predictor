#!/usr/bin/env bash

# Locations that contain "extreme events"
 
python weather_trafficcount_timeseries.py \
  --lat 49.98188665 --lon 7.945090557 --radius 20 \
  --outfile ~/companion-data/reduced_germany_2015_timeseries/extreme1_20km.pddf.hdf5 \
  ~/companion-data/

python weather_trafficcount_timeseries.py \
  --lat 48.09601721 --lon 9.823858097 --radius 20 \
  --outfile ~/companion-data/reduced_germany_2015_timeseries/extreme2_20km.pddf.hdf5 \
  ~/companion-data/

python weather_trafficcount_timeseries.py \
  --lat 47.82897834 --lon 12.69356186 --radius 20 \
  --outfile ~/companion-data/reduced_germany_2015_timeseries/extreme3_20km.pddf.hdf5 \
  ~/companion-data/

python weather_trafficcount_timeseries.py \
  --lat 48.02506997 --lon 10.71590852 --radius 20 \
  --outfile ~/companion-data/reduced_germany_2015_timeseries/extreme4_20km.pddf.hdf5 \
  ~/companion-data/

python weather_trafficcount_timeseries.py \
  --lat 48.04131603 --lon 10.79041623 --radius 20 \
  --outfile ~/companion-data/reduced_germany_2015_timeseries/extreme5_20km.pddf.hdf5 \
  ~/companion-data/

python weather_trafficcount_timeseries.py \
  --lat 50.58242127 --lon 7.479772229 --radius 20 \
  --outfile ~/companion-data/reduced_germany_2015_timeseries/extreme6_20km.pddf.hdf5 \
  ~/companion-data/
