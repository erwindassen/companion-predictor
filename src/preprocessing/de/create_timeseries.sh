#!/usr/bin/env bash


# Big cities
for i in 5 10 20 50
do
  #Munich
  python weather_trafficcount_timeseries.py \
    --lat 48.138384 --lon 11.579089 --radius $i \
    --outfile ~/companion-data/reduced_germany_2015_timeseries/munich_${i}km.pddf.hdf5 \
    ~/companion-data/

  # Frankfurt
  python weather_trafficcount_timeseries.py \
    --lat 50.114305 --lon 8.685025 --radius $i \
    --outfile ~/companion-data/reduced_germany_2015_timeseries/frankfurt_${i}km.pddf.hdf5 \
    ~/companion-data/

  # Berlin
  python weather_trafficcount_timeseries.py \
    --lat 52.514563 --lon 13.405539 --radius $i \
    --outfile ~/companion-data/reduced_germany_2015_timeseries/berlin_${i}km.pddf.hdf5 \
    ~/companion-data/

  # Hamburg
  python weather_trafficcount_timeseries.py \
    --lat 53.547776 --lon 10.000747 --radius $i \
    --outfile ~/companion-data/reduced_germany_2015_timeseries/hamburg_${i}km.pddf.hdf5 \
    ~/companion-data/

  # Cologne
  python weather_trafficcount_timeseries.py \
    --lat 50.939396 --lon 6.961496 --radius $i \
    --outfile ~/companion-data/reduced_germany_2015_timeseries/cologne_${i}km.pddf.hdf5 \
    ~/companion-data/

  # Stuttgart
  python weather_trafficcount_timeseries.py \
    --lat 48.774643 --lon 9.176101 --radius $i \
    --outfile ~/companion-data/reduced_germany_2015_timeseries/stuttgart_${i}km.pddf.hdf5 \
    ~/companion-data/

done


# Other areas

# Near Walsrode, intersection between highway 7 and 27
python weather_trafficcount_timeseries.py \
  --lat 52.852745 --lon 9.700032 --radius 20 \
  --outfile ~/companion-data/reduced_germany_2015_timeseries/walsrode_20km.pddf.hdf5 \
  ~/companion-data/

# Near Rosenheim, intersection between highway 8 and 93
python weather_trafficcount_timeseries.py \
  --lat 47.814378 --lon 12.102973 --radius 20 \
  --outfile ~/companion-data/reduced_germany_2015_timeseries/rosenheim_20km.pddf.hdf5 \
  ~/companion-data/

# Kempten
python weather_trafficcount_timeseries.py \
  --lat 47.721482 --lon 10.318352 --radius 20 \
  --outfile ~/companion-data/reduced_germany_2015_timeseries/kempten_20km.pddf.hdf5 \
  ~/companion-data/
