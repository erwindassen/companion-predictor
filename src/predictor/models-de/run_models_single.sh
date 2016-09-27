#!/usr/bin/env bash

BASEDIR="/home/michael/companion-data/nn-munich-50km"
SERIES1="/home/michael/companion-data/reduced_germany_2015_timeseries/munich_50km.pddf.hdf5"

for nn in 32 64 128 32-32 128-32 128-64-32 32-128-32 32-64-128-64-32
do
  echo "Network configuration: $nn"
  d="$BASEDIR/$nn"
  mkdir -p $d

  python nn_keras.py \
    --outdir $d \
    --nn $nn \
    $SERIES1 > $d/w_weather.log
  
  python nn_keras.py \
    --outdir $d \
    --nn $nn \
    --dropout 0.25 \
    $SERIES1 > $d/w_weather_dropout.log
  
  python nn_keras.py \
    --exclude-weather \
    --outdir $d \
    --nn $nn \
    $SERIES1 > $d/wo_weather.log
done

