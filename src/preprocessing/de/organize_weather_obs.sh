#!/bin/bash
# 
# This scripts unzips all DWD weather observations and concatenantes them
# into a single large text file with the header intact.
#
# You must pass a list of zipped stundenwerte_*.zip files.
#
# Call Signature:
# $ bash organize_weather_obs.sh < fnames
# $ cat fnames
# stundenwerte_FF_00003_19370101_20110331_hist.zip
# stundenwerte_FF_00011_19800901_20151231_hist.zip
# ...
#
# To generate the file list, use
# $ ls -1 stundenwerte_*.zip > fnames
#

if [[ -d "tmp" ]]; then
    echo "!! Tmp Exists -- Aborting"
    exit 1;
fi

mkdir tmp
echo "// Extracting Files"
while read line; do
    zip=$line
    unzip -q -d tmp $zip
done

echo "// Concatenating Observations"
cd tmp
tail -q -n +2 produkt_* > tmp_data
head -1 produkt_* | head -2 | tail -1 > tmp_header
cat tmp_header tmp_data > dwd_observations_full.txt
cd ..
