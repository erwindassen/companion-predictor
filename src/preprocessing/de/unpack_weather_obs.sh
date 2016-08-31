#!/bin/bash
# 
# This scripts unzips all DWD weather observations.
#
# You must pass a list of zipped stundenwerte_*.zip files.
#
# Call Signature:
# $ bash unpack_weather_obs.sh < fnames
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

echo "// Removing Useless Files"
rm tmp/Stationsmetadaten*
rm tmp/Beschreibung*
