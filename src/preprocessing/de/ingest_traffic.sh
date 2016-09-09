#!/bin/bash
#
# This loops over all compressed traffic data we have, unzips them, runs the
# Python ingest, then deletes the directory.
#
# You must pass a list of zipped trafficIncidents*.zip files.
#
# Call Signature:
# $ bash ingest_traffic.sh < fnames
# $ cat fnames
# trafficIncidents_2014_09_29.zip
# trafficIncidents_2016_06_02.zip
# ...
#
# To generate the file list, use
# $ ls -1 trafficIncidents_*.zip > fnames
#

# Base Directory [MBA Volker]
basedir='/Users/volker/Work_ST/Companion/Source'

# Base Directory [Companion-VM Volker]
# basedir='/home/volker/Source'

# Python Script
pythonscript=${basedir}'/Preprocessing/DE/ingest_traffic.py'

# Loop Lines w/ Specified ZIP Files
while read line; do 
    zip=$line
    echo ""
    echo "***** Unpacking ${zip}"
    echo ""
    date=${zip:17:10}
    mkdir $date
    unzip -q $zip -d $date
    echo ""
    echo "***** Processing ${zip}"
    echo ""
    python $pythonscript --date $date
    echo ""
    echo "***** Removing ${zip}"
    echo ""
    rm -rf $date
done
