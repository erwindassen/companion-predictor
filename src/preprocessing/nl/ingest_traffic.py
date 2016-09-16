"""
Ingest Traffic Data from TomTom.

Make sure to edit the path where you dropped in PyLR. PyLR also needs the
bitstring module, which is available in several Anaconda channels (bioconda).

XML Parsing:
http://www.saltycrane.com/blog/2011/07/example-parsing-xml-lxml-objectify/

TMC Event Codes:
http://wiki.openstreetmap.org/wiki/TMC/Event_Code_List

@todo: Be a bit more clever with TMC event codes (alertCEventCode).
@todo: Include PyLR and bitstream as submodules.
@todo: Vectorize: 1/ Only location_openlr_base64 in loop
                  2/ Use Apply PyLR.serialize + Extracting of Lat/Lon Pairs
                  3/ Midpoint Computation
"""

import glob
import gzip
import xml.etree.cElementTree as et
import pandas as pd

import numpy as np
import argparse
from datetime import datetime
from collections import namedtuple


# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True,
                    help="Date/Directory To Process (YYYY_MM_DD) or (DD_MM_YYYY)")
parser.add_argument("--type", required=True,
                    help="Type A is YYYY_MM_DD and type B is DD_MM_YYYY")
args = parser.parse_args()

# Define record type
columns = ['node_id', 'timestamp', 'measurement', 'measurement_type']
Record = namedtuple('Record', columns)


###############################################################################
###############################################################################
# FUNCTIONS
###############################################################################
###############################################################################

if args.type == 'A':

    def process(xml):
        site_measurements = xml[0][0][1][4:]  # if xml[0] == 'Header' else xml[1][0][1][4:]

        for site_measurement in site_measurements:
            node_id = site_measurement[0].attrib.get("id")

            if not node_id.startswith('RWS'):
                continue

            timestamp = datetime.strptime(site_measurement[1].text[:-1], '%Y-%m-%dT%H:%M:%S')

            for subelement in site_measurement:
                for value in subelement:

                    measurement = list(value[0][0])

                    if (len(measurement) != 1):
                        continue
                    else:
                        type = value[0][0].tag[31:]
                        value = float(measurement[0].text)
                        value = np.NaN if value < 0 else value
                        data.append(Record(node_id, timestamp, value, type))

        return data

elif args.type == 'B':

    def process(xml):

        site_measurements = xml[0][0][1][4:]  # if xml[0] == 'Header' else xml[1][0][1][4:]

        for site_measurement in site_measurements:
            node_id = site_measurement[0].attrib.get("id")

            if not node_id.startswith('RWS'):
                continue

            timestamp = datetime.strptime(site_measurement[1].text[:-1], '%Y-%m-%dT%H:%M:%S')

            for subelement in site_measurement:
                for value in subelement:

                    measurement = list(value[0][-1])

                    if (len(measurement) != 1):
                        continue
                    else:
                        type = value[0][-1].tag[31:]
                        value = float(measurement[0].text)
                        value = np.NaN if value < 0 else value
                        data.append(Record(node_id, timestamp, value, type))

        return data


###############################################################################
###############################################################################
# MAIN CODE
###############################################################################
###############################################################################

# Debug
print("// Processing Data for Date %s" % (args.date))

# Build List of Files to Ingest
globs = glob.iglob("%s/*raffic*.gz" % args.date)  # This gets relevant files of both types

dfs = list()
# Loop XML Files
for gg in globs:
    # print "// Processing %s" % gg

    data = []

    # Load XML
    with gzip.open(gg, 'r') as fxml:

        # Some XML files are corrupted, skip them silently
        try:
            xml = et.parse(fxml).getroot()
        except:
            xml = None

        if xml is not None:
            data = process(xml)

    print('.', end='')

    if len(data) > 0:

        df = pd.DataFrame(data=data, columns=columns)
        df['measurement_type'] = df['measurement_type'].astype('category')
        dfs.append(df)

# Throw into Dataframe
if len(dfs) > 0:

    df = pd.concat(dfs, axis=0)

    print('')  # Add a new line
    # Drop Duplicates
    print("// Dropping Duplicates")
    df.drop_duplicates(inplace=True)

    # Store Dataframe
    fname = args.date.split('/')
    fname = fname[-2] if fname[-1] == '' else fname[-1]
    fpath = "./Traffic_%s.hdf5" % fname
    print("// Saving %i Records to %s" % (len(df.index), fpath))
    # There's a bug for very long dataframes when using the fixed format
    # Cf. https://github.com/PyTables/PyTables/issues/531
    # Let's use the table format. This more also compatible with vanilla HDF5.
    # Cf. http://stackoverflow.com/a/20256692
    # Cf. http://stackoverflow.com/a/30787168
    df.to_hdf(fpath, 'df', mode='w', format='table')

else:
    print("!! No Records. Not Saving.")
