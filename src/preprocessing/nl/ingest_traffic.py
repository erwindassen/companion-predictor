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
args = parser.parse_args()

# Define record type
columns = ['node_id', 'timestamp', 'measurement', 'measurement_type']
Record = namedtuple('Record', columns)

# Define XML namespaces

ns = {'datex': 'http://datex2.eu/schema/2/2_0',
      'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
      'xsd': 'http://www.w3.org/2001/XMLSchema',
      'xsi': 'http://www.w3.org/2001/XMLSchema-instance'}

###############################################################################
###############################################################################
# FUNCTIONS
###############################################################################
###############################################################################

def process(xml):

    site_measurements = xml.iterfind('.//datex:siteMeasurements', ns)

    for site_measurement in site_measurements:
        node_id = site_measurement[0].attrib.get("id")

        if not node_id.startswith('RWS'):
            continue

        timestamp = site_measurement.find('datex:measurementTimeDefault', ns).text
        time = datetime.strptime(timestamp[:-1], '%Y-%m-%dT%H:%M:%S')

        # We inline the two types of measuremtens to reduce branching

        vf = site_measurement.iterfind('.//datex:vehicleFlow', ns)
        for mp in vf:
            if mp.find('.datex:dataError', ns) is None:  # If is None error is not present
                measurement = mp.find('.datex:vehicleFlowRate', ns)
                if measurement is not None:
                    value = float(measurement.text)
                    value = np.NaN if value < 0 else value
                    data.append(Record(node_id, time, value, 'vehicleFlowRate'))
                else:
                    print('Malformed node... skipping.')

        ts = site_measurement.iterfind('.//datex:averageVehicleSpeed', ns)
        for mp in ts:
            if mp.find('.datex:dataError', ns) is None:  # If is None error is not present
                measurement = mp.find('.datex:speed', ns)
                if measurement is not None:
                    value = float(measurement.text)
                    value = np.NaN if value < 0 else value
                    data.append(Record(node_id, time, value, 'averageVehicleSpeed'))
                else:
                    print('Malformed node... skipping.')

    return data


###############################################################################
###############################################################################
# MAIN CODE
###############################################################################
###############################################################################

# Debug
print("// Processing Data for date folder %s" % (args.date))

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

        df = pd.DataFrame(data=data, columns=columns).dropna()  # We could decide to do something else
        df = df.groupby(by=['node_id', 'timestamp', 'measurement_type'], as_index=False).mean()
        df = df.pivot_table(index=['node_id', 'timestamp'], columns=['measurement_type'], values=['measurement'])

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
    with pd.HDFStore(fpath, 'w') as store:
        store['df'] = df

else:
    print("!! No Records. Not Saving.")
