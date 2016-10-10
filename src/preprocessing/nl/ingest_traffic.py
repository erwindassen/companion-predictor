"""
Ingest Traffic Data from NDW.
Copyright S[&]T
"""

from deco import concurrent, synchronized
from boltons.iterutils import chunked_iter

import begin

import pandas as pd
import numpy as np

import pathlib2 as pl
import gzip
import xml.etree.cElementTree as et
from datetime import datetime
from collections import namedtuple
from multiprocessing import cpu_count
import logging

from tables.exceptions import HDF5ExtError

DFS_PER_FILE = 5000

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

@concurrent(processes=cpu_count()-2)
def process(fpath):
    """
    Process one given NDW .xml.gz file
    :param fpath: pathlib.Path specifying input file
    :return: list of Record objects with parsed data
    """

    logging.info("// Processing %s" % str(fpath))
    data = list()

    # Load XML
    with gzip.open(str(fpath), 'r') as fxml:

        # Some XML files are corrupted, skip them silently
        try:
            xml = et.parse(fxml).getroot()
        except:
            xml = None

        if xml is None:
            return data

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
                        logging.warning('Malformed node in {}... skipping.'.format(str(fpath)))

            ts = site_measurement.iterfind('.//datex:averageVehicleSpeed', ns)
            for mp in ts:
                if mp.find('.datex:dataError', ns) is None:  # If is None error is not present
                    measurement = mp.find('.datex:speed', ns)
                    if measurement is not None:
                        value = float(measurement.text)
                        value = np.NaN if value < 0 else value
                        data.append(Record(node_id, time, value, 'averageVehicleSpeed'))
                    else:
                        logging.warning('Malformed node in {}... skipping.'.format(str(fpath)))

    if len(data) > 0:
        df = pd.DataFrame(data=data, columns=columns).dropna()  # We could decide to do something else

        df = df.groupby(by=['timestamp', 'node_id', 'measurement_type'], as_index=False).mean()
        df = df.pivot_table(index=['timestamp', 'node_id'], columns=['measurement_type'], values=['measurement'])

        # Pandas cannot write multiindex columns to hdf yet. We have to drop it.
        df.columns = ['averageVehicleSpeed', 'vehicleFlowRate']

        # Recast indices to collumns (also needed for dask)
        df.reset_index(inplace=True)

    else:
        df = pd.DataFrame()  # We need to return a DataFrame object even if empty

    return df


###############################################################################
###############################################################################
# MAIN CODE
###############################################################################
###############################################################################

@begin.start
@begin.logging
@begin.convert(_automatic=True)
def run(input:  'Input dir to look recursively for NDW traffic xml.gz files' = '.',
        output: 'Output dir to write dataframes to' = '.'):
    """
    Processes NDW xml files extrating traffic speed and flow.
    """

    # "Date/Directory To Process (YYYY_MM_DD) or (DD_MM_YYYY)")

    # Debug
    logging.info("// Processing Data for date folder {}".format(str(input)))

    # Build List of Files to Ingest
    input = pl.Path(input)
    output = pl.Path(output)

    if not (input.exists() and input.is_dir() and output.exists() and output.is_dir()):
        raise IOError('Invalid input/output directories passed.')

    globs = list(input.rglob('*raffic*.gz'))  # This gets relevant files of both types

    @synchronized
    def orchestrate(globs):
        dfs = dict()

        # Loop XML Files
        for fpath in globs:
            dfs[fpath.name] = process(fpath)
            # assert isinstance(dfs[fpath.name], pd.DataFrame)

        return dfs

    # Call the orchestrator
    dfs = orchestrate(globs)

    # Throw into Dataframe
    if len(dfs) > 0:

        for sdfs in chunked_iter(dfs.values(), DFS_PER_FILE):
            df = pd.concat(sdfs, axis=0)

            # Drop Duplicates
            df.drop_duplicates(inplace=True)

            # Categorize node_id
            df.node_id = df.node_id.astype('category')

            # Store Dataframe
            fname = output / pl.Path('Traffic-{}.hdf5'.format(datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')))
            logging.info("// Saving %i Records to %s" % (len(df), fname))

            try:
                with pd.HDFStore(str(fname), mode='w', format='table') as store:
                    store.put('range', pd.Series([df.timestamp.min(), df.timestamp.max()]), format='table')
                    store.put('df', df, format='table')
            except HDF5ExtError:
                logging.warning("HDF Exception caught. Probably harmless but check file size.")

            logging.info("// Done")

    else:
        logging.info("!! No Records. Not Saving.")
