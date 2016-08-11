import h5py
import pandas as pd
import pathlib2 as pl
import logging
from sys import stdout
import mmh3

from docopt import docopt
from deco import synchronized, concurrent

from multiprocessing import cpu_count


pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 1000)

_INPUT_PATH = pl.Path("./hdf_data/")
# _INPUT_PATH = pl.Path("/Volumes/CompanionEx/Data/hdf/")
_OUTPUT_PATH = pl.Path("./dfs_data/")

# _PROCESSES = 1
_PROCESSES = max(1, cpu_count() - 1)

# Configure logging
handler = logging.StreamHandler(stream=stdout)
logging.basicConfig(handlers=(handler,), format='%(levelname)s %(asctime)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


@concurrent(processes=_PROCESSES)
def preprocess_file(file, input=_INPUT_PATH, mode='pandas'):
    """
    Creates a generator that return each preprocessed file as a DataFrame one at a time.

    :param file: file name to process in the given input path.
    :param input: Input path where HDFs are found.
    :param mode: Determines the output format: either 'pandas' (DataFrame) or 'numpy' (array).
    :return: A dataset generator (each is a tuple (index, features, target_flow, target_speed)
    where all but the first is a numpy array and index is a pandas dataframe).
    """

    assert file is not None
    filepath = input / file

    try:
        logger.info('Preprocessing %s' % str(file.name))
        stdout.flush()

        fdf = pd.DataFrame(data=[],columns=['timestamp_start'])
        f = h5py.File(str(filepath), 'r')
        for site in iter(f):
            sdf = pd.DataFrame(data=[],columns=['timestamp_start'])
            site = f[site]
            for measurements in iter(site):
                measurements = site[measurements]
                if 'units' in measurements.attrs:
                    units = measurements.attrs['units'].decode().split(",")
                    columns = list(map(lambda n, u: str.strip(n + u),
                                       ('', '', measurements.name.split("/")[-1]),
                                       units))
                    df = pd.DataFrame(data=measurements.value, columns=columns)\
                        .drop(['timestamp_end'], axis=1)\
                        .dropna(axis=0, how='all')  # Drops the empty rows included for completeness in the HDF

                    # Merge df into sdf
                    sdf = pd.merge(sdf, df, on='timestamp_start', how='outer')

            # Add site name as column
            sdf['site'] = site.name[1:].lower()  # To remove the leading "/"
            fdf = fdf.append(sdf, ignore_index=True)

        # Close the file
        f.close()

        # If we end up with an empty dataframe skip
        if fdf.index.size == 0:
            logger.info('Empty DataFrame found... skipping.')
            return

        # Hash the site column for efficiency in ML down the pipe
        # fdf['site_hash'] = fdf['site'].apply(hash)  # The standard python hash of object... very fast... but not consistent among sessions
        fdf['site_hash'] = fdf['site'].apply(lambda s: mmh3.hash64(s)[-1])  # Murmur hash v3. Should be consistent and fast.
        site_counts = fdf['site'].value_counts()

        # # Categorize the sites
        # fdf['site'] = fdf['site'].astype('category')

        # Convert timestamp to int as it should be
        fdf['timestamp_start'] = fdf['timestamp_start'].astype('int64')

        # Make a new index of datetime objects
        fdf['datetime_start'] = fdf['timestamp_start'].astype('M8[s]')

        # Use timestamp and aggregate to day
        datetime_index = pd.DatetimeIndex(fdf['datetime_start'])
        fdf['day'] = datetime_index.to_period(freq='d')
        fdf['day'] = fdf['day'].apply(str)
        day_counts = fdf['day'].value_counts()

        # Set index to (site, datetime_start) and sort before filling NA
        fdf = fdf.set_index(['site', 'datetime_start']).sort_index()

        # Fill NA values: first propagate forward then backwards for the initial NA values
        fdf = fdf.fillna(method='ffill').fillna(method='bfill')

        # It is better for dask to have a standard int index.
        fdf = fdf.reset_index()

        # Set filename as metadata
        name = 'PP_' + file.name

        if mode == 'numpy':
            fdf = fdf.reset_index()

            index = fdf[['site', 'datetime_start', 'site_hash']]

            features = fdf[['site_hash', 'timestamp_start', 'precipitation mm/h', 'temperature C', 'windspeed m/s']]\
                .as_matrix()
            features_desc = \
            '''
            site_hash (int), timestamp_start (int), precipitation (mm/h, float), temperature (C, float), windspeed (m/s, float)
            '''

            target_flow = fdf[['trafficflow counts/h']].as_matrix()
            target_flow_desc = \
            '''
            traffic_flow (counts/min, float)
            '''

            target_speed = fdf[['trafficspeed km/h']].as_matrix()
            target_speed_desc = \
            '''
            traffic_speed (km/h, float)
            '''

            metadata = {'name': name,
                        'features': features_desc,
                        'target_flow': target_flow_desc,
                        'target_speed': target_speed_desc,
                        'day_counts': day_counts,
                        'site_counts': site_counts}

            # Yield the current data set
            return index, features, target_flow, target_speed, metadata

        elif mode == 'pandas':
            return name, fdf, day_counts, site_counts

        else:
            raise ValueError('Expected "pandas" or "numpy"')

        return

    except TypeError:
        logger.error('Input file list not iterable... skipping.')


@concurrent(processes=_PROCESSES)
def write_file(ds, output=_OUTPUT_PATH, mode='pandas'):

    if mode == 'pandas':
        name, df, day_counts, site_counts = ds
        filepath = output / pl.Path(name)
        store = pd.HDFStore(str(filepath), mode='a')

        # Write out
        store.open()
        df.to_hdf(store, key='dataset', format='table', mode='a')
        day_counts.to_hdf(store, key='day_counts', format='table', mode='a')
        site_counts.to_hdf(store, key='site_counts', format='table', mode='a')
        store.close()

        with h5py.File(str(filepath), 'a') as store:
            start = name.split('_')[-2]
            end = name.split('_')[-1]

            store.attrs['name'] = name
            store.attrs['datetime_start'] = start
            store.attrs['datetime_end'] = end

    elif mode == 'numpy':
        index, features, target_flow, target_speed, description = ds  # Unpack

        # Write out
        name = description['name']
        filepath = output / pl.Path(name)
        store = pd.HDFStore(str(filepath), mode='a')

        store.open()
        index.to_hdf(store, key='index', format='table', mode='a')
        day_counts = description['day_counts']
        day_counts.to_hdf(store, key='day_counts', format='table', mode='a')
        site_counts = description['site_counts']
        site_counts.to_hdf(store, key='site_counts', format='table', mode='a')
        store.close()

        with h5py.File(str(filepath), 'a') as store:
            store.create_dataset('features_weather', data=features)
            store.create_dataset('target_flow', data=target_flow)
            store.create_dataset('target_speed', data=target_speed)

            store['/features_weather'].attrs['columns'] = description['features']
            store['/target_flow'].attrs['columns'] = description['target_flow']
            store['/target_speed'].attrs['columns'] = description['target_speed']

            start = name.split('_')[-2]
            end = name.split('_')[-1]

            store.attrs['name'] = name
            store.attrs['datetime_start'] = start
            store.attrs['datetime_end'] = end

    else:
        raise ValueError('Expected "pandas" or "numpy"')

@synchronized
def preprocess(input=_INPUT_PATH, output=_OUTPUT_PATH, files=None, mode='pandas'):
    """
    Preprocess HDFs from the companion-risk-factors into a pandas DataFrame usable by the predictor.

    :param input: Input path where HDFs are found
    :param output: Output path were to store DataFrames, or the str "iter" in which case it returns an iterator over the input files.
    :param files: Iterable object with list of file names to process in the given input path.
    :param mode: Determines the output format: either 'pandas' (DataFrame) or 'numpy' (array).
    """

    if files is None:
        files = list(input.glob('*.hdf'))

    ds = dict()
    for file in files:
        file = pl.Path(file)
        ds[str(file)] = preprocess_file(file, input=input, mode=mode)

    preprocess_file.wait()

    for file in ds.keys():
        write_file(ds[file], output=output, mode=mode)


if __name__ == '__main__':

    __doc__ = \
    """Usage: pp.py  [--numpy] [--quiet] [--input <input_path>] [--output <output_path>] [--files <file_list>]

    -h, --help   show this
    -n, --numpy  exports output into numpy arrays instead of pandas DataFrames
    -q, --quiet  suppress info output, errors will be shown
    -i <input_path>, --input <input_path>  input path with HDFs [default: ./hdf_data]
    -o <output_path>, --output <output_path>  output path for DataFrames [default: ./dfs_data]
    -f <file_list>, --files <file_list>  a file with the list of files to process in the given input path [default: *]
    """

    opts = docopt(__doc__)

    if opts['--numpy']:
        mode = 'numpy'
    else:
        mode = 'pandas'

    if opts['--quiet']:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    if opts['--files'] != '*':
        file = pl.Path(opts['--files'])
        with file.open() as f:
            files = f.read().splitlines()
    else:
        files = None

    logger.info('Looking for files in %s...' % str(opts['--input']))
    logger.info('and writing to %s...' % str(opts['--output']))

    if len(files) > 1:
        logger.info('%i file(s) to process...' % len(files))

    logger.info('Using %i processes.' % _PROCESSES)
    handler.flush()

    preprocess(input=pl.Path(opts['--input']), output=pl.Path(opts['--output']),
               files=files, mode=mode)

    logger.info('...done!')
    logging.shutdown()
