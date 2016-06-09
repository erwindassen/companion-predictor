import h5py
import pandas as pd
import pathlib2 as pl
import logging
import sys
from docopt import docopt
from deco import synchronized, concurrent


pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 1000)

_INPUT_PATH = pl.Path("./hdf_data/")
# _INPUT_PATH = pl.Path("/Volumes/CompanionEx/Data/hdf/")
_OUTPUT_PATH = pl.Path("./dfs_data/")

# Configure logging
handler = logging.StreamHandler(stream=sys.stdout)
logging.basicConfig(handlers=(handler,), format='%(levelname)s %(asctime)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def preprocessing_generator(input=_INPUT_PATH, files=None):
    """
    Creates a generator that return each preprocessed file as a DataFrame one at a time.

    :param input: Input path where HDFs are found.
    :param files: Iterable object with list of file names to process in the given input path.
    :return: A dataset generator (each is a tuple (index, features, target_flow, target_speed)
    where all but the first is a numpy array and index is a pandas dataframe).
    """

    if files is None:
        files = list(input.glob('*.hdf'))
    else:
        files = [pl.Path(input) / pl.Path(f) for f in files]

    logger.info('%i file(s) to process...' % len(files))

    try:
        for filepath in files:
            logger.info('Processing %s' % str(filepath.name))
            sys.stdout.flush()

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
                sdf['site'] = site.name[1:]  # To remove the leading "/"
                fdf = fdf.append(sdf, ignore_index=True)

            # Close the file
            f.close()

            # If we end up with an empty dataframe skip
            if fdf.index.size == 0:
                logger.info('Empty DataFrame found... skipping.')
                continue

            # Categorize the sites
            fdf['site'] = fdf['site'].astype('category')

            # Convert timestamp to int as it should be
            fdf['timestamp_start'] = fdf['timestamp_start'].astype('int64')

            # Make a new index of datetime objects
            fdf['datetime_start'] = fdf['timestamp_start'].astype('M8[s]')

            # Set index to (site, datetime_start) and sort before filling NA
            fdf = fdf.set_index(['site', 'datetime_start']).sort_index()

            # Fill NA values: first propagate forward then backwards for the initial NA values
            fdf = fdf.fillna(method='ffill').fillna(method='bfill')

            # Attach the filepath name to the dataframe as __name__
            fdf.__name__ = 'PP_' + filepath.name

            fdf = fdf.reset_index()
            fdf['site_hash'] = fdf['site'].apply(hash)  # The standard python hash of object... very fast...

            index = fdf[['site', 'datetime_start', 'site_hash']]
            index.__name__ = fdf.__name__

            features = fdf[['site_hash', 'timestamp_start', 'precipitation mm/h', 'temperature C', 'windspeed m/s']]\
                .as_matrix()
            features.__doc__ = \
            '''
            Contains, in order:

            site_hash (int), timestamp_start (int), precipitation (mm/h, float), temperature (C, float), windspeed (m/s, float)
            '''

            target_flow = fdf[['trafficflow counts/h']].as_matrix()
            target_flow.__doc__ = \
            '''
            Contains, in order:

            traffic_flow (counts/min, float)
            '''

            target_speed = fdf[['trafficspeed km/h']].as_matrix()
            target_speed.__doc__ = \
            '''
            Contains, in order:

            traffic_speed (km/h, float)
            '''

            # Yield the current DataFrame
            yield (index, features, target_flow, target_speed)

        return

    except TypeError:
        logger.error('Input file list not iterable... skipping.')


# @synchronized
def preprocess(input=_INPUT_PATH, output=_OUTPUT_PATH, files=None):
    """
    Preprocess HDFs from the companion-risk-factors into a pandas DataFrame usable by the predictor.

    :param input: Input path where HDFs are found
    :param output: Output path were to store DataFrames, or the str "iter" in which case it returns an iterator over the input files.
    :param files: Iterable object with list of file names to process in the given input path.
    """

    generator = preprocessing_generator(input=input, files=files)

    for index, features, target_flow, target_speed in generator:
        # Write out

        filepath = output / pl.Path(index.__name__)
        store = pd.HDFStore(str(filepath), mode='a')

        store.open()
        index.to_hdf(store, key='index', mode='a')
        store.close()

        with h5py.File(str(filepath), 'a') as store:
            store.create_dataset('features_weather', data=features)
            store.create_dataset('target_flow', data=target_flow)
            store.create_dataset('target_speed', data=target_speed)

            store['/features_weather'].attrs['doc'] = features.__doc__
            store['/target_flow'].attrs['doc'] = target_flow.__doc__
            store['/target_speed'].attrs['doc'] = target_speed.__doc__

            name = filepath.stem
            start = name.split('_')[-2]
            end = name.split('_')[-1]

            store.attrs['name'] = name
            store.attrs['datetime_start'] = start
            store.attrs['datetime_end'] = end


if __name__ == '__main__':

    __doc__ = \
    """Usage: pp.py [--quiet] [--input <input_path>] [--output <output_path>] [--files <file_list>]

    -h, --help   show this
    -q, --quiet  suppress info output, errors will be shown
    -i <input_path>, --input <input_path>  input path with HDFs [default: ./hdf_data]
    -o <output_path>, --output <output_path>  output path for DataFrames [default: ./dfs_data]
    -f <file_list>, --files <file_list>  a file with the list of files to process in the given input path [default: *]

    """

    opts = docopt(__doc__)
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
    handler.flush()

    preprocess(input=pl.Path(opts['--input']), output=pl.Path(opts['--output']), files=files)

    logger.info('...done!')
    logging.shutdown()
