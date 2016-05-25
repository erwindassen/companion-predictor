import h5py
import pandas as pd
import pathlib2 as pl
from docopt import docopt

pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 1000)

_INPUT_PATH = pl.Path("./hdf_data/")
# _INPUT_PATH = pl.Path("/Volumes/CompanionEx/Data/hdf/")
_OUTPUT_PATH = pl.Path("./dfs_data/")

def preprocessing_generator(input=_INPUT_PATH):
    """
    Creates a generator that return each preprocessed file as a DataFrame one at a time.

    :param input: Input path where HDFs are found
    :return: A DataFrame generator
    """

    for filepath in input.glob('*.hdf'):
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

                    # Merge df into odf
                    sdf = pd.merge(sdf, df, on='timestamp_start', how='outer')

            # Add site name as column
            sdf['site'] = site.name[1:]  # To remove the leading "/"
            fdf = fdf.append(sdf, ignore_index=True)

        # Close the file
        f.close()

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

        # Yield the current DataFrame
        yield fdf


def preprocess(input=_INPUT_PATH, output=_OUTPUT_PATH):
    """
    Preprocess HDFs from the companion-risk-factors into a pandas DataFrame usable by the predictor.

    :param input: Input path where HDFs are found
    :param output: Output path were to store DataFrames, or the str "iter" in which case it returns an iterator over the input files.
    :return:
    """

    generator = preprocessing_generator(input=input)

    for df in generator:
        # Write out
        filepath = output / pl.Path(df.__name__)
        df.to_hdf(str(filepath), key='dataset')
        with h5py.File(str(filepath), 'r+') as f:
            name = filepath.stem
            start = name.split('_')[-2]
            end = name.split('_')[-1]

            f.attrs['name'] = name
            f.attrs['datetime_start'] = start
            f.attrs['datetime_end'] = end


if __name__ == '__main__':

    import sys

    __doc__ = \
    """Usage: pp.py [_INPUT_PATH] [_OUTPUT_PATH]

    -h, --help   show this
    -i, --input  _INPUT_PATH input path with HDFs [default: ./hdf_data]
    -o, --output _OUTPUT_PATH output path for DataFrames [default: ./dfs_data]

    """

    opts = docopt(__doc__)
    opts['_INPUT_PATH'] = _INPUT_PATH if opts['_INPUT_PATH'] is None else pl.Path(opts['_INPUT_PATH'])
    opts['_OUTPUT_PATH'] = _OUTPUT_PATH if opts['_OUTPUT_PATH'] is None else pl.Path(opts['_OUTPUT_PATH'])

    print('Looking for files in %s...' % str(opts['_INPUT_PATH']))
    print('and writing to %s...' % str(opts['_OUTPUT_PATH']))
    sys.stdout.flush()

    preprocess(input=opts['_INPUT_PATH'], output=opts['_OUTPUT_PATH'])

    print('...done!')
