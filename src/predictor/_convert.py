"""
Short script to convert older dataframes to a dask-compatible variant. Shouldn't be needed anymore.

Just edit the OUTPUT_PATH variable to point to where the DataFrames are.
"""

import pandas as pd
import pathlib2 as pl
import h5py

# OUTPUT_PATH = pl.Path("/Volumes/CompanionEx/Data/dfs/")
OUTPUT_PATH = pl.Path("./dfs_data/")

files = list(OUTPUT_PATH.glob('*.hdf'))

for filepath in files:
    filepath = OUTPUT_PATH / pl.Path(filepath)
    print(str(filepath))
    df = pd.read_hdf(str(filepath), key='dataset')
    filepath.unlink()

    store = pd.HDFStore(str(filepath), mode='a')
    store.open()
    df.to_hdf(store, key='dataset', format='table', mode='a')
    store.close()

    with h5py.File(str(filepath), 'a') as f:
        name = filepath.stem
        start = name.split('_')[2]
        end = name.split('_')[3]

        f.attrs['name'] = name
        f.attrs['datetime_start'] = start
        f.attrs['datetime_end'] = end

