"""
Short script to convert older dataframes to a dask-compatible variant. Shouldn't be needed anymore.

Just edit the OUTPUT_PATH variable to point to where the DataFrames are.
"""

import pandas as pd
import pathlib2 as pl
import h5py
from hashlib import md5

OUTPUT_PATH = pl.Path("/Volumes/CompanionEx/Data/dfs/")
# OUTPUT_PATH = pl.Path("./dfs_data/")

files = list(OUTPUT_PATH.glob('*.hdf'))

for filepath in files:
    print(str(filepath))
    df = pd.read_hdf(str(filepath), key='dataset')
    filepath.unlink()

    df = df.reset_index()
    df['site_hash'] = df['site'].apply(hash)  # The standard python hash of object... very fast...
    np_features = df[['site_hash', 'timestamp_start', 'precipitation mm/h', 'temperature C', 'windspeed m/s']].as_matrix()
    np_target_flow = df[['trafficflow counts/h']].as_matrix()
    np_target_speed = df[['trafficspeed km/h']].as_matrix()

    store = pd.HDFStore(str(filepath), mode='a')
    store.open()
    df[['site', 'datetime_start', 'site_hash']].to_hdf(store, key='index', mode='a')
    store.close()

    with h5py.File(str(filepath), 'a') as store:
        store.create_dataset('features_weather', data=np_features)
        store.create_dataset('target_flow', data=np_target_flow)
        store.create_dataset('target_speed', data=np_target_speed)

        store['/features_weather'].attrs['units'] = '(mm/h, C, m/s)'
        store['/target_flow'].attrs['units'] = '(counts/m)'
        store['/target_speed'].attrs['units'] = '(km/h)'

        name = filepath.stem
        start = name.split('_')[2]
        end = name.split('_')[3]

        store.attrs['name'] = name
        store.attrs['datetime_start'] = start
        store.attrs['datetime_end'] = end
