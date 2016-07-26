import dask
from dask.multiprocessing import get
from dask import dataframe as dd
from dask import array as da
from dask import delayed
import pandas as pd
import pathlib2 as pl

pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 1000)

# dask.set_options(get=get);

DF_DIR = pl.Path('/Volumes/CompanionEx/Data/dfs/')
# DF_DIR = pl.Path('./df_data/')

glob = str(DF_DIR) + '/*.hdf'

df = dd.read_hdf(glob, key='dataset')

print(dask.__version__)
print(df.npartitions)

samples = df.sample(1e-6).compute()
print(samples.head())

# print(df.count())
