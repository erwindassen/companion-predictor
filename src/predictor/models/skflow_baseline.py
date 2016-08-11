# import h5py
# import dask
import pandas as pd
# import pathlib2 as pl
# import mmh3  # The hash function used to hash sites. See the preprocessor script.
import tensorflow.contrib.learn as skflow

# from dask import array as da
from dask import dataframe as dd
# from dask import delayed
# from dask.multiprocessing import get
from dask.diagnostics.progress import ProgressBar
from sklearn import metrics


pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 1000)
# dask.set_options(get=get)  # Due to a bug we can't read files in different processes so set this option after reading.

progress_bar = ProgressBar()
progress_bar.register()   # turn on progressbar globally

CHUNK_SIZE = 10000
RANDOM_STATE = 2376452
DF_FILES = '/Volumes/CompanionEx/Data/dfs_pandas/PP_TS_2016-01-16*.hdf'
DF_FILE = '/Volumes/CompanionEx/Data/dfs_pandas/PP_TS_2016-01-21-06_2016-01-21-13_0-200_20160601144931.hdf'


# Function definitions
######################

def split_xy(dataset):
    features = dataset[['site_hash', 'timestamp_start', 'precipitation mm/h', 'temperature C', 'windspeed m/s']]
    target = dataset[['trafficspeed km/h']]  # , 'trafficflow counts/h']]

    return  (features, target)

def train(features, target):
    dnn_reg = skflow.TensorFlowDNNRegressor(hidden_units=[20, 40, 20],
                                            batch_size=500, steps=50, learning_rate=0.00001, dropout=None,
                                            optimizer='Adagrad', continue_training=False, verbose=1)
    dnn_reg.fit(features, target, logdir='../tf_logs/baseline/')

    return  dnn_reg

def predict(model, features):
    predict_target = model.predict(features)

    return predict_target

def score(target, prediction):
    mae = metrics.mean_absolute_error(target, prediction)

    return mae


# Try with a pandas dataframe first
###################################

# Load data
data = pd.read_hdf(DF_FILE, key='dataset').reset_index()
# data = dd.read_hdf('/Volumes/CompanionEx/Data/dfs_pandas/PP_TS_2016-01-01-06_2016-01-01-13.hdf')
data = dd.from_pandas(data, chunksize=CHUNK_SIZE)

# Split
# Due to a bug (dask.dataframe does not support multi-indices as of version 0.9.0) we have to reset_index on each dataframe
train_data, test_data, validation_data = tuple(map(lambda df: df.compute(), data.random_split([0.7, 0.20, 0.10], random_state=RANDOM_STATE)))
# train_data, test_data, validation_data = data.random_split([0.7, 0.20, 0.10], random_state=2376452)
# Note that these are pandas dataframes due to the compute() call

# Train
train_features, train_target = split_xy(train_data)
assert not train_features.isnull().values.any(), 'Data contains NAN'
assert not train_target.isnull().values.any(), 'Data contains NAN'
model = train(train_features, train_target)

# Predict
test_features, test_target = split_xy(test_data)
test_prediction = predict(model, test_features)

# Score
mae = score(test_target, test_prediction)
print(mae)

# Now try with dask.dataframe
#############################

# Load data
data = dd.read_hdf(DF_FILES, key='dataset', chunksize=CHUNK_SIZE)
# data = pd.read_hdf('/Volumes/CompanionEx/Data/dfs_pandas/PP_TS_2016-01-01-06_2016-01-01-13.hdf')

# Split
# Due to a bug (dask.dataframe does not support multi-indices as of version 0.9.0) we have to reset_index on each dataframe
train_data, test_data, validation_data = tuple(map(lambda df: df.reset_index(), data.random_split([0.7, 0.20, 0.10], random_state=RANDOM_STATE)))
# train_data, test_data, validation_data = data.random_split([0.7, 0.20, 0.10], random_state=2376452)

# Train
train_features, train_target = split_xy(train_data)
model = train(train_features, train_target)

# Predict
test_features, test_target = split_xy(test_data)
test_prediction = predict(model, test_features)

# Score
mae = score(test_target, test_prediction)
print(mae)
