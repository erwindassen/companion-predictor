# from datetime import date, datetime, timedelta
from itertools import product
# from random import shuffle

# import h5py

import pathlib2 as pl
# import mmh3  # The hash function used to hash sites. See the preprocessor script.

import lasagne  # Ignore any errors for now
import numpy as np
import theano
import theano.tensor as T
# theano.config.exception_verbosity = 'high'
theano.config.openmp = True
theano.config.openmp_elemenwise_minsize = 200000

import pandas as pd
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 1000)
# dask.set_options(get=get)  # Due to a bug we can't read files in different processes so set this option after reading.

RANDOM_STATE = 2376452

CHUNK_SIZE = int(1e5)
DF_FILES = pl.Path('/Volumes/CompanionEx/Data/dfs_pandas/PP_TS_2016-05-24-00_2016-06-01-00*.hdf')
DF_FILE = pl.Path('/Volumes/CompanionEx/Data/dfs_pandas/PP_TS_2016-05-24-00_2016-06-01-00_0-200_20160624101922.hdf')

FEATURES = ['site_hash', 'timestamp_start', 'precipitation mm/h', 'temperature C', 'windspeed m/s']
TARGETS = ['trafficspeed km/h']#, 'trafficflow counts/h']

PANDAS = True

# Function definitions
######################

def filter_for_day(data, day, complement=False):
    datetime_index = data.index.get_level_values('datetime_start')
    datetime_index = pd.DatetimeIndex(datetime_index)

    if complement:
        return data[(datetime_index.year != day.year) | (datetime_index.month != day.month) | (datetime_index.day != day.day)]
    else:
        return data[(datetime_index.year == day.year) & (datetime_index.month == day.month) & (datetime_index.day == day.day)]

def batches(source_df, sites=None, days=None,
            max_batches=1000, max_batch_length=100, max_seq_length=24*60):
    if sites is None:
        site_bag = set(source_df.index.get_level_values(0))

    if days is None:
        day_bag = set(source_df['day'].unique())

    sample_bag = product(sites, days)
    #     sample_bag = list(sample_bag)  # Takes too long
    #     shuffle(sample_bag)  # Takes too long

    for i in range(max_batches):
        samples = list()
        for j in range(max_batch_length):
            try:
                samples.append(next(sample_bag))
            except StopIteration:
                break

        if len(samples) == 0:
            #             print("No samples at batch %i" % i)
            raise StopIteration

        # Prepare batch
        batch_length = len(samples)
        batch = np.zeros([batch_length, max_seq_length, len(FEATURES)], dtype='float64')
        mask = np.zeros([batch_length, max_seq_length], dtype='float64')
        target = np.zeros([batch_length, max_seq_length, len(TARGETS)], dtype='float64')

        for j in range(batch_length):
            site, period = samples[j]

            # query measurements
            data = source_df.query("site == '%s'" % site)
            data = data[data['day'] == period]

            data_f = data[FEATURES].values
            data_t = data[TARGETS].values

            seq_length = data.shape[0]
            assert seq_length <= max_seq_length, "Error: sequence longer than `max_seq_length` found"

            batch[j, :seq_length, :] = data_f
            target[j, :seq_length, :] = data_t
            mask[j, :seq_length] = np.ones([seq_length])

        yield i, batch, target, mask

def preprocess(data, filter_query=None):

    # Optional filtering
    if filter_query is not None:
        data = data.query(filter_query)

    # Create 'day' column
    datetime_index = data.index.get_level_values('datetime_start')
    datetime_index = pd.DatetimeIndex(datetime_index)
    data['day'] = datetime_index.to_period(freq='d')

    # Select the test and validation days
    days = data['day'].value_counts()
    test_day = days.keys()[-2]
    validation_day = days.keys()[-3]
    print('Daterange: {} to {}.'.format(datetime_index.min(), datetime_index.max()))
    print('Test day: {}.'.format(str(test_day)))
    print('Validation day: {}.'.format(str(validation_day)))

    # Split
    test_data = filter_for_day(data, test_day)
    validation_data = filter_for_day(data, validation_day)
    train_data = filter_for_day(data, test_day, complement=True)  # Exclude the test day...
    train_data = filter_for_day(train_data, validation_day, complement=True)  # ... and the validation day.

    print('Dataset sizes:')
    print('Train: {}'.format(train_data.size))
    print('Test: {}'.format(test_data.size))
    print('Validation: {}'.format(validation_data.size))

    return train_data, test_data, validation_data, test_day, validation_day


if __name__ == '__main__':
    # Parameters
    ############
    PANDAS = True

    # We'll train the network with 10 epochs of a maximum of `max_batches` each
    num_epochs = 10
    max_batches = 1000
    max_batch_length = 15  # Maximum number os day-long measurement sequence (of one site) per batch
    max_seq_length = 24 * 60  # Maximum number of measurements per site per day

    num_lstm_units = 20
    max_grad = 5.0

    # Load
    ######
    if PANDAS:
        data = pd.read_hdf(str(DF_FILE))
    else:
        # from dask import array as da
        from dask import dataframe as dd
        # from dask import delayed
        # from dask.multiprocessing import get
        from dask.diagnostics.progress import ProgressBar

        progress_bar = ProgressBar()
        progress_bar.register()  # turn on progressbar globally
        data = dd.read_hdf(str(DF_FILES), key='dataset', chunksize=CHUNK_SIZE)

    # Query
    #######
    SITES = []

    # Preprocess
    ############
    train_data, test_data, validation_data, test_day, validation_day = preprocess(data,
                                                                                  filter_query="site in @SITES")

    # Find sites in the data sets (needed to generate batches)
    site_bag = set(train_data.index.get_level_values(0))
    test_site_bag = set(test_data.index.get_level_values(0))
    validation_site_bag = set(validation_data.index.get_level_values(0))
    tdt = pd.DatetimeIndex(train_data.index.get_level_values('datetime_start'))
    day_bag = set(tdt.to_period(freq='d'))

    # Model
    #######
    input_var = T.tensor3('input', dtype=theano.config.floatX)
    target_values = T.tensor3('target', dtype=theano.config.floatX)

    l_in = lasagne.layers.InputLayer(shape=(None, None, 5), input_var=input_var, name='input_layer')
    l_mask = lasagne.layers.InputLayer(shape=(None, None), name='mask')

    l_lstm = lasagne.layers.LSTMLayer(l_in, num_units=num_lstm_units,
                                      gradient_steps=-1, grad_clipping=max_grad, unroll_scan=False,
                                      mask_input=l_mask, name='l_lstm_1')

    # We want to combine the LSTM with a dense layer and need to reshape the input. We dot this with a `ReshapeLayer`
    # First, retrieve symbolic variables for the input shape
    n_batch, n_time_steps, n_features = l_in.input_var.shape

    # Now, squash the n_batch and n_time_steps dimensions
    l_reshape_in = lasagne.layers.ReshapeLayer(l_lstm, (-1, num_lstm_units))

    # Now, we can apply feed-forward layers as usual.
    l_dense_1 = lasagne.layers.DenseLayer(l_reshape_in, num_units=20, nonlinearity=lasagne.nonlinearities.tanh,
                                          name='l_dense_1')
    l_dense_2 = lasagne.layers.DenseLayer(l_dense_1, num_units=1, nonlinearity=lasagne.nonlinearities.tanh,
                                          name='l_dense_2')
    # Now, the shape will be n_batch*n_timesteps, 1.  We can then reshape to
    # n_batch, n_timesteps to get a single value for each timstep from each sequence
    l_reshape_out = lasagne.layers.ReshapeLayer(l_dense_2, (n_batch, n_time_steps, 1), name='output_layer')

    # Train
    #######
    # lasagne.layers.get_output produces an expression for the output of the net
    network_output = lasagne.layers.get_output(l_reshape_out)
    # The value we care about is the final value produced for each sequence
    # so we simply slice it out.
    predicted_values = network_output#[:, -1]

    # Our cost will be mean-squared error
    loss = T.mean(lasagne.objectives.squared_error(predicted_values, target_values))

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_reshape_out)

    # Compute adam updates for training
    updates = lasagne.updates.adam(loss, all_params)

    # Theano function
    train = theano.function([l_in.input_var, target_values, l_mask.input_var], loss, updates=updates)
    compute_cost = theano.function([l_in.input_var, target_values, l_mask.input_var], loss)
    ide = theano.function([target_values], outputs=[target_values])  # To obtain target value output from network (DEBUG)
    ff = theano.function([l_in.input_var, l_mask.input_var], outputs=[predicted_values])  # For inference

    # Train loop
    ############

    for epoch in range(num_epochs):
        print('TRAIN', end=' ')
        for batch_num, batch, target, mask in batches(train_data, sites=site_bag, days=day_bag,
                                                      max_batches=max_batches, max_batch_length=max_batch_length):
            train(batch, target, mask)
            if batch_num % 10 == 0:
                if batch_num % 100 == 0:
                    print(batch_num, end='')
                print(".", end='')

        print('')

        cost_val = 0.0
        print('TEST', end=' ')
        for batch_num, batch, target, mask in batches(test_data, days=set((test_day,)), sites=test_site_bag):
            cost_val += compute_cost(batch, target, mask)
            if batch_num % 10 == 0:
                print(batch_num, end='')

        cost_val = cost_val / (batch_num + 1)
        print('')

        print("Epoch {} validation cost = {}".format(epoch + 1, cost_val))
