# from datetime import date, datetime, timedelta
from itertools import product
from pickle import load
from random import sample, shuffle

# import h5py

import pathlib2 as pl
import mmh3  # The hash function used to hash sites. See the preprocessor script.

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

RANDOM_SEED = 26715
CHUNK_SIZE = int(2e5)

NUM_SITES = 3000

MAX_BATCHES = 1000
MAX_BATCH_LENGTH = 100  # Maximum number os day-long measurement sequence (of one site) per batch
MAX_SEQ_LENGTH = 24 * 60  # Maximum number of measurements per site per day
SUBSEQ_LENGTH = 60  # 1 measurement/min a subsquence of 1h

# DF_FILES = pl.Path('/Volumes/CompanionEx/Data/dfs_pandas/PP_TS_2016-05-24-00_2016-06-01-00*.hdf')
DF_FILES = pl.Path('./dfs_data/*.hdf')
DF_FILE = pl.Path('/Volumes/CompanionEx/Data/dfs_pandas/PP_TS_2016-05-24-00_2016-06-01-00_0-200_20160624101922.hdf')

FEATURES = ['site_hash', 'timestamp_start', 'precipitation mm/h', 'temperature C', 'windspeed m/s']
TARGETS = ['trafficspeed km/h']#, 'trafficflow counts/h']

DASK = True

# Function definitions
######################

def filter_for_days(data, days, complement=False):
    if complement:
        return data[~data['day'].isin(days)]
    else:
        return data[data['day'].isin(days)]

def batches(source_df, sites, days, dask=False,
            max_batches=MAX_BATCHES, max_batch_length=MAX_BATCH_LENGTH, subseq_length=SUBSEQ_LENGTH):
    sample_bag = product(sites, days)

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
        batch = np.zeros([batch_length, subseq_length, len(FEATURES)], dtype='float64')
        mask = np.zeros([batch_length, subseq_length], dtype='float64')
        target = np.zeros([batch_length, subseq_length, len(TARGETS)], dtype='float64')

        max_seq_length = 0
        for j in range(batch_length):
            site, period = samples[j]
            site_hash = mmh3.hash64(site)[-1]

            # query measurements
            data = source_df[(source_df.site_hash == site_hash) & (source_df.day == period)]
            if dask:
                data = data.compute()

            data_f = data[FEATURES].values
            data_t = data[TARGETS].values

            this_length = data_f.shape[0]
            max_seq_length = max(max_seq_length, this_length)
            if this_length < subseq_length:
                #                 print('Warning: batch to small: {}'.format(this_length))

                batch[j, :this_length, :] = data_f
                target[j, :this_length, :] = data_t
                mask[j, :this_length] = np.ones([this_length])
            else:
                # Select a random subsequence
                ind = np.random.random_integers(0, high=(this_length - subseq_length))
                data_f = data_f[ind:ind + subseq_length, :]
                data_t = data_t[ind:ind + subseq_length, :]

                batch[j, :, :] = data_f
                target[j, :, :] = data_t
                mask[j, :] = np.ones([subseq_length])

                #         print('Max sequence length in this batch: {}'.format(max_seq_length))
        yield i, batch, target, mask


if __name__ == '__main__':
    # Parameters
    ############

    # We'll train the network with 10 epochs of a maximum of `max_batches` each
    num_lstm_units = 20
    max_grad = 5.0

    # Load
    ######
    if DASK:
        # from dask import array as da
        from dask import dataframe as dd
        # from dask import delayed
        # from dask.multiprocessing import get
        from dask.diagnostics.progress import ProgressBar

        progress_bar = ProgressBar()
        progress_bar.register()  # turn on progressbar globally
        data = dd.read_hdf(str(DF_FILES), key='dataset', chunksize=CHUNK_SIZE)
        days = set(dd.read_hdf(str(DF_FILES), key='day_counts').compute().index)
        sites = set(dd.read_hdf(str(DF_FILES), key='site_counts').compute().index)
    else:
        raise NotImplementedError

    # Preselected sites
    with open('./selected_sites.pkl', mode='rb') as fname:
        selected_sites = set(load(fname))

    sites = sites - selected_sites
    sampled_sites = sample(sites, NUM_SITES)
    hashes = set([mmh3.hash64(s)[-1] for s in sampled_sites])

    # Split sites
    train_sites = set(sample(hashes, int(0.8*NUM_SITES)))
    rest = hashes - train_sites
    test_sites = set(sample(rest, int(0.1*NUM_SITES)))
    validation_sites = rest - test_sites

    test_days = sample()
    validation_days = day_counts.sample(1).values()

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
