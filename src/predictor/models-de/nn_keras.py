"""NN traffic+weather time series using Keras"""

import sys
import os
import argparse

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.engine.training import slice_X

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as md

# Add sanitation module to path
sys.path.append(os.path.join(sys.path[0],
                             '../../',
                             'preprocessing/'
                             'de'))
from sanitize_timeseries import sanitize


HIDDEN_SIZE = 128
TEST_FRACTION = 0.1

FEATURES = [
    'precipitation',
    'precipitation_amount',
    'precipitation_kind',
    'temperature',
    'relative_humidity'
#    'wind_speed'
]
OUTPUT = [
    'incidents_total'
]


def prepare(df, features=FEATURES):
    dates = df.index.to_pydatetime()
    month = np.matrix([d.day for d in dates]).T
    day = np.matrix([d.weekday()+1 for d in dates]).T
    hour = np.matrix([d.hour for d in dates]).T
    if features:
        feat = df[features].as_matrix()
        x = np.hstack((month, day, hour, feat))
    else:
        x = np.hstack((month, day, hour))
    y = df[OUTPUT].as_matrix()
    return x, y

def train_nn(filename, shuffle=True):
    # Prepare data
    df = sanitize(filename, normalize=True, interpolate_limit=None)
    x, y = prepare(df)
    assert x.shape[0] == y.shape[0]
    n_samples = x.shape[0]
    
    # Shuffle data
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    
    # Split data into train/test
    split_at = int(n_samples*TEST_FRACTION)
    (x_train, x_val) = (slice_X(x, 0, split_at), slice_X(x, split_at))
    (y_train, y_val) = (y[:split_at], y[split_at:])
    print("X_train shape: %s" % repr(x_train.shape))
    print("Y_train shape: %s" % repr(y_train.shape))
    
    model = Sequential([
        Dense(HIDDEN_SIZE, input_dim=x.shape[1], init='uniform', activation='relu'),
        Dense(10, init='uniform', activation='relu'),
        Dense(y.shape[1], init='uniform', activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val))
    scores = model.evaluate(x_val, y_val)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    # Predict
    df_res = pd.DataFrame()
    predictions = model.predict(x_val, verbose=0)
    predictions[predictions<0.0] = 0.0
    
    # Visualize
    dates = np.array(df.index.to_pydatetime())[indices][split_at:]
    datenums = md.date2num(dates) 
    plt.xticks(rotation=25)
    ax = plt.gca()
    DATEFMT = '%d/%m-%Y %H'
    xfmt = md.DateFormatter(DATEFMT)
    ax.xaxis.set_major_formatter(xfmt)
    plt.plot(datenums, predictions.tolist(), 'r.')
    plt.plot(datenums, y_val.tolist(), 'b.')
    plt.show()
    

def train_nn_multicity(filename_train, filename_test, features=FEATURES):
    # Prepare data
    df_train = sanitize(filename_train, normalize=True, interpolate_limit=None)
    df_test = sanitize(filename_test, normalize=True, interpolate_limit=None)
    x_train, y_train = prepare(df_train, features=FEATURES)
    x_test, y_test = prepare(df_test, features=FEATURES)
    assert x_train.shape[0] == y_train.shape[0]
    n_samples = x_train.shape[0]
    
    # NOTE experimental network configuration
    model = Sequential([
        Dense(HIDDEN_SIZE, input_dim=x_train.shape[1], init='uniform', activation='relu'),
        #Dense(64, activation='linear'),
        Dense(10, init='uniform', activation='relu'),
        Dense(y_train.shape[1], init='uniform', activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test))
    scores = model.evaluate(x_test, y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    # Get the original max of incident counts
    df_train_orig = sanitize(filename_train, normalize=False, interpolate_limit=None)
    df_test_orig = sanitize(filename_test, normalize=False, interpolate_limit=None)
    max_incidents = df_test_orig['incidents_total'].max()
    ratio = df_test_orig['incidents_total'].mean() / df_train_orig['incidents_total'].mean()
    print("Max incidents_total: %d" % max_incidents)
    print("Ratio: %.5f" % ratio)
    
    # Predict
    predictions = model.predict(x_test, verbose=0)
    predictions[predictions<0.0] = 0.0
    predictions *= max_incidents*ratio
    
    # Visualize
    df_train_orig[['incidents_total']].plot()
    df_test_orig['predictions'] = predictions.flatten().tolist()
    df_test_orig[['incidents_total', 'predictions']].plot()
    
    # Store predictions
    basename = filename_test.split('.')[0]
    if features:
        outfile = basename + '_w_weather.pddf.hdf5'
    else:
        outfile = basename + '_wo_weather.pddf.hdf5'
    df_test_orig.to_hdf(outfile, 'df', mode='w')


def main(args):
    if len(args.filename) == 1:
        print("Training/testing on single data set")
        train_nn(args.filename[0])
    elif len(args.filename) == 2:
        print("Training/testing using two separate data sets")
        train_nn_multicity(args.filename[0], args.filename[1]) # With weather
        train_nn_multicity(args.filename[0], args.filename[1], features=[]) # Without features
        plt.show()
    else:
        print("bad input")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename',
                        nargs='*',
                        help="DE traffic+weather training data (.pddf.hdf5)")

    try:
        sys.exit(main(parser.parse_args()))
    except Exception as e:
        print("Failure: %s" % e)
        sys.exit(1)
        
