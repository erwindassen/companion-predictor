"""NN traffic+weather time series using Keras"""

import sys
import os
import argparse

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.engine.training import slice_X
from keras.utils.visualize_util import plot as k_plt
from keras.utils.visualize_util import model_to_dot

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


def plot_model(model, filename=None):
    if filename:
        k_plt(model, to_file=filename, show_shapes=True)
    #graph = model_to_dot(model)
    #graph.write_png(filename)
    
    # Print sum of weights from input nodes to first layer
    w1 = model.layers[0].get_weights()[0]
    #print(w1.shape)
    for i in range(w1.shape[0]):
        print("Input node %d: mean=%.4f, std=%.4f" % (i, w1[i,:].mean(), w1[i,:].std()))

    #for layer in model.layers:
    #    for n in layer.get_weights():
    #        print(n.shape)
        #print(layer.get_weights().shape)


def train_nn(filename, nnconf, features=FEATURES, dropout=None, outdir=None):
    # Prepare data
    print("Reading data from %s ..." % filename)
    df = sanitize(filename, normalize=True, interpolate_limit=None)
    _x, _y = prepare(df, features=features)
    n_samples = _x.shape[0]
    print("Samples: %d" % n_samples)

    # Split data (same split as Volker used for other models)
    x_train = np.concatenate((_x[0:3534], _x[3705::]))
    y_train = np.concatenate((_y[0:3534], _y[3705::]))
    x_test = _x[3534:3705]
    y_test = _y[3534:3705]

    assert x_train.shape[0] == y_train.shape[0]
    print("Samples: %d" % n_samples)
    
    # Build network
    layers = [
        Dense(nnconf[0], input_dim=x_train.shape[1], init='uniform', activation='relu'),
    ]
    if dropout:
        print("Add dropout layer, p=%.3f" % dropout)
        layers += [Dropout(dropout)]
    layers += [Dense(n, init='uniform', activation='relu') for n in nnconf[1::]]
    layers += [Dense(y_train.shape[1], init='uniform', activation='linear')]
    model = Sequential(layers)
    
    plot_model(model)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test))
    scores = model.evaluate(x_test, y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    # Get the original max of incident counts
    #df_train_orig = sanitize(filename_train, normalize=False, interpolate_limit=None)
    df_test_orig = sanitize(filename, normalize=False, interpolate_limit=None)
    max_incidents = df_test_orig['incidents_total'].max()
    ratio = 1.0
    #ratio = df_test_orig['incidents_total'].mean() / df_train_orig['incidents_total'].mean()
    #print("Max incidents_total: %d" % max_incidents)
    #print("Ratio: %.5f" % ratio)
    
    # Predict
    _predictions = model.predict(x_test, verbose=0)
    _predictions[_predictions<0.0] = 0.0
    predictions = np.array([_predictions[i-3534] if 3534 <= i < 3705 else np.nan for i in range(n_samples)])
    #predictions *= max_incidents*ratio
    df_test_orig['predictions'] = predictions.flatten().tolist()
    
    if not outdir:
        return

    # Store predictions
    if features:
        basename = 'w_weather'
    else:
        basename = 'wo_weather'
    if dropout is not None:
        basename += '_dropout'
    df_test_orig.to_hdf(os.path.join(outdir, basename + '.pddf.hdf5'), 'df', mode='w')


def train_nn_multicity(filenames_train, filename_test, nnconf,
                       features=FEATURES, dropout=None, outdir=None):
    # Prepare data
    _first= filenames_train[0]
    print("Reading %s ..." % _first)
    df_train = sanitize(_first, normalize=True, interpolate_limit=None)
    x_train, y_train = prepare(df_train, features=features)
    for fname in filenames_train[1::]:
        print("Reading %s ..." % fname)
        df_train = sanitize(fname, normalize=True, interpolate_limit=None)
        _x, _y = prepare(df_train, features=features)
        x_train = np.concatenate((x_train, _x))
        y_train = np.concatenate((y_train, _y))

    print("Reading test data %s ..." % filename_test)
    df_test = sanitize(filename_test, normalize=True, interpolate_limit=None)
    x_test, y_test = prepare(df_test, features=features)
    assert x_train.shape[0] == y_train.shape[0]
    n_samples = x_train.shape[0]
    print("Samples: %d" % n_samples)
    
    # Build network
    layers = [
        Dense(nnconf[0], input_dim=x_train.shape[1], init='uniform', activation='relu'),
    ]
    if dropout:
        print("Add dropout layer, p=%.3f" % dropout)
        layers += [Dropout(dropout)]
    layers += [Dense(n, init='uniform', activation='relu') for n in nnconf[1::]]
    layers += [Dense(y_train.shape[1], init='uniform', activation='linear')]
    model = Sequential(layers)
    
    plot_model(model)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test))
    scores = model.evaluate(x_test, y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    # Get the original max of incident counts
    #df_train_orig = sanitize(filename_train, normalize=False, interpolate_limit=None)
    df_test_orig = sanitize(filename_test, normalize=False, interpolate_limit=None)
    max_incidents = df_test_orig['incidents_total'].max()
    ratio = 1.0
    #ratio = df_test_orig['incidents_total'].mean() / df_train_orig['incidents_total'].mean()
    #print("Max incidents_total: %d" % max_incidents)
    #print("Ratio: %.5f" % ratio)
    
    # Predict
    predictions = model.predict(x_test, verbose=0)
    predictions[predictions<0.0] = 0.0
    predictions *= max_incidents*ratio
    df_test_orig['predictions'] = predictions.flatten().tolist()
    
    if not outdir:
        return

    # Store predictions
    if features:
        basename = 'w_weather'
    else:
        basename = 'wo_weather'
    if dropout is not None:
        basename += '_dropout'
    df_test_orig.to_hdf(os.path.join(outdir, basename + '.pddf.hdf5'), 'df', mode='w')

    # Make plots
    plot_model(model, os.path.join(outdir, basename + '_model.png'))
    ax = df_test_orig[['incidents_total', 'predictions']].plot()
    fig = ax.get_figure()
    fig.savefig(os.path.join(outdir, basename + '_predictions.png'))

    ax = df_test_orig[['incidents_total', 'predictions']].plot()
    ax.set_xlim(pd.Timestamp('2015-05-01'), pd.Timestamp('2015-06-01'))
    fig = ax.get_figure()
    fig.savefig(os.path.join(outdir, basename + '_predictions_zoom.png'))


def main(args):
    nnconf = [int(i) for i in args.nn.split('-')]
    if len(args.filename) == 1:
        print("Training/testing on single data set")
        if args.exclude_weather:
            print("Excluding weather features")
            train_nn(args.filename[0], nnconf,
                     features=[], dropout=args.dropout,
                     outdir=args.outdir) # Without features
        else:
            train_nn(args.filename[0], nnconf,
                     dropout=args.dropout,
                     outdir=args.outdir) # With weather
    else:
        print("Training/testing using several data sets")
        if args.exclude_weather:
            print("Excluding weather features")
            train_nn_multicity(args.filename[0:-1], args.filename[-1], nnconf,
                               features=[], dropout=args.dropout,
                               outdir=args.outdir) # Without features
        else:
            train_nn_multicity(args.filename[0:-1], args.filename[-1], nnconf,
                               dropout=args.dropout,
                               outdir=args.outdir) # With weather
    return 0

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename',
                        nargs='*',
                        help="DE traffic+weather training data (.pddf.hdf5)")
    parser.add_argument('--nn',
                        required=True,
                        help="Network confguration, e.g. 128-32 for two hidden layers with 128 and 32 nodes respectively")
    parser.add_argument('--outdir',
                        required=False,
                        help="Directory to save data and plots")
    parser.add_argument('--exclude-weather',
                        required=False,
                        action='store_true',
                        help="Exclude weather features")
    parser.add_argument('--dropout',
                        required=False,
                        type=float,
                        help="Factor for dropout layer. If ommitted, no dropout layer is added.")
    main(parser.parse_args())
    sys.exit(0)

    try:
        sys.exit(main(parser.parse_args()))
    except Exception as e:
        print("Failure: %s" % e)
        sys.exit(1)
        
