"""
Sanitize DE time series data:
    - Replace nan's in incident counts with 0
    - Add column incidents_total
    - Optionally normalize indident counts
    - Replace precipitation_kind=-999 with nan
    - Interpolate over <limit> nan's
"""

import datetime as dt
import pandas as pd
import numpy as np



def normalize_series(df, key):
    s = df[key]
    df[key] = s / (s.max() - s.min())


def sanitize(filename, normalize=False, interpolate_limit=3):
    df = pd.read_hdf(filename, 'df')
    
    # Replace nan's in in incident counts
    df.loc[np.isnan(df.incidents_queueing), 'incidents_queueing'] = 0
    df.loc[np.isnan(df.incidents_slow), 'incidents_slow'] = 0
    df.loc[np.isnan(df.incidents_stationary), 'incidents_stationary'] = 0

    # Add incidents_total
    df['incidents_total'] = df['incidents_queueing'] + \
                            df['incidents_slow'] + \
                            df['incidents_stationary']

    if normalize:
        normalize_series(df, 'incidents_queueing')
        normalize_series(df, 'incidents_slow')
        normalize_series(df, 'incidents_stationary')
        normalize_series(df, 'incidents_total')

    # Interpolate
    df.interpolate(limit=interpolate_limit, inplace=True)

    return df

