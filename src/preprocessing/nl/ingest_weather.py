"""
Ingest KNMI (Dutch) Weather Stations & Observations.
Cf. http://localhost:8888/notebooks/NL/_ImportKNMIStations.ipynb
Cf. http://projects.knmi.nl/klimatologie/uurgegevens/
Cf. http://projects.knmi.nl/klimatologie/uurgegevens/selectie.cgi

The hour refers to observations of the previous hour, i.e. hour = 1 indicates
observations to have taken place from 00:00 to 01:00. Times are UTC.

@todo: Currently only works for parsing all Dutch stations.
       Change to autodetect the appropriate skiprows (line 95).
@todo: Fold hour and date together.

Call Signature:
$ python ingest_weather.py --fname KNMI_20160915_hourly.txt
"""

import pandas as pd
import argparse


###############################################################################
###############################################################################
# FUNCTIONS
###############################################################################
###############################################################################

def extract_weather_station(fname):
    """
    Extract Weather Stations, Store to Dataframe.
    @param: fname - File name of KNMI weather data. Header has station data.
    @returns: df_stations - Dataframe with Weather Station Data.
    """

    # Debug
    print "// Extracting Stations (%s)" % fname

    # Read Lines
    with open(fname) as f:
        lines = f.readlines()

    # Scan Beginning of Station ID Block
    for iline, line in enumerate(lines):
        if line == '# STN      LON(east)   LAT(north)     ALT(m)  NAME\r\n':
            iline_lo = iline+1

    # Scan End of Station ID Block
    for iline, line in enumerate(lines[iline_lo:]):
        if line == '# YYYYMMDD = datum (YYYY=jaar,MM=maand,DD=dag); \r\n':
            iline_hi = iline_lo + iline - 1
        if iline == 50:
            break

    # Extract Station Lines
    lines_stations = lines[iline_lo:iline_hi]

    # Build Dataframe with Stations
    lon = []; lat = []; alt = []; loc = []; station_id = []
    for line_station in lines_stations:
        station_id.append(int(line_station.strip()[2:].split(':')[0]))
        line_station_data = line_station.strip()[2:].split(':')[1]
        lon.append(float(line_station_data[:14].strip()))
        lat.append(float(line_station_data[15:27].strip()))
        alt.append(float(line_station_data[28:38].strip()))
        loc.append(line_station_data[39:].strip().title())

    # Make Dataframe
    df = pd.DataFrame(data={ 'station_id': station_id, \
                             'latitude_deg': lat, 'longitude_deg': lon, \
                             'altitude_m': alt, 'location': loc}, \
                      columns=['station_id', \
                               'latitude_deg', 'longitude_deg', \
                               'altitude_m', 'location'])

    # Return
    return df


def extract_weather_observations(fname):
    """
    Extract Weather Observations.
    """

    # Column Descriptors
    names = [ 'station', 'date', 'hour', \
              'wind_direction_deg', 'wind_speed_m_s', \
              'wind_speed_last10min_m_s', 'wind_speed_peak_m_s', \
              'temperature_C', 'temperature_min_C', 'dew_point_C', \
              'sunshine_frac_hour', 'total_incident_radiation_J_cm2', \
              'precipitation_duration_frac_hour', \
              'precipitation_mm_hr', \
              'pressure_kPa', \
              'visibility_intcode', 'cloud_cover_intcode', \
              'rh_percent', \
              'weather_intcode', 'measurement_intcode', \
              'mist_bool', 'rain_bool', \
              'snow_bool', 'severe_weather_bool', 'ice_bool' ]

    # Read Files
    print '// Loading Weather Observations'
    df = pd.read_csv(fname, skiprows=71, header=None, \
                     sep=',', names=names, \
                     parse_dates=[1,3], \
                     skipinitialspace=True)

    # Convert
    df.sunshine_frac_hour *= 0.1
    df.wind_speed_m_s *= 0.1
    df.wind_speed_peak_m_s *= 0.1
    df.wind_speed_last10min_m_s *= 0.1
    df.temperature_min_C *= 0.1
    df.temperature_C *= 0.1
    df.dew_point_C *= 0.1
    df.precipitation_duration_frac_hour *= 0.1
    df.precipitation_mm_hr *= 0.1
    df.pressure_kPa *= 0.1/10.0

    # Return
    return df


###############################################################################
###############################################################################
# MAIN CODE
###############################################################################
###############################################################################

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fname', required=True, \
                    help='KNMI Weather Observations.')
args = parser.parse_args()

# Weather Stations
df_stations = extract_weather_station(args.fname)

# Weather Observations
df_obs = extract_weather_observations(args.fname)

# Store Station Dataframe
print "// Storing Station Locations (%i Stations)" % len(df_stations.index)
df_stations.to_hdf('weather_stations.pddf.hdf5', 'df', mode='w', \
                   format='table')

# Store Observations Dataframe
print "// Storing Observations (%i)" % len(df_obs.index)
df_obs.to_hdf('weather_observations.pddf.hdf5', 'df', mode='w', format='table')
