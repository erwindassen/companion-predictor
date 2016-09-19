"""
Loads Traffic Stations KML into a Pandas Dataframes.
"""

import lxml.objectify
import urllib2
import pandas as pd


# Load Raw Data
print '// Loading KML with Measurement Sites'
url = 'https://raw.githubusercontent.com/erwindassen'
url += '/companion-predictor/erwin/sites.kml'
xml = lxml.objectify.parse(urllib2.urlopen(url)).getroot()

# Prepare Dataframe
stations = []
lons = []
lats = []
for iplace, place in enumerate(xml.Document.Placemark):
    stations.append(place.attrib['id'].lower())
    lons.append(float(place.Point.coordinates.text.split(',')[0]))
    lats.append(float(place.Point.coordinates.text.split(',')[1]))

# Assemble Dataframe
df = pd.DataFrame(data={'station': stations, \
                        'latitude': lats, 'longitude': lons }, \
                  columns=['station', 'latitude', 'longitude'])

# Store
fname = 'traffic_stations.pddf.hdf5'
print "// Storing Dataframe to %s" % fname
df.to_hdf(fname, 'df', mode='w', format='table')
