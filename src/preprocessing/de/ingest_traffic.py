"""
Ingest Traffic Data from TomTom.

Make sure to edit the path where you dropped in PyLR. PyLR also needs the
bitstring module, which is available in several Anaconda channels (bioconda).

XML Parsing:
http://www.saltycrane.com/blog/2011/07/example-parsing-xml-lxml-objectify/

TMC Event Codes:
http://wiki.openstreetmap.org/wiki/TMC/Event_Code_List

@todo: Be a bit more clever with TMC event codes (alertCEventCode).
@todo: Include PyLR and bitstream as submodules.
@todo: Vectorize: 1/ Only location_openlr_base64 in loop
                  2/ Use Apply PyLR.serialize + Extracting of Lat/Lon Pairs
                  3/ Midpoint Computation
"""

import glob
import lxml.objectify
import urllib2
import pandas as pd
import sys; sys.path.append('/Users/volker/Work_ST/Companion/External/PyLR')
# import sys; sys.path.append('/home/volker/Source/External/PyLR')
import pylr
import numpy as np
import argparse


###############################################################################
###############################################################################
# FUNCTIONS
###############################################################################
###############################################################################

# Convenience
r2d = 180.0 / np.pi
d2r = np.pi / 180.0

def lat_lon_mid(lat_01, lon_01, lat_02, lon_02):
    """
    Finds the Lat/Lon Midpoint. Input/Output Angles in Degree.

    @param: lat_01 -- Latitude  of First Point [Deg]
    @param: lon_01 -- Longitude of First Point [Deg]
    @param: lat_02 -- Latitude  of Second Point [Deg]
    @param: lon_02 -- Longitude of Second Point [Deg]
    @return: lat_03 -- Latitude  of Midpoint [Deg]
    @return: lon_03 -- Longitude of Midpoint [Deg]
    """

    # Convert...
    lat_01 *= d2r
    lat_02 *= d2r
    lon_01 *= d2r
    lon_02 *= d2r
    
    # http://www.movable-type.co.uk/scripts/latlong.html
    Bx = np.cos(lat_02) * np.cos(lon_02-lon_01)
    By = np.cos(lat_02) * np.sin(lon_02-lon_01)
    lat_03 = np.arctan2(np.sin(lat_01) + np.sin(lat_02), \
                        np.sqrt( (np.cos(lat_01) + Bx) * \
                            (np.cos(lat_01) + Bx) + By * By ) )
    lon_03 = lon_01 + np.arctan2(By, np.cos(lat_01) + Bx)
    
    # Convert Back
    lat_03 *= r2d
    lon_03 *= r2d
    
    # Return
    return lat_03, lon_03


###############################################################################
###############################################################################
# MAIN CODE
###############################################################################
###############################################################################

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True, \
                    help="Date/Directory To Process (YYYY_MM_DD)")
args = parser.parse_args()

# Debug
print "// Processing Data for Date %s" % (args.date)

# Setup Lists
# NB: Apparently, appending lists in Python us an O(1) op and Python allocates
#     room for the next 8 list items on every append. So there's no need for
#     preallocating the list.
situation_id = []
timestamp = []
event_code = []
event_type = []
average_speed = []
location_openlr = []
location_lat_decimal = []
location_lon_decimal = []

# Build List of Files to Ingest
globs = glob.glob("%s/trafficIncidents*" % args.date)
print "// Processing %i Traffic XML Files" % len(globs)

# Loop XML Files
for gg in globs:
    # print "// Processing %s" % gg

    # Load XML
    with open(gg, 'r') as fxml:

        # Some XML files are corrupted, skip them silently
        try:
            xml = lxml.objectify.parse(fxml).getroot()
        except:
            xml = None

        if xml is not None:
            
            # Loop Situations
            for situation in xml.payloadPublication.situation:
                
                # Codes 701 < X < 854 Are Maintenance Codes
                # Codes 500 Are Lanes Closed
                # Codes <200 Are Interesting to Us
                # Codes 100 to 200 Are Interesting
                # 0 to 100 - Might be interesting,
                #            but refer to more general congestion

                if (int(situation.situationRecord.\
                        situationRecordExtension.\
                        alertCEventCode.text) >= 100) and \
                   (int(situation.situationRecord.\
                        situationRecordExtension.\
                        alertCEventCode.text) < 200):
                    
                    # We should count in these tags being here
                    situation_id.append(situation.situationRecord.attrib['id'])
                    timestamp.append(situation.situationRecord.\
                                     situationRecordCreationTime.\
                                     text)
                    event_code.append(situation.situationRecord.\
                                      situationRecordExtension.\
                                      alertCEventCode.\
                                      text)
                    location_openlr.append(situation.situationRecord.\
                                           groupOfLocations.\
                                           locationContainedInGroup.\
                                           locationExtension.openlr.binary.\
                                           text)

                    # Sometimes, some tags aren't there, even if they should...
                    tags_present = []
                    for e in situation.situationRecord.iterchildren():
                        tags_present.append(e.tag[33:])

                    if 'abnormalTrafficType' in tags_present:
                        event_type.append(situation.situationRecord.\
                                          abnormalTrafficType.text)
                    else:
                        event_type.append('Unknown')

                    if 'abnormalTrafficExtension' in tags_present:
                        average_speed.append(situation.situationRecord.\
                                             abnormalTrafficExtension.\
                                             averageSpeed.\
                                             text)
                    else:
                        average_speed.append(np.nan)
                    
                    # Parse Location to Lat/Lon Pairs
                    location_tmp = pylr.parse_binary(situation.\
                                                     situationRecord.\
                                                     groupOfLocations.\
                                                     locationContainedInGroup.\
                                                     locationExtension.openlr.\
                                                     binary.text, \
                                                     base64=True)

                    # Determine Midpoint
                    lat_mid, lon_mid = \
                        lat_lon_mid(location_tmp.flrp.coords.lat, \
                                    location_tmp.flrp.coords.lon, \
                                    location_tmp.llrp.coords.lat, \
                                    location_tmp.llrp.coords.lon)
                    
                    location_lat_decimal.append(lat_mid)
                    location_lon_decimal.append(lon_mid)

# Throw into Dataframe
if len(globs) > 0:
    data = { 'situation_id': situation_id, \
             'timestamp': pd.to_datetime(timestamp), \
             'event_code': event_code, \
             'event_type': event_type, \
             'average_speed': np.asarray(average_speed, dtype=np.float64), \
             'location_openlr_base64': location_openlr, \
             'location_lat_decimal': np.asarray(location_lat_decimal, \
                                                dtype=np.float64), \
             'location_lon_decimal': np.asarray(location_lon_decimal, \
                                                dtype=np.float64) }
    columns = data.keys()
    df = pd.DataFrame(data=data, columns=columns)

    # Drop Duplicates
    print "// Dropping Duplicates"
    df.drop_duplicates(inplace=True)

    # Store Dataframe
    fname = "Traffic_%s.hdf5" % args.date
    print "// Saving %i Records to %s" % (len(df.index), fname)
    with pd.HDFStore(fname, 'w') as store:
        store['df'] = df

else:
    print "!! No Records. Not Saving."
