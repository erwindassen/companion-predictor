# Preprocessing Scripts -- German Data

## Pipeline Description

These scripts take TomTom Traffic Incident XML files [1], historic DWD weather data [2], and merge them into a Pandas dataframe of incidents where each incident is annotated with weather data from the nearest weather station for a given observation at the nearest point in time.

| Script | Description |
| --- | --- |
| ingest_traffic.py | Ingest TomTom XML traffic event data into Pandas dataframe (HDF5 wrapped). |
| ingest_weather_obs.py | Ingest DWD historic weather station data into Pandas dataframe (HDF5 wrapped.) |
| ingest_weather_stations.py | Ingest DWD historic weather station locations into Pandas dataframes (HDF5 wrapped). |
| match_weather_to_traffic.py | Aligns weather observations with traffic incidents. This is computationally expensive and could probably done more efficiently. It takes about three days to ingest a month of data. |
| clean_traffic_with_weather.py | Cleans up traffic data by, e.g., throwing away incidents that have the weather station too far away or have missing values. See the file header for details. |

All scripts list call signatures in the header.

The pipeline is a bit complicated. In short, it works as follows:

1. Weather station locations are extracted from CSV files and stored in a dataframe (see ingest_weather_stations.py for details).
2. Weather observation data is unpacked (see/use unpack_weather_obs.sh).
3. Weather observations are processed into dataframes (see/use ingest_weather_obs.py)
4. Traffic data is ingested into dataframes. We do this on a day-by-day basis (iterating over the zipped XMLs) because the data XMLs are incredibly redundant (and therefore huge). See ingest_traffic.sh.
5. Traffic data is merged with station locations and observations (match_weather_to_traffic.py)
6. Merged traffic and weather data is cleaned up (clean_traffic_with_weather.py).

This pipeline operates on a few days of data at a time (dictated by the range of the a zipped TomTom traffic advisory collection). For each range, a HDF5-wrapped dataframe is generated. To join a number of these, use cat_pddf.py.

We mentioned that the TomTom XML files are incredibly redundant. At their highest granularity, they basically dump **all presently active traffic advisories in Germany** once every minute. Of course, situations usually persist minutes or hours, so this is a lot of repetitive data.

## Output Dataframe Description

The fields and units in the final dataframe are as follows.

| Key | Description | Unit |
| --- | ---- | --- |
| situation_id | | |
| event_type | Type of event. One of queingTraffic, slowTraffic, stationaryTraffic. |
| event_code | TMC event code [3]. We only store event codes 100<=X<200. | |
| location_lat_decimal | Event location latitude. | Degree Decimal |
| location_lon_decimal | Event location longitude. | Degree Decimal |
| timestamp | Timestamp of event. | UTC/Zulu |
| timestamp_seconds | Timestamp of event. | Unix Epoch | Seconds | 
| average_speed | Average absolute speed. | km/h |
| precipitation_closest_weather_station_id | ID of Closest Precipiation Measurement Station | |
| precipitation_closest_weather_station_dxdy | Geodesic Distance to Closest Precipiation Measurement Station | km |
| wind_closest_weather_station_id | ID of Closest Wind Measurement Station | |
| wind_closest_weather_station_dxdy | Geodesic Distance to Closest Wind Measurement Station | km |
| temperature_closest_weather_station_id | ID of Closest Temperature Measurement Station | |
| temperature_closest_weather_station_dxdy | Geodesic Distance to Closest Temperature Measurement Station | km |
| precipitation_dt | Time Delta to Nearest Precipitation Observations | Seconds |
| precipitation | Was there precipitation? | Bool
| precipitation_amount | Amount of precipitation | mm/h
| precipitation_kind | Type of precipitation | Integer
| temperature_dt | Time Delta to Nearest Temperature/Humidity Observations | Seconds |
| temperature | Observed Temperature | Deg C |
| relative_humidity | Observed Relative Humidity (100% ~ Fog) | Per cent |
| wind_dt | Time Delta to Nearest Wind Observation | Seconds |
| wind_speed | Wind Speed | m/s |
| wind_direction | Wind Direction | Degree |

## References/Sources

1. TomTom Data -- ???
2. DWD Data -- ftp://ftp-cdc.dwd.de/pub/CDC/observations_germany/climate/hourly/
3. TMC Codes -- http://wiki.openstreetmap.org/wiki/TMC/Event_Code_List


