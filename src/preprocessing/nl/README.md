# Preprocessing Scripts -- Dutch Data

## Pipeline Description

These scripts take the raw HDF5 output of the preprocesser for the NDW/KNMI road/weather data, and turn them into an easily loaded Pandas dataframe.

Originally, weather data is reported at hourly intervals, but traffic data reported on the minute. The scripts downsample the traffic data to the hour by averaging.

| Script | Description |
| --- | --- |
| downsample.py | Script to downsample the data. The file takes a list of HDF5 files from stdin. The call signature and layout of the list file is shown in the header of downsample.py. |
| hdf5_to_pddf.py | Script to convert the downsampled data in the HDF5 and convert into a Pandas dataframe (which is stored in an HDF5 container), use hdf5_to_pddf.py. |

The pipeline should be executed in the order above.

## Output Dataframe Description

The fields and units in the final dataframe are as follows.

| Key | Description | Unit |
| --- | ---- | --- |
| station | ID String of Measurement Station | |
| timestamp_start | Begin of Measurement Period | TZ Unclear -- GMT? UTC? |
| timestamp_end | End of Measurement Period | TZ Unclear -- GMT? UTC? |
| trafficspeed | Avg Traffic Speed | km/h |
| trafficflow | Avg Traffic Flow | cars/h |
| temperature | Temperature | Deg C |
| precipitation | Amount of Precipitation | mm/h |
| windspeed | Wind Speed | m/s |
