# companion-predictor
A ML based travel time predictor for Companion NDW data  

The predictor reads HDF5 files as exported by the Java code (see the linked repository Companion-Risk-Factors). These files contain
weather and traffic data for NDW traffic measurement sites around the Netherlands. Each file contains the measurements for one site,
separated for the different data (traffic speed, traffic flow, temperature, wind speed, humidity, precipitation, etc...). 

The predictor reads these HDF5 files and generates Pandas DataFrame objects which combine all the different data into one frame and  
subsequently exports to HDF5 as well. Note that each exported file will contain the data for one measurement site at a specified time 
interval.
  
These exported DataFrame objects will be read and combined using the Dask library as this supports distributed processing. At this 
step the measurements potentially for multiple time intervals per measurement site will all be merged into chunks of specified size
still per measurement site. 

Finally these chunks are fed to the TFLearn library for learning from the data. 


# Install
Python 3 is required to run the software. The repository contains a requirements.txt file containing all the packages required to
run the predictor software. Typically this requirements file can be used to install from Anaconda (http://www.continuum.io ). After
installing Anaconda (Python 3+) use the following command to install the packages:
conda create -n environment_name --file requirements.txt

 
 
# Background and Explanation
The NDW has currently (dd 2016-06-06) more than 27.000 measurement sites all over the Netherlands, with a focus on the highways
(approximately 16.000 measurement sites). In the Java tool a filter has already been applied to select only the measurement
sites on highways, as they will be the (initial) focus of this study. As an initial step in the learning process all measurement 
sites are considered to behave similarly to weather events. This ensures the availability of large amounts of data and limits 
the scope of the problem. At a later stage (possibly outside the scope of this particular study) a classification of measurement 
sites can be considered to possibly improve the forecasting abilities of the learning system.
