# Data README

This doc has all of the sources for the data sets contained.

* `travel_times` - This contains travel time data for census tracts in the city of Chicago. This data comes from the Center for Spatial Data Science and was calculated with OpenTripPlanner (OTP). The data can be found here: https://pysal.org/access/resources.html. It was accessed August 9th, 2022.
* `svi` - This data includes American Community Survey (ACS) and Social Vulnerability Index (SVI). The data comes from the CDC/ATDSR and can be downloaded here: https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html. We chose 2018 (the most recent at the time), Illinois, Census Tracts, and Shapefile for the options. The data was accessed August 9th, 2022.

**NOTE:** Some of the data may need to be decompressed. The [decompress-data.sh](decompress-data.sh) script can do this for you.
