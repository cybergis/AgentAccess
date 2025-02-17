#!/bin/bash
# Author: Alexander Michels
# Purpose: Some of the data is too large to have
# in Github decompressed, so this script is designed
# to decompress all of that data for you!
curr_dir=$(pwd)

cd travel_times/
bzip2 -dk *.bz2
cd $curr_dir

# unzip the SVI data
cd svi/
unzip -o SVI2018_US_CT.zip
unzip -o SVI_2022_US_CT.zip
cd $curr_dir
