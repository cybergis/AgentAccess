#!/bin/bash
# Author: Alexander Michels
# Purpose: Some of the data is too large to have
# in Github decompressed, so this script is designed
# to decompress all of that data for you!
curr_dir=$(pwd)
cd travel_times/
bzip2 -dk *.bz2
cd $curr_dir