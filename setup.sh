#!/bin/bash
# Author: Alex Michels
# Purpose: Sets up the repo/environment after a fresh clone

# get the curr_dir
curr_dir=$(pwd)

# unpack some data
cd data/
bash decompress-data.sh

cd $curr_dir
