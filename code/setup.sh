#!/usr/bin/env bash

# create conda environment
conda create -n shoerinsics python=3.9

# activate environment
conda activate shoerinsics

# install required packages
conda install -c pytorch -c conda-forge -c anaconda  --file requirements.txt


