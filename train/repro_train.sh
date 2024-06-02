#!/bin/bash

# Define Paths ==================
home_root='/home/th748/UNICORN'
script_dir="${home_root}/seq2cells/scripts"

config_train="${home_root}/train/test_configs/config_pbmcanndata_fine_tune.yml"

# Run Training ==================
python ${script_dir}/training/fine_tune_on_anndata.py \
config_file=${config_train}

