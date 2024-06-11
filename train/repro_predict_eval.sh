#!/bin/bash

# Define Paths ==================
home_root='/home/th748/UNICORN'
script_dir="${home_root}/seq2cells/scripts"

config_eval="${home_root}/train/test_configs/config_anndata_eval.yml"

# Run Predictions & Evaluation ===
python ${script_dir}/training/eval_single_cell_model_group.py \
config_file=${config_eval} \
checkpoint_file=$1
