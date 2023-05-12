#!/bin/bash


#COBALT -n 1
#COBALT -A Performance
#COBALT -t 5:00:00

export https_proxy="https://proxy:3128"
export http_proxy="http://proxy:3128"
export ftp_proxy="ftp://proxy:3128"

source ~/miniconda3/etc/profile.d/conda.sh
conda list envs
conda activate torch
# python get_data.py all
python py_run.py --save_dir "Res/NASH_mnist" --opt "NASH" --json_file "mnist.json" --total_runs 1
source deactivate 
