#!/bin/bash


#COBALT -n 1
#COBALT -q dgx
#COBALT -A Performance
#COBALT -t 5:00:00

#export https_proxy="https://proxy:3128"
#export http_proxy="http://proxy:3128"
#export ftp_proxy="ftp://proxy:3128"

#source activate torch
python py_run.py --save_file "mnist_MER" --opt "MER" --json_file "mnist.json" --total_runs 1
#source deactivate 
