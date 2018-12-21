#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4GB

ml TensorFlow/1.10.1-fosscuda-2018a-Python-3.6.4

python -u acgan64.py
