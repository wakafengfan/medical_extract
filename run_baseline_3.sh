#!/bin/sh

cd medical_extract
export PYTHONPATH="."
export ROOT_DIR="root"


/root/anaconda3/envs/py36/bin/python baseline_3/extract_3.py --train_file train_0729.json
