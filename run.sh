#!/bin/sh

cd medical_extract
export PYTHONPATH="."
export ROOT_DIR="root"


/root/anaconda3/envs/py36/bin/python baseline/extract.py
#/root/anaconda3/envs/py36/bin/python baseline_2/extract.py


#/root/anaconda3/envs/py36/bin/python baseline/eval.py

