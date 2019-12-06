#!/usr/bin/env bash

REMOTE_DIR="/home/martin/projects/replaceNet";
REMOTE_PY="/home/martin/anaconda3/bin/python3"

cd "$REMOTE_DIR"; "$REMOTE_PY" train_replace_net.py 2>&1 | tee replaceNet.log;


