#!/usr/bin/env bash
set -xe

date;
MACHINE="martin@celsius.smn.cs.brown.edu";
REMOTE_DIR="/home/martin/projects/replaceNet";
REMOTE_PY="/home/martin/anaconda3/bin/python3"

scp *.py "$MACHINE:$REMOTE_DIR"
#scp -r "$(pwd)/" "$MACHINE:$REMOTE_DIR"
ssh "$MACHINE" "screen -S replaceNet -dm bash -c 'cd $REMOTE_DIR; $REMOTE_PY harmonize.py 2>&1 | tee rawT.log' "


