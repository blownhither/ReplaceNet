#!/usr/bin/env bash
set -xe

date;
MACHINE="martin@alchemist.smn.cs.brown.edu";
REMOTE_DIR="/home/martin/projects/replaceNet";
REMOTE_PY="/home/martin/anaconda3/bin/python3"

scp *.py *.yml *.yaml "$MACHINE:$REMOTE_DIR"

#scp -r "$(pwd)/model_logs" "$MACHINE:$REMOTE_DIR"
#scp -r "$(pwd)/image_data" "$MACHINE:$REMOTE_DIR"
#scp -r "tmp/model-20191202162813" "$MACHINE:$REMOTE_DIR/tmp"
#scp "image_data/bool_mask_sep_inst.zip" "$MACHINE:$REMOTE_DIR/image_data"


#ssh "$MACHINE" "screen -S replaceNet -dm bash -c 'cd $REMOTE_DIR; $REMOTE_PY train_replace_net.py 2>&1 | tee rawT.log' "


