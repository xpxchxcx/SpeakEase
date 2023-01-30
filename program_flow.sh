#!/bin/bash

cd "raw_video_pipeline"
peekingduck run
echo "raw video capture complete"
# read -p "Press Enter to continue" </dev/tty

cd ".."
~/miniconda/envs/aisg/bin/python raw_transfer.py
echo "source as raw video on analysis pipeline complete"
# read -p "Press Enter to continue" </dev/tty

cd "analysis_video_pipeline"
peekingduck run
echo "analysis complete"
read -p "Press Enter to continue" </dev/tty