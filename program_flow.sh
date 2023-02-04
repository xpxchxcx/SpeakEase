#!/bin/bash

cd "raw_video_pipeline"
peekingduck run
echo "raw video capture complete"
# read -p "Press Enter to continue" </dev/tty

cd ".."
python raw_transfer.py
echo "source as raw video on analysis pipeline complete"
# read -p "Press Enter to continue" </dev/tty

cd "analysis_video_pipeline"
peekingduck run --viewer
echo "analysis complete"
read -p "Press Enter to continue" </dev/tty