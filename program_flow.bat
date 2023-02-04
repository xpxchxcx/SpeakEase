@echo off

cd raw_video_pipeline
peekingduck run
echo "raw video capture complete"
rem pause

cd ..
python raw_transfer.py
echo "source as raw video on analysis pipeline complete"
rem pause

cd analysis_video_pipeline
peekingduck run --viewer
echo "analysis complete"
pause