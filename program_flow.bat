@echo off
cd %~dp0raw_video_pipeline
peekingduck run
echo raw video capture complete
pause
cd ..
python raw_transfer.py
echo source as raw video on analysis pipeline complete
pause
cd %~dp0analysis_video_pipeline
peekingduck run
echo analysis complete
pause