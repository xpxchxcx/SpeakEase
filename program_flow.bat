@echo off
cd C:\Users\mynam\Downloads\AISG_Challenge-main\raw_video_pipeline
peekingduck run
echo raw video capture complete
pause
python C:\Users\mynam\Downloads\AISG_Challenge-main\raw_transfer.py
echo source as raw video on analysis pipeline complete
pause

cd C:\Users\mynam\Downloads\AISG_Challenge-main\analysis_video_pipeline
peekingduck run

echo analysis complete
pause