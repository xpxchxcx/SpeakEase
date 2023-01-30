import os
import re

# Find the newest .mp3 file in the directory
directory = r'C:\Users\mynam\Downloads\AISG_Challenge-main\raw_video_pipeline\raw_videos'
newest_mp3 = max(os.listdir(directory), key=lambda f: os.path.getctime(os.path.join(directory, f)))
newest_mp3_path = os.path.join(directory, newest_mp3).replace("\\", "\\\\")

# Read the contents of the yml file
with open(r'C:\Users\mynam\Downloads\AISG_Challenge-main\analysis_video_pipeline\pipeline_config.yml', 'r') as file:
    contents = file.read()

# Use regular expression to replace only the text after "source: " on the specific line
contents = re.sub(r"source:.*", f'source: {newest_mp3_path}', contents)


# Write the modified contents back to the yml file
with open(r'C:\Users\mynam\Downloads\AISG_Challenge-main\analysis_video_pipeline\pipeline_config.yml', 'w') as file:
    file.write(contents)

