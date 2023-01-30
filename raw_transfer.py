import os
import re

ROOT_DIR = os.path.abspath(os.curdir)
MP3_DIR = ROOT_DIR + "./raw_video_pipeline/raw_videos"
YML_DIR = ROOT_DIR + "./analysis_video_pipeline/pipeline_config.yml"

# Find the newest .mp3 file in the directory
directory = MP3_DIR
newest_mp3 = max(os.listdir(directory), key=lambda f: os.path.getctime(os.path.join(directory, f)))
newest_mp3_path = os.path.join(directory, newest_mp3).replace("\\", "\\\\")

# Read the contents of the yml file
with open(YML_DIR, 'r') as file:
    contents = file.read()

# Use regular expression to replace only the text after "source: " on the specific line
contents = re.sub(r"source:.*", f'source: {newest_mp3_path}', contents)


# Write the modified contents back to the yml file
with open(YML_DIR, 'w') as file:
    file.write(contents)

