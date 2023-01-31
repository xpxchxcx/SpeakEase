"""Python script to edit the pipeline config YML file in the analysis pipeline

The script locates the recorded .mp4 file created from the raw video pipeline
and edits the source location in the YML file to the filepath of the located file.
"""


import os
import re


def main():
    ROOT_DIR = os.path.abspath(os.curdir)
    MP3_DIR = f'{ROOT_DIR}/raw_video_pipeline/raw_videos'
    YML_DIR = f'{ROOT_DIR}/analysis_video_pipeline/pipeline_config.yml'

    # Find the newest .mp4 file in the directory
    newest_mp3 = max(os.listdir(MP3_DIR), key=lambda f: os.path.getctime(os.path.join(MP3_DIR, f)))
    newest_mp3_path = os.path.join(MP3_DIR, newest_mp3).replace("\\", "\\\\")

    # Read the contents of the yml file
    with open(YML_DIR, 'r') as file:
        contents = file.read()

    # Use regular expression to replace only the text after "source: " on the specific line
    contents = re.sub(r"source:.*", f'source: {newest_mp3_path}', contents)

    # Write the modified contents back to the yml file
    with open(YML_DIR, 'w') as file:
        file.write(contents)


if __name__ == '__main__':
    main()
