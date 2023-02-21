<br />

<img src="https://user-images.githubusercontent.com/87000020/216827457-1d03e644-73ef-4f94-a3fb-6150c2c1437a.svg">

<p align="center">
  <img src=https://img.shields.io/badge/python-3.8%20%7C%203.9-blue.svg alt="Python Versions"/>
  <img src=https://github.com/xpxchxcx/SpeakEase/actions/workflows/pylint.yml/badge.svg alt="Pylint" />
  <img src=https://github.com/xpxchxcx/SpeakEase/actions/workflows/ci.yml/badge.svg alt="Unit Tests" />
  <a href=http://hits.dwyl.com/xpxchxcx/SpeakEase><img src=https://hits.dwyl.com/xpxchxcx/SpeakEase.svg?style=flat-square&show=unique alt="Hit Count" /></a>
</p>

<h4 align="center">
  <a href="https://youtu.be/AtQ3xYFboaA/">Video</a>
  <span> · </span>
  <a href="https://good-looking-ostrich.static.app/">Documentation</a>
  <span> · </span>
  <a href="https://github.com/xpxchxcx/SpeakEase/issues">Report a bug</a>
</h4>

---

**SpeakEase** helps users develop public speaking skills using computer vision to detect poor body postures. 
An accessible, cost-effective, and smart digital solution, 
SpeakEase significantly reduces reliance on physical mentorship and/or expensive courses 
and empowers users to become confident speakers.

## Features

### Real-time Feedback
SpeakEase records and analyses live presentations / presentation attempts, 
providing instant feedback to its users about their presentation postures. 

### Accurate Detection of Body Languages
SpeakEase leverages on [PeekingDuck](https://github.com/aisingapore/PeekingDuck) to obtain pose estimations 
and determines if the pose **violates common bad practices during public speaking**.
Some of these bad practices currently supported are:

- **Folded Arms** - When the arms are crossed and/or tucked behind the elbows
- **Touching Face** - When the hands are touching the face
- **Leaning** - When the user is leaning and/or swaying excessively

<p align="center">
 <img src=https://user-images.githubusercontent.com/87000020/216830154-196a2202-8040-4f20-b64b-6c821b8b7fb0.gif alt="Sample Video 1"/>
</p>

### Supports Multi-Person Analyses
SpeakEase supports group presentations by tracking multiple users simultaneously.

<p align="center">
 <img src=https://user-images.githubusercontent.com/87000020/216838269-130c9264-aab2-4d03-a589-a9aca221ddfe.gif alt="Sample Video 2"/>
</p>

### Instant Playback
SpeakEase returns a playback of the analysed presentations after each session to the user.

Reviews can occur at any time after the session, either complementing feedback from professional mentors, 
or allowing users to record their progress and track their improvements over time.

## Usage

### Installation

1. Refer to the [PeekingDuck installation guide](https://peekingduck.readthedocs.io/en/stable/getting_started/index.html) to install PeekingDuck.
2. (optional) To satisfy additional dependencies used for code testing, install `matplotlib` via `pip` using the terminal

```
pip install matplotlib
```

3. [Fork](https://github.com/xpxchxcx/SpeakEase/fork) or clone this repository to access the source code.

### Physical set-up

4. Prepare the physical environment. Some recommendations for setting up prior to deploying SpeakEase include:

- Place the video-capturing device on a level surface at a suitable distance away from the user(s). Ideally, the video should capture the **full body of the user(s) presenting**.
- Capture the video against a monotone surface such as a wall or a green screen, if possible.
- The body of each presenter should face the camera directly at all times, if possible. Avoid turning to the side as this will mess with the detection algorithm.

### Deployment

5. Navigate to the root directory in the terminal and run

```
[Windows] bash program_flow.bat
[MacOS/Linux] sh program_flow.sh
```

The program will start by capturing a live video feed without additional analytics. Execute your presentation attempt during this window. Once complete, press the `q` key to terminate the recording.

The program will then automatically open PeekingDuck viewer to output the result of the processed video. Use the scaling controls to fit the resultant video to the display window and use the horizontal scroller to analyse each frame.

## Acknowledgements

This project is an undertaking of the National AI Student Challenge (NAISC) 2022 organised by AI Singapore. 
Read the proposal [here](https://docs.google.com/document/d/1LbN1IhLCAH8XIzLqoWbnzlq0TzxpQlO-R37CZbm0naw/edit#).
