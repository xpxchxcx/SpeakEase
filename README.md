<p align="center">
 <img src=https://user-images.githubusercontent.com/87000020/216827457-1d03e644-73ef-4f94-a3fb-6150c2c1437a.svg alt="SpeakEase" width="650" height="300"/>
</p>

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

**SpeakEase** helps users overcome their fear of public speaking using computer vision to detect body language and provide feedback. 
It's accessible, cost-effective, and eliminates need for physical mentors. 
Improving public speaking skills, especially for those affected by glossophobia, SpeakEase empowers users to become confident speakers.

## Features

### Real-time Feedback to Users
When deployed, SpeakEase records and analyses live presentations / presentation attempts, 
providing instant feedback to its users about their presentation postures. 

### Accurate Tracking of Body Languages
SpeakEase makes use of [PeekingDuck](https://github.com/aisingapore/PeekingDuck) to obtain pose estimations 
and determines if the pose **violates common bad practices during public speaking**.
Some of these bad practices currently supported are:

- **Folded Arms** - When the arms are crossed and/or tucked behind the elbows
- **Touching Face** - When the hands are touching the face
- **Leaning** - When the user is leaning and/or swaying excessively

### Instant Playback for Users
SpeakEase sends back the users a playback of the analysed presentations after each session and 
allows them to review the footage in their free time.

<p align="center">
 <img src=https://user-images.githubusercontent.com/87000020/216830154-196a2202-8040-4f20-b64b-6c821b8b7fb0.gif alt="Sample Video"/>
</p>

## Installation

1. Refer to the [PeekingDuck installation guide](https://peekingduck.readthedocs.io/en/stable/getting_started/index.html) to install PeekingDuck.
2. (optional) To satisfy additional dependencies used for code testing, install `matplotlib` via `pip` using the terminal

```
pip install matplotlib
```

3. [Fork](https://github.com/xpxchxcx/SpeakEase/fork) or clone this repository to access the source code.
Navigate to the root directory in the terminal and run

```
[Windows] bash program_flow.bat
[MacOS/Linux] sh program_flow.sh
```

While the video feed window is displayed during the execution of the program, simply press the `q` key to terminate the video.

## Acknowledgements

This project is an undertaking of the National AI Student Challenge (NAISC) 2022 organised by AI Singapore.
