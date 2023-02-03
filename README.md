# SpeakEase [![Python version](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue.svg)](https://pypi.org/project/peekingduck/)

<h4 align="center">
  <a href="https://good-looking-ostrich.static.app/">API Docs</a>
  <span> · </span>
  <a href="https://github.com/xpxchxcx/SpeakEase/issues">Report a bug</a>
</h4>

## Introduction

SpeakEase helps users overcome their fear of public speaking using computer vision to detect body language and provide feedback. 
It's accessible, cost-effective, and eliminates need for physical mentors. 
Improving public speaking skills, especially for those affected by glossophobia, SpeakEase empowers users to become confident speakers.

### How it Works

When deployed, the application records and analyses live presentations / presentation attempts, 
providing instant feedback to its users about their presentation postures. 
SpeakEase makes use of [PeekingDuck](https://github.com/aisingapore/PeekingDuck) to obtain pose estimations 
and determines if the pose **violates common bad practices during public speaking**.
Some of these bad practices currently supported are:

- **Folded Arms** - When the arms are crossed and/or tucked behind the elbows
- **Touching Face** - When the hands are touching the face
- **Leaning** - When the user is leaning and/or swaying excessively

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