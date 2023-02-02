"""Docstring for the helper.py script

This script processes poses saved to a temporary text file ``test.txt``.
It checks whether the poses satisfy any of the pose checkers defined in the custom ``Node`` class.

Usage
-----
Navigate to this file in the terminal and run
    ```python -m helper.py```
"""


from itertools import product
import re
from typing import List, Mapping, Optional, Tuple

from analysis_video_pipeline.src.custom_nodes.dabble.movement import Node
import matplotlib.pyplot as plt
from yaml import safe_load


# Initialise constants
filepath = 'test.txt'
Coord = Tuple[int, int]


def get_node() -> Node:
    with open('analysis_video_pipeline/src/custom_nodes/configs/dabble/movement.yml', 'r') as config_file:
        config = safe_load(config_file)
        return Node(config=config)


def get_arms_folded_poses(
        node: Node,
        keypoints: List[Optional[Coord]],
        to_draw: bool
) -> Optional[Tuple[Coord, ...]]:
    """TODO documentation"""

    # Initialise joints
    left_shoulder = keypoints[node._KP_LEFT_SHOULDER]
    left_elbow = keypoints[node._KP_LEFT_ELBOW]
    left_wrist = keypoints[node._KP_LEFT_WRIST]
    right_shoulder = keypoints[node._KP_RIGHT_SHOULDER]
    right_elbow = keypoints[node._KP_RIGHT_ELBOW]
    right_wrist = keypoints[node._KP_RIGHT_WRIST]

    # Check if arms are folded
    res = node.are_arms_folded(
        left_shoulder,
        left_elbow,
        left_wrist,
        right_shoulder,
        right_elbow,
        right_wrist
    )
    if not res:
        return
    elif to_draw:
        plot_arm_folded_pose(
            left_shoulder,  # type: ignore
            left_elbow,  # type: ignore
            left_wrist,  # type: ignore
            right_shoulder,  # type: ignore
            right_elbow,  # type: ignore
            right_wrist,  # type: ignore
            res
        )
    return left_shoulder, \
        left_elbow, \
        left_wrist, \
        right_shoulder, \
        right_elbow, \
        right_wrist    # type: ignore


def get_leaning_poses(
        node: Node,
        keypoints: List[Optional[Coord]],
        to_draw: bool
) -> Optional[Tuple[Coord, ...]]:
    """TODO documentation"""

    # Initialise joints
    left_shoulder = keypoints[node._KP_LEFT_SHOULDER]
    right_shoulder = keypoints[node._KP_RIGHT_SHOULDER]
    left_hip = keypoints[node._KP_LEFT_HIP]
    right_hip = keypoints[node._KP_RIGHT_HIP]
    
    # Check if leaning
    res = node.is_leaning(
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip
    )
    if not res:
        return
    elif to_draw:
        plot_leaning_pose(
            left_shoulder,  # type: ignore
            right_shoulder,  # type: ignore
            left_hip,  # type: ignore
            right_hip,  # type: ignore
            res
        )
    return left_shoulder, \
        right_shoulder, \
        left_hip, \
        right_hip  # type: ignore


def get_touching_face_poses(
        node: Node,
        keypoints: List[Optional[Coord]],
        to_draw: bool
) -> Optional[Tuple[Optional[Coord], ...]]:
    """TODO documentation"""

    # Initialise joints
    left_elbow = keypoints[node._KP_LEFT_ELBOW]
    right_elbow = keypoints[node._KP_RIGHT_ELBOW]
    left_wrist = keypoints[node._KP_LEFT_WRIST]
    right_wrist = keypoints[node._KP_RIGHT_WRIST]
    nose = keypoints[node._KP_NOSE]
    left_eye = keypoints[node._KP_LEFT_EYE]
    right_eye = keypoints[node._KP_RIGHT_EYE]
    left_ear = keypoints[node._KP_LEFT_EAR]
    right_ear = keypoints[node._KP_RIGHT_EAR]
    
    # Check if leaning
    res = node.is_face_touched(
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        nose,
        left_eye,
        right_eye,
        left_ear,
        right_ear
    )
    if not res:
        return
    elif to_draw:
        plot_touching_face_pose(
            left_elbow,
            right_elbow,
            left_wrist,
            right_wrist,
            nose,
            left_eye,
            right_eye,
            left_ear,
            right_ear,
            res
        )
    return left_elbow, \
        right_elbow, \
        left_wrist, \
        right_wrist, \
        nose, \
        left_eye, \
        right_eye, \
        left_ear, \
        right_ear


def get_correct_poses() -> Tuple[Mapping[int, int], Mapping[int, int], Mapping[int, int]]:
    """TODO documentation"""

    # Initialise constants
    node = get_node()
    draw_limit = 5

    with open(filepath, 'r') as f:
        arm_folded_poses = {}
        leaning_poses = {}
        touching_face_poses = {}

        for i, lst in enumerate(f.readlines()):
            lst = [
                None if x == 'None' else tuple(int(y) for y in x[1:-1].split(', ')) \
                for x in re.split(r'(?<=[^0-9]), (?<=[^0-9])', lst[1:-2])
            ]
            r1 = get_arms_folded_poses(node, lst, len(arm_folded_poses) < draw_limit)
            r2 = get_leaning_poses(node, lst, len(leaning_poses) < draw_limit)
            r3 = get_touching_face_poses(node, lst, len(touching_face_poses) < draw_limit)
            if r1 is not None:
                arm_folded_poses[i + 1] = r1
            if r2 is not None:
                leaning_poses[i + 1] = r2
            if r3 is not None:
                touching_face_poses[i + 1] = r3

        return arm_folded_poses, leaning_poses, touching_face_poses


def plot_arm_folded_pose(
        left_shoulder: Coord,
        left_elbow: Coord,
        left_wrist: Coord,
        right_shoulder: Coord,
        right_elbow: Coord,
        right_wrist: Coord,
        arms_folded: bool
) -> None:
    """TODO documentation"""

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.plot(*zip(right_shoulder, left_shoulder, left_elbow, right_elbow, right_shoulder))
    plt.plot(*zip(left_shoulder, left_elbow, left_wrist), marker='o')
    plt.plot(*zip(right_shoulder, right_elbow, right_wrist), marker='o')
    plt.title('Arms Are ' + ('' if arms_folded else 'Not ') + 'Folded')
    plt.show()


def plot_leaning_pose(
        left_shoulder: Coord,
        right_shoulder: Coord,
        left_hip: Coord,
        right_hip: Coord,
        leaning: bool
) -> None:
    """TODO documentation"""
    
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.plot(*zip(right_shoulder, left_shoulder, left_hip, right_hip, right_shoulder), marker='o')
    plt.title('Pose Is ' + ('' if leaning else 'Not ') + 'Leaning')
    plt.show()


def plot_touching_face_pose(
        left_elbow: Optional[Coord],
        right_elbow: Optional[Coord],
        left_wrist: Optional[Coord],
        right_wrist: Optional[Coord],
        nose: Optional[Coord],
        left_eye: Optional[Coord],
        right_eye: Optional[Coord],
        left_ear: Optional[Coord],
        right_ear: Optional[Coord],
        touching_face: bool
) -> None:
    """TODO documentation"""

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    if left_elbow is not None and left_wrist is not None:
        plt.plot(*zip(left_elbow, left_wrist), marker='o')
    if right_elbow is not None and right_wrist is not None:
        plt.plot(*zip(right_elbow, right_wrist), marker='o')
    if left_eye is not None:
        if nose is not None:
            plt.plot(*zip(nose, left_eye), marker='o')
        if left_ear is not None:
            plt.plot(*zip(left_eye, left_ear), marker='o')
    if right_eye is not None:
        if nose is not None:
            plt.plot(*zip(nose, right_eye), marker='o')
        if right_ear is not None:
            plt.plot(*zip(right_eye, right_ear), marker='o')
    
    for c1, c2 in product(
            (left_elbow, left_wrist, right_elbow, right_wrist),
            (nose, left_eye, left_ear, right_eye, right_ear)
    ):
        if c1 is None or c2 is None:
            continue
        plt.plot(*zip(c1, c2), marker='o')

    plt.title('Face ' + ('' if touching_face else 'Not ') + 'Touched')
    plt.show()


if __name__ == '__main__':
    arms_folded, leaning, touching_face = get_correct_poses()
    print(f'Arms Folded Poses: {arms_folded}\n')
    print(f'Leaning Poses: {leaning}\n')
    print(f'Touching Face: {touching_face}')
