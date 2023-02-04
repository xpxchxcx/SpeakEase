"""Docstring for the are_arms_folded.py module

This module implements a custom dabble Node class for analysing a given pose.

Usage
-----
This module should be part of a package that follows the file structure as specified by the
[PeekingDuck documentation](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html).

Navigate to the root directory of the package and run the following line on the terminal:

```
peekingduck run
```
"""

# pylint: disable=logging-format-interpolation

from math import pi, sqrt
from typing import Any, Mapping, Optional

from peekingduck.pipeline.nodes.abstract_node import AbstractNode

from custom_nodes.dabble.utils import (
    Coord,
    ERROR_OUTPUT,
    KP_LEFT_ELBOW,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_RIGHT_ELBOW,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
    angle_between_vectors_in_rad,
    obtain_keypoint
)


class Node(AbstractNode):
    """Custom Node class to determine whether the arms are folded for every given pose

    Methods
    -------
    are_arms_folded : bool
        Determines if the arms of the given pose are folded
    run : dict
        Returns the dictionary of folded arms for each given pose
    """

    def __init__(
            self,
            config: Optional[Mapping[str, Any]] = None,
            **kwargs
    ) -> None:
        """Initialises the custom Node class

        Parameters
        ----------
        config : dict, optional
            Node custom configuration

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments for instantiating the AbstractNode parent class
        """

        super().__init__(config, node_path=__name__, **kwargs)  # type: ignore

    def are_arms_folded(
            self,
            left_shoulder: Optional[Coord],
            left_elbow: Optional[Coord],
            left_wrist: Optional[Coord],
            right_shoulder: Optional[Coord],
            right_elbow: Optional[Coord],
            right_wrist: Optional[Coord]
    ) -> bool:
        """Determines if the arms of the given pose are folded

        Parameters
        ----------
        left_shoulder : `Coord`, optional
            \\( (x, y) \\) coordinate of the left shoulder
        left_elbow : `Coord`, optional
            \\( (x, y) \\ coordinate of the left elbow
        left_wrist : `Coord`, optional
            \\( (x, y) \\ coordinate of the left wrist
        right_shoulder : `Coord`, optional
            \\( (x, y) \\ coordinate of the right shoulder
        right_elbow : `Coord`, optional
            \\( (x, y) \\ coordinate of the right elbow
        right_wrist : `Coord`, optional
            \\( (x, y) \\ coordinate of the right wrist

        Returns
        -------
        bool
            ``True`` if both arms are folded, ``False`` otherwise

        Notes
        -----
        The line from the shoulder to the elbow intersects the line from the wrist to the elbow at the elbow.

        An arm is considered folded if:

        - The angle that the two lines make with each other is less than \\( 120 90^{\\circ} \\)
        - The x-coordinate of the wrist lies in between the x-coordinates of the shoulders
        - The y-coordinate of the wrist lies below the y-coordinate of either shoulder
        - The distance between the wrist and the elbow is at least half that between the shoulders
        """

        # Check if keypoints are defined
        if left_shoulder is None or \
                left_elbow is None or \
                left_wrist is None or \
                right_shoulder is None or \
                right_elbow is None or \
                right_wrist is None:
            return False

        # Obtain coordinates
        left_shoulder_x, left_shoulder_y = left_shoulder
        right_shoulder_x, right_shoulder_y = right_shoulder
        left_elbow_x, left_elbow_y = left_elbow
        right_elbow_x, right_elbow_y = right_elbow
        left_wrist_x, left_wrist_y = left_wrist
        right_wrist_x, right_wrist_y = right_wrist

        # Calculate relevant vectors
        left_shoulder_elbow_vec = (
            left_shoulder_x - left_elbow_x,
            left_shoulder_y - left_elbow_y
        )
        left_wrist_elbow_vec = (
            left_wrist_x - left_elbow_x,
            left_wrist_y - left_elbow_y
        )
        right_shoulder_elbow_vec = (
            right_shoulder_x - right_elbow_x,
            right_shoulder_y - right_elbow_y
        )
        right_wrist_elbow_vec = (
            right_wrist_x - right_elbow_x,
            right_wrist_y - right_elbow_y
        )

        # Calculate angle made between the left shoulder, left elbow, and left wrist
        left_angle = angle_between_vectors_in_rad(
            *left_shoulder_elbow_vec,
            *left_wrist_elbow_vec
        )
        # Calculate angle made between the right shoulder, right elbow, and right wrist
        right_angle = angle_between_vectors_in_rad(
            *right_shoulder_elbow_vec,
            *right_wrist_elbow_vec
        )
        if left_angle == ERROR_OUTPUT or right_angle == ERROR_OUTPUT:
            # Needs debugging
            self.logger.warning(
                'Either one or both the calculated angles in are_arms_folded() has returned an error.' +
                f'\nAngle calculated between left shoulder to left elbow {left_shoulder_elbow_vec} ' +
                f'and left wrist to left elbow {left_wrist_elbow_vec} is {left_angle} radians.' +
                f'\nAngle calculated between right shoulder to right elbow {right_shoulder_elbow_vec} ' +
                f'and right wrist to right elbow {right_wrist_elbow_vec} is {right_angle} radians.'
            )
            return False

        # Calculate distance from the left elbow to the left wrist
        left_dist = sqrt(
            (left_wrist_x - left_elbow_x) * (left_wrist_x - left_elbow_x) + \
            (left_wrist_y - left_elbow_y) * (left_wrist_y - left_elbow_y)
        )
        # Calculate distance from the right elbow to the right wrist
        right_dist = sqrt(
            (right_wrist_x - right_elbow_x) * (right_wrist_x - right_elbow_x) + \
            (right_wrist_y - right_elbow_y) * (right_wrist_y - right_elbow_y)
        )
        # Calculate distance between the two shoulders
        shoulder_dist = sqrt(
            (right_shoulder_x - left_shoulder_x) * (right_shoulder_x - left_shoulder_x) + \
            (right_shoulder_y - left_shoulder_y) * (right_shoulder_y - left_shoulder_y)
        )

        threshold = 2 * pi / 3  # 2/3pi rad or 120 deg
        # Check if left arm is folded
        left_folded = left_angle < threshold and \
            right_shoulder_x <= left_wrist_x <= left_shoulder_x and \
            left_wrist_y > min(left_shoulder_y, right_shoulder_y) and \
            left_dist * 2 >= shoulder_dist
        # Check if right arm is folded
        right_folded = right_angle < threshold and \
            right_shoulder_x <= right_wrist_x <= left_shoulder_x and \
            right_wrist_y > min(left_shoulder_y, right_shoulder_y) and \
            right_dist * 2 >= shoulder_dist
        # Check if both arms are folded
        return left_folded and right_folded

    def run(
            self,
            inputs: Mapping[str, Any]
    ) -> Mapping[str, Mapping[int, int]]:
        """Returns the dictionary of folded arms for each given pose

        Parameters
        ----------
        inputs : dict
            Dictionary with the following keys:
            
            - 'img' - given image to be displayed
            - 'keypoints' - keypoints from PoseNet model
            - 'obj_attrs' - to obtain tracking IDs from PoseNet model

        Returns
        -------
        dict
            Dictionary with the following keys:
            
            - 'arms_folded' - output of the current run
        """

        # Initialise error message
        error_msg = 'The input dictionary does not contain the {} key.'

        # Check if required inputs are in pipeline
        if 'img' not in inputs:
            # There must be an image to display
            self.logger.error(error_msg.format("'img'"))
            return {'arms_folded': {}}  # type: ignore
        elif 'keypoints' not in inputs:
            self.logger.warning(error_msg.format("'keypoints'"))
        
        # Get required inputs from pipeline
        height, width, *_ = inputs['img'].shape
        all_ids = inputs.get('obj_attrs', {}).get('ids', [])
        all_keypoints = inputs.get('keypoints', [])

        # Handle the detection of each person
        arms_folded = {}
        for curr_id, keypoints in zip(all_ids, all_keypoints):

            # Store and display PoseNet keypoints
            keypoint_list = [
                obtain_keypoint(
                    *keypoint.tolist(),
                    img_width=width,
                    img_height=height
                ) \
                for keypoint in keypoints
            ]

            # Update the relevant ID in the output dictionary
            arms_folded[curr_id] = int(
                self.are_arms_folded(
                    keypoint_list[KP_LEFT_SHOULDER],
                    keypoint_list[KP_LEFT_ELBOW],
                    keypoint_list[KP_LEFT_WRIST],
                    keypoint_list[KP_RIGHT_SHOULDER],
                    keypoint_list[KP_RIGHT_ELBOW],
                    keypoint_list[KP_RIGHT_WRIST]
                )
            )

        return {'arms_folded': arms_folded}


if __name__ == '__main__':
    pass
