"""Docstring for the is_leaning.py module

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

from math import pi
from typing import Any, Mapping, Optional

from peekingduck.pipeline.nodes.abstract_node import AbstractNode

from custom_nodes.dabble.utils import (
    Coord,
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    angle_between_vectors_in_rad,
    obtain_keypoint
)


class Node(AbstractNode):
    """Custom Node class to determine whether each pose is leaning

    Methods
    -------
    is_leaning : bool
        Determines if the given pose is leaning towards one side
    run : dict
        Returns the dictionary of leaning poses for each given pose
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

    def is_leaning(
            self,
            left_shoulder: Optional[Coord],
            right_shoulder: Optional[Coord],
            left_hip: Optional[Coord],
            right_hip: Optional[Coord]
    ) -> bool:
        """Determines if the given pose is leaning towards one side

        Parameters
        ----------
        left_shoulder : `Coord`, optional
            Tuple containing the \\( (x, y) \\) coordinates of the left shoulder
        right_shoulder : `Coord`, optional
            Tuple containing the \\( (x, y) \\) coordinates of the right shoulder
        left_hip : `Coord`, optional
            Tuple containing the \\( (x, y) \\) coordinates of the left hip
        right_hip : `Coord`, optional
            Tuple containing the \\( (x, y) \\) coordinates of the right hip
        left_knee : `Coord`, optional
            Tuple containing the \\( (x, y) \\) coordinates of the left knee
        right_knee : `Coord`, optional
            Tuple containing the \\( (x, y) \\) coordinates of the right knee

        Returns
        -------
        bool
            ``True`` if the pose is leaning towards one side, ``False`` otherwise

        Notes
        -----
        The line from the shoulder to the hip intersects the hip line.

        A pose is considered leaning if the angle that the two lines make with each other
        falls outside a \\( 15^{\\circ} \\) tolerance.
        """

        # Check if keypoints are defined
        if left_shoulder is None or \
                left_hip is None or \
                right_shoulder is None or \
                right_hip is None:
            return False

        # Initialize buffer
        # Get left and right shoulder keypoints
        left_shoulder_x, left_shoulder_y = left_shoulder
        right_shoulder_x, right_shoulder_y = right_shoulder
        # Get left and right hip keypoints
        left_hip_x, left_hip_y = left_hip
        right_hip_x, right_hip_y = right_hip

        # Calculate the angle between left shoulder to left hip to left knee
        left_angle = angle_between_vectors_in_rad(
            left_shoulder_x - left_hip_x,
            left_shoulder_y - left_hip_y,
            right_hip_x - left_hip_x,
            right_hip_y - left_hip_y
        )
        # Calculate the angle between right shoulder to right hip to right knee
        right_angle = angle_between_vectors_in_rad(
            right_shoulder_x - right_hip_x,
            right_shoulder_y - right_hip_y,
            left_hip_x - right_hip_x,
            left_hip_y - right_hip_y
        )

        # Check if either side has crossed the threshold for leaning
        sway_threshold = 15 * pi / 180  # 15 deg
        return left_angle < pi/2 - sway_threshold or \
            right_angle < pi/2 - sway_threshold or \
            left_angle > pi/2 + sway_threshold or \
            right_angle > pi/2 + sway_threshold

    def run(
            self,
            inputs: Mapping[str, Any]
    ) -> Mapping[str, Mapping[int, int]]:
        """Returns the dictionary of leaning poses for each given pose

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

            - 'is_leaning' - output of the current run
        """

        # Initialise error message
        error_msg = 'The input dictionary does not contain the {} key.'

        # Check if required inputs are in pipeline
        if 'img' not in inputs:
            # There must be an image to display
            self.logger.error(error_msg.format("'img'"))
            return {'is_leaning': {}}  # type: ignore
        elif 'keypoints' not in inputs:
            self.logger.warning(error_msg.format("'keypoints'"))

        # Get required inputs from pipeline
        height, width, *_ = inputs['img'].shape
        all_ids = inputs.get('obj_attrs', {}).get('ids', [])
        all_keypoints = inputs.get('keypoints', [])

        # Handle the detection of each person
        is_leaning = {}
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
            is_leaning[curr_id] = int(
                self.is_leaning(
                    keypoint_list[KP_LEFT_SHOULDER],
                    keypoint_list[KP_RIGHT_SHOULDER],
                    keypoint_list[KP_LEFT_HIP],
                    keypoint_list[KP_RIGHT_HIP]
                )
            )

        return {'is_leaning': is_leaning}


if __name__ == '__main__':
    pass
