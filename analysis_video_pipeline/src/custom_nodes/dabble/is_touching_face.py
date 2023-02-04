"""Docstring for the is_touching_face.py module

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

from math import sqrt
from typing import Any, Mapping, Optional

from peekingduck.pipeline.nodes.abstract_node import AbstractNode

from custom_nodes.dabble.utils import (
    Coord,
    KP_LEFT_EAR,
    KP_LEFT_ELBOW,
    KP_LEFT_EYE,
    KP_LEFT_WRIST,
    KP_NOSE,
    KP_RIGHT_EAR,
    KP_RIGHT_ELBOW,
    KP_RIGHT_EYE,
    KP_RIGHT_WRIST,
    obtain_keypoint
)


class Node(AbstractNode):
    """Custom Node class to determine whether the pose is touching the face

    Methods
    -------
    are_arms_folded : bool
        Determines if the hands of the given pose is touching the face
    run : dict
        Returns the dictionary of faces touched for each given pose
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

    def is_face_touched(
            self,
            left_elbow: Optional[Coord],
            right_elbow: Optional[Coord],
            left_wrist: Optional[Coord],
            right_wrist: Optional[Coord],
            nose: Optional[Coord],
            left_eye: Optional[Coord],
            right_eye: Optional[Coord],
            left_ear: Optional[Coord],
            right_ear: Optional[Coord]
    ) -> bool:
        """Determines if the hands of the given pose is touching the face

        Parameters
        ----------
        left_elbow : `Coord`, optional
            \\( (x, y) \\) coordinate of the left elbow
        right_elbow : `Coord`, optional
            \\( (x, y) \\) coordinate of the right elbow
        left_wrist : `Coord`, optional
            \\( (x, y) \\) coordinate of the left wrist
        right_wrist : `Coord`, optional
            \\( (x, y) \\) coordinate of the right wrist
        nose : `Coord`, optional
            \\( (x, y) \\) coordinate of the nose
        left_eye : `Coord`, optional
            \\( (x, y) \\) coordinate of the left eye
        right_eye : `Coord`, optional
            \\( (x, y) \\) coordinate of the right eye
        left_ear : `Coord`, optional
            \\( (x, y) \\) coordinate of the left ear
        right_ear : `Coord`, optional
            \\( (x, y) \\) coordinate of the right ear

        Returns
        -------
        bool
            ``True`` if either hand is touching the face, ``False`` otherwise

        Notes
        -----
        A pose is considered to be touching the face if the following conditions are satisfied:

        - Any of the coordinates for the elbows or wrists are defined
        - Any of the coordinates for the nose, eyes, or ears are defined
        - The distance from the facial feature(s) to the arm feature(s) is sufficiently small
        """

        # Define threshold for distance
        threshold = 100

        # Obtain coordinates of defined limb
        for limb_coordinate in [left_elbow, right_elbow, left_wrist, right_wrist]:
            if limb_coordinate is None:
                continue

            # Obtain coordinates of defined facial feature
            for face_coordinate in [nose, left_eye, right_eye, left_ear, right_ear]:
                if face_coordinate is None:
                    continue

                # Calculate distance between limb joint and facial feature joint
                limb_x, limb_y = limb_coordinate
                face_x, face_y = face_coordinate
                dist = sqrt(
                    (limb_x - face_x) * (limb_x - face_x) + \
                    (limb_y - face_y) * (limb_y - face_y)
                )

                # Check if limb is touching the face
                if dist < threshold:
                    return True

        # No pair of coordinates exist that satisfy the distance threshold
        return False

    def run(
            self,
            inputs: Mapping[str, Any]
    ) -> Mapping[str, Mapping[int, int]]:
        """Returns the dictionary of faces touched for each given pose

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

            - 'is_touching_face' - output of the current run
        """

        # Initialise error message
        error_msg = 'The input dictionary does not contain the {} key.'

        # Check if required inputs are in pipeline
        if 'img' not in inputs:
            # There must be an image to display
            self.logger.error(error_msg.format("'img'"))
            return {'is_touching_face': {}}  # type: ignore
        elif 'keypoints' not in inputs:
            self.logger.warning(error_msg.format("'keypoints'"))

        # Get required inputs from pipeline
        height, width, *_ = inputs['img'].shape
        all_ids = inputs.get('obj_attrs', {}).get('ids', [])
        all_keypoints = inputs.get('keypoints', [])

        # Handle the detection of each person
        touching_face = {}
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
            touching_face[curr_id] = int(
                self.is_face_touched(
                    keypoint_list[KP_LEFT_ELBOW],
                    keypoint_list[KP_RIGHT_ELBOW],
                    keypoint_list[KP_LEFT_WRIST],
                    keypoint_list[KP_RIGHT_WRIST],
                    keypoint_list[KP_NOSE],
                    keypoint_list[KP_LEFT_EYE],
                    keypoint_list[KP_RIGHT_EYE],
                    keypoint_list[KP_LEFT_EAR],
                    keypoint_list[KP_RIGHT_EAR]
                )
            )

        return {'is_touching_face': touching_face}


if __name__ == '__main__':
    pass
