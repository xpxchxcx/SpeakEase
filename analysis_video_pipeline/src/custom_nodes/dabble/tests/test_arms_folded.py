"""Docstring for the test_arms_folded.py module

This module implements unit tests for the custom Node class in `are_arms_folded.py`.

Usage
-----
From the terminal, navigate to the `analysis_video_pipeline/src` directory and run the following:

```
python -m unittest [-v] custom_nodes/dabble/tests/test_arms_folded.py
```

_Note that -v is an optional flag to increase the verbosity of the unit test outputs._

For more information, refer to the
[official unit testing documentation](https://docs.python.org/3/library/unittest.html#test-discovery).
"""

# pylint: disable=invalid-name

from itertools import chain, combinations
from typing import Optional
import unittest
from yaml import safe_load

from custom_nodes.dabble.are_arms_folded import Node
from custom_nodes.dabble.utils import (
    Coord,
    KP_LEFT_ELBOW,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_RIGHT_ELBOW,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
)
from custom_nodes.visualise import (
    ARE_ARMS_FOLDED_NEGATIVE_CASES_OUTSTRETCHED_ARMS,
    ARE_ARMS_FOLDED_NEGATIVE_CASES_TOUCHING_FACE,
    ARE_ARMS_FOLDED_POSITIVE_CASES_HALF_CROSS,
    ARE_ARMS_FOLDED_POSITIVE_CASES_FULL_CROSS
)


class TestNode(unittest.TestCase):
    """Python unit testing class to test the custom Node class in `are_arms_folded.py`

    Methods
    -------
    error_msg : str
        Template for unit test error messages
    test_are_arms_folded_missing
        Checks that the function returns ``False`` if at least one input is missing
    test_are_arms_folded_touching_face
        Checks that the arms are not considered folded if the pose is touching the face
    test_are_arms_folded_outstretched_arms
        Check that outstretched arms are not considered folded
    test_are_arms_folded_half_cross
        Checks that half-crossed arms are considered folded
    test_are_arms_folded_full_cross
        Checks that fully-crossed arms are considered folded
    """

    def setUp(self) -> None:
        """Initialises the Node object used for unit testing"""

        # Initialise YML filepath
        filepath = 'custom_nodes/configs/dabble/are_arms_folded.yml'

        # Obtain Node instance
        with open(filepath, 'r', encoding='utf-8') as config_file:
            config = safe_load(config_file)
            self.node = Node(config=config)

    def error_msg(
            self,
            left_shoulder: Optional[Coord],
            left_elbow: Optional[Coord],
            left_wrist: Optional[Coord],
            right_shoulder: Optional[Coord],
            right_elbow: Optional[Coord],
            right_wrist: Optional[Coord],
            output: bool,
            expected: bool,
            test: str
    ) -> str:
        """Template for unit test error messages

        Parameters
        ----------
        left_shoulder : `Coord`, optional
            The left shoulder coordinate \\( (x, y) \\) passed into the function
        left_elbow : `Coord`, optional
            The left elbow coordinate \\( (x, y) \\) passed into the function
        left_wrist : `Coord`, optional
            The left wrist coordinate \\( (x, y) \\) passed into the function
        right_shoulder : `Coord`, optional
            The right shoulder coordinate \\( (x, y) \\) passed into the function
        right_elbow : `Coord`, optional
            The right elbow coordinate \\( (x, y) \\) passed into the function
        right_wrist : `Coord`, optional
            The right wrist coordinate \\( (x, y) \\) passed into the function
        output : bool
            The obtained output from the function
        expected : bool
            The expected output of the function
        test : str
            The type of unit test performed

        Returns
        -------
        str
            The formatted error message
        """

        return f'\n\
                =============================================\n\
                 Test Type: {test}\n\
                Parameters: left_shouler = {left_shoulder}\n\
                            left_elbow = {left_elbow}\n\
                            left_wrist = {left_wrist}\n\
                            right_shoulder = {right_shoulder}\n\
                            right_elbow = {right_elbow}\n\
                            right_wrist = {right_wrist}\n\
                    Output: {output}\n\
                  Expected: {expected}\n\
                ============================================='

    def test_are_arms_folded_missing(self) -> None:
        """Checks that the function returns ``False`` if at least one input is missing

        The function requires all coordinates to perform the check.
        """

        # Initialise coordinates
        pose = [(i, i) for i in range(17)]
        required_coordinates = {
            KP_LEFT_SHOULDER,
            KP_LEFT_ELBOW,
            KP_LEFT_WRIST,
            KP_RIGHT_SHOULDER,
            KP_RIGHT_ELBOW,
            KP_RIGHT_WRIST
        }
        coordinates = {joint: coordinate for joint, coordinate in enumerate(pose) if joint in required_coordinates}

        # Set one or more coordinates to None
        for i, selected_coordinates in enumerate(chain(
                *(combinations(required_coordinates, num + 1) for num in range(len(required_coordinates)))
        )):
            left_shoulder = None if KP_LEFT_SHOULDER in selected_coordinates else \
                coordinates[KP_LEFT_SHOULDER]
            left_elbow = None if KP_LEFT_ELBOW in selected_coordinates else \
                coordinates[KP_LEFT_ELBOW]
            left_wrist = None if KP_LEFT_WRIST in selected_coordinates else \
                coordinates[KP_LEFT_WRIST]
            right_shoulder = None if KP_RIGHT_SHOULDER in selected_coordinates else \
                coordinates[KP_RIGHT_SHOULDER]
            right_elbow = None if KP_RIGHT_ELBOW in selected_coordinates else \
                coordinates[KP_RIGHT_ELBOW]
            right_wrist = None if KP_RIGHT_WRIST in selected_coordinates else \
                coordinates[KP_RIGHT_WRIST]

            # Obtain the result
            res = self.node.are_arms_folded(
                left_shoulder,
                left_elbow,
                left_wrist,
                right_shoulder,
                right_elbow,
                right_wrist
            )

            # Perform assertion check
            self.assertFalse(
                res,
                self.error_msg(
                    left_shoulder,
                    left_elbow,
                    left_wrist,
                    right_shoulder,
                    right_elbow,
                    right_wrist,
                    res,
                    False,
                    f'One or more Missing Inputs (Case {i + 1})'
                )
            )

    def test_are_arms_folded_touching_face(self) -> None:
        """Checks that the arms are not considered folded if the pose is touching the face"""

        for i, coordinates in enumerate(ARE_ARMS_FOLDED_NEGATIVE_CASES_TOUCHING_FACE):

            # Initialise coordinates
            left_shoulder, \
                left_elbow, \
                left_wrist, \
                right_shoulder, \
                right_elbow, \
                right_wrist = coordinates

            # Obtain the result
            res = self.node.are_arms_folded(
                left_shoulder,
                left_elbow,
                left_wrist,
                right_shoulder,
                right_elbow,
                right_wrist
            )

            # Perform assertion check
            self.assertFalse(
                res,
                self.error_msg(
                    left_shoulder,
                    left_elbow,
                    left_wrist,
                    right_shoulder,
                    right_elbow,
                    right_wrist,
                    res,
                    False,
                    f'Touching Face instead of Folding Arms (Case {i + 1})'
                )
            )

    def test_are_arms_folded_outstretched_arms(self) -> None:
        """Check that outstretched arms are not considered folded"""

        for i, coordinates in enumerate(ARE_ARMS_FOLDED_NEGATIVE_CASES_OUTSTRETCHED_ARMS):

            # Initialise coordinates
            left_shoulder, \
                left_elbow, \
                left_wrist, \
                right_shoulder, \
                right_elbow, \
                right_wrist = coordinates

            # Obtain the result
            res = self.node.are_arms_folded(
                left_shoulder,
                left_elbow,
                left_wrist,
                right_shoulder,
                right_elbow,
                right_wrist
            )

            # Perform assertion check
            self.assertFalse(
                res,
                self.error_msg(
                    left_shoulder,
                    left_elbow,
                    left_wrist,
                    right_shoulder,
                    right_elbow,
                    right_wrist,
                    res,
                    False,
                    f'Arms Outstretched (Case {i + 1})'
                )
            )

    def test_are_arms_folded_half_cross(self) -> None:
        """Checks that half-crossed arms are considered folded"""

        for i, coordinates in enumerate(ARE_ARMS_FOLDED_POSITIVE_CASES_HALF_CROSS):

            # Initialise coordinates
            left_shoulder, \
                left_elbow, \
                left_wrist, \
                right_shoulder, \
                right_elbow, \
                right_wrist = coordinates

            # Obtain the result
            res = self.node.are_arms_folded(
                left_shoulder,
                left_elbow,
                left_wrist,
                right_shoulder,
                right_elbow,
                right_wrist
            )

            # Perform assertion check
            self.assertTrue(
                res,
                self.error_msg(
                    left_shoulder,
                    left_elbow,
                    left_wrist,
                    right_shoulder,
                    right_elbow,
                    right_wrist,
                    res,
                    True,
                    f'Arms Half-Crossed (Case {i + 1})'
                )
            )

    def test_are_arms_folded_full_cross(self) -> None:
        """Checks that fully-crossed arms are considered folded"""

        for i, coordinates in enumerate(ARE_ARMS_FOLDED_POSITIVE_CASES_FULL_CROSS):

            # Initialise coordinates
            left_shoulder, \
                left_elbow, \
                left_wrist, \
                right_shoulder, \
                right_elbow, \
                right_wrist = coordinates

            # Obtain the result
            res = self.node.are_arms_folded(
                left_shoulder,
                left_elbow,
                left_wrist,
                right_shoulder,
                right_elbow,
                right_wrist
            )

            # Perform assertion check
            self.assertTrue(
                res,
                self.error_msg(
                    left_shoulder,
                    left_elbow,
                    left_wrist,
                    right_shoulder,
                    right_elbow,
                    right_wrist,
                    res,
                    True,
                    f'Arms Full-Crossed (Case {i + 1})'
                )
            )


if __name__ == '__main__':
    unittest.main()
