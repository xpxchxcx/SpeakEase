"""Docstring for the test_is_touching_face.py module

This module implements unit tests for the custom Node class in `is_touching_face.py`.

Usage
-----
From the terminal, navigate to the `analysis_video_pipeline/src` directory and run the following:

```
python -m unittest [-v] custom_nodes/dabble/tests/test_is_touching_face.py
```

_Note that -v is an optional flag to increase the verbosity of the unit test outputs._

For more information, refer to the
[official unit testing documentation](https://docs.python.org/3/library/unittest.html#test-discovery).
"""

# pylint: disable=invalid-name, logging-format-interpolation

from itertools import product
from typing import Optional
import unittest
from yaml import safe_load

from custom_nodes.dabble.is_touching_face import Node
from custom_nodes.dabble.utils import Coord
from custom_nodes.visualise import IS_TOUCHING_FACE_POSITIVE_CASES_ALL_DEFINED


class TestNode(unittest.TestCase):
    """Python unit testing class to test the custom Node class in `is_touching_face.py`

    Methods
    -------
    error_msg : str
        Template for unit test error messages
    test_is_face_touched_all_missing : None
        hecks that undefined poses return ``False``
    test_is_face_touched_negative_some_defined : None
        Checks that poses are undefined if all of the keypoints in a required set are undefined
    test_is_face_touched_negative_all_defined : None
        Checks that poses that do not meet the requirement are considered not touching face
    test_is_face_touched_positive_some_defined : None
        Checks that poses are considered as touching face even with some undefined keypoints
    test_is_face_touched_all_defined : None
        Checks that poses where the face is touched is considered as touching face
    """

    def setUp(self) -> None:
        """Initialises the Node object used for unit testing"""

        # Initialise YML filepath
        filepath = 'custom_nodes/configs/dabble/is_touching_face.yml'

        # Obtain Node instance
        with open(filepath, 'r', encoding='utf-8') as config_file:
            config = safe_load(config_file)
            self.node = Node(config=config)

    def error_msg(
            self,
            left_elbow: Optional[Coord],
            right_elbow: Optional[Coord],
            left_wrist: Optional[Coord],
            right_wrist: Optional[Coord],
            nose: Optional[Coord],
            left_eye: Optional[Coord],
            right_eye: Optional[Coord],
            left_ear: Optional[Coord],
            right_ear: Optional[Coord],
            output: bool,
            expected: bool,
            test: str
    ) -> str:
        """Template for unit test error messages

        Parameters
        ----------
        left_elbow : `Coord`, optional
            The left elbow coordinate \\( (x, y) \\) passed into the function
        right_elbow : `Coord`, optional
            The right elbow coordinate \\( (x, y) \\) passed into the function
        left_wrist : `Coord`, optional
            The left wrist coordinate \\( (x, y) \\) passed into the function
        right_wrist : `Coord`, optional
            The right wrist coordinate \\( (x, y) \\) passed into the function
        nose : `Coord`, optional
            The nose coordinate \\( (x, y) \\) passed into the function
        left_eye : `Coord`, optional
            The left eye coordinate \\( (x, y) \\) passed into the function
        right_eye : `Coord`, optional
            The right eye coordinate \\( (x, y) \\) passed into the function
        left_ear : `Coord`, optional
            The left ear coordinate \\( (x, y) \\) passed into the function
        right_ear : `Coord`, optional
            The right ear coordinate \\( (x, y) \\) passed into the function
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
                Parameters: left_elbow = {left_elbow}\n\
                            right_elbow = {right_elbow}\n\
                            left_wrist = {left_wrist}\n\
                            right_wrist = {right_wrist}\n\
                            nose = {nose}\n\
                            left_eye = {left_eye}\n\
                            right_eye = {right_eye}\n\
                            left_ear = {left_ear}\n\
                            right_ear = {right_ear}\n\
                    Output: {output}\n\
                  Expected: {expected}\n\
                ============================================='

    def test_is_face_touched_all_missing(self) -> None:
        """Checks that undefined poses return ``False``"""

        # Obtain the resut
        res = self.node.is_face_touched(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )

        # Perform assertion check
        self.assertFalse(
            res,
            self.error_msg(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                res,
                False,
                'Touching Face All Missing'
            )
        )

    def test_is_face_touched_negative_some_defined(self) -> None:
        """Checks that poses are undefined if all of the keypoints in a required set are undefined

        The function requires that at least one keypoint in each set of keypoints is defined.
        """

        # Select coordinates to define
        for i, (limb_coordinate, face_coordinate) in enumerate(((None, (0, 0)), ((0, 0), None))):

            # Initialise coordinates
            left_elbow = right_elbow = left_wrist = right_wrist = limb_coordinate
            nose = left_eye = right_eye = left_ear = right_ear = face_coordinate

            # Obtain the result
            res = self.node.is_face_touched(
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

            # Perform assertion check
            self.assertFalse(
                res,
                self.error_msg(
                    left_elbow,
                    right_elbow,
                    left_wrist,
                    right_wrist,
                    nose,
                    left_eye,
                    right_eye,
                    left_ear,
                    right_ear,
                    res,
                    False,
                    f'Touching Face Negative Some Defined (Case {i + 1})'
                )
            )

    def test_is_face_touched_negative_all_defined(self) -> None:
        """Checks that poses that do not meet the requirement are considered not touching face"""

        # Initialise fake coordinates for limb features and facial features
        limb_coordinate = (250, 250)
        face_coordinate = (0, 0)

        # Initialise coordinates
        left_elbow = right_elbow = left_wrist = right_wrist = limb_coordinate
        nose = left_eye = right_eye = left_ear = right_ear = face_coordinate

        # Obtain the result
        res = self.node.is_face_touched(
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

        # Perform assertion check
        self.assertFalse(
            res,
            self.error_msg(
                left_elbow,
                right_elbow,
                left_wrist,
                right_wrist,
                nose,
                left_eye,
                right_eye,
                left_ear,
                right_ear,
                res,
                False,
                'Touching Face Negative All Defined'
            )
        )

    def test_is_face_touched_positive_some_defined(self) -> None:
        """Checks that poses are considered as touching face even with some undefined keypoints

        The function is expected to check every possible pair of points
        for one that meets the requirement for touching face.
        """

        # Initialise fake coordinates for limb features and facial features
        limb_coordinate = (25, 25)
        face_coordinate = (0, 0)

        # Select coordinates to define
        for case, (i, j) in enumerate(product(range(4), range(4, 9))):

            # Initialise coordinate list
            # Define limb coordinate at index i and facial coordinate at index j
            coordinates = [None] * 9
            coordinates[i] = limb_coordinate  # type: ignore
            coordinates[j] = face_coordinate  # type: ignore

            # Initialise specific coordinates
            left_elbow, \
                right_elbow, \
                left_wrist, \
                right_wrist, \
                nose, \
                left_eye, \
                right_eye, \
                left_ear, \
                right_ear, = coordinates

            # Obtain the result
            res = self.node.is_face_touched(
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

            # Perform assertion check
            self.assertTrue(
                res,
                self.error_msg(
                    left_elbow,
                    right_elbow,
                    left_wrist,
                    right_wrist,
                    nose,
                    left_eye,
                    right_eye,
                    left_ear,
                    right_ear,
                    res,
                    True,
                    f'Touching Face Positive Some Defined (Case {case + 1})'
                )
            )

    def test_is_face_touched_all_defined(self) -> None:
        """Checks that poses where the face is touched is considered as touching face"""

        for i, coordinates in enumerate(IS_TOUCHING_FACE_POSITIVE_CASES_ALL_DEFINED):

            # Initialise coordinates
            left_elbow, \
                right_elbow, \
                left_wrist, \
                right_wrist, \
                nose, \
                left_eye, \
                right_eye, \
                left_ear, \
                right_ear, = coordinates

            # Obtain the result
            res = self.node.is_face_touched(
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

            # Perform assertion check
            self.assertTrue(
                res,
                self.error_msg(
                    left_elbow,
                    right_elbow,
                    left_wrist,
                    right_wrist,
                    nose,
                    left_eye,
                    right_eye,
                    left_ear,
                    right_ear,
                    res,
                    True,
                    f'Touching Face All Defined (Case {i + 1})'
                )
            )


if __name__ == '__main__':
    unittest.main()
