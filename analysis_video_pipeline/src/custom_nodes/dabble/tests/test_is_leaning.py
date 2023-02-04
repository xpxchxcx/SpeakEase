"""Docstring for the test_is_leaning.py module

This module implements unit tests for the custom Node class in `is_leaning.py`.

Usage
-----
From the terminal, navigate to the `analysis_video_pipeline/src` directory and run the following:

```
python -m unittest [-v] custom_nodes/dabble/tests/test_is_leaning.py
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

from custom_nodes.dabble.is_leaning import Node
from custom_nodes.dabble.utils import (
    Coord,
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER
)
from custom_nodes.visualise import (
    IS_LEANING_NEGATIVE_CASES,
    IS_LEANING_POSITIVE_CASES
)


class TestNode(unittest.TestCase):
    """Python unit testing class to test the custom Node class in `is_leaning.py`

    Methods
    -------
    error_msg : str
        Template for unit test error messages
    test_is_leaning_folded_missing : None
        hecks that the function returns ``False`` if at least one input is missing
    test_is_leaning_negative : None
        Checks that non-leaning poses are considered not leaning
    test_is_leaning_positive : None
        Checks that true leaning poses are considered leaning
    """

    def setUp(self) -> None:
        """Initialises the Node object used for unit testing"""

        # Initialise YML filepath
        filepath = 'custom_nodes/configs/dabble/is_leaning.yml'

        # Obtain Node instance
        with open(filepath, 'r', encoding='utf-8') as config_file:
            config = safe_load(config_file)
            self.node = Node(config=config)

    def error_msg(
            self,
            left_shoulder: Optional[Coord],
            right_shoulder: Optional[Coord],
            left_hip: Optional[Coord],
            right_hip: Optional[Coord],
            output: bool,
            expected: bool,
            test: str
    ) -> str:
        """Template for unit test error messages

        Parameters
        ----------
        left_shoulder : `Coord`, optional
            The left shoulder coordinate \\( (x, y) \\) passed into the function
        right_shoulder : `Coord`, optional
            The right shoulder coordinate \\( (x, y) \\) passed into the function
        left_hip : `Coord`, optional
            The left hip coordinate \\( (x, y) \\) passed into the function
        right_hip : `Coord`, optional
            The right hip coordinate \\( (x, y) \\) passed into the function
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
                            right_shoulder = {right_shoulder}\n\
                            left_hip = {left_hip}\n\
                            right_hip = {right_hip}\n\
                    Output: {output}\n\
                  Expected: {expected}\n\
                ============================================='

    def test_is_leaning_folded_missing(self) -> None:
        """Checks that the function returns ``False`` if at least one input is missing

        The function requires all coordinates to perform the check.
        """

        # Initialise coordinates
        pose = [(i, i) for i in range(17)]
        required_coordinates = {
            KP_LEFT_SHOULDER,
            KP_RIGHT_SHOULDER,
            KP_LEFT_HIP,
            KP_RIGHT_HIP
        }
        coordinates = {joint: coordinate for joint, coordinate in enumerate(pose) if joint in required_coordinates}

        # Set one or more coordinates to None
        for i, selected_coordinates in enumerate(chain(
                *(combinations(required_coordinates, num + 1) for num in range(len(required_coordinates)))
        )):
            left_shoulder = None if KP_LEFT_SHOULDER in selected_coordinates else \
                coordinates[KP_LEFT_SHOULDER]
            right_shoulder = None if KP_RIGHT_SHOULDER in selected_coordinates else \
                coordinates[KP_RIGHT_SHOULDER]
            left_hip = None if KP_LEFT_HIP in selected_coordinates else \
                coordinates[KP_LEFT_HIP]
            right_hip = None if KP_RIGHT_HIP in selected_coordinates else \
                coordinates[KP_RIGHT_HIP]

            # Obtain the result
            res = self.node.is_leaning(
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip
            )

            # Perform assertion check
            self.assertFalse(
                res,
                self.error_msg(
                    left_shoulder,
                    right_shoulder,
                    left_hip,
                    right_hip,
                    res,
                    False,
                    f'One or more Missing Inputs (Case {i + 1})'
                )
            )

    def test_is_leaning_negative(self) -> None:
        """Checks that non-leaning poses are considered not leaning"""

        for i, coordinates in enumerate(IS_LEANING_NEGATIVE_CASES):

            # Initialise coordinates
            left_shoulder, \
                right_shoulder, \
                left_hip, \
                right_hip = coordinates

            # Obtain the result
            res = self.node.is_leaning(
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip
            )

            # Perform assertion check
            self.assertFalse(
                res,
                self.error_msg(
                    left_shoulder,
                    right_shoulder,
                    left_hip,
                    right_hip,
                    res,
                    False,
                    f'Non-Leaning Pose (Case {i + 1})'
                )
            )

    def test_is_leaning_positive(self) -> None:
        """Checks that true leaning poses are considered leaning"""

        for i, coordinates in enumerate(IS_LEANING_POSITIVE_CASES):

            # Initialise coordinates
            left_shoulder, \
                right_shoulder, \
                left_hip, \
                right_hip = coordinates

            # Obtain the result
            res = self.node.is_leaning(
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip
            )

            # Perform assertion check
            self.assertTrue(
                res,
                self.error_msg(
                    left_shoulder,
                    right_shoulder,
                    left_hip,
                    right_hip,
                    res,
                    True,
                    f'Leaning Pose (Case {i + 1})'
                )
            )


if __name__ == '__main__':
    unittest.main()
