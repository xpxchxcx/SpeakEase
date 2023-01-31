"""Docstring for the test_node.py module

This module implements unit tests for the custom node class.

Usage
-----
    From the terminal, navigate to the directory where this file is located.

    Proceed to execute either of the following commands:
        ```
        python -m unittest [-v] test_node.py
        python -m unittest discover
        ```
    Note that -v is an optional flag to increase the verbosity of the unit test outputs.
    
    For more information, refer to the `official unit testing documentation <https://docs.python.org/3/library/unittest.html#test-discovery>`_.
"""

# pyright: reportInvalidStringEscapeSequence=false

# Main libraries for unit testing
from analysis_video_pipeline.src.custom_nodes.dabble.movement import Node
import unittest

# Supporting libraries for unit testing
from itertools import chain, combinations
from math import pi
from typing import Optional, Tuple
from yaml import safe_load

from analysis_video_pipeline.tests.visualise import \
    ARE_ARMS_FOLDED_NEGATIVE_CASES_OUTSTRETCHED_ARMS, \
    ARE_ARMS_FOLDED_NEGATIVE_CASES_TOUCHING_FACE, \
    ARE_ARMS_FOLDED_POSITIVE_CASES_HALF_CROSS, \
    ARE_ARMS_FOLDED_POSITIVE_CASES_FULL_CROSS


Coord = Tuple[int, int]  # Type-hinting alias for coordinates


class TestNode(unittest.TestCase):

    # Define decimal point precision
    _DECIMAL_PRECISION = 6

    def setUp(self):
        with open('analysis_video_pipeline/src/custom_nodes/configs/dabble/movement.yml', 'r') as config_file:
            config = safe_load(config_file)
            self.node = Node(config=config)
    
    """Test cases for _angle_between_vectors_in_rad()"""

    def error_msg_angle_between_vectors_in_rad(
            self,
            v1: Coord,
            v2: Coord,
            output: float,
            expected: float,
            test: str,
            *,
            precision: int = _DECIMAL_PRECISION
    ) -> str:
        """Template for unit test error messages for _angle_between_vectors_in_rad()

        Parameters
        ----------
            v1 : `Coord`
                The first vector (`x`, `y`) parameter passed into the function
            v2 : `Coord`
                The second vector (`x`, `y`) parameter passed into the function
            output : float
                The obtained output from the function
            expected : float
                The expected output of the function
            test : str
                The type of unit test performed
            precision : int, default=`_DECIMAL_PRECISION`
                The expected decimal precision of the outputs
        
        Returns
        -------
            str
                The formatted error message
        """

        return f'\n\
                =============================================\n\
                  Function: _angle_between_vectors_in_rad()\n\
                 Test Type: {test}\n\
                Parameters: v1 = {v1}, v2 = {v2}\n\
                 Precision: {precision}\n\
                    Output: {output}\n\
                  Expected: {expected}\n\
                ============================================='

    def test_angle_between_vectors_in_rad_zero(self) -> None:
        """Checks if passing a zero vector into the function results in an error
        
        The function is expected to output ``Node.ERROR_OUTPUT``.
        """

        # Initialise zero vector zero and non-zero vector vec
        zero = (0, 0)
        vec = (0, 1)

        # Obtain results
        r1 = self.node._angle_between_vectors_in_rad(*zero, *vec)
        r2 = self.node._angle_between_vectors_in_rad(*vec, *zero)
        r3 = self.node._angle_between_vectors_in_rad(*zero, *zero)

        # Perform assertion checks
        self.assertEqual(
            r1,
            self.node.ERROR_OUTPUT,
            msg=self.error_msg_angle_between_vectors_in_rad(
                zero,
                vec,
                r1,
                self.node.ERROR_OUTPUT,
                'First Parameter is Zero Vector',
                precision=0
            )
        )
        self.assertEqual(
            r2,
            self.node.ERROR_OUTPUT,
            msg=self.error_msg_angle_between_vectors_in_rad(
                vec,
                zero,
                r2,
                self.node.ERROR_OUTPUT,
                'Second Parameter is Zero Vector',
                precision=0
            )
        )
        self.assertEqual(
            r3,
            self.node.ERROR_OUTPUT,
            msg=self.error_msg_angle_between_vectors_in_rad(
                zero,
                zero,
                r3,
                self.node.ERROR_OUTPUT,
                'Both Parameters are Zero Vectors',
                precision=0
            )
        )
    
    def test_angle_between_vectors_in_rad_associativity(self) -> None:
        """Checks that the function is associative
        
        The order in which the vectors are passed into the function should not matter.
        """

        # Initialise vectors v1 and v2
        v1 = (1, 3)
        v2 = (-2, 5)

        # Obtain results
        r1 = self.node._angle_between_vectors_in_rad(*v1, *v2)
        r2 = self.node._angle_between_vectors_in_rad(*v2, *v1)

        # Perform assertion checks
        self.assertAlmostEqual(
            r1,
            r2,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg_angle_between_vectors_in_rad(
                v1,
                v2,
                r1,
                r2,
                'Function Associativity'
            )
        )

    def test_angle_between_vectors_in_rad_orthogonal(self) -> None:
        """Checks the resultant angle between two orthogonal non-zero vectors

        Two non-zero orthogonal vectors should make an angle of
        :math:`\frac {\pi} {2}` radians between them.
        """

        # Initialise vectors v1 and v2 that are orthogonal to each other
        v1 = (0, 1)
        v2 = (1, 0)

        # Obtain results
        res = self.node._angle_between_vectors_in_rad(*v1, *v2)

        # Perform assertion checks
        self.assertAlmostEqual(
            res,
            pi / 2,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg_angle_between_vectors_in_rad(
                v1,
                v2,
                res,
                pi / 2,
                'Orthogonal Non-Zero Vectors'
            )
        )
    
    def test_angle_between_vectors_in_rad_identical(self) -> None:
        """Checks the resultant angle between two identical non-zero vectors
        
        Two identical non-zero vectors should make an angle of
        :math:`0` radians between them.
        """

        # Initialise vector vec
        vec = (0, 1)

        # Obtain results
        res = self.node._angle_between_vectors_in_rad(*vec, *vec)

        # Perform assertion checks
        self.assertAlmostEqual(
            res,
            0,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg_angle_between_vectors_in_rad(
                vec,
                vec,
                res,
                0,
                'Identical Non-Zero Vectors'
            )
        )
    
    def test_angle_between_vectors_in_rad_opposite(self) -> None:
        """Checks the resultant angle between two non-zero vectors of equal magnitude but opposite direction
        
        Two non-zero vectors of equal magnitude but opposite direction
        should make an angle of :math:`\pi` radians between them.
        """

        # Initialise opposite vectors v1 and v2
        v1 = (0, 1)
        v2 = (0, -1)

        # Obtain results
        res = self.node._angle_between_vectors_in_rad(*v1, *v2)

        # Perform assertion checks
        self.assertAlmostEqual(
            res,
            pi,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg_angle_between_vectors_in_rad(
                v1,
                v2,
                res,
                pi,
                'Opposite Non-Zero Vectors'
            )
        )

    def test_angle_between_vectors_in_rad_acute(self) -> None:
        """Checks the resultant angle between vectors (1, 0) and (1, 1)

        Notes
        -----
              ^
            2 |
              |    (1, 1)
            1 |   x
              |
            0 ----x--------->
                   (1, 0)
              0   1   2   3
            
            The angle between
                :math:`\overrightarrow{ V_{1} } = \begin{pmatrix} 1 \\ 0 \end{pmatrix}` and
                :math:`\overrightarrow{ V_{2} } = \begin{pmatrix} 1 \\ 1 \end{pmatrix}`
            should be :math:`\frac {\pi} {4}` radians.
        """

        # Initialise vectors v1 and v2
        v1 = (1, 0)
        v2 = (1, 1)

        # Obtain results
        res = self.node._angle_between_vectors_in_rad(*v1, *v2)

        # Perform assertion checks
        self.assertAlmostEqual(
            res,
            pi / 4,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg_angle_between_vectors_in_rad(
                v1,
                v2,
                res,
                pi / 4,
                'Acute Angle'
            )
        )
    
    def test_angle_between_vectors_in_rad_obtuse(self) -> None:
        """Checks the resultant angle between vectors (1, 0) and (-1, 1)

        Notes
        -----
                   ^
                   | 2
        (-1, 1)    |
               x   | 1
                   |
           ------------x--------->
                        (1, 0)
           -2  -1  0   1   2   3
            
            The angle between
                :math:`\overrightarrow{ V_{1} } = \begin{pmatrix} 1 \\ 0 \end{pmatrix}` and
                :math:`\overrightarrow{ V_{2} } = \begin{pmatrix} -1 \\ 1 \end{pmatrix}`
            should be :math:`\frac {3 \pi} {4}` radians.
        """

        # Initialise vectors v1 and v2
        v1 = (1, 0)
        v2 = (-1, 1)

        # Obtain results
        res = self.node._angle_between_vectors_in_rad(*v1, *v2)

        # Perform assertion checks
        self.assertAlmostEqual(
            res,
            3 * pi / 4,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg_angle_between_vectors_in_rad(
                v1,
                v2,
                res,
                3 * pi / 4,
                'Obtuse Angle'
            )
        )
    
    """Test cases for are_arms_folded()"""

    def error_msg_are_arms_folded(
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
        """Template for unit test error messages for are_arms_folded()

        Parameters
        ----------
            left_shoulder : `Coord`, optional
                The left shoulder coordinate (`x`, `y`) passed into the function
            left_elbow : `Coord`, optional
                The left elbow coordinate (`x`, `y`) passed into the function
            left_wrist : `Coord`, optional
                The left wrist coordinate (`x`, `y`) passed into the function
            right_shoulder : `Coord`, optional
                The right shoulder coordinate (`x`, `y`) passed into the function
            right_elbow : `Coord`, optional
                The right elbow coordinate (`x`, `y`) passed into the function
            right_wrist : `Coord`, optional
                The right wrist coordinate (`x`, `y`) passed into the function
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
                  Function: are_arms_folded()\n\
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
        """Checks that the function returns False if at least one input is missing

        The function requires all coordinates to perform the check.
        """
        
        # Initialise coordinates
        pose = [(i, i) for i in range(17)]
        required_coordinates = {
            self.node._KP_LEFT_SHOULDER,
            self.node._KP_LEFT_ELBOW,
            self.node._KP_LEFT_WRIST,
            self.node._KP_RIGHT_SHOULDER,
            self.node._KP_RIGHT_ELBOW,
            self.node._KP_RIGHT_WRIST
        }
        coordinates = {joint: coordinate for joint, coordinate in enumerate(pose) if joint in required_coordinates}

        # Set one or more coordinates to None
        for i, selected_coordinates in enumerate(chain(
                *(combinations(required_coordinates, num + 1) for num in range(len(required_coordinates)))
        )):
            left_shoulder = None if self.node._KP_LEFT_SHOULDER in selected_coordinates else \
                coordinates[self.node._KP_LEFT_SHOULDER]
            left_elbow = None if self.node._KP_LEFT_ELBOW in selected_coordinates else \
                coordinates[self.node._KP_LEFT_ELBOW]
            left_wrist = None if self.node._KP_LEFT_WRIST in selected_coordinates else \
                coordinates[self.node._KP_LEFT_WRIST]
            right_shoulder = None if self.node._KP_RIGHT_SHOULDER in selected_coordinates else \
                coordinates[self.node._KP_RIGHT_SHOULDER]
            right_elbow = None if self.node._KP_RIGHT_ELBOW in selected_coordinates else \
                coordinates[self.node._KP_RIGHT_ELBOW]
            right_wrist = None if self.node._KP_RIGHT_WRIST in selected_coordinates else \
                coordinates[self.node._KP_RIGHT_WRIST]

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
                self.error_msg_are_arms_folded(
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
                self.error_msg_are_arms_folded(
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
                self.error_msg_are_arms_folded(
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
                self.error_msg_are_arms_folded(
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
                self.error_msg_are_arms_folded(
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

    """Test cases for is_face_touched()"""

    def test_is_face_touched(self):
        pass  # TODO implement this unit test

    """Test cases for is_leaning()"""

    def test_is_leaning(self):
        pass  # TODO implement this unit test


if __name__ == '__main__':
    unittest.main()
