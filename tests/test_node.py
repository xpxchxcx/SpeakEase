"""Docstring for the test_node.py module

This module implements unit tests for the custom node class.

Usage
-----
    From the terminal, navigate to the directory where this file is located.

    Proceed to execute either of the following commands:
        python -m unittest [-v] test_node.py
        python -m unittest [-v] discover
    Note that -v is an optional flag to increase the verbosity of the unit test outputs.
    
    For more information, refer to the `official unit testing documentation <https://docs.python.org/3/library/unittest.html#test-discovery>`_.
"""

# pyright: reportInvalidStringEscapeSequence=false

# Main libraries for unit testing
from src.custom_nodes.dabble.movement import Node
import unittest

# Supporting libraries for unit testing
from math import pi
from typing import Optional, Tuple
from yaml import safe_load

class TestNode(unittest.TestCase):

    # Define decimal point precision
    _DECIMAL_PRECISION = 6

    def setUp(self):
        with open('src/custom_nodes/configs/dabble/movement.yml', 'r') as config_file:
            config = safe_load(config_file)
            self.node = Node(config=config)
    
    """Test cases for _angle_between_vectors_in_rad()"""

    def error_msg_angle_between_vectors_in_rad(
            self,
            v1: Tuple[int, int],
            v2: Tuple[int, int],
            output: float,
            *,
            expected: Optional[float] = None,
            precision: int = _DECIMAL_PRECISION,
            test: Optional[str] = 'No Description'
    ) -> str:
        """Template for unit test error messages for _angle_between_vectors_in_rad()

        Parameters
        ----------
            v1 : tuple of ints
                The first vector (`x`, `y`) parameter passed into the function
            v2 : tuple of ints
                The second vector (`x`, `y`) parameter passed into the function
            output : float
                The obtained output from the function
            expected : float, optional, default=None
                The expected output of the function
            precision : int, default=`_DECIMAL_PRECISION`
                The expected decimal precision of the outputs
            test : str, optional, default='No Description'
                The type of unit test performed
        
        Returns
        -------
            str
                The formatted error message
        """

        expected_str = 'Not Specified' if expected is None else str(expected)
        return f'\n\
                =============================================\n\
                  Function: _angle_between_vectors_in_rad()\n\
                 Test Type: {test}\n\
                Parameters: v1 = {v1}, v2 = {v2}\n\
                 Precision: {precision}\n\
                    Output: {output}\n\
                  Expected: {expected_str}\n\
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
                expected=self.node.ERROR_OUTPUT,
                precision=0,
                test='First Parameter is Zero Vector'
            )
        )
        self.assertEqual(
            r2,
            self.node.ERROR_OUTPUT,
            msg=self.error_msg_angle_between_vectors_in_rad(
                vec,
                zero,
                r2,
                expected=self.node.ERROR_OUTPUT,
                precision=0,
                test='Second Parameter is Zero Vector'
            )
        )
        self.assertEqual(
            r3,
            self.node.ERROR_OUTPUT,
            msg=self.error_msg_angle_between_vectors_in_rad(
                zero,
                zero,
                r3,
                expected=self.node.ERROR_OUTPUT,
                precision=0,
                test='Both Parameters are Zero Vectors'
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
                expected=r2,
                test='Function Associativity'
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
                expected=pi / 2,
                test='Orthogonal Non-Zero Vectors'
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
                expected=0,
                test='Identical Non-Zero Vectors'
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
                expected=pi,
                test='Opposite Non-Zero Vectors'
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
                expected=pi / 4,
                test='Acute Angle'
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
                expected=3 * pi / 4,
                test='Obtuse Angle'
            )
        )
    
    """Test cases for _are_arms_folded()"""
    
    def test_are_arms_folded(self):
        pass  # TODO implement this unit test

    """Test cases for _is_face_touched()"""

    def test_is_face_touched(self):
        pass  # TODO implement this unit test

    """Test cases for _is_leaning()"""

    def test_is_leaning(self):
        pass  # TODO implement this unit test


if __name__ == '__main__':
    unittest.main()
