"""Docstring for the test_utils.py module

This module implements unit tests for functions in `utils.py`.

Usage
-----
From the terminal, navigate to the `analysis_video_pipeline/src` directory and run the following:

```
python -m unittest [-v] custom_nodes/dabble/tests/test_utils.py
```

_Note that -v is an optional flag to increase the verbosity of the unit test outputs._

For more information, refer to the
[official unit testing documentation](https://docs.python.org/3/library/unittest.html#test-discovery).
"""

# pylint: disable=invalid-name

from math import pi
import unittest

from custom_nodes.dabble.utils import (
    Coord,
    ERROR_OUTPUT,
    angle_between_vectors_in_rad
)


class TestNode(unittest.TestCase):
    """Python unit testing class to test functions in `utils.py`

    Attributes
    ----------
    _DECIMAL_PRECISION
        Decimal point precision for approximations

    Methods
    -------
    error_msg : str
        Template for unit test error messages
    test_angle_between_vectors_in_rad_zero : None
        Checks if passing a zero vector into the function results in an error
    test_angle_between_vectors_in_rad_associativity : None
        Checks that the function is associative
    test_angle_between_vectors_in_rad_orthogonal : None
        Checks the resultant angle between two orthogonal non-zero vectors
    test_angle_between_vectors_in_rad_identical : None
        Checks the resultant angle between two identical non-zero vectors
    test_angle_between_vectors_in_rad_opposite : None
        Checks the resultant angle between two non-zero vectors of equal magnitude but opposite direction
    test_angle_between_vectors_in_rad_acute : None
        Checks the resultant angle between vectors \\( (1, 0) \\) and \\( (1, 1) \\)
    test_angle_between_vectors_in_rad_obtuse : None
        Checks the resultant angle between vectors \\( (1, 0) \\) and \\( (-1, 1) \\)
    """

    """Define decimal point precision for approximations"""  # pylint: disable=pointless-string-statement
    _DECIMAL_PRECISION = 6

    def error_msg(
            self,
            v1: Coord,
            v2: Coord,
            output: float,
            expected: float,
            test: str,
            *,
            precision: int = _DECIMAL_PRECISION
    ) -> str:
        """Template for unit test error messages

        Parameters
        ----------
        v1 : `Coord`
            The first vector \\( (x, y) \\) parameter passed into the function
        v2 : `Coord`
            The second vector \\( (x, y) \\) parameter passed into the function
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
                 Test Type: {test}\n\
                Parameters: v1 = {v1}, v2 = {v2}\n\
                 Precision: {precision}\n\
                    Output: {output}\n\
                  Expected: {expected}\n\
                ============================================='

    def test_angle_between_vectors_in_rad_zero(self) -> None:
        """Checks if passing a zero vector into the function results in an error

        The function is expected to output ``ERROR_OUTPUT``.
        """

        # Initialise zero vector zero and non-zero vector vec
        zero = (0, 0)
        vec = (0, 1)

        # Obtain results
        r1 = angle_between_vectors_in_rad(*zero, *vec)
        r2 = angle_between_vectors_in_rad(*vec, *zero)
        r3 = angle_between_vectors_in_rad(*zero, *zero)

        # Perform assertion checks
        self.assertEqual(
            r1,
            ERROR_OUTPUT,
            msg=self.error_msg(
                zero,
                vec,
                r1,
                ERROR_OUTPUT,
                'First Parameter is Zero Vector',
                precision=0
            )
        )
        self.assertEqual(
            r2,
            ERROR_OUTPUT,
            msg=self.error_msg(
                vec,
                zero,
                r2,
                ERROR_OUTPUT,
                'Second Parameter is Zero Vector',
                precision=0
            )
        )
        self.assertEqual(
            r3,
            ERROR_OUTPUT,
            msg=self.error_msg(
                zero,
                zero,
                r3,
                ERROR_OUTPUT,
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
        r1 = angle_between_vectors_in_rad(*v1, *v2)
        r2 = angle_between_vectors_in_rad(*v2, *v1)

        # Perform assertion checks
        self.assertAlmostEqual(
            r1,
            r2,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg(
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
        \\( \\frac {\\pi} {2} \\) radians between them.
        """

        # Initialise vectors v1 and v2 that are orthogonal to each other
        v1 = (0, 1)
        v2 = (1, 0)

        # Obtain results
        res = angle_between_vectors_in_rad(*v1, *v2)

        # Perform assertion checks
        self.assertAlmostEqual(
            res,
            pi / 2,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg(
                v1,
                v2,
                res,
                pi / 2,
                'Orthogonal Non-Zero Vectors'
            )
        )

    def test_angle_between_vectors_in_rad_identical(self) -> None:
        """Checks the resultant angle between two identical non-zero vectors

        Two identical non-zero vectors should make an angle of \\( 0 \\) radians between them.
        """

        # Initialise vector vec
        vec = (0, 1)

        # Obtain results
        res = angle_between_vectors_in_rad(*vec, *vec)

        # Perform assertion checks
        self.assertAlmostEqual(
            res,
            0,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg(
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
        should make an angle of \\( \\pi \\) radians between them.
        """

        # Initialise opposite vectors v1 and v2
        v1 = (0, 1)
        v2 = (0, -1)

        # Obtain results
        res = angle_between_vectors_in_rad(*v1, *v2)

        # Perform assertion checks
        self.assertAlmostEqual(
            res,
            pi,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg(
                v1,
                v2,
                res,
                pi,
                'Opposite Non-Zero Vectors'
            )
        )

    def test_angle_between_vectors_in_rad_acute(self) -> None:
        """Checks the resultant angle between vectors \\( (1, 0) \\) and \\( (1, 1) \\)

        Notes
        -----
        ```
          ^
        2 |
          |    (1, 1)
        1 |   x
          |
        0 ----x--------->
               (1, 0)
          0   1   2   3
        ```

        The angle between
            \\( \\overrightarrow{ V_{1} } = \\begin{pmatrix} 1 \\\\ 0 \\end{pmatrix} \\) and
            \\( \\overrightarrow{ V_{2} } = \\begin{pmatrix} 1 \\\\ 1 \\end{pmatrix} \\)
        should be \\( \\frac {\\pi} {4} \\) radians.
        """

        # Initialise vectors v1 and v2
        v1 = (1, 0)
        v2 = (1, 1)

        # Obtain results
        res = angle_between_vectors_in_rad(*v1, *v2)

        # Perform assertion checks
        self.assertAlmostEqual(
            res,
            pi / 4,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg(
                v1,
                v2,
                res,
                pi / 4,
                'Acute Angle'
            )
        )

    def test_angle_between_vectors_in_rad_obtuse(self) -> None:
        """Checks the resultant angle between vectors \\( (1, 0) \\) and \\( (-1, 1) \\)

        Notes
        -----
        ```
                   ^
                   | 2
        (-1, 1)    |
               x   | 1
                   |
           ------------x--------->
                        (1, 0)
           -2  -1  0   1   2   3
        ```

        The angle between
            \\( \\overrightarrow{ V_{1} } = \\begin{pmatrix} 1 \\\\ 0 \\end{pmatrix} \\) and
            \\( \\overrightarrow{ V_{2} } = \\begin{pmatrix} -1 \\\\ 1 \\end{pmatrix} \\)
        should be \\( \\frac {3 \\pi} {4} \\) radians.
        """

        # Initialise vectors v1 and v2
        v1 = (1, 0)
        v2 = (-1, 1)

        # Obtain results
        res = angle_between_vectors_in_rad(*v1, *v2)

        # Perform assertion checks
        self.assertAlmostEqual(
            res,
            3 * pi / 4,
            places=self._DECIMAL_PRECISION,
            msg=self.error_msg(
                v1,
                v2,
                res,
                3 * pi / 4,
                'Obtuse Angle'
            )
        )


if __name__ == '__main__':
    unittest.main()
