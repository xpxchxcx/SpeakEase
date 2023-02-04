"""Docstring for the dabble/utils.py script

This script contains constants and other miscellaneous functions
for the rest of the scripts in the dabble module to use.

Usage
-----
This script is not meant to be used independently.
"""

# pylint: disable=invalid-name, logging-format-interpolation

from math import acos, sqrt
import logging
from typing import Tuple


"""Defines the logger for this script"""  # pylint: disable=pointless-string-statement
_logger = logging.getLogger(__name__)


"""Type-hinting alias for coordinates"""
Coord = Tuple[int, int]


"""Defines the default output for encountered errors as \\( -1 \\)

Computational functions will return non-negative input.
"""
ERROR_OUTPUT = -1


"""Defines the list of skeletal keypoints as returned by the PoseNet model

![PoseNet diagram](https://discuss.tensorflow.org/uploads/default/original/2X/9/951fd6aaf5fec83500fe2e9891348416e13b66dd.png)
"""  # pylint: disable=line-too-long
(
    KP_NOSE,
    KP_LEFT_EYE,
    KP_RIGHT_EYE,
    KP_LEFT_EAR,
    KP_RIGHT_EAR,
    KP_LEFT_SHOULDER,
    KP_RIGHT_SHOULDER,
    KP_LEFT_ELBOW,
    KP_RIGHT_ELBOW,
    KP_LEFT_WRIST,
    KP_RIGHT_WRIST,
    KP_LEFT_HIP,
    KP_RIGHT_HIP,
    KP_LEFT_KNEE,
    KP_RIGHT_KNEE,
    KP_LEFT_FOOT,
    KP_RIGHT_FOOT
) = range(17)


def obtain_keypoint(
        rel_x: float,
        rel_y: float,
        img_width: int,
        img_height: int
) -> Coord:
    """Obtains the coordinates of a detected PoseNet keypoint on a given image

    Parameters
    ----------
    rel_x : float
        Relative horizontal position of the coordinate; \\( 0 \\le x_{rel} \\le 1 \\)
    rel_y : float
        Relative vertical position of the coordinate; \\( 0 \\le y_{rel} \\le 1 \\)
    img_width : int
        Width of the image in pixels
    img_height : int
        Height of the image in pixels

    Returns
    -------
    `Coord`
        Absolute coordinate \\( (x_{abs}, y_{abs}) \\) on the image;
        \\( 0 \\le x_{abs} \\le width \\) and \\( 0 \\le y_{abs} \\le height \\).
    """

    return round(rel_x * img_width), round(rel_y * img_height)


def angle_between_vectors_in_rad(
        x1: int,
        y1: int,
        x2: int,
        y2: int
) -> float:
    """Obtains the (smaller) angle between two non-zero vectors in radians

    The angle \\( \\theta \\) is computed via the cosine rule (see the Notes section).

    Parameters
    ----------
    x1 : int
        The magnitude of vector \\( \\overrightarrow{ V_{1} } \\) in the x-axis
    y1 : int
        The magnitude of vector \\( \\overrightarrow{ V_{1} } \\) in the y-axis
    x2 : int
        The magnitude of vector \\( \\overrightarrow{ V_{2} } \\) in the x-axis
    y2 : int
        The magnitude of vector \\( \\overrightarrow{ V_{2} } \\) in the y-axis

    Returns
    -------
    float
        The (smaller) angle \\( \\theta \\) between
            \\( \\overrightarrow{ V_{1} } \\) and
            \\( \\overrightarrow{ V_{2} } \\).

        Returns ``ERROR_OUTPUT`` instead if:

        - Either one or both of
            \\( \\overrightarrow{ V_{1} } = \\overrightarrow{0} \\) and
            \\( \\overrightarrow{ V_{2} } = \\overrightarrow{0} \\)
        - \\( \\cos \\theta \\notin [-1, 1] \\)

    Notes
    -----
    The cosine of the (smaller) angle between two vectors
        \\( \\overrightarrow{ V_{1} } = \\begin{pmatrix} x_{1} \\\\ y_{1} \\end{pmatrix} \\) and
        \\( \\overrightarrow{ V_{2} } = \\begin{pmatrix} x_{2} \\\\ y_{2} \\end{pmatrix} \\)
    is calculated by:

    $$
    \\cos \\theta
        = \\frac { \\overrightarrow{ V_{1} } \\cdot \\overrightarrow{ V_{2} } }
                {\\left\\| \\overrightarrow{V_{1}} \\right\\| \\left\\| \\overrightarrow{V_{2}} \\right\\|}
        = \\frac {x_{1}x_{2} + y_{1}y_{2}}
                {\\sqrt {{x_{1}}^2 + {y_{1}}^2} \\sqrt {{x_{2}}^2 + {y_{2}}^2}}
        ,\\ 0 \\le \\theta \\le \\pi
    $$

    Hence, the angle \\( \\theta \\) between
        \\( \\overrightarrow{ V_{1} } \\) and
        \\( \\overrightarrow{ V_{2} } \\)
    can be calculated by taking the inverse cosine of the result.

    Examples
    --------
    The angle between two orthogonal vectors is \\( \\frac {\\pi} {2} \\).

    >>> v1 = (0, 1)
    >>> v2 = (1, 0)
    >>> angle_between_vectors_in_rad(*v1, *v2)
    1.5707963267948966
    >>> angle_between_vectors_in_rad(*v2, *v1)
    1.5707963267948966

    The angle between two parallel vectors is \\( 0 \\).

    >>> angle_between_vectors_in_rad(*v1, *v1)
    0.0
    """

    # Check for zero vectors
    if x1 == y1 == 0 or x2 == y2 == 0:
        _logger.error(
            f'One or more zero vectors v1 = ({x1}, {y1}) and ' +
            f'v2 = ({x2}, {y2}) were passed into angle_between_vectors_in_rad().'
        )
        return ERROR_OUTPUT

    # Compute the cosine value
    dot_prod = x1 * x2 + y1 * y2
    v1_magnitude = sqrt(x1 * x1 + y1 * y1)
    v2_magnitude = sqrt(x2 * x2 + y2 * y2)
    cos_value = dot_prod / (v1_magnitude * v2_magnitude)

    # Check if the cosine value is within acos domain of [-1, 1]
    if abs(cos_value) > 1:
        _logger.error(
            f'angle_between_vectors_in_rad() obtained cosine value {cos_value} ' +
            'that is not within acos domain [-1, 1]. ' +
            f'v1 · v2 = {dot_prod}, ||v1|| = {v1_magnitude}, ||v2|| = {v2_magnitude}'
        )
        return ERROR_OUTPUT

    return acos(cos_value)


if __name__ == '__main__':
    pass
