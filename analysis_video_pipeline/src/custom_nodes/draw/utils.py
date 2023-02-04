"""Docstring for the draw/utils.py script

This script contains constants and other miscellaneous functions
for the rest of the scripts in the draw module to use.

Usage
-----
This script is not meant to be used independently.
"""

# pylint: disable=invalid-name, logging-format-interpolation

from typing import Tuple

import cv2


"""Type-hinting alias for coordinates"""  # pylint: disable=pointless-string-statement
Coord = Tuple[int, int]


"""Define colours for display purposes

Note: OpenCV loads file in BGR format
"""  # pylint: disable=pointless-string-statement

"""Defines the colour white for display purposes"""  # pylint: disable=pointless-string-statement
WHITE = (255, 255, 255)

"""Defines the colour blue for display purposes"""  # pylint: disable=pointless-string-statement
BLUE = (255, 0, 0)


"""Define font properties for display purposes"""  # pylint: disable=pointless-string-statement

"""Defines the font family for display purposes"""  # pylint: disable=pointless-string-statement
_FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX  # pylint: disable=no-member

"""Defines the font scale for display purposes"""  # pylint: disable=pointless-string-statement
_FONT_SCALE = 1

"""Defines the font thickness for display purposes"""  # pylint: disable=pointless-string-statement
_FONT_THICKNESS = 2


def display_text(
        img,
        abs_x: int,
        abs_y: int,
        text: str,
        font_colour: Tuple[int, int, int],
        *,
        font_face: int = _FONT_FACE,
        font_scale: float = _FONT_SCALE,
        font_thickness: int = _FONT_THICKNESS
) -> None:
    """Displays text at a specified coordinate on top of the displayed image

    Parameters
    ----------
    img
        The image to display
    abs_x : int
        Absolute x-coordinate to display the text at
    abs_y : int
        Absolute y-coordinate to display the text at
    text : str
        Text to display
    font_colour : tuple of ints
        Colour of the text to display, specified in BGR format

    Other Parameters
    ----------------
    font_face : int, default=`_FONT_FACE`
        Font type of the text to display.

        Limited to a subset of Hershey Fonts as
        [supported by OpenCV](https://stackoverflow.com/questions/371910008/load-truetype-font-to-opencv).
    font_scale : float
        Relative size of the text to display
    font_thickness : int
        Relative thickness of the text to display
    """

    cv2.putText(  # pylint: disable=no-member
        img=img,
        text=text,
        org=(abs_x, abs_y),
        fontFace=font_face,
        fontScale=font_scale,
        color=font_colour,
        thickness=font_thickness
    )


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


def display_bbox_info(
        img,
        bbox: Tuple[int, int, int, int],
        score: float,
        *,
        arms_folded: bool = False,
        is_leaning: bool = False,
        face_touched: bool = False
) -> None:
    """Displays the information associated with the given bounding box

    Parameters
    ----------
    img
        The image to display; it has a \\( width \\) and a \\( height \\).
    bbox : tuple of ints
        Bounding box, represented by its top-left \\( (x_{1}, y_{1}) \\)
        and bottom-right \\( (x_{2}, y_{2}) \\) coordinates.

        The parameter `bbox` is specified in the format \\( (x_{1}, y_{1}, x_{2}, y_{2}) \\);
        \\( 0 \\le x_{1} \\le x_{2} \\le width \\) and
        \\( 0 \\le y_{1} \\le y_{2} \\le height \\).
    score : float
        Confidence score of the bounding box
    arms_folded : bool, default=False
        ``True`` if the person is detected to have folded their arms, ``False`` otherwise
    is_leaning : bool, default=False
        ``True``if the person is detected to be leaning to one side, ``False`` otherwise
    face_touched : bool, default=False
        ``True`` if the person is detected to be touching their face, ``False`` otherwise

    Notes
    -----
    All information will be displayed in the bottom-left corner
    \\( (x_{1}, y_{2}) \\) of the bounding box.

    This information includes the following:

    - Confidence score of the bounding box
    - Whether the arms are folded
    - Whether the person is leaning too much to one side
    - Whether the person is touching his/her face
    """

    # Initialise constants
    num_lines = 3  # default number of 'lines' above the bottom-left coordinate
    line_height = round(30 * _FONT_SCALE)  # height of each 'line' in pixels; 30 is arbitrary
    img_height, img_width, *_ = img.shape

    # Obtain bottom-left coordinate of bounding box
    x1, y1, x2, y2 = bbox
    x, _ = obtain_keypoint(x1, y1, img_width, img_height)
    _, y = obtain_keypoint(x2, y2, img_width, img_height)

    # Display information
    display_text(img, x, y - num_lines * line_height, f'BBox {score:0.2f}', WHITE)
    if arms_folded:
        num_lines += 1
        display_text(img, x, y - num_lines * line_height, 'Arms Folded', BLUE)
    if is_leaning:
        num_lines += 1
        display_text(img, x, y - num_lines * line_height, 'Leaning', BLUE)
    if face_touched:
        num_lines += 1
        display_text(img, x, y - num_lines * line_height, 'Touching Face', BLUE)


if __name__ == '__main__':
    pass
