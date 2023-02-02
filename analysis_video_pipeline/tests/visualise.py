"""Docstring for the visualise.py module

This module implements a visualisation method using matplotlib for the unit test cases.

Usage
-----
This script is to be used internally for testing and debugging purposes only.

Uncomment the relevant line(s) in the main function `main()` and run this script in the terminal via

```
python visualise.py
```

Dependencies
------------
This script requires matplotlib to run.

To install matplotlib, run the following line in the terminal:

```
pip install matplotlib
```
"""

from itertools import product
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


Coord = Tuple[int, int]  # Type-hinting alias for coordinates


"""Unit test cases for are_arms_folded()

Each test case returns 6 coordinates (`x`, `y`) in the following order:
    - Left shoulder
    - Left elbow
    - Left wrist
    - Right shoulder
    - Right elbow
    - Right wrist
"""

ARE_ARMS_FOLDED_POSITIVE_CASES_HALF_CROSS: \
        List[Tuple[Coord, Coord, Coord, Coord, Coord, Coord]] = [
    (
        (1153, 239),
        (1184, 437),
        (1067, 425),
        (919, 235),
        (864, 425),
        (1054, 425)
    ),
    (
        (1158, 239),
        (1186, 434),
        (1054, 431),
        (917, 236),
        (862, 424),
        (1055, 424)
    ),
    (
        (1158, 237),
        (1190, 432),
        (1051, 432),
        (916, 234),
        (861, 423),
        (1052, 422)
    ),
    (
        (1160, 238),
        (1192, 431),
        (1063, 433),
        (922, 233),
        (862, 420),
        (1012, 417)
    ),
    (
        (1154, 236),
        (1189, 433),
        (1064, 430),
        (919, 233),
        (860, 420),
        (1013, 416)
    )
]


ARE_ARMS_FOLDED_POSITIVE_CASES_FULL_CROSS: \
        List[Tuple[Coord, Coord, Coord, Coord, Coord, Coord]] = [
    (
        (1153, 239),
        (1184, 437),
        (950, 390),
        (919, 235),
        (864, 425),
        (1100, 390)
    ),
    (
        (1158, 239),
        (1186, 434),
        (950, 390),
        (917, 236),
        (862, 424),
        (1100, 390)
    )
]


ARE_ARMS_FOLDED_NEGATIVE_CASES_TOUCHING_FACE: \
        List[Tuple[Coord, Coord, Coord, Coord, Coord, Coord]] = [
    (
        (1140, 254),
        (1197, 359),
        (1105, 237),
        (934, 252),
        (897, 361),
        (946, 252)
    ),
    (
        (1140, 256),
        (1194, 356),
        (1104, 239),
        (930, 259),
        (889, 360),
        (944, 255)
    ),
    (
        (1137, 255),
        (1191, 359),
        (1108, 236),
        (925, 262),
        (885, 363),
        (944, 254)
    ),
    (
        (1139, 255),
        (1194, 358),
        (1108, 237),
        (925, 262),
        (890, 362),
        (945, 254)
    )
]


ARE_ARMS_FOLDED_NEGATIVE_CASES_OUTSTRETCHED_ARMS: \
        List[Tuple[Coord, Coord, Coord, Coord, Coord, Coord]] = [
    (
        (964, 573),
        (979, 715),
        (1010, 862),
        (783, 573),
        (786, 728),
        (742, 889)
    ),
    (
        (1076, 530),
        (1124, 704),
        (1110, 925),
        (872, 532),
        (807, 700),
        (822, 844)
    ),
    (
        (1069, 525),
        (1116, 687),
        (1110, 850),
        (846, 537),
        (822, 697),
        (820, 843)
    ),
    (
        (1054, 461),
        (1139, 686),
        (1193, 893),
        (804, 470),
        (790, 702),
        (685, 940)
    )
]


"""Unit test cases for is_leaning()

Each test case returns 4 coordinates (`x`, `y`) in the following order:
    - Left shoulder
    - Left elbow
    - Left hip
    - Right hip
"""

IS_LEANING_POSITIVE_CASES: \
        List[Tuple[Coord, Coord, Coord, Coord]] = [
    (
        (1005, 567),
        (825, 564),
        (967, 876),
        (881, 840)
    ),
    (
        (1041, 576),
        (824, 567),
        (1004, 875),
        (900, 853)
    ),
    (
        (968, 574),
        (795, 574),
        (977, 875),
        (848, 863)
    ),
    (
        (995, 572),
        (805, 565),
        (994, 886),
        (846, 867)
    ),
    (
        (1041, 575),
        (821, 567),
        (1010, 884),
        (886, 868)
    ),
    (
        (1051, 577),
        (844, 571),
        (1013, 885),
        (897, 859)
    )
]


IS_LEANING_NEGATIVE_CASES: \
        List[Tuple[Coord, Coord, Coord, Coord]] = [
    (
        (990, 573),
        (805, 567),
        (994, 860),
        (865, 854)
    ),
    (
        (979, 576),
        (806, 576),
        (986, 882),
        (845, 867)
    ),
    (
        (977, 578),
        (786, 574),
        (976, 865),
        (840, 859)
    ),
    (
        (964, 573),
        (783, 573),
        (975, 863),
        (837, 856)
    ),
    (
        (968, 574),
        (784, 573),
        (981, 861),
        (835, 854)
    )
]


"""Unit test cases for is_touching_face()

Each test case returns 9 coordinates (`x`, `y`) in the following order:
    - Left elbow
    - Right elbow
    - Left wrist
    - Right wrist
    - Nose
    - Left eye
    - Right eye
    - Left ear
    - Right ear
Each coordinate may or may not be defined (defaults to None).
"""

IS_TOUCHING_FACE_POSITIVE_CASES_ALL_DEFINED: \
        List[Tuple[Coord, Coord, Coord, Coord, Coord, Coord, Coord, Coord, Coord]] = [
    (
        (962, 290),
        (721, 310),
        (880, 145),
        (743, 198),
        (818, 57),
        (854, 43),
        (803, 40),
        (881, 55),
        (762, 57)
    ),
    (
        (947, 276),
        (732, 290),
        (892, 161),
        (772, 172),
        (815, 67),
        (847, 50),
        (794, 46),
        (891, 67),
        (752, 65)
    ),
    (
        (932, 272),
        (729, 281),
        (889, 140),
        (759, 163),
        (800, 51),
        (845, 46),
        (799, 45),
        (859, 51),
        (763, 68)
    ),
    (
        (934, 272),
        (743, 288),
        (890, 150),
        (780, 170),
        (799, 50),
        (836, 42),
        (792, 43),
        (882, 72),
        (755, 68)
    ),
    (
        (935, 268),
        (738, 288),
        (891, 144),
        (778, 170),
        (809, 53),
        (847, 45),
        (801, 44),
        (869, 49),
        (761, 64)
    )
]


"""Visualisation functions"""

def visualise(
        title: str,
        *coord_sets: Tuple[Optional[Coord], ...],
        invert_xaxis: bool = True,
        invert_yaxis: bool = True
) -> None:
    """Visualise sets of coordinates in matplotlib
    
    Parameters
    ----------
    title : str
        Title of the matplotlib graph
    *coord_sets : tuple of optional `Coord`s
        Sets of coordinates that need to be plotted.

        Refer to the Notes section for more information on how the coordinates are plotted.
    invert_xaxis : bool, default=True
        ``True`` if the x-axis of the graph should be inverted, ``False`` otherwise.

        The x-axis of the graph is inverted by default due to the OpenCV coordinate system.
    invert_yaxis : bool, default=True
        ``True`` if the y-axis of the graph should be inverted, ``False`` otherwise.

        The y-axis of the graph is inverted by default due to the OpenCV coordinate system.
    
    Notes
    -----
    Each set `((x1, y1), (x2, y2), ..., (xN, yN), ...)` of coordinates in `coord_sets` 
        will be plotted in the order that they are passed; i.e., 
        a segment line will be drawn first between `(x1, y1)` and `(x2, y2)`, 
        then another segment line will be drawn between `(x2, y2)` and `(x3, y3)`, 
        etc., until all the coordinates are plotted onto the graph.
    
    The sets of coordinates in `coord_sets` will also be plotted in the order that they are passed.
    """

    # Configure graph metadata
    plt.title(title)
    if invert_xaxis:
        plt.gca().invert_xaxis()
    if invert_yaxis:
        plt.gca().invert_yaxis()
    
    # Plot each set of coordinates
    for coord_set in coord_sets:
        if coord_set is None:
            continue
        plt.plot(*zip(*coord_set), marker='o')  # type: ignore
    
    plt.show()


def visualise_are_arms_folded(
        title: str,
        left_shoulder: Optional[Coord],
        left_elbow: Optional[Coord],
        left_wrist: Optional[Coord],
        right_shoulder: Optional[Coord],
        right_elbow: Optional[Coord],
        right_wrist: Optional[Coord]
) -> None:
    """Visualise unit test inputs for `are_arms_folded()` in matplotlib
    
    Parameters
    ----------
    title : str
        Title of the graph
    left_shoulder : `Coord`, optional
        (`x`, `y`) coordinate of the left shoulder
    left_elbow : `Coord`, optional
        (`x`, `y`) coordinate of the left elbow
    left_wrist : `Coord`, optional
        (`x`, `y`) coordinate of the left wrist
    right_shoulder : `Coord`, optional
        (`x`, `y`) coordinate of the right shoulder
    right_elbow : `Coord`, optional
        (`x`, `y`) coordinate of the right elbow
    right_wrist : `Coord`, optional
        (`x`, `y`) coordinate of the right wrist
    """

    visualise(
        title,
        (right_shoulder, left_shoulder, left_elbow, right_elbow, right_shoulder),
        (left_shoulder, left_elbow, left_wrist),
        (right_shoulder, right_elbow, right_wrist)
    )


def visualise_is_leaning(
        title: str,
        left_shoulder: Optional[Coord],
        right_shoulder: Optional[Coord],
        left_hip: Optional[Coord],
        right_hip: Optional[Coord]
) -> None:
    """Visualise unit test inputs for `is_leaning()` in matplotlib
    
    Parameters
    ----------
    title : str
        Title of the graph
    left_shoulder : `Coord`, optional
        (`x`, `y`) coordinate of the left shoulder
    right_shoulder : `Coord`, optional
        (`x`, `y`) coordinate of the right shoulder
    left_hip : `Coord`, optional
        (`x`, `y`) coordinate of the left hip
    right_hip : `Coord`, optional
        (`x`, `y`) coordinate of the right hip
    """

    visualise(
        title,
        (right_shoulder, left_shoulder, left_hip, right_hip, right_shoulder)
    )


def visualise_is_face_touched(
        title: str,
        left_elbow: Optional[Coord],
        right_elbow: Optional[Coord],
        left_wrist: Optional[Coord],
        right_wrist: Optional[Coord],
        nose: Optional[Coord],
        left_eye: Optional[Coord],
        right_eye: Optional[Coord],
        left_ear: Optional[Coord],
        right_ear: Optional[Coord]
) -> None:
    """Visualise unit test inputs for `is_face_touched()` in matplotlib
    
    Parameters
    ----------
    title : str
        Title of the graph
    left_elbow : `Coord`, optional
        (`x`, `y`) coordinate of the left elbow
    right_elbow : `Coord`, optional
        (`x`, `y`) coordinate of the right elbow
    left_wrist : `Coord`, optional
        (`x`, `y`) coordinate of the left wrist
    right_wrist : `Coord`, optional
        (`x`, `y`) coordinate of the right wrist
    nose : `Coord`, optional
        (`x`, `y`) coordinate of the nose
    left_eye : `Coord`, optional
        (`x`, `y`) coordinate of the left eye
    right_eye : `Coord`, optional
        (`x`, `y`) coordinate of the right eye
    left_ear : `Coord`, optional
        (`x`, `y`) coordinate of the left ear
    right_ear : `Coord`, optional
        (`x`, `y`) coordinate of the right ear
    """
    
    coord_sets = []

    # Plot elbow to wrist joints
    if left_elbow is not None and left_wrist is not None:
        coord_sets.append((left_elbow, left_wrist))
    if right_elbow is not None and right_wrist is not None:
        coord_sets.append((right_elbow, right_wrist))
    
    # Plot facial features
    if left_eye is not None:
        if nose is not None:
            coord_sets.append((nose, left_eye))
        if left_ear is not None:
            coord_sets.append((left_eye, left_ear))
    if right_eye is not None:
        if nose is not None:
            coord_sets.append((nose, right_eye))
        if right_ear is not None:
            coord_sets.append((right_eye, right_ear))
    
    # Plot limbs to facial features linkage(s)
    for c1, c2 in product(
            (left_elbow, left_wrist, right_elbow, right_wrist),
            (nose, left_eye, left_ear, right_eye, right_ear)
    ):
        if c1 is None or c2 is None:
            continue
        coord_sets.append((c1, c2))
    
    # Call the plotting function
    visualise(title, *coord_sets)


"""Main function"""

def main() -> None:
    """Uncomment the relevant lines to visualise the input(s)"""
    
    """
    for i, test_case in enumerate(ARE_ARMS_FOLDED_POSITIVE_CASES_HALF_CROSS):
        visualise_are_arms_folded(
            f'Half Fold Test Case {i + 1} for are_arms_folded()',
            *test_case
        )
    """

    """
    for i, test_case in enumerate(ARE_ARMS_FOLDED_NEGATIVE_CASES_TOUCHING_FACE):
        visualise_are_arms_folded(
            f'Touching Face Test Case {i + 1} for are_arms_folded()',
            *test_case
        )
    """
    
    """
    for i, test_case in enumerate(ARE_ARMS_FOLDED_NEGATIVE_CASES_OUTSTRETCHED_ARMS):
        visualise_are_arms_folded(
            f'Outstretched Arms Test Case {i + 1} for are_arms_folded()',
            *test_case
        )
    """

    """
    for i, test_case in enumerate(ARE_ARMS_FOLDED_POSITIVE_CASES_FULL_CROSS):
        visualise_are_arms_folded(
            f'Full Fold Test Case {i + 1} for are_arms_folded()',
            *test_case
        )
    """

    """
    for i, test_case in enumerate(IS_LEANING_POSITIVE_CASES):
        visualise_is_leaning(
            f'Positive Test Case {i + 1} for is_leaning()',
            *test_case
        )
    """

    """
    for i, test_case in enumerate(IS_LEANING_NEGATIVE_CASES):
        visualise_is_leaning(
            f'Negative Test Case {i + 1} for is_leaning()',
            *test_case
        )
    """

    """
    for i, test_case in enumerate(IS_TOUCHING_FACE_POSITIVE_CASES_ALL_DEFINED):
        visualise_is_face_touched(
            f'All Defined Positive Test Case {i + 1} for is_face_touched()',
            *test_case
        )
    """

    pass

if __name__ == '__main__':
    main()
