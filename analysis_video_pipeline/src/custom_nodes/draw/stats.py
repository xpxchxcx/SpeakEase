"""Docstring for the stats.py module

This module implements a custom draw Node class for outputting the coordinates of PoseNet keypoints.

Usage
-----
This module should be part of a package that follows the file structure as specified by the
[PeekingDuck documentation](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html).

Navigate to the root directory of the package and run the following line on the terminal:

```
peekingduck run
```
"""

# pylint: disable=invalid-name, logging-format-interpolation

from collections import defaultdict
from typing import Any, Mapping, Optional

from peekingduck.pipeline.nodes.abstract_node import AbstractNode

from custom_nodes.draw.utils import (
    BLUE,
    WHITE,
    display_bbox_info,
    display_text,
    obtain_keypoint
)


class Node(AbstractNode):
    """Custom Node class to output the coordinates of PoseNet keypoints.

    Attributes
    ----------
    _frame_tracker : dict
        Tracks the number of processed frames by ID

    Methods
    -------
    run : dict
        Outputs the coordinates of PoseNet keypoints
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

        # Initialise counters
        self._total_frames = defaultdict(int)
        self._arms_folded_frames = defaultdict(int)
        self._leaning_frames = defaultdict(int)
        self._touching_face_frames = defaultdict(int)

    def run(
            self,
            inputs: Mapping[str, Any]
    ) -> Mapping:
        """Outputs the coordinates of PoseNet keypoints

        Parameters
        ----------
        inputs : dict
            Dictionary with the following keys:

            - 'img' - given image to be displayed
            - 'keypoints' - keypoints from PoseNet model

        Returns
        -------
        dict
            An empty dictionary
        """

        # Initialise error message
        error_msg = 'The input dictionary does not contain the {} key.'

        # Check if required inputs are in pipeline
        if 'img' not in inputs:
            # There must be an image to display
            self.logger.error(error_msg.format("'img'"))
            return {}
        for key in ('bboxes', 'bbox_scores', 'keypoints', 'obj_attrs'):
            if key not in inputs:
                # One or more metadata inputs are missing
                self.logger.warning(error_msg.format(f"'{key}'"))

        # Get required inputs from pipeline
        img = inputs['img']
        height, width, *_ = img.shape
        bboxes = inputs.get('bboxes', [])
        bbox_scores = inputs.get('bbox_scores', [])
        all_ids = inputs.get('obj_attrs', {}).get('ids', [])
        all_keypoints = inputs.get('keypoints', [])
        all_arms_folded = inputs.get('arms_folded', {})
        all_leaning = inputs.get('is_leaning', {})
        all_touching_face = inputs.get('is_touching_face', {})

        # Handle the detection of each person
        for curr_id, bbox, bbox_score, keypoints in \
                zip(all_ids, bboxes, bbox_scores, all_keypoints):

            # Output the coordinates
            for keypoint in keypoints:
                x, y = obtain_keypoint(*keypoint.tolist(), img_width=width, img_height=height)
                display_text(
                    img,
                    x,
                    y,
                    text=f'({x}, {y})',
                    font_colour=WHITE,
                    font_scale=0.5,
                    font_thickness=1
                )

            # Calculate the statistics
            self._total_frames[curr_id] += 1
            self._arms_folded_frames[curr_id] += all_arms_folded.get(curr_id, 0)
            self._leaning_frames[curr_id] += all_leaning.get(curr_id, 0)
            self._touching_face_frames[curr_id] += all_touching_face.get(curr_id, 0)

            # Output the bounding box
            display_bbox_info(
                img,
                bbox,
                bbox_score,
                arms_folded=all_arms_folded.get(curr_id, False),
                is_leaning=all_leaning.get(curr_id, False),
                face_touched=all_touching_face.get(curr_id, False)
            )

        # Initialise statistics display
        line_height = 30
        display_text(
            img,
            50,
            50,
            ' ID ArmFold Leaning FaceTouch  Total',
            WHITE
        )

        # Output the statistics for each detected person
        for i, curr_id in enumerate(sorted(self._total_frames)):
            arm_fold_stats = f'{self._arms_folded_frames[curr_id] / self._total_frames[curr_id] * 100:.2f}%'
            lean_stats = f'{self._leaning_frames[curr_id] / self._total_frames[curr_id] * 100:.2f}%'
            touching_face_stats = f'{self._touching_face_frames[curr_id] / self._total_frames[curr_id] * 100:.2f}%'
            display_text(
                img,
                50,
                round(50 + line_height * (i + 1)),
                ' '.join((
                    str(curr_id).center(3),
                    arm_fold_stats.center(7),
                    lean_stats.center(7),
                    touching_face_stats.center(9),
                    str(self._total_frames[curr_id]).center(5)
                )),
                BLUE
            )

        return {}


if __name__ == '__main__':
    pass
