"""Docstring for the movement.py module

This module implements a custom Node class extended from PeekingDuck's AbstractNode.

Usage
-----
    This module should be part of a package that follows the file structure
    as specified by the `PeekingDuck documentation <https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html>`_.
"""

# pyright: reportInvalidStringEscapeSequence=false

import logging
from math import acos, pi, sqrt
from typing import Any, Mapping, Optional, Tuple

import cv2
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """Custom node to display PoseNet's skeletal key points

    TODO write documentation for custom node class
    """

    # Define colours for display purposes
    # Note: OpenCV loads file in BGR format
    _WHITE = (255, 255, 255)
    _YELLOW = (0, 255, 255)
    _BLUE = (255, 0, 0)

    # Define font properties for display purposes
    _FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 1.5
    _FONT_THICKNESS = 3

    # Define constants for PoseNet keypoint detection
    _THRESHOLD = 0.6
    # PoseNet's skeletal key points
    # https://discuss.tensorflow.org/uploads/default/original/2X/9/951fd6aaf5fec83500fe2e9891348416e13b66dd.png
    (
        _KP_NOSE,
        _KP_LEFT_EYE,
        _KP_RIGHT_EYE,
        _KP_LEFT_EAR,
        _KP_RIGHT_EAR,
        _KP_LEFT_SHOULDER,
        _KP_RIGHT_SHOULDER,
        _KP_LEFT_ELBOW,
        _KP_RIGHT_ELBOW,
        _KP_LEFT_WRIST,
        _KP_RIGHT_WRIST,
        _KP_LEFT_HIP,
        _KP_RIGHT_HIP,
        _KP_LEFT_KNEE,
        _KP_RIGHT_KNEE,
        _KP_LEFT_FOOT,
        _KP_RIGHT_FOOT
    ) = range(17)

    def __init__(
            self,
            config: Optional[Mapping[str, Any]] = None,
            **kwargs
    ) -> None:
        """Initialises the custom node class

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
        self._logger = logging.getLogger(__name__)

        # Define variables used for tracking
        self._img = None
        self._arm_fold_count = self._leaning_count = self._face_touch_count = self._total_frame_count = 0

    @property
    def height(self) -> int:
        # Height of the displayed image, specified in pixels
        return self.ERROR_OUTPUT if self._img is None else self._img.shape[0]

    @property
    def width(self) -> int:
        # Width of the displayed image, specified in pixels
        return self.ERROR_OUTPUT if self._img is None else self._img.shape[1]
    
    @property
    def ERROR_OUTPUT(self) -> int:
        # Define error return value as -1
        # Computational functions will return non-negative input
        return -1

    def _map_coord_onto_img(
            self,
            x: float,
            y: float
    ) -> Tuple[int, int]:
        """Maps relative coordinates onto the displayed image

        Parameters
        ----------
            x : float
                Relative horizontal position of the coordinate; 0 <= `x` <= 1
            y : float
                Relative vertical position of the coordinate; 0 <= `y` <= 1

        Returns
        -------
            tuple of int
                Absolute coordinate (`x1`, `y1`) on the image; 0 <= `x1` <= `width` and 0 <= `y1` <= `height`
       """

        return round(x * self.width), round(y * self.height)
    
    def _obtain_keypoint(
            self,
            keypoint: Tuple[float, float],
            score: float
    ) -> Optional[Tuple[int, int]]:
        """Obtains a detected PoseNet keypoint if its confidence score meets or exceeds the threshold

        Parameters
        ----------
            keypoint : tuple of floats
                Relative coordinate (`x`, `y`) of the detected PoseNet keypoint;
                0 <= `x` <= 1 and 0 <= `y` <= 1
            score : float
                Confidence score of the detected PoseNet keypoint

        Returns
        -------
            tuple of ints, optional
                Absolute coordinates of the detected PoseNet keypoint
                if its confidence score meets or exceeds the threshold confidence
        """

        return None if score < self._THRESHOLD else self._map_coord_onto_img(*keypoint)

    def _display_text(
            self,
            x: int,
            y: int,
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
            x : int
                x-coordinate to display the text at
            y : int
                y-coordinate to display the text at
            text : str
                Text to display
            font_colour : tuple of ints
                Colour of the text to display, specified in BGR format
        
        Other Parameters
        ----------------
            font_face : int, default=`_FONT_FACE`
                Font type of the text to display
                Limited to a subset of Hershey Fonts as supported by OpenCV
                https://stackoverflow.com/questions/371910008/load-truetype-font-to-opencv
            font_scale : float
                Relative size of the text to display
            font_thickness : int
                Relative thickness of the text to display
        """

        cv2.putText(
            img=self._img,  # type: ignore
            text=text,
            org=(x, y),
            fontFace=font_face,
            fontScale=font_scale,
            color=font_colour,
            thickness=font_thickness
        )

    def _display_bbox_info(
            self,
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
            bbox : tuple of ints
                Bounding box, represented by its top-left (`x1`, `y1`) and bottom-right (`x2`, `y2`) coordinates
                The parameter `bbox` is specified in the format (`x1`, `y1`, `x2`, `y2`);
                0 <= `x1` <= `x2` <= `width` and 0 <= `y1` <= `y2` <= `height`
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
        All information will be displayed in the bottom-left corner (`x1`, `y2`) of the bounding box.
        This information includes the following:
            - Confidence score of the bounding box
            - Whether the arms are folded
            - Whether the person is leaning too much to one side
            - Whether the person is touching his/her face
            - TODO include more information to display
        """

        # Initialise constants
        num_lines = 3  # default number of 'lines' above the bottom-left coordinate
        line_height = round(30 * self._FONT_SCALE)  # height of each 'line' in pixels; 30 is arbitrary

        # Obtain bottom-left coordinate of bounding box
        x1, y1, x2, y2 = bbox
        x, _ = self._map_coord_onto_img(x1, y1)
        _, y = self._map_coord_onto_img(x2, y2)

        # Display information
        self._display_text(x, y - num_lines * line_height, f'BBox {score:0.2f}', self._WHITE)
        if arms_folded:
            num_lines += 1
            self._display_text(x, y - num_lines * line_height, 'Arms Folded', self._BLUE)
        if is_leaning:
            num_lines += 1
            self._display_text(x, y - num_lines * line_height, 'Leaning', self._BLUE)
        if face_touched:
            num_lines += 1
            self._display_text(x, y - num_lines * line_height, 'Touching Face', self._BLUE)

    def _angle_between_vectors_in_rad(
            self,
            x1: int,
            y1: int,
            x2: int,
            y2: int
    ) -> float:
        """Obtains the (smaller) angle between two non-zero vectors in radians

        The angle is computed via the cosine rule (see the Notes section).

        Parameters
        ----------
            x1 : int
                The magnitude of vector :math:`\overrightarrow{ V_{1} }` in the x-axis
            y1 : int
                The magnitude of vector :math:`\overrightarrow{ V_{1} }` in the y-axis
            x2 : int
                The magnitude of vector :math:`\overrightarrow{ V_{2} }` in the x-axis
            y2 : int
                The magnitude of vector :math:`\overrightarrow{ V_{2} }` in the y-axis

        Returns
        -------
            float
                The (smaller) angle :math:`\Theta` between
                    :math:`\overrightarrow{ V_{1} }` and
                    :math:`\overrightarrow{ V_{2} }`.
                If either vectors are zero vectors, returns ``ERROR_OUTPUT``.
        
        Notes
        -----
        The cosine of the (smaller) angle between two vectors
            :math:`\overrightarrow{ V_{1} } = \begin{pmatrix} x_{1} \\ y_{1} \end{pmatrix}` and
            :math:`\overrightarrow{ V_{2} } = \begin{pmatrix} x_{2} \\ y_{2} \end{pmatrix}`
        is calculated by:

        .. math::
            \cos\Theta
                = \frac { \overrightarrow{ V_{1} } \cdot \overrightarrow{ V_{2} } } 
                        {\left\| \overrightarrow{V_{1}} \right\| \left\| \overrightarrow{V_{2}} \right\|}
                = \frac {x_{1}x_{2} + y_{1}y_{2}}
                        {\sqrt {{x_{1}}^2 + {y_{1}}^2} \sqrt {{x_{2}}^2 + {y_{2}}^2}}
                ,\ 0 \le \Theta \le \pi
        
        Hence, the angle :math:`\Theta` between
            :math:`\overrightarrow{ V_{1} }` and
            :math:`\overrightarrow{ V_{2} }`
        can be calculated by taking the inverse cosine of the result.

        Examples
        --------
        `_angle_between_vectors_in_rad()` is a private function and should only be called internally.
        Note that these examples serve the purpose of documentation and/or unit testing only.

        >>> node = Node()

        **The angle between two orthogonal vectors is :math:`\frac {\pi} {2}`.**

        >>> v1 = (0, 1)
        >>> v2 = (1, 0)
        >>> node._angle_between_vectors_in_rad(*v1, *v2)
        **1.5707963267948966**
        >>> node._angle_between_vectors_in_rad(*v2, *v1)
        **1.5707963267948966**

        **The angle between two parallel vectors is :math:`0`.**

        >>> node._angle_between_vectors_in_rad(*v1, *v1)
        **0.0**
        """

        # Check for zero vectors
        if x1 == y1 == 0 or x2 == y2 == 0:
            return self.ERROR_OUTPUT

        # Compute the inverse cosine value
        dot_prod = x1 * x2 + y1 * y2
        v1_magnitude = sqrt(x1 * x1 + y1 * y1)
        v2_magnitude = sqrt(x2 * x2 + y2 * y2)
        return acos(dot_prod / (v1_magnitude * v2_magnitude))

    def are_arms_folded(
            self,
            left_shoulder: Optional[Tuple[int, int]],
            left_elbow: Optional[Tuple[int, int]],
            left_wrist: Optional[Tuple[int, int]],
            right_shoulder: Optional[Tuple[int, int]],
            right_elbow: Optional[Tuple[int, int]],
            right_wrist: Optional[Tuple[int, int]]
    ) -> bool:
        """Determines if the arms of the given pose are folded

        Parameters
        ----------
            left_shoulder : tuple of ints, optional
                (`x`, `y`) coordinate of the left shoulder
            left_elbow : tuple of ints, optional
                (`x`, `y`) coordinate of the left elbow
            left_wrist : tuple of ints, optional
                (`x`, `y`) coordinate of the left wrist
            right_shoulder : tuple of ints, optional
                (`x`, `y`) coordinate of the right shoulder
            right_elbow : tuple of ints, optional
                (`x`, `y`) coordinate of the right elbow
            right_wrist : tuple of ints, optional
                (`x`, `y`) coordinate of the right wrist

        Returns
        -------
            bool
                ``True`` if both arms are folded, ``False`` otherwise
        
        Notes
        -----
        The line from the shoulder to the elbow intersects the line from the wrist to the elbow at the elbow
        An arm is considered folded if:
            - The angle that the two lines make with each other is less than 120 degrees
            - The x-coordinate of the wrist lies in between the x-coordinates of the shoulders
            - The distance between the wrist and the elbow is at least half that between the shoulders
        """

        # Check if keypoints are defined
        if left_shoulder is None or \
                left_elbow is None or \
                left_wrist is None or \
                right_shoulder is None or \
                right_elbow is None or \
                right_wrist is None:
            return False

        # Obtain coordinates
        left_shoulder_x, left_shoulder_y = left_shoulder
        right_shoulder_x, right_shoulder_y = right_shoulder
        left_elbow_x, left_elbow_y = left_elbow
        right_elbow_x, right_elbow_y = right_elbow
        left_wrist_x, left_wrist_y = left_wrist
        right_wrist_x, right_wrist_y = right_wrist

        # Calculate angle made between the left shoulder, left elbow, and left wrist
        left_angle = self._angle_between_vectors_in_rad(
            left_shoulder_x - left_elbow_x,
            left_shoulder_y - left_elbow_y,
            left_wrist_x - left_elbow_x,
            left_wrist_y - left_elbow_y
        )
        # Calculate angle made between the right shoulder, right elbow, and right wrist
        right_angle = self._angle_between_vectors_in_rad(
            right_shoulder_x - right_elbow_x,
            right_shoulder_y - right_elbow_y,
            right_wrist_x - right_elbow_x,
            right_wrist_y - right_elbow_y
        )

        # Calculate distance from the left elbow to the left wrist
        left_dist = sqrt(
            (left_wrist_x - left_elbow_x) * (left_wrist_x - left_elbow_x) + \
            (left_wrist_y - left_elbow_y) * (left_wrist_y - left_elbow_y)
        )
        # Calculate distance from the right elbow to the right wrist
        right_dist = sqrt(
            (right_wrist_x - right_elbow_x) * (right_wrist_x - right_elbow_x) + \
            (right_wrist_y - right_elbow_y) * (right_wrist_y - right_elbow_y)
        )
        # Calculate distance between the two shoulders
        shoulder_dist = sqrt(
            (right_shoulder_x - left_shoulder_x) * (right_shoulder_x - left_shoulder_x) + \
            (right_shoulder_y - left_shoulder_y) * (right_shoulder_y - left_shoulder_y)
        )

        threshold = 2 * pi / 3  # 2/3pi rad or 120 deg
        # Check if left arm is folded
        left_folded = left_angle < threshold and \
            right_shoulder_x <= left_wrist_x <= left_shoulder_x and \
            left_dist * 2 >= shoulder_dist
        # Check if right arm is folded
        right_folded = right_angle < threshold and \
            right_shoulder_x <= right_wrist_x <= left_shoulder_x and \
            right_dist * 2 >= shoulder_dist
        # Check if both arms are folded
        return left_folded and right_folded

    def is_face_touched(
            self,
            left_elbow: Optional[Tuple[int, int]],
            right_elbow: Optional[Tuple[int, int]],
            left_wrist: Optional[Tuple[int, int]],
            right_wrist: Optional[Tuple[int, int]],
            nose: Optional[Tuple[int, int]]
    ) -> bool:
        """Determines if the hands of the given pose is touching the face

        Parameters
        ----------
            left_elbow : tuple of ints, optional
                (`x`, `y`) coordinate of the left elbow
            right_elbow : tuple of ints, optional
                (`x`, `y`) coordinate of the right elbow
            left_wrist : tuple of ints, optional
                (`x`, `y`) coordinate of the left wrist
            right_wrist : tuple of ints, optional
                (`x`, `y`) coordinate of the right wrist
            nose : tuple of ints, optional
                (`x`, `y`) coordinate of the nose

        Returns
        -------
            bool
                ``True`` if either hand is touching the face, ``False`` otherwise
        
        Notes
        -----
        TODO define the conditions for a hand to touch the face
        """
        
        # Check if keypoints are defined
        if left_elbow is None or \
                right_elbow is None or \
                left_wrist is None or \
                right_wrist is None or \
                nose is None:
            return False

        # Obtain coordinates
        left_elbow_x, left_elbow_y = left_elbow
        right_elbow_x, right_elbow_y = right_elbow
        left_wrist_x, left_wrist_y = left_wrist
        right_wrist_x, right_wrist_y = right_wrist
        nose_x, nose_y = nose

        # Calculate distances from the respective joints to the nose
        left_elbow_dist = sqrt(
            (left_elbow_x - nose_x) * (left_elbow_x - nose_x) + \
            (left_elbow_y - nose_y) * (left_elbow_y - nose_y)
        )
        right_elbow_dist = sqrt(
            (right_elbow_x - nose_x) * (right_elbow_x - nose_x) + \
            (right_elbow_y - nose_y) * (right_elbow_y - nose_y)
        )
        left_wrist_dist = sqrt(
            (left_wrist_x - nose_x) * (left_wrist_x - nose_x) + \
            (left_wrist_y - nose_y) * (left_wrist_y - nose_y)
        )
        right_wrist_dist = sqrt(
            (right_wrist_x - nose_x) * (right_wrist_x - nose_x) + \
            (right_wrist_y - nose_y) * (right_wrist_y - nose_y)
        )

        # Check if either hand is touching the face
        threshold = 150
        return left_elbow_dist < threshold or \
            right_elbow_dist < threshold or \
            left_wrist_dist < threshold or \
            right_wrist_dist < threshold

    def is_leaning(
            self,
            left_shoulder: Optional[Tuple[int, int]],
            right_shoulder: Optional[Tuple[int, int]],
            left_hip: Optional[Tuple[int, int]],
            right_hip: Optional[Tuple[int, int]],
            left_knee: Optional[Tuple[int, int]],
            right_knee: Optional[Tuple[int, int]]
    ) -> bool:
        """Determines if the given pose is leaning towards one side

        Parameters
        ----------
            left_shoulder : tuple of ints, optional
                Tuple containing the `x` and `y` coordinates of the left shoulder
            right_shoulder : tuple of ints, optional
                Tuple containing the `x` and `y` coordinates of the right shoulder
            left_hip : tuple of ints, optional
                Tuple containing the `x` and `y` coordinates of the left hip
            right_hip : tuple of ints, optional
                Tuple containing the `x` and `y` coordinates of the right hip
            left_knee : tuple of ints, optional
                Tuple containing the `x` and `y` coordinates of the left knee
            right_knee : tuple of ints, optional
                Tuple containing the `x` and `y` coordinates of the right knee
        
        Returns
        -------
            bool
                ``True`` if the pose is leaning towards one side, ``False`` otherwise
        
        Notes
        -----
        TODO define the conditions for leaning
        """

        # Check if keypoints are defined
        if left_shoulder is None or \
                left_hip is None or \
                left_knee is None or \
                right_shoulder is None or \
                right_hip is None or \
                right_knee is None:
            return False

        # Initialize buffer
        # Get left and right shoulder keypoints
        left_shoulder_x, left_shoulder_y = left_shoulder
        right_shoulder_x, right_shoulder_y = right_shoulder
        # Get left and right hip keypoints
        left_hip_x, left_hip_y = left_hip
        right_hip_x, right_hip_y = right_hip
        # Get left and right knee keypoints
        left_knee_x, left_knee_y = left_knee
        right_knee_x, right_knee_y = right_knee

        # Calculate the angle between left shoulder to left hip to left knee
        left_angle = self._angle_between_vectors_in_rad(
            left_shoulder_x - left_hip_x,
            left_shoulder_y - left_hip_y,
            left_knee_x - left_hip_x,
            left_knee_y - left_hip_y
        )
        # Calculate the angle between right shoulder to right hip to right knee
        right_angle = self._angle_between_vectors_in_rad(
            right_shoulder_x - right_hip_x,
            right_shoulder_y - right_hip_y,
            right_knee_x - right_hip_x,
            right_knee_y - right_hip_y
        )

        # Check if either side has crossed the threshold for leaning
        sway_threshold = 20 * pi / 180  # 20 deg
        return left_angle < pi - sway_threshold or \
            right_angle < pi - sway_threshold or \
            left_angle > pi + sway_threshold or \
            right_angle > pi + sway_threshold

    def run(
            self,
            inputs: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Displays calculated PoseNet keypoints and TODO write documentation for custom node class

        Parameters
        ----------
            inputs : dict
                Input dictionary with keys "img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores"

        Returns
        -------
            dict
                Empty dictionary
        """

        # Initialise constants
        error_msg = 'The input dictionary does not contain the {} key.'
        frames_before_summary = 200

        # Check if required inputs are in pipeline
        if 'img' not in inputs:
            # There must be an image to display
            # Otherwise, `self._display_text()` will raise an error
            self._logger.error(error_msg.format("'img'"))
            return {}
        for key in ('bboxes', 'bbox_scores', 'keypoints', 'keypoint_scores'):
            if key not in inputs:
                # One or more metadata inputs are missing
                self._logger.warning(error_msg.format(f"'{key}'"))
        
        # Get required inputs from pipeline
        self._img = inputs['img']
        bboxes = inputs.get('bboxes', [])
        bbox_scores = inputs.get('bbox_scores', [])
        all_keypoints = inputs.get('keypoints', [])
        all_keypoint_scores = inputs.get('keypoint_scores', [])

        self._total_frame_count += 1

        # Handle the detection of each person
        for bbox, bbox_score, keypoints, keypoint_scores in \
                zip(bboxes, bbox_scores, all_keypoints, all_keypoint_scores):

            # Store and display PoseNet keypoints
            keypoint_list = []
            for keypoint, keypoint_score in zip(keypoints, keypoint_scores):
                result = self._obtain_keypoint(keypoint.tolist(), keypoint_score)
                keypoint_list.append(result)
                if result is None:
                    continue
                x, y = result
                self._display_text(
                    x,
                    y,
                    f'({x}, {y})',
                    self._WHITE,
                    font_scale=0.5,
                    font_thickness=1
                )

            # Determine if the pose violates any bad presentation poses
            arms_folded = self.are_arms_folded(
                keypoint_list[self._KP_LEFT_SHOULDER],
                keypoint_list[self._KP_LEFT_ELBOW],
                keypoint_list[self._KP_LEFT_WRIST],
                keypoint_list[self._KP_RIGHT_SHOULDER],
                keypoint_list[self._KP_RIGHT_ELBOW],
                keypoint_list[self._KP_RIGHT_WRIST]
            )
            face_touched = self.is_face_touched(
                keypoint_list[self._KP_LEFT_ELBOW],
                keypoint_list[self._KP_RIGHT_ELBOW],
                keypoint_list[self._KP_LEFT_WRIST],
                keypoint_list[self._KP_RIGHT_WRIST],
                keypoint_list[self._KP_NOSE]
            )
            is_leaning = self.is_leaning(
                keypoint_list[self._KP_LEFT_SHOULDER],
                keypoint_list[self._KP_RIGHT_SHOULDER],
                keypoint_list[self._KP_LEFT_HIP],
                keypoint_list[self._KP_RIGHT_HIP],
                keypoint_list[self._KP_LEFT_KNEE],
                keypoint_list[self._KP_RIGHT_KNEE]
            )

            # Increment relevant counters
            self._arm_fold_count += arms_folded
            self._face_touch_count += face_touched
            self._leaning_count += is_leaning

            # Display the results on the image
            self._display_bbox_info(
                bbox,
                bbox_score,
                arms_folded=arms_folded,
                is_leaning=is_leaning,
                face_touched=face_touched
            )
            if self._total_frame_count > 0 and self._total_frame_count % frames_before_summary == 0:
                self._logger.info(f'\n \
                    ------------------------\n \
                       Current Statistics\n \
                    ------------------------\n \
                      Arm Folding - {self._arm_fold_count / self._total_frame_count * 100:0.3f}%\n \
                          Leaning - {self._leaning_count / self._total_frame_count * 100:0.3f}%\n \
                    Touching Face - {self._face_touch_count / self._total_frame_count * 100:0.3f}%\n \
                    [ Frame Count : {self._total_frame_count} ]'
                )
        
        return {}


if __name__ == '__main__':
    pass
