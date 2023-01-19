import cv2
from math import acos, pi, sqrt
from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from typing import Any, Mapping, Optional, Tuple


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
    _FONT_SCALE = 0.6
    _FONT_THICKNESS = 2

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

    def __init__(self, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        """Initialises the custom node class

        Parameters
            config : Optional[Mapping[str, Any]]
                Node custom configuration
            kwargs : Any
                Keyword arguments for instantiating the AbstractNode parent class
        """

        super().__init__(config, node_path=__name__, **kwargs)
        self.img = None

    @property
    def height(self) -> int:
        # Height of the displayed image, specified in pixels
        return 0 if self.img is None else self.img.shape[0]

    @property
    def width(self) -> int:
        # Width of the displayed image, specified in pixels
        return 0 if self.img is None else self.img.shape[1]

    def _map_coord_onto_img(self,
                            x: float,
                            y: float) -> Tuple[int, int]:
        """Maps relative coordinates onto the displayed image

        Parameters
            x : float
                Relative horizontal position of the coordinate; 0 <= x <= 1
            y : float
                Relative vertical position of the coordinate; 0 <= y <= 1

        Returns
            Tuple[int, int]
                Absolute coordinate (x1, y1) on the image; 0 <= x1 <= width and 0 <= y1 <= height
       """

        return round(x * self.width), round(y * self.height)

    def _display_text(self,
                      x: int,
                      y: int,
                      text: str,
                      font_colour: Tuple[int, int, int],
                      *,
                      font_face: Optional[int] = _FONT_FACE,
                      font_scale: Optional[float] = _FONT_SCALE,
                      font_thickness: Optional[int] = _FONT_THICKNESS) -> None:
        """Displays text at a specified coordinate on top of the displayed image

        Parameters
            x : int
                x-coordinate to display the text at
            y : int
                y-coordinate to display the text at
            text : str
                Text to display
            font_colour : Tuple[int, int, int]
                Colour of the text to display
            font_face : Optional[int]
                Font type of the text to display
                Limited to a subset of Hershey Fonts as supported by OpenCV
                https://stackoverflow.com/questions/371910008/load-truetype-font-to-opencv
            font_scale : Optional[float]
                Relative size of the text to display
            font_thickness : Optional[int]
                Relative thickness of the text to display
        """

        cv2.putText(img=self.img,
                    text=text,
                    org=(x, y),
                    fontFace=font_face,
                    fontScale=font_scale,
                    color=font_colour,
                    thickness=font_thickness)

    def _display_bbox_info(self,
                           bbox: Tuple[int, int, int, int],
                           score: float,
                           *,
                           arms_folded: Optional[bool] = False) -> None:
        """Displays the information associated with the given bounding box

        All information will be displayed in the bottom-left corner (x1, y2) of the bounding box
        This information includes the following:
            - Confidence score of the bounding box
            - Whether the arms are folded
            - TODO include more information to display

        Parameters
            bbox : Tuple[int, int, int, int]
                Bounding box, represented by its top-left (x1, y1) and bottom-right (x2, y2) coordinates
                The parameter is specified in the format (x1, y1, x2, y2);
                0 <= x1 <= x2 <= self.width and 0 <= y1 <= y2 <= self.height
            score : float
                Confidence score of the bounding box
            arms_folded : Optional[bool]
                True if the person is detected to have folded their arms, False otherwise
        """

        # Obtain bottom-left coordinate of bounding box
        x1, y1, x2, y2 = bbox
        x, _ = self._map_coord_onto_img(x1, y1)
        _, y = self._map_coord_onto_img(x2, y2)

        # Display information
        self._display_text(x, y - 3 * round(30 * self._FONT_SCALE), f'BBox {score:0.2f}', self._WHITE)
        if arms_folded:
            self._display_text(x, y - 2 * round(30 * self._FONT_SCALE), 'Arms Folded', self._BLUE)

    def _obtain_keypoint(self,
                         keypoint: Tuple[int, int],
                         score: float) -> Optional[Tuple[int, int]]:
        """Obtains a detected PoseNet keypoint if its confidence score meets or exceeds the threshold

        Parameters
            keypoint : Tuple[int, int]
                Relative coordinate of the detected PoseNet keypoint
            score : float
                Confidence score of the detected PoseNet keypoint

        Returns
            Optional[Tuple[int, int]]
                Absolute coordinates of the detected PoseNet keypoint
                if its confidence score meets or exceeds the threshold confidence
        """

        return None if score < self._THRESHOLD else self._map_coord_onto_img(*keypoint)

    @staticmethod
    def _angle_between_vectors_in_rad(x1: int,
                                      y1: int,
                                      x2: int,
                                      y2: int) -> float:
        """Obtains the angle between two vectors (in radians) via the cosine rule

        Let v1 = (x1, y1), v2 = (x2, y2), and 0 <= angle <= 180 is the angle between v1 and v2
        The cosine of the angle is the dot product of v1 and v2 divided by the product of their magnitudes;
            cos(angle) = (x1 * x2 + y1 * y2) / [sqrt(x1^2 + y1^2) * sqrt(x2^2 + y2^2)]

        Parameters
            x1 : int
                The magnitude of vector v1 in the x-axis
            y1 : int
                The magnitude of vector v1 in the y-axis
            x2 : int
                The magnitude of vector v2 in the x-axis
            y2 : int
                The magnitude of vector v2 in the y-axis

        Returns
            float
                The angle between vectors v1 and v2
        """

        dot_prod = x1 * x2 + y1 * y2
        v1_mag = sqrt(x1 * x1 + y1 * y1)
        v2_mag = sqrt(x2 * x2 + y2 * y2)
        return acos(dot_prod / (v1_mag * v2_mag))

    def are_arms_folded(self,
                        left_shoulder: Optional[Tuple[int, int]],
                        left_elbow: Optional[Tuple[int, int]],
                        left_wrist: Optional[Tuple[int, int]],
                        right_shoulder: Optional[Tuple[int, int]],
                        right_elbow: Optional[Tuple[int, int]],
                        right_wrist: Optional[Tuple[int, int]]) -> bool:
        """Determines if the arms of the given pose are folded

        The line from the shoulder to the elbow intersects the line from the wrist to the elbow at the elbow
        An arm is considered folded if:
            - The angle that the two lines make with each other is less than 120 degrees
            - The x-coordinate of the wrist lies in between the x-coordinates of the shoulders
            - The distance between the wrist and the elbow is at least half that between the shoulders

        Parameters
            left_shoulder : Optional[Tuple[int, int]]
                (x, y) coordinate of the left shoulder
            left_elbow : Optional[Tuple[int, int]]
                (x, y) coordinate of the left elbow
            left_wrist : Optional[Tuple[int, int]]
                (x, y) coordinate of the left wrist
            right_shoulder : Optional[Tuple[int, int]]
                (x, y) coordinate of the right shoulder
            right_elbow : Optional[Tuple[int, int]]
                (x, y) coordinate of the right elbow
            right_wrist : Optional[Tuple[int, int]]
                (x, y) coordinate of the right wrist

        Returns
            bool
                True if both arms are folded, False otherwise
        """

        # Check if keypoints are defined
        if left_shoulder is None or left_elbow is None or left_wrist is None or \
                right_shoulder is None or right_elbow is None or right_wrist is None:
            return False

        # Obtain coordinates
        x1, y1 = left_shoulder
        x2, y2 = right_shoulder
        x3, y3 = left_elbow
        x4, y4 = right_elbow
        x5, y5 = left_wrist
        x6, y6 = right_wrist

        # Calculate angles
        left_angle = self._angle_between_vectors_in_rad(x1 - x3, y1 - y3, x5 - x3, y5 - y3)
        right_angle = self._angle_between_vectors_in_rad(x2 - x4, y2 - y4, x6 - x4, y6 - y4)
        print(f'left angle: {left_angle}, right angle: {right_angle}')  # TODO remove

        # Calculate distances
        left_dist = sqrt((x5 - x3) * (x5 - x3) + (y5 - y3) * (y5 - y3))
        right_dist = sqrt((x6 - x4) * (x6 - x4) + (y6 - y4) * (y6 - y4))
        shoulder_dist = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        print(f'left distance: {left_dist}, right distance: {right_dist}, shoulder_dist: {shoulder_dist}')  # TODO remove

        # Check if either arm is folded
        threshold = 2 * pi / 3
        left_folded = left_angle < threshold and x2 <= x5 <= x1 and left_dist * 2 >= shoulder_dist
        right_folded = right_angle < threshold and x2 <= x6 <= x1 and right_dist * 2 >= shoulder_dist
        return left_folded and right_folded

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Displays calculated PoseNet keypoints and TODO write documentation for custom node class

        Parameters
            inputs : Mapping[str, Any]
                Input dictionary with keys "img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores"

        Returns
            Mapping[str, Any]
                Empty dictionary
        """

        # Get required inputs from pipeline
        self.img = inputs['img']  # assert that there is an image to display
        bboxes = inputs.get('bboxes', [])
        bbox_scores = inputs.get('bbox_scores', [])
        all_keypoints = inputs.get('keypoints', [])
        all_keypoint_scores = inputs.get('keypoint_scores', [])

        # Handle the detection of each person
        for bbox, bbox_score, keypoints, keypoint_scores in \
                zip(bboxes, bbox_scores, all_keypoints, all_keypoint_scores):

            # Store and display PoseNet keypoints
            keypoint_list = []
            for i, (keypoint, keypoint_score) in enumerate(zip(keypoints, keypoint_scores)):
                result = self._obtain_keypoint(keypoint.tolist(), keypoint_score)
                keypoint_list.append(result)
                if result is not None:
                    x, y = result
                    self._display_text(
                        x, y, f'({x}, {y})',
                        self._YELLOW if self._KP_LEFT_SHOULDER <= i <= self._KP_RIGHT_WRIST else self._WHITE)

            # Determine if the pose violates any bad presentation poses
            arms_folded = self.are_arms_folded(keypoint_list[self._KP_LEFT_SHOULDER],
                                               keypoint_list[self._KP_LEFT_ELBOW],
                                               keypoint_list[self._KP_LEFT_WRIST],
                                               keypoint_list[self._KP_RIGHT_SHOULDER],
                                               keypoint_list[self._KP_RIGHT_ELBOW],
                                               keypoint_list[self._KP_RIGHT_WRIST])

            # Display the results on the image
            self._display_bbox_info(bbox, bbox_score, arms_folded=arms_folded)

        return {}
