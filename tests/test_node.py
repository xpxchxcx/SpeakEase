# Main libraries for unit testing
from src.custom_nodes.dabble.movement import Node
import unittest

# Supporting libraries for unit testing
from math import pi
from yaml import safe_load

class TestNode(unittest.TestCase):

    def setUp(self):
        with open('src/custom_nodes/configs/dabble/movement.yml', 'r') as config_file:
            config = safe_load(config_file)
            self.node = Node(config=config)

    def test_angle_between_vectors_in_rad(self):
        # Test Case 1
        # The angle between two orthogonal vectors is pi/2 rad
        v1 = (0, 1)
        v2 = (1, 0)
        self.assertEqual(self.node._angle_between_vectors_in_rad(*v1, *v2), pi / 2)
        self.assertEqual(self.node._angle_between_vectors_in_rad(*v2, *v1), pi / 2)

        # Test Case 2
        # The angle between two parallel vectors is 0 rad
        self.assertEqual(self.node._angle_between_vectors_in_rad(*v1, *v1), 0)
    
    def test_are_arms_folded(self):
        pass  # TODO implement this unit test

    def test_is_face_touched(self):
        pass  # TODO implement this unit test

    def test_is_leaning(self):
        pass  # TODO implement this unit test


if __name__ == '__main__':
    unittest.main()
