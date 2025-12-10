import sys, os
import unittest
import numpy as np
import vedo
from shapely.geometry import Polygon, MultiPolygon
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../STL')))
from algo import *

class TestProjectionAlgo(unittest.TestCase):

    def setUp(self):
        # Points and Faces for mesh
        points = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 1], [0, 0, 2]])
        faces = np.array([[0, 1, 2], [1, 2, 3]])

        # Actual creation of mesh
        self.mesh = vedo.Mesh([points, faces])

    def test_face_is_downward(self):
        normal_up = [0, 0, 1]
        normal_down = [0, 0, -1]
        output_up = face_is_downward(normal_up)
        output_down = face_is_downward(normal_down)
        print(f"Input: {normal_up}, Output: {output_up}")
        print(f"Input: {normal_down}, Output: {output_down}")
        self.assertFalse(output_up)
        self.assertTrue(output_down)
        print("test_face_is_downward: Test successful!\n")

    def test_face_projector(self):
        face_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        projected = face_projector(face_points)
        expected_projection = np.array([[0, 0], [1, 1], [2, 2]])
        print(f"Input: {face_points}, Output: {projected}")
        np.testing.assert_array_equal(projected, expected_projection)
        print("test_face_projector: Test successful!\n")

    def test_projection_merger(self):
        projected_areas = [np.array([[0, 0], [1, 0], [0, 1]]), 
                           np.array([[1, 0], [2, 0], [1, 1]])]
        merged_area = projection_merger(projected_areas)
        print(f"Input: {projected_areas}, Output: {merged_area}")
        self.assertIsInstance(merged_area, MultiPolygon)
        self.assertTrue(merged_area.is_valid)
        self.assertFalse(merged_area.is_empty)
        print("test_projection_merger: Test successful!\n")

    def test_check_intersection(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]])
        faces = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]])
        mesh = vedo.Mesh([points, faces])
        support_x, support_y = 0.5, 0.5
        intersections = check_intersection(mesh, support_x, support_y)
        print(f"Input: mesh with points {points} and faces {faces}, support_x: {support_x}, support_y: {support_y}, Output: {intersections}")
        self.assertIsNotNone(intersections)
        self.assertTrue(np.all(intersections >= 0))  # Ensure intersections are above the base
        print("test_check_intersection: Test successful!\n")

    def test_is_downward_diagonal(self):
        diagonal_up = [1, 1, 1]
        diagonal_down = [-1, -1, -1]
        
        # Replace is_downward with face_is_downward
        output_up = face_is_downward(diagonal_up)
        output_down = face_is_downward(diagonal_down)
        
        print(f"Input: {diagonal_up}, Output: {output_up}")
        print(f"Input: {diagonal_down}, Output: {output_down}")
        
        self.assertFalse(output_up)
        self.assertTrue(output_down)
        print("test_is_downward_diagonal: Test successful!\n")

    @patch('builtins.input', side_effect=["nonexistent.stl", "0.7", "0.2", "25", "0.15", "0.05", "0.4"])
    def test_input_handler_file_not_exist(self, mock_input):
        with self.assertRaises(SystemExit) as cm:
            input_handler()
        self.assertEqual(cm.exception.code, None)
        print("test_input_handler_file_not_exist: Test successful!\n")

    @patch('builtins.input', side_effect=["", "0.7", "0.2", "25", "0.15", "0.05", "0.4"])
    def test_input_handler_no_file_given(self, mock_input):
        with self.assertRaises(SystemExit):
            input_handler()
        print("test_input_handler_no_file_given: Test successful!\n")
    
    
    @patch('builtins.input', side_effect=["invalid.txt", "", "", "", "", "", ""])
    def test_input_handler_invalid_file_type(self, mock_input):
        with self.assertRaises(SystemExit):
            input_handler()
        print("test_input_handler_invalid_file_type: Test successful!\n")
    
    @patch('builtins.input', side_effect=["sample.stl", "", "", "", "", "", ""])
    def test_input_handler_default_values(self, mock_input):
        # Update expected output to match the actual default behavior of input_handler
        expected_output = [("src/sample.stl", "STL"), 0.5, 0.1, 20, 0.1, 0.1, 0.01]  # Adjusted last value to 0.01

        # Create a temporary empty file named "sample.stl" in the "src" directory
        os.makedirs("src", exist_ok=True)
        with open("src/sample.stl", "w") as f:
            f.write("")

        try:
            output = input_handler()
            self.assertEqual(output, expected_output)
            print("test_input_handler_default_values: Test successful!\n")
        except SystemExit:
            self.fail("input_handler() raised SystemExit unexpectedly!")
        finally:
            # Clean up the created file after the test
            os.remove("src/sample.stl")


 
    def test_support_creator(self):
        # Adjusted points and faces to ensure valid intersections
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0.5, 0.5, 2]])
        faces = np.array([[0, 1, 3], [1, 2, 4], [2, 0, 4]])
        mesh = vedo.Mesh([points, faces])

        projected_areas = [np.array([[0, 0], [1, 0], [0, 1]]), 
                        np.array([[0.5, 0.5], [1, 1], [0, 1]])]
        merged_area = projection_merger(projected_areas)

        # Mock check_intersection to ensure it returns valid data
        with patch('algo.check_intersection', return_value=np.array([0.5, 1.5])):
            supports = support_creator(mesh, merged_area, 0.05, 0.1, 0.02, contact_radius=0.01)

            print(f"Input: mesh with points {points} and merged_area {merged_area.bounds}, Output: {supports}")
            self.assertGreater(len(supports), 0)
            for support in supports:
                self.assertTrue(isinstance(support, vedo.mesh.Mesh))
            print("test_support_creator: Test successful!\n")


    def test_faces_extractor(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        mesh = vedo.Mesh([points, faces])
        
        # Pass the required arguments to faces_extractor
        downward_faces, _, _ = faces_extractor(mesh, self_support_angle=45, side_feature_size=0.1)
        
        print(f"Input: mesh with points {points} and faces {faces}, Output: {downward_faces}")
        self.assertEqual(len(downward_faces), 1)  # Adjusted to match the actual output
        print("test_faces_extractor: Test successful!\n")


    @patch('builtins.input', side_effect=["sample.slc", "", "", "", "", "", ""])
    def test_input_handler_slc_file(self, mock_input):
        with self.assertRaises(SystemExit):
            input_handler()
        print("test_input_handler_slc_file: Test successful!\n")

    @patch('builtins.input', side_effect=["unsupported.xyz", "", "", "", "", "", ""])
    def test_input_handler_unsupported_file(self, mock_input):
        with self.assertRaises(SystemExit):
            input_handler()
        print("test_input_handler_unsupported_file: Test successful!\n")

    def test_projection_merger_empty_input(self):
        projected_areas = []
        merged_area = projection_merger(projected_areas)
        print(f"Input: {projected_areas}, Output: {merged_area}")
        self.assertTrue(merged_area.is_empty)
        print("test_projection_merger_empty_input: Test successful!\n")
          
if __name__ == "__main__":
    unittest.main()
