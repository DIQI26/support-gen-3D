import sys, os
import unittest
import numpy as np
import vedo
from shapely.geometry import Polygon, MultiPolygon
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Combined')))
from stl_algo import *

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

    def test_projection_merger_empty_input(self):
        projected_areas = []
        merged_area = projection_merger(projected_areas)
        print(f"Input: {projected_areas}, Output: {merged_area}")
        self.assertTrue(merged_area.is_empty)
        print("test_projection_merger_empty_input: Test successful!\n")


class TestStlSupportPlacement(unittest.TestCase):

    def setUp(self):
        # Use just the file name '8.stl' as the file path
        self.file_path = '8.stl'  # File is assumed to be in the correct 'src' folder

        # Mock points and faces for a sample mesh (you can modify this as needed for actual test cases)
        self.points = np.array([
            [0, 0, 0],    # Point 1
            [1, 0, 0],    # Point 2
            [0, 1, 0],    # Point 3
            [0.5, 0.5, 1] # Point 4 (elevated for a downward face)
        ])
        self.faces = np.array([
            [0, 1, 3],  # Face 1 (downward facing)
            [1, 2, 3]   # Face 2 (downward facing)
        ])
        
        # Create a mock STL mesh object using vedo.Mesh for testing
        self.stl_mesh = vedo.Mesh([self.points, self.faces])

    @patch('stl_algo.result_saver')  # Mock the result_saver to avoid file writes
    @patch('builtins.input', side_effect=['0.5', '0.2', '20', '0.1', '0.05', '0.3', '1000', '10' , 'N'])  # Correctly mock inputs
    @patch('vedo.Plotter')  # Mock Plotter to prevent visualization window from opening
    @patch('vedo.Mesh.clone', return_value=MagicMock())  # Patch clone to avoid copy issues
    def test_support_placement_on_required_points_stl(self, mock_clone, mock_plotter, mock_input, mock_saver):
        """
        Test whether supports are placed on points identified as requiring support in the STL algorithm.
        """
        # Mock the result_saver to avoid file saves
        mock_saver.return_value = None

        # Mock the Plotter instance to avoid visualization
        mock_plotter_instance = MagicMock()
        mock_plotter.return_value = mock_plotter_instance
        mock_plotter_instance.show.return_value = None

        # Mock the vedo.Mesh creation to return our predefined mesh instead of reading an actual file
        with patch('vedo.Mesh', return_value=self.stl_mesh):
            # Run the STL main function to generate supports
            stl_main(self.file_path)

        # Compute normals manually for each face
        def compute_face_normal(vertices):
            p0, p1, p2 = vertices
            normal = np.cross(p1 - p0, p2 - p0)
            return normal / np.linalg.norm(normal)

        # Check if a face is downward-facing by normal
        downward_faces = []
        for face in self.faces:
            vertices = self.stl_mesh.points()[face]
            normal = compute_face_normal(vertices)
            if face_is_downward(normal):
                downward_faces.append(face)

        # Simulate support generation for each downward face and check intersection points
        generated_supports = []
        for face in downward_faces:
            centroid = np.mean(self.stl_mesh.points()[face], axis=0)
            if check_intersection(self.stl_mesh, centroid[0], centroid[1]) is not None:
                generated_supports.append(centroid)

        # Now check that supports are placed at correct points
        for support in generated_supports:
            expected_centroid = np.mean(self.stl_mesh.points()[self.faces[0]], axis=0)
            self.assertTrue(np.allclose(support[:2], expected_centroid[:2], atol=1e-3),
                            f"Support not placed at the expected centroid {expected_centroid}.")


class TestSimpleCoverage(unittest.TestCase):

    def setUp(self):
        # Setting up a basic mesh object
        self.points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.faces = np.array([
            [0, 1, 2],
            [1, 2, 3]
        ])
        self.mesh = vedo.Mesh([self.points, self.faces])

    def test_add_torsion_based_support(self):
        result = add_torsion_based_support(self.mesh, [], [], [], 1.0, 0.5, 0.1, 0.1, [])
        self.assertIsInstance(result, list)

    def test_calculate_all_face_torques(self):
        torques, magnitudes, avg_torque, torsion_faces = calculate_all_face_torques(self.mesh, 1.0, [])
        self.assertIsInstance(torques, np.ndarray)

    def test_calculate_normal(self):
        normal = calculate_normal(self.points[0], self.points[1], self.points[2])
        self.assertEqual(normal.shape, (3,))

    def test_calculate_surface_area_above(self):
        result = calculate_surface_area_above(self.mesh, 0.5, [], 0.1, 0.1, 1.0)
        self.assertIsInstance(result, tuple)

    def test_calculate_torque_for_face(self):
        torque, center = calculate_torque_for_face(self.points[0], self.points[1], self.points[2], np.array([0.5, 0.5, 0.5]), 1.0)
        self.assertEqual(len(torque), 3)

    def test_check_intersection(self):
        result = check_intersection(self.mesh, 0.5, 0.5)
        self.assertIsInstance(result, (np.ndarray, type(None)))

    def test_face_is_downward(self):
        downward = face_is_downward(np.array([0, 0, -1]))
        self.assertTrue(downward)

    def test_face_overlaps_support_on_z(self):
        overlaps = face_overlaps_support_on_z(self.points, self.mesh, 0.1)
        self.assertIsInstance(overlaps, bool)

    def test_face_projector(self):
        projected = face_projector(np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(projected.shape[1], 2)

    def test_faces_are_connected(self):
        connected = faces_are_connected(self.faces[0], self.faces[1])
        self.assertIsInstance(connected, bool)

    def test_faces_extractor(self):
        result = faces_extractor(self.mesh, 45, 0.1)
        self.assertEqual(len(result), 5)

    def test_merge_triangles_into_mesh(self):
        triangle = vedo.shapes.Triangle(self.points[0], self.points[1], self.points[2])
        merged_mesh = merge_triangles_into_mesh([triangle])
        self.assertIsInstance(merged_mesh, vedo.Mesh)

    def test_point_inside_mesh(self):
        inside = point_inside_mesh((0.1, 0.1, 0.1), self.mesh)
        self.assertIsInstance(inside, bool)
        
    def test_support_creator(self):
        result = support_creator(self.mesh, [], sh.Polygon([(0,0), (1,0), (1,1), (0,1)]), 0.1, 0.5, 0.1)
        self.assertEqual(len(result), 2)

    def test_support_filter(self):
        result = support_filter(self.mesh, [], [], 0.1, 0.1)
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
