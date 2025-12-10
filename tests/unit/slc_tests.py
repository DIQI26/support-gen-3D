import unittest
import numpy as np
import sys, os
from pathlib import Path
from shapely.geometry import Polygon, Point, MultiPolygon
from io import BytesIO
import tempfile
from unittest.mock import patch, MagicMock, mock_open
import unittest
import struct

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Combined')))
from slc_support_structure import SLCReader, SupportGenerator,supports_overlap, calculate_angle_with_vertical, check_intersection, merge_polygons, SupportParameterBuilder, get_user_input, slc_main

class TestSLCReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.valid_file_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Combined/src/8.slc')))
        cls.invalid_file_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Combined/src/invalid.slc')))

    def setUp(self):
        self.reader = SLCReader(self.valid_file_path)

    def tearDown(self):
        self.reader.close()

    def test_initialization(self): # verifies correct initialisation
        self.assertEqual(self.reader.file_path, self.valid_file_path) # checks file path, scale and offset
        self.assertEqual(self.reader.scale, 1.0)
        np.testing.assert_array_equal(self.reader.offset, np.array([0.0, 0.0, 0.0]))

    def test_device_readable(self): # checks if file stream is readable
        self.assertTrue(self.reader.device_readable())

    def test_read_header_valid(self): # checks header for specified entires
        self.assertTrue(self.reader.read_header())
        self.assertIn('-TYPE', self.reader.header)
        self.assertIn('-UNIT', self.reader.header)


    def test_read_slice_valid(self): # reads a slice and ensures valid data is returned
        self.reader.read_header()  # ensure header is read
        slice_data = self.reader.read_slice()
        self.assertIsNotNone(slice_data)
        self.assertIn('z', slice_data)
        self.assertIn('contours', slice_data)



    def test_read_slices(self): # multiple slices
        self.reader.read_header()  # ensure header is read
        slices = self.reader.read_slices()
        self.assertIsInstance(slices, list)
        self.assertGreater(len(slices), 0, "No slices were read from the SLC file.")


    def test_next_z_value(self): # verifies a float is returned
        self.reader.read_header()  # Ensure header is read
        z_value = self.reader.next_z_value()
        self.assertIsInstance(z_value, float)

    def test_next_z_value_no_read(self): 
        # Create a mock stream to simulate invalid read scenario
        self.reader.stream = MagicMock()
        self.reader.stream.readable.return_value = False
        z_value = self.reader.next_z_value()
        self.assertTrue(np.isnan(z_value), "Expected NaN for unreadable stream.")

    def test_scale_vertices(self): #checks that vertex scaling works correctly with different scale values
        vertices = np.array([[0.0, 0.0], [1.0, 1.0]])
        scaled_vertices = self.reader._scale(vertices)
        np.testing.assert_array_equal(scaled_vertices, vertices)

        self.reader.scale = 2.0
        scaled_vertices = self.reader._scale(vertices)
        expected_scaled_vertices = np.array([[0.0, 0.0], [2.0, 2.0]])
        np.testing.assert_array_equal(scaled_vertices, expected_scaled_vertices)


    def test_close(self): # checks that file stream is closed properly
        self.reader.close()
        with self.assertRaises(ValueError):
            self.reader.stream.read()  # Ensure the stream is closed and raises an error

    def test_invalid_file_path(self): # check for invalid path of slc file
        with self.assertRaises(FileNotFoundError):
            SLCReader(self.invalid_file_path)

class TestSupportParameter(unittest.TestCase):

    def test_default_initialization(self): # verfies default values for parameters are as expected
        builder = SupportParameterBuilder()
        settings = builder.build()
        self.assertEqual(settings.spacing_between_supports, 0.5)
        print("test_default_initialization: Passed")
        self.assertEqual(settings.maximum_column_diameter, 0.2)
        self.assertEqual(settings.self_support_angle, 20)
        self.assertEqual(settings.spacing_from_model, 0.5)
        self.assertEqual(settings.contact_point_diameter, 0.2)
        self.assertEqual(settings.support_radius, 0.1)

    def test_custom_initialization(self):   # tests builders ability to enter custom values
        builder = SupportParameterBuilder()
        settings = builder.set_spacing_between_supports(1.0) \
                          .set_maximum_column_diameter(0.3) \
                          .set_self_support_angle(35) \
                          .set_spacing_from_model(0.4) \
                          .set_contact_point_diameter(0.25) \
                          .build()

        self.assertEqual(settings.spacing_between_supports, 1.0)
        print("test_custom_initialization: Passed")
        self.assertEqual(settings.maximum_column_diameter, 0.3)
        self.assertEqual(settings.self_support_angle, 35)
        self.assertEqual(settings.spacing_from_model, 0.4)
        self.assertEqual(settings.contact_point_diameter, 0.25)
        self.assertEqual(settings.support_radius, 0.15)

class TestSupportGenerator(unittest.TestCase):

    def setUp(self):
        # Define the attributes used in the tests
        self.spacing_from_model = 1.0
        self.spacing_between_supports = 2.0
        self.self_support_angle = 30.0
        
        # Create a sample slices list and merged_area
        self.slices = [
            {'z': 0, 'contours': [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])]},  # Example square contour
            {'z': 1, 'contours': [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])]}   # Example square contour
        ]
        self.merged_area = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])  # Example polygon

        # Create an instance of SupportSettings using the builder
        self.settings = SupportParameterBuilder() \
                        .set_spacing_between_supports(self.spacing_between_supports) \
                        .set_self_support_angle(self.self_support_angle) \
                        .set_spacing_from_model(self.spacing_from_model) \
                        .build()

        # Create an instance of SupportGenerator
        self.generator = SupportGenerator(self.slices, self.merged_area, self.settings)

    def test_create_support_for_empty_slices(self):
        # Mocking an empty polygon area and settings
        empty_generator = SupportGenerator([], Polygon(), SupportParameterBuilder().build())
        supports = empty_generator.create_support_for_angle()
        self.assertEqual(supports, [])
        print("test_create_support_for_empty_slices: Passed")

    def test_create_support_for_invalid_merged_area(self):
        invalid_area = Polygon()
        invalid_generator = SupportGenerator(self.slices, invalid_area, self.settings)
        supports = invalid_generator.create_support_for_angle()
        self.assertEqual(supports, [])
        print("test_create_support_for_invalid_merged_area: Passed")

    def test_supports_overlap(self): # tests both overlap and non overlap scenarios
        existing_points = [[0, 0], [1, 1]]
        new_point = [0.05, 0.05]  # this should overlap with [0, 0]
        self.assertTrue(supports_overlap(existing_points, new_point, 0.1))
        print("test_supports_overlap (overlap): Passed")

        new_point = [2, 2]  # this should not overlap with any existing points
        self.assertFalse(supports_overlap(existing_points, new_point, 0.1)) # 0.1 is min distance
        print("test_supports_overlap (no overlap): Passed")

    def test_all_slices_below_angle(self): # no supports generated for below self support angle
        slices_below_angle = [
            {'z': 0, 'contours': [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])]},  # all angles below self_support_angle
            {'z': 1, 'contours': [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])]}
        ]
        generator = SupportGenerator(slices_below_angle, self.merged_area, self.settings)
        supports = generator.create_support_for_angle()
        self.assertEqual(supports, [])
        self.assertFalse(generator.any_support_needed)
        print("test_all_slices_below_angle: Passed")

    def test_negative_height_support(self): # supports with negative height should not be generated
        slices_negative_height = [
            {'z': 0, 'contours': [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])]},  # setup that causes negative height
            {'z': 1, 'contours': [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])]}
        ]
        generator = SupportGenerator(slices_negative_height, self.merged_area, self.settings)
        supports = generator.create_support_for_angle()
        self.assertEqual(len(supports), 0)
        print("test_negative_height_support: Passed")

    def test_zero_support_radius(self): # no supports generated when support radius is zero
        self.settings.support_radius = 0  # set support radius to zero
        generator = SupportGenerator(self.slices, self.merged_area, self.settings)
        supports = generator.create_support_for_parameters()
        self.assertEqual(len(supports), 0)
        print("test_zero_support_radius: Passed")

class TestCalculateAngleWithVertical(unittest.TestCase):

    def test_vertical_line(self): # angle with vertical line should be 0
        contour = np.array([[1, 0], [1, 1]])  # vertical line
        angle = calculate_angle_with_vertical(contour)
        self.assertAlmostEqual(angle, 0.0, places=6)  # expecting 0 degrees
        print("test_vertical_line: Passed")

    def test_horizontal_line(self): # angle with horizontal line should be 90
        contour = np.array([[0, 0], [1, 0]])  # Horizontal line
        angle = calculate_angle_with_vertical(contour)
        self.assertAlmostEqual(angle, 90.0, places=6)  # Expecting 90 degrees
        print("test_horizontal_line: Passed")

    def test_diagonal_line(self): 
        contour = np.array([[0, 0], [1, 1]])  # 45-degree diagonal line
        angle = calculate_angle_with_vertical(contour)
        self.assertAlmostEqual(angle, 45.0, places=6)  # Expecting 45 degrees
        print("test_diagonal_line: Passed")

    def test_single_point(self):
        contour = np.array([[0, 0]])  # Single point
        angle = calculate_angle_with_vertical(contour)
        self.assertAlmostEqual(angle, 0.0, places=6)  # Expecting 0 degrees
        print("test_single_point: Passed")

    def test_empty_contour(self):
        contour = np.array([])  # Empty contour
        angle = calculate_angle_with_vertical(contour)
        self.assertEqual(angle, 0.0)  # Expecting 0
        print("test_empty_contour: Passed")

class TestCheckIntersection(unittest.TestCase):
    def setUp(self):
        # Setting up test data for the slices
        self.slices = [
            {
                'z': 0,
                'contours': [np.array([[0, 0], [1, 2], [2, 1]])]  # Triangle
            },
            {
                'z': 1,
                'contours': [np.array([[2, 0], [3, 3], [4, 2]])]  # Another triangle
            },
            {
                'z': 2,
                'contours': []  # No contours
            }
        ]

    def test_point_inside_first_contour(self):
        result = check_intersection(self.slices, 1, 1)
        self.assertEqual(result, [0], "Expected z value for the first slice")

    def test_point_inside_second_contour(self):
        result = check_intersection(self.slices, 3, 1)
        self.assertEqual(result, [1], "Expected z value for the second slice")

    def test_point_on_edge_first_contour(self):
        result = check_intersection(self.slices, 1, 2)
        self.assertEqual(result, [0], "Expected z value for the first slice")

    def test_point_on_edge_second_contour(self):
        result = check_intersection(self.slices, 4, 2)
        self.assertEqual(result, [1], "Expected z value for the second slice")

    def test_point_outside_all_contours(self):
        result = check_intersection(self.slices, 5, 5)  # Outside all contours
        self.assertEqual(result, [], "Expected no intersection with any contours")
    
    def test_point_inside_multiple_contours(self):
        result = check_intersection(self.slices, 0.5, 0.5)  # Inside the first triangle
        self.assertEqual(result, [0], "Expected z value for the first slice with multiple contours")

    def test_point_on_corner_of_first_contour(self):
        result = check_intersection(self.slices, 0, 0)  # Corner of the first triangle
        self.assertEqual(result, [0], "Expected z value for the first slice at corner")

    def test_point_on_corner_of_second_contour(self):
        result = check_intersection(self.slices, 4, 2)  # Corner of the second triangle
        self.assertEqual(result, [1], "Expected z value for the second slice at corner")

    def test_empty_slices(self):
        empty_slices = []
        result = check_intersection(empty_slices, 0, 0)
        self.assertEqual(result, [], "Expected no intersection with empty slices")
        
    def test_partial_intersection_with_invalid_contour(self):
        # Using a contour that might be invalid
        invalid_contour = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])  # Self-intersecting
        self.slices.append({
            'z': 4,
            'contours': [invalid_contour]  # Self-intersecting contour
        })
        result = check_intersection(self.slices, 0.5, 0.5)  # Inside valid contour
        self.assertEqual(result, [0, 4], "Expected z values for valid intersection with an invalid contour")

class TestMergePolygons(unittest.TestCase):

    def test_non_overlapping_polygons(self):
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        merged = merge_polygons([poly1, poly2])
        
        # Check if the merged result is a MultiPolygon and count the number of geometries
        if isinstance(merged, MultiPolygon):
            self.assertEqual(len(merged.geoms), 2, "Should have two non-overlapping polygons")
        else:
            self.fail("Expected a MultiPolygon for non-overlapping input")

    def test_overlapping_polygons(self):
        polygon1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        polygon2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        merged = merge_polygons([polygon1, polygon2])
        
        # Check if the result is a single Polygon
        self.assertTrue(isinstance(merged, Polygon), "Should have one merged polygon")
        
        print("test_overlapping_polygons: Passed")

    def test_touching_polygons(self):
        polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        polygon2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        merged = merge_polygons([polygon1, polygon2])
        
        # Check if the result is a single Polygon
        self.assertTrue(isinstance(merged, Polygon), "Should merge touching polygons into one")
        
        print("test_touching_polygons: Passed")
    
    def test_empty_polygons(self):
        empty_poly = Polygon()
        valid_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        merged = merge_polygons([empty_poly, valid_poly])
        
        # Should return the valid polygon since the other is empty
        self.assertEqual(merged, valid_poly, "Should return the valid polygon when one is empty")
    
    def test_all_empty_polygons(self):
        empty_poly = Polygon()
        merged = merge_polygons([empty_poly, empty_poly])
        
        # Should return None as all are empty
        self.assertIsNone(merged, "Should return None when all polygons are empty")
    
    def test_invalid_polygon(self):
        # Create a polygon that is invalid (self-intersecting)
        invalid_poly = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
        valid_poly = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        merged = merge_polygons([invalid_poly, valid_poly])
        
        # Should return the valid polygon since the other is invalid
        self.assertEqual(merged, valid_poly, "Should return the valid polygon when one is invalid")
    
    def test_multiple_polygons(self):
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        poly3 = Polygon([(4, 4), (5, 4), (5, 5), (4, 5)])
        merged = merge_polygons([poly1, poly2, poly3])
        
        # Check if the result is a MultiPolygon and has the correct number of geometries
        if isinstance(merged, MultiPolygon):
            self.assertEqual(len(merged.geoms), 2, "Should have two merged polygons")
        else:
            self.fail("Expected a MultiPolygon for multiple input polygons")

    def test_identical_polygons(self):
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        merged = merge_polygons([poly1, poly2])
        
        # Should return a single polygon
        self.assertTrue(isinstance(merged, Polygon), "Should merge identical polygons into one")

    def test_mixed_validity(self):
        valid_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        invalid_poly = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])  # Self-intersecting
        empty_poly = Polygon()
        
        merged = merge_polygons([valid_poly, invalid_poly, empty_poly])
        
        # Should return only the valid polygon
        self.assertEqual(merged, valid_poly, "Should return the valid polygon when mixed with invalid and empty") 

class TestGetUserInput(unittest.TestCase):

    @patch('builtins.input', side_effect=["", ""])
    def test_default_input(self, mock_input):
        result = get_user_input("Enter a value: ", 1.0)
        self.assertEqual(result, 1.0)

    @patch('builtins.input', side_effect=["invalid", "2.5"])
    def test_invalid_string_input(self, mock_input):
        result = get_user_input("Enter a value: ", 1.0)
        self.assertEqual(result, 2.5)

    @patch('builtins.input', side_effect=["-1", "2.0"])
    def test_negative_input(self, mock_input):
        result = get_user_input("Enter a value: ", 2.0)
        self.assertEqual(result, 2.0)

    @patch('builtins.input', side_effect=["0", "0.0"])
    def test_zero_input(self, mock_input):
        result = get_user_input("Enter a value: ", 0.0)
        self.assertEqual(result, 0.0)

class TestSLCMain(unittest.TestCase):

    @patch('builtins.print')  # Mocking print to suppress output during tests
    @patch('slc_support_structure.get_user_input', side_effect=[0.5, 0.2, 20, 0.5, 0.2])
    @patch('slc_support_structure.SLCReader')
    @patch('slc_support_structure.SupportGenerator')
    @patch('slc_support_structure.merge_polygons')
    @patch('slc_support_structure.visualize_slices_and_supports')  # Mocking visualization
    def test_slc_main(self, mock_visualize, mock_merge_polygons, mock_support_generator, mock_slc_reader, mock_get_user_input, mock_print):
        # Setup mock for SLCReader
        mock_reader_instance = MagicMock()
        
        # Mock read_slices to include the expected structure with 'z'
        mock_reader_instance.read_slices.return_value = [
            {'contours': [[(0, 0), (1, 1), (1, 0), (0, 0)]], 'z': 0.0}
        ]
        mock_slc_reader.return_value = mock_reader_instance

        # Setup mock for SupportGenerator
        mock_support_instance = MagicMock()
        mock_support_instance.create_support_for_angle.return_value = ['angle_support']  # 1 support
        mock_support_instance.create_support_for_parameters.return_value = ['parameter_support']  # 1 support
        mock_support_generator.return_value = mock_support_instance
        
        # Setup mock for merge_polygons
        mock_merge_polygons.return_value = 'merged_polygon'
        
        # Define a dummy file path
        dummy_file_path = Path('dummy_path.slc')
        
        # Call the function
        slc_main(dummy_file_path)

        # Assert that the SLCReader was initialized with the correct file path
        mock_slc_reader.assert_called_once_with(dummy_file_path)
        
        # Assert that read_slices was called on the SLCReader
        mock_reader_instance.read_slices.assert_called_once_with(join_gaps=True)
        
        # Assert that merge_polygons was called with the correct parameters
        mock_merge_polygons.assert_called_once()
        
        # Assert that SupportGenerator was initialized with correct parameters
        mock_support_generator.assert_called_once()
        
        # Assert that the correct methods were called on SupportGenerator
        mock_support_instance.create_support_for_angle.assert_called_once()
        mock_support_instance.create_support_for_parameters.assert_called_once()
        
        # Assert that print was called with the total number of supports generated
        mock_print.assert_called_once_with("Total number of supports generated: 2")  # Updated expectation

        # Ensure the visualization function was called, but we don't care about its parameters
        mock_visualize.assert_called_once()  # This checks that the visualization function was called

class TestSupportsOverlap(unittest.TestCase):

    def test_no_existing_points(self):
        existing_points = []
        new_point = [0.0, 0.0]
        min_distance = 1.0
        
        result = supports_overlap(existing_points, new_point, min_distance)
        self.assertFalse(result)
        print("test_no_existing_points: Passed")

    def test_no_overlap(self):
        existing_points = [[0.0, 0.0], [2.0, 2.0]]
        new_point = [3.0, 3.0]
        min_distance = 1.0
        
        result = supports_overlap(existing_points, new_point, min_distance)
        self.assertFalse(result)
        print("test_no_overlap: Passed")

    def test_overlap(self):
        existing_points = [[0.0, 0.0], [2.0, 2.0]]
        new_point = [0.5, 0.5]  # Distance < sqrt(2) ≈ 1.41
        min_distance = 1.0
        
        result = supports_overlap(existing_points, new_point, min_distance)
        self.assertTrue(result)
        print("test_overlap: Passed")

    def test_on_boundary(self):
        existing_points = [[0.0, 0.0], [2.0, 2.0]]
        new_point = [1.0, 1.0]  # Distance = sqrt(2) ≈ 1.41
        min_distance = np.sqrt(2)  # Exact distance
        
        result = supports_overlap(existing_points, new_point, min_distance)
        self.assertFalse(result)  # No overlap, as distance is exactly equal to min_distance
        print("test_on_boundary: Passed")

    def test_multiple_points_some_overlap(self):
        existing_points = [[0.0, 0.0], [3.0, 3.0], [5.0, 5.0]]
        new_point = [2.0, 2.0]  # Should overlap with [3.0, 3.0]
        min_distance = 2.0
        
        result = supports_overlap(existing_points, new_point, min_distance)
        self.assertTrue(result)  # Overlaps with [3.0, 3.0]
        print("test_multiple_points_some_overlap: Passed")


if __name__ == '__main__':
    unittest.main()