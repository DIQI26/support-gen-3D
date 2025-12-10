import shapely.geometry as geo
import math
import sys, os
import unittest
import numpy as np
import vedo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Practice/PointFinder_HY')))
from main import points_finder, support_creator, calculate_distance, point_in_slice, need_support

class TestMainFunctions(unittest.TestCase):

    def setUp(self):
        self.slice_thickness = 0.05
        self.support_angle = 20
        
        # Creation of slices
        points1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])  
        points2 = np.array([[0.1, 0.1, 0.05], [1.1, 0.1, 0.05], [1.1, 1.1, 0.05], [0.1, 1.1, 0.05]])
        slice1 = vedo.Mesh([points1])
        slice2 = vedo.Mesh([points2])
        self.slices = [slice1, slice2]
        self.slice1 = slice1
        self.slice2 = slice2

    def test_points_finder(self):
        points_need_support, points_nearest = points_finder(self.slices, self.slice_thickness, self.support_angle)
        print(f"Input slices: {self.slices}, Slice Thickness: {self.slice_thickness}, Support Angle: {self.support_angle}")
        print(f"Points needing support: {points_need_support}")
        print(f"Nearest points: {points_nearest}")
        
        # Assert points needing support and nearest points found
        self.assertGreater(len(points_need_support), 0)
        self.assertGreater(len(points_nearest), 0)
        print("test_points_finder: Test successful!\n")

    def test_points_finder_empty(self):
        # Empty slice test
        slices_empty = []
        points_need_support, points_nearest = points_finder(slices_empty, self.slice_thickness, self.support_angle)
        print(f"Input: Empty slices, Output Points needing support: {points_need_support}, Nearest points: {points_nearest}")
        self.assertEqual(len(points_need_support), 0)
        self.assertEqual(len(points_nearest), 0)
        print("test_points_finder_empty: Test successful!\n")

    def test_points_finder_no_support_needed(self):
        # No support needed test
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        slice1 = vedo.Mesh([points])
        slice2 = vedo.Mesh([points + [0, 0, self.slice_thickness]])
        slices = [slice1, slice2]
        points_need_support, points_nearest = points_finder(slices, self.slice_thickness, self.support_angle)
        print(f"Input slices: {slices}, Slice Thickness: {self.slice_thickness}, Support Angle: {self.support_angle}")
        print(f"Points needing support: {points_need_support}")
        print(f"Nearest points: {points_nearest}")
        self.assertEqual(len(points_need_support), 0)
        self.assertEqual(len(points_nearest), 0)
        print("test_points_finder_no_support_needed: Test successful!\n")

    def test_support_creator(self):
        # Testing creation of a support with normal parameters
        center = (0, 0, 0)
        height = 1
        radius = 0.1
        support = support_creator(center=center, height=height, radius=radius)
        print(f"Input: center={center}, height={height}, radius={radius}, Output: {support}")
        self.assertIsInstance(support, vedo.Mesh)
        print("test_support_creator: Test successful!\n")

    def test_calculate_distance(self):
        point1 = (0, 0, 0)
        point2 = (3, 4, 0)
        distance = calculate_distance(point1, point2)
        print(f"Input: point1={point1}, point2={point2}, Output: {distance}")
        self.assertEqual(distance, 5)
        print("test_calculate_distance: Test successful!\n")

    def test_point_in_slice(self):
        slice = vedo.Mesh([np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])])
        point_inside = (0.5, 0.5, 0)
        point_outside = (2, 2, 0)
        result_inside = point_in_slice(point_inside, slice)
        result_outside = point_in_slice(point_outside, slice)
        print(f"Input: slice points={slice.points()}, point_inside={point_inside}, point_outside={point_outside}")
        print(f"Output: result_inside={result_inside}, result_outside={result_outside}")
        self.assertTrue(result_inside)
        self.assertFalse(result_outside)
        print("test_point_in_slice: Test successful!\n")

    def test_need_support(self):
        up_point = (0.5, 0.5, 0.1)
        result = need_support(up_point, self.slice2, self.slice_thickness, self.support_angle)
        print(f"Input: up_point={up_point}, Slice Thickness: {self.slice_thickness}, Support Angle: {self.support_angle}, Slice2 points={self.slice2.points()}")
        print(f"Output: result={result}")
        
        # Result checking
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)

        if result[1] is not None:
            self.assertIsInstance(result[1], tuple)
        else:
            self.assertIsNone(result[1])

        # Does point need support, is nearest point valid?
        self.assertTrue(result[0] or result[1] is None)
        print("test_need_support: Test successful!\n")

    def test_need_support_edge_case(self):
        # Point is at threshold distance
        up_point = (0.05, 0.05, 0.05)
        result = need_support(up_point, self.slice1, self.slice_thickness, self.support_angle)
        print(f"Input: up_point={up_point}, Slice Thickness: {self.slice_thickness}, Support Angle: {self.support_angle}, Slice1 points={self.slice1.points()}")
        print(f"Output: result={result}")
        
        # Lowest layer doesn't need support
        self.assertFalse(result[0])
        print("test_need_support_edge_case: Test successful!\n")

if __name__ == '__main__':
    unittest.main()
