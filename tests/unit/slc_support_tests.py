# File path: tests/unit/slc_support_tests.py
import unittest
import numpy as np
import sys, os
from pathlib import Path
from shapely.geometry import Polygon


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../SLC')))
from slc_support_structure_1 import SLCReader, SupportSettings, SupportGenerator, merge_polygons

class TestSlcAcceptance(unittest.TestCase):

    def setUp(self):
        # Setup the file path for testing
        self.file_path = Path(__file__).parent.parent.parent / 'SLC/slc_samples/8.slc'
        
        # Initialize the SlcReader with file path
        self.slc_reader = SLCReader(
            file_path=self.file_path
        )
        
        # Create support settings
        self.support_settings = SupportSettings(
            spacing_between_supports=0.5,
            maximum_column_diameter=0.1,
            self_support_angle=20,
            spacing_from_model=0.1,
            contact_point_diameter=0.1
        )

    def tearDown(self):
        # Close the file stream after each test
        self.slc_reader.close()

    def test_support_generation_from_slices(self):
        """
        User Story #1: Test if supports are generated from the SLC file.
        """
        # Read slices from the SLC file
        slices = self.slc_reader.read_slices()
        self.assertGreater(len(slices), 0, "No slices read from the file.")
        
        # Generate supports using SupportGenerator
        merged_area = merge_polygons([Polygon(c) for s in slices for c in s['contours']])
        support_generator = SupportGenerator(slices, merged_area, self.support_settings)
        supports = support_generator.create_support_for_parameters()

        # Check if supports were generated
        self.assertGreater(len(supports), 0, "No supports generated for the slices.")

    def test_custom_support_parameters(self):
        """
        User Story #3: Test if custom support parameters are applied correctly.
        """
        # Parameters for customization
        custom_settings = SupportSettings(
            spacing_between_supports=1.0,
            maximum_column_diameter=0.2,
            self_support_angle=30,
            spacing_from_model=0.2,
            contact_point_diameter=0.15
        )
        
        # Initialize a new SlcReader with custom parameters
        custom_slc_reader = SLCReader(file_path=self.file_path)
        
        # Read slices and generate supports with custom parameters
        slices = custom_slc_reader.read_slices()
        merged_area = merge_polygons([Polygon(c) for s in slices for c in s['contours']])
        support_generator = SupportGenerator(slices, merged_area, custom_settings)
        supports = support_generator.create_support_for_parameters()
        
        # Check if supports were generated with custom parameters
        self.assertGreater(len(supports), 0, "No supports generated with custom parameters.")

        # Cleanup
        custom_slc_reader.close()

class TestSlcSupportPlacement(unittest.TestCase):

    def setUp(self):
        # Setup the file path for testing
        self.file_path = Path(__file__).parent.parent.parent / 'Combined/src/8.slc'
        
        # Initialize the SlcReader with default parameters
        self.slc_reader = SLCReader(
            file_path=self.file_path
        )
        
        # Create support settings
        self.support_settings = SupportSettings(
            spacing_between_supports=0.5,
            maximum_column_diameter=0.1,
            self_support_angle=20,
            spacing_from_model=0.1,
            contact_point_diameter=0.1
        )

    def tearDown(self):
        # Close the file stream after each test
        self.slc_reader.close()

    def test_support_placement_on_required_points_slc(self):
        """
        Client Request: Test whether supports are placed on points identified as requiring support.
        """
        # Read slices from the SLC file
        slices = self.slc_reader.read_slices()
        self.assertGreater(len(slices), 0, "No slices read from the file.")
        
        # Generate supports
        merged_area = merge_polygons([Polygon(c) for s in slices for c in s['contours']])
        support_generator = SupportGenerator(slices, merged_area, self.support_settings)
        supports = support_generator.create_support_for_parameters()
        
        # Ensure supports are placed correctly
        self.assertGreater(len(supports), 0, "No supports generated for the slices.")

if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])  # Exclude first argument to avoid unittest processing it
