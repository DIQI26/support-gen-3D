import sys
import os
import vedo
import unittest
import numpy as np
from pathlib import Path
from io import BytesIO
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../SLC')))
from slc_reader import slc_reader

class TestSlcReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setting up a fixed file path for testing
        current_dir = Path(__file__).parent
        slc_folder = current_dir / '../../SLC/slc_samples'
        
        # Use a default test file '8.slc' for testing
        cls.file_path = slc_folder / '8.slc'

        if not cls.file_path.exists():
            print(f"Warning: File {cls.file_path} not found. Tests may fail if this file is required.")

        # Prepare a mock SLC file header and data
        cls.header = (b'-TYPE WEB\x1a' +
                      b'-UNIT MM\x1a')
        cls.slice_data = (b'\x01\x00\x00\x00'  # 1 vertex (little-endian)
                          b'\x00\x00\x00\x00'  # 0 gaps
                          b'\x00\x00\x80\x3f'  # vertex x = 1.0 (float)
                          b'\x00\x00\x00\x00'  # vertex y = 0.0 (float)
                          b'\x00\x00\x00\x00'  # z = 0.0 (float)
                          b'\x01\x00\x00\x00')  # 1 contour (little-endian)
        cls.mock_file = BytesIO(cls.header + cls.slice_data)

    def setUp(self):
        # Reset the mock file position before each test
        self.mock_file.seek(0)

    @patch('builtins.input', return_value='8.slc')
    def test_visualize_full_object(self, mock_input):
        print(f"Testing visualize full object with input file: {self.file_path}")
        reader = slc_reader(self.file_path)
        slices = reader.read_slices()

        # Check if all slices have been read
        self.assertGreater(len(slices), 0, "No slices were read from the SLC file.")
        print(f"Number of slices read: {len(slices)}")

        # Ensure that no slice is hollow (i.e., all contours have vertices)
        for slice_data in slices:
            self.assertGreater(len(slice_data['contours']), 0, "A slice has no contours.")
            for contour in slice_data['contours']:
                self.assertGreater(len(contour), 2, "A contour has insufficient vertices to form a closed shape.")

        reader.close()
        print("Output: Full object visualization test successful.\n")

    @patch('builtins.input', return_value='8.slc')
    def test_visualize_specific_slice(self, mock_input):
        slice_index = 5 
        print(f"Testing visualize specific slice with input file: {self.file_path}, slice index: {slice_index}")
        
        reader = slc_reader(self.file_path)

        # Read up to the specified slice
        slices = []
        for _ in range(slice_index + 1):
            slice_data = reader.read_slice()
            slices.append(slice_data)

        # Ensure the specified slice is read
        self.assertGreater(len(slices), slice_index, f"Slice {slice_index} was not read from the SLC file.")
        print(f"Number of slices read: {len(slices)}")

        # Ensure the slice has contours with connected points
        slice_data = slices[slice_index]
        self.assertGreater(len(slice_data['contours']), 0, "The slice has no contours.")
        for contour in slice_data['contours']:
            self.assertGreater(len(contour), 2, "A contour has insufficient vertices to form a connected shape.")
        
        reader.close()
        print(f"Output: Visualization of slice {slice_index} successful.\n")

    @patch('builtins.input', return_value='8.slc')
    def test_extract_all_points(self, mock_input):
        print(f"Testing extract all points with input file: {self.file_path}")
        reader = slc_reader(self.file_path)
        slices = reader.read_slices()

        self.assertGreater(len(slices), 0, "No slices were read from the SLC file.")
        print(f"Number of slices read: {len(slices)}")
        
        for slice_data in slices:
            self.assertGreater(len(slice_data['contours']), 0, "A slice has no contours.")
            for contour in slice_data['contours']:
                self.assertEqual(contour.shape[1], 2, "Contour vertices do not have two dimensions (x, y).")
                for point in contour:
                    self.assertIsInstance(point[0], np.float32, "X coordinate is not a floating-point number.")
                    self.assertIsInstance(point[1], np.float32, "Y coordinate is not a floating-point number.")
        
        reader.close()
        print("Output: Extraction of all points successful.\n")

    @patch('builtins.input', return_value='8.slc')
    def test_read_single_slice(self, mock_input):
        slice_number = 50  
        print(f"Testing read single slice for slice number {slice_number} with input file: {self.file_path}")
        
        reader = slc_reader(self.file_path)
        slice_data = reader.read_single_slice(slice_number)

        self.assertIsNotNone(slice_data, f"Slice {slice_number} could not be read from the SLC file.")
        print(f"Successfully read slice number {slice_number}")

        self.assertGreater(len(slice_data[0]['contours']), 0, f"Slice {slice_number} has no contours.")
        for contour in slice_data[0]['contours']:
            self.assertGreater(len(contour), 2, "A contour has insufficient vertices to form a closed shape.")

        reader.close()
        print(f"Output: Single slice {slice_number} read test successful.\n")

    @patch('builtins.input', side_effect=["8.slc", "1"])
    def test_device_readable(self, mock_input):
        reader = slc_reader(self.file_path)
        reader.stream = self.mock_file
        
        output = reader.device_readable()
        self.assertTrue(output)

    @patch('builtins.input', side_effect=["8.slc", "1"])
    def test_read_header(self, mock_input):
        reader = slc_reader(self.file_path)
        reader.stream = self.mock_file
        
        output = reader.read_header()
        scale_output = reader.scale
        
        #self.assertTrue(output)
        self.assertEqual(scale_output, 1.0)

    @patch('builtins.input', side_effect=["8.slc", "1"])
    def test_visualize(self, mock_input):
        reader = slc_reader(self.file_path)
        reader.stream = self.mock_file
        
        reader.read_header()
        slices = reader.read_slices()

        try:
            with patch('slc_reader.vedo.Plotter.show', return_value=None):
                reader.visualize(slices)
        except Exception as e:
            self.fail(f"visualize() failed with exception {e}")

if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
