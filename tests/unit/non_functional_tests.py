import unittest
from unittest.mock import patch, MagicMock
import sys, os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Combined')))
from combined import main
from stl_algo import stl_main
from slc_support_structure import slc_main



class TestPerformance(unittest.TestCase):
    
    
    @patch('stl_algo.result_saver')  # Patch where result_saver is used, in the stl_algo module
    @patch('builtins.input', side_effect=['8.stl', '', '', '', '', '', '', '', '', 'N'])
    @patch('vedo.Plotter')  # Mock the entire Plotter class to prevent visualizations
    def test_8_stl_time(self, mock_plotter, mock_input, mock_saver):
 
        # Start timing
        start_time = time.time()
        
        # Mock result_saver to avoid saving the result during testing
        mock_saver.return_value = None

        # Mock the Plotter instance and its methods
        mock_plotter_instance = MagicMock()
        mock_plotter.return_value = mock_plotter_instance
        mock_plotter_instance.show.return_value = None  # Ensure show() does nothing

        # Run the combined main function which handles both STL and SLC files
        main()
        
        # Stop timing
        end_time = time.time()
        duration = end_time - start_time
        print(f"8.stl has finished processing in {duration:.2f} seconds")
        
        # Ensure that the result_saver was called and that 'N' was entered to not save
        mock_saver.assert_called_once()

        # Ensure that the Plotter was instantiated
        mock_plotter.assert_called()

        # Check that the input function was called for each input
        self.assertEqual(mock_input.call_count, 9)
  

    @patch('stl_algo.result_saver')  # Patch where result_saver is used, in the stl_algo module
    @patch('builtins.input', side_effect=['star.stl', '', '', '', '', '', '', '', '', 'N'])
    @patch('vedo.Plotter')  # Mock the entire Plotter class to prevent visualizations
    def test_star_stl_time(self, mock_plotter, mock_input, mock_saver):
 
        # Start timing
        start_time = time.time()
        
        # Mock result_saver to avoid saving the result during testing
        mock_saver.return_value = None

        # Mock the Plotter instance and its methods
        mock_plotter_instance = MagicMock()
        mock_plotter.return_value = mock_plotter_instance
        mock_plotter_instance.show.return_value = None  # Ensure show() does nothing

        # Run the combined main function which handles both STL and SLC files
        main()
        
        # Stop timing
        end_time = time.time()
        duration = end_time - start_time
        print(f"star.stl has finished processing in {duration:.2f} seconds")
        
        # Ensure that the result_saver was called and that 'N' was entered to not save
        mock_saver.assert_called_once()

        # Ensure that the Plotter was instantiated
        mock_plotter.assert_called()

        # Check that the input function was called for each input
        self.assertEqual(mock_input.call_count, 9)


    @patch('builtins.input', side_effect=['8.slc', '', '', '', '', '', ''])
    @patch('vedo.Plotter.show')  # Only patch the 'show' method of the Plotter class
    def test_8_slc_time(self, mock_show, mock_input):
        
        start_time = time.time()
        
        # Mock 'show()' to prevent the visualization window from appearing
        mock_show.return_value = None

        # Run the combined main function which handles both STL and SLC files
        main()
        
        # Stop timing
        end_time = time.time()
        duration = end_time - start_time
        print(f"8.slc has finished processing in {duration:.2f} seconds")
        
        # Ensure that the input function was called for each input
        self.assertEqual(mock_input.call_count, 6)

        # Ensure that the 'show()' method was called to confirm that visualization was triggered
        mock_show.assert_called()
        
    @patch('builtins.input', side_effect=['star.slc', '', '', '', '', '', ''])
    @patch('vedo.Plotter.show')  # Only patch the 'show' method of the Plotter class
    def test_star_slc_time(self, mock_show, mock_input):

        start_time = time.time()
        
        # Mock 'show()' to prevent the visualization window from appearing
        mock_show.return_value = None

        # Run the combined main function which handles both STL and SLC files
        main()
        
        # Stop timing
        end_time = time.time()
        duration = end_time - start_time
        print(f"star.slc has finished processing in {duration:.2f} seconds")
        
        # Ensure that the input function was called for each input
        self.assertEqual(mock_input.call_count, 6)

        # Ensure that the 'show()' method was called to confirm that visualization was triggered
        mock_show.assert_called()
        
if __name__ == '__main__':
    unittest.main()