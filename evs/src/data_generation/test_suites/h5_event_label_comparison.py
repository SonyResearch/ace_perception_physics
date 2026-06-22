"""
@brief This class contains unit tests for comparing H5 label event files with their corresponding event files.

@author Asude Aydin (asude.aydin@sony.com)
@date 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import unittest
import os
import h5py
import sys


class H5LabelEventComparison(unittest.TestCase):
    """
    This class contains unit tests for comparing H5 label event files with their corresponding event files.

    Attributes:
        None

    Methods:
        test_h5_event_label_comparison: Test method to compare H5 label event files with their event files.
    """

    def test_h5_event_label_comparison(self):
        """Test method to compare H5 label event files with their corresponding event files."""
        # Traverse through all subdirectories in the mat_base_dir
        for root, dirs, files in os.walk(self.H5_BASE_DIR):
            for cur_dir in dirs:
                h5_dir = os.path.join(root, cur_dir)

                # Check if the directory contains any files
                if not os.listdir(h5_dir):
                    continue  # Skip empty folders

                # Count ms_to_idx arrays in h5 files
                label_h5_files = sorted(
                    [f for f in os.listdir(h5_dir) if f.endswith("label.h5")]
                )
                for label_h5_filename in label_h5_files:
                    event_h5_filename = label_h5_filename.replace("_label", "")
                    event_h5 = h5py.File(os.path.join(h5_dir, event_h5_filename), "r")
                    label_h5 = h5py.File(os.path.join(h5_dir, label_h5_filename), "r")

                    self.assertEqual(
                        len(label_h5["points"]), len(label_h5["points_orig"])
                    )
                    self.assertEqual(
                        len(label_h5["radius"]), len(label_h5["radius_orig"])
                    )
                    self.assertEqual(
                        len(label_h5["velocities"]), len(label_h5["points"])
                    )
                    self.assertEqual(
                        len(event_h5["events/x"]), len(event_h5["events/y"])
                    )
                    self.assertEqual(
                        len(event_h5["events/t"]), len(event_h5["events/p"])
                    )

                    end_seq = int(label_h5_filename.split("_")[3])
                    start_seq = int(label_h5_filename.split("_")[2])

                    last_ms = len(label_h5["points"]) - 1
                    last_ms_idx = event_h5["ms_to_idx"][last_ms]

                    last_bin_duration = (
                        event_h5["events/t"][-1] - event_h5["events/t"][last_ms_idx]
                    )
                    # There can be upto 0.8 ms events missing or extra
                    self.assertAlmostEqual(last_bin_duration, 1000, delta=800)

                    total_duration_seq = round((end_seq - start_seq) / 2 + 1)  # ms
                    total_duration_t = round(
                        (event_h5["events/t"][-1] - event_h5["events/t"][0]) / 1e3
                    )

                    self.assertAlmostEqual(
                        total_duration_seq, total_duration_t, delta=1
                    )

    def test_h5_event_label_counts(self):
        """Test method to compare count of H5 label and event files."""

        label_count = 0
        event_count = 0
        for root, dirs, files in os.walk(self.H5_BASE_DIR):
            for file in files:
                if file.endswith("label.h5"):
                    label_count += 1
                elif file.endswith("h5"):
                    event_count += 1
        self.assertEqual(event_count, label_count)
        self.assertNotEqual(event_count, 0)


if __name__ == "__main__":
    H5LabelEventComparison.H5_BASE_DIR = sys.argv.pop()
    unittest.main()
