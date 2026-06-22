"""
@brief Sanity check for matching length between h5 labels and events from EVS.

@file sequence_counter.py
@author Asude Aydin
@date August 2024
@version 0.0
@copyright SPDX-License-Identifier: MIT
"""


import os
import sys
import unittest
from pathlib import Path

import h5py


def count_label_elements(root_dir):
    """Count and compare label and event lengths."""
    results = []
    tot_ms_counter = 0
    camera_ms_counter = {}
    for root, _, files in os.walk(root_dir):
        files = os.listdir(root)
        if any(file.endswith(".h5") for file in files):
            subfolder_name = Path(root).relative_to(root_dir)
            camera_name = subfolder_name.name

            if camera_name not in camera_ms_counter.keys():
                camera_ms_counter[subfolder_name.name] = 0

            sequence_ms_counter = 0
            # Iterate over all files in the given directory
            for label_filename in sorted(os.listdir(root)):
                # Check if the file ends with '_label.h5'
                if label_filename.endswith("_label.h5"):
                    label_filepath = os.path.join(root, label_filename)

                    # Open the file and read the "label" branch
                    with h5py.File(label_filepath, "r") as h5file:
                        label_length = len(h5file["points"])
                        sequence_ms_counter += len(h5file["points"])

                    event_filepath = os.path.join(root, label_filename.replace("_label", ""))

                    with h5py.File(event_filepath, "r") as h5file:
                        ms_to_idx_length = len(h5file["ms_to_idx"])

                    results.append((subfolder_name, label_length, ms_to_idx_length))

                    camera_ms_counter[subfolder_name.name] += label_length

                    assert (
                        label_length - 1 == ms_to_idx_length
                        or label_length == ms_to_idx_length
                        or label_length == ms_to_idx_length - 1
                    )

            print(subfolder_name.name, sequence_ms_counter)
            tot_ms_counter += sequence_ms_counter
    print(camera_ms_counter)
    print("Total ms: ", tot_ms_counter)
    return results


class TestCountLabelElements(unittest.TestCase):
    """
    Unit test for the count_label_elements function.
    """

    def setUp(self):
        """Sets up the test environment by initializing the root directory."""

        self.root_dir = self.ROOT_DIR

    def test_count_label_elements(self):
        """Tests that the label lengths and ms_to_idx lengths meet the expected conditions."""
        results = count_label_elements(self.root_dir)
        for subfolder_name, label_length, ms_to_idx_length in results:
            self.assertTrue(
                label_length - 1 == ms_to_idx_length
                or label_length == ms_to_idx_length
                or label_length == ms_to_idx_length - 1,
                f"Failed for {subfolder_name} with label length {label_length} and ms_to_idx length {ms_to_idx_length}",
            )


if __name__ == "__main__":
    TestCountLabelElements.ROOT_DIR = sys.argv.pop()

    unittest.main()
