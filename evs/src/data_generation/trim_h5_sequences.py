# pylint: disable=undefined-loop-variable
# TODO(asude): clean pylint
"""
@brief A script for data post processing to remove events and corresponding labels from h5 files.

@file trim_h5_sequences
@author Asude Aydin (asude.aydin@sony.com)
@date 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import csv
import os
from pathlib import Path

import fire
import h5py
from data_generation.tools.event_data_format import Events
from data_generation.tools.h5_writer import H5Writer, H5WriterLabel


def find_filenames_by_sequence(directory, sequence_number):
    """Finds the filename with the given sequence number in directory.

    Args:
        directory (str): Directory containing h5 files.
        sequence_number (float): Number of sequence.

    Returns:
        list: List containing paths of file names with matching sequence number.
    """
    # Format the sequence number to match the file naming convention
    sequence_formatted = f"seq_{int(sequence_number):03d}"  # Pads the sequence number with zeros

    # Initialize lists to hold the matching filenames
    data_filenames = []
    label_filenames = []

    # Walk through the directory
    for filename in os.listdir(directory):
        # Check if the filename starts with the formatted sequence number
        if filename.startswith(sequence_formatted):
            # Check if it's a label file or a data file
            if filename.endswith("_label.h5"):
                label_filenames.append(filename)
            elif filename.endswith(".h5"):
                data_filenames.append(filename)

    # Return the full paths of the data and label files
    h5_path = os.path.join(directory, data_filenames[0])
    h5_label_path = os.path.join(directory, label_filenames[0])
    return h5_path, h5_label_path


def read_csv_files(file_path: str):
    """Read csv file from given path and return list containing column values.

    Args:
        file_path (str): Path to the csv file

    Returns:
        list: Lists containing values for relevant columns.
    """
    # Initialize lists for each column
    recording_names = []
    camera_names = []
    sequence_numbers = []
    trim_starts = []
    trim_ends = []

    # Open the CSV file for reading
    with open(file_path, "r", encoding="UTF-8") as file:
        # Create a CSV reader object using a comma as the delimiter
        csv_reader = csv.reader(file, delimiter=",")

        # Skip the header line
        next(csv_reader, None)

        # Iterate over the rows in the CSV file
        for row in csv_reader:
            assert len(row) == 5

            # Skip empty rows or rows that are headers repeated
            if not row or "Recording name" in row[0]:
                continue

            # Append data from each row to the respective lists
            recording_names.append(row[0].strip())
            camera_names.append(row[1].strip())
            sequence_numbers.append(row[2].strip())
            trim_starts.append(row[3].strip() if row[3].strip() != "" else 0)
            trim_ends.append(row[4].strip() if row[4].strip() != "" else -1)

    assert len(recording_names) == len(camera_names) == len(sequence_numbers) == len(trim_starts) == len(trim_ends)

    return recording_names, camera_names, sequence_numbers, trim_starts, trim_ends


def trim_h5(old_data_root: str, new_data_root: str, trim_csv_path: str):
    """
    This function loads a sequence from an H5 file, trims it to only include data between
    the specified start and end milliseconds, and saves the new trimmed dataset
    to the output path provided.

    Parameters:
        old_data_root: Root directory of h5 sequences to be trimmed.
        new_data_root: Root directory of trimmed h5 sequences to be saved.
        trim_csv_path: Path to csv file containing metadata on trim durations.

    Returns:
    - None

    """

    recording_names, camera_names, sequence_numbers, trim_starts, trim_ends = read_csv_files(trim_csv_path)

    for i, rec_name in enumerate(recording_names):
        print(i, rec_name, camera_names[i], sequence_numbers[i], trim_starts[i], trim_ends[i])
        seq_root = Path(old_data_root) / rec_name / camera_names[i]

        # Create new h5 folder if it doesn't exist
        new_h5_folder = Path(new_data_root) / rec_name / camera_names[i]
        new_h5_folder.mkdir(parents=True, exist_ok=True)

        old_h5_path, old_h5_label_path = find_filenames_by_sequence(seq_root, sequence_numbers[i])

        assert Path(old_h5_path).exists(), "Previous h5 event file doesn't exist."
        assert Path(old_h5_label_path).exists(), "Previous h5 label file doesn't exist."

        # Check if new h5 file already exists
        new_h5_path = new_h5_folder / Path(old_h5_path).name
        if new_h5_path.exists():
            continue

        # Read from old h5 event and label files
        old_h5 = h5py.File(old_h5_path, "r")
        old_h5_label = h5py.File(old_h5_label_path, "r")

        start_trim_ms = int(trim_starts[i])
        end_trim_ms = int(trim_ends[i])

        remove_idx_start = old_h5["ms_to_idx"][start_trim_ms]
        remove_idx_end = old_h5["ms_to_idx"][end_trim_ms]

        new_events = {
            "t": old_h5["events"]["t"][remove_idx_start : remove_idx_end + 1]
            - int(start_trim_ms * 1e3),  # New bug fix -> not tested yet
            "x": old_h5["events"]["x"][remove_idx_start : remove_idx_end + 1],
            "y": old_h5["events"]["y"][remove_idx_start : remove_idx_end + 1],
            "p": old_h5["events"]["p"][remove_idx_start : remove_idx_end + 1],
        }

        new_t0 = old_h5["t0"][0] + old_h5["events"]["t"][remove_idx_start]

        # Save metadata
        metadata_dict = {}
        metadata_dict["t0"] = new_t0
        metadata_dict["trim_start_index"] = start_trim_ms
        metadata_dict["trim_end_index"] = end_trim_ms

        # Initialize new h5 file
        h5writer = H5Writer(new_h5_path)

        # Convert from dict to Events class for h5 writer
        events_class = Events(new_events["x"], new_events["y"], new_events["p"], new_events["t"])

        # Write the new h5 data
        h5writer.add_data(events_class)
        h5writer.add_metadata(metadata_dict)

        new_label_dict = {}
        for key_name in old_h5_label.keys():
            if end_trim_ms == -1:
                new_label_dict[key_name] = old_h5_label[key_name][start_trim_ms:]
            else:
                new_label_dict[key_name] = old_h5_label[key_name][start_trim_ms:end_trim_ms]

        # Initialize the new h5 label writer
        new_h5_label_path = new_h5_folder / Path(old_h5_label_path).name
        h5_writer_label = H5WriterLabel(new_h5_label_path, default_shape=new_label_dict[key_name].shape[0])

        h5_writer_label.add_data(new_label_dict)


def main():
    """Main script."""
    fire.Fire(trim_h5)


if __name__ == "__main__":
    main()
