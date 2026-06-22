"""
@brief Script for reformatting.

@file process_raw_files.py
@author Claudio Fanconic, Asude Aydin (maintainer: asude.aydin@sony.com)
@date 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import argparse
import os

from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from calibration import python_module as calibration
from evs.utilities.event_readers import FixedSizeTriggerEventReader

from evs.src.data_generation.tools.event_data_format import Events
from evs.src.data_generation.tools.h5_writer import H5Writer


def parse_args():
    """
    Parse input arguments.

    @return: Parsed argument struct
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evs_raw_dir",
        type=str,
        help="Path to folder containing evs raw recordings.",
        required=True,
    )

    parser.add_argument(
        "--h5_dir",
        type=str,
        help="Folder path for new h5 folders to be saved.",
        required=True,
    )

    parser.add_argument(
        "--metadata_path",
        type=str,
        help="Folder path for trigger and triangulated metadata.",
        required=True,
    )

    parser.add_argument(
        "--cam_params_path",
        type=str,
        help="Path to yaml file containing camera parameters.",
        required=True,
    )

    parser.add_argument(
        "--ball_radius",
        type=float,
        help="Radius of the ball",
        required=False,
        default=0.02,
    )

    parser.add_argument(
        "--accumulation_time",
        type=float,
        help="Accumulation time for the event representation",
        required=False,
        default=0.001,
    )

    args = parser.parse_args()

    return args


def get_csv_txt(metadata_path: Path):
    """
    Retrieve data from CSV and TXT files in a metadata directory.

    Args:
    - metadata_path (str): The path to the metadata directory.

    Returns:
    - dataframe (pandas DataFrame): DataFrame containing data from the 'points_3d.csv' file.
    - triggers (list): List of triggers read from the 'triggers.txt' file.
    - triggers_file_path (str): The full path to the 'triggers.txt' file.
    """

    # Fix to getting directly the sequence folder
    points_3d_file_path = metadata_path / "points_3d.csv"
    points_3d_all_file_path = metadata_path / "points_3d_all.csv"

    if points_3d_all_file_path.exists():
        points_3d_existing_path = points_3d_all_file_path
    elif points_3d_file_path.exists():
        points_3d_existing_path = points_3d_file_path
    else:
        raise FileNotFoundError("Neither points_3d.csv nor points_3d_all.csv was found.")

    triggers_file_path = metadata_path / "triggers.txt"

    assert points_3d_file_path.suffix == ".csv"
    assert triggers_file_path.suffix == ".txt"

    dataframe = pd.read_csv(points_3d_existing_path)

    triggers = []
    with open(triggers_file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Remove leading and trailing whitespace and add the value to the list
            triggers.append(int(line.strip()))

    return dataframe, triggers, triggers_file_path


def get_h5_event_path(new_cam_path: Path, sequences: list, seq_id: int, first_frameid: int) -> Path:
    """
    Generate path name for the event (.h5) file associated with a specific sequence.

    Args:
        new_cam_path (Path): The path to the new camera folder.
        sequences (list): A list of sequences.
        seq_id (int): The sequence ID counter.
        first_frameid: ID of first frame in recording

    Returns:
        Path: The path to the event (.h5) file.
    """
    new_cam_path.mkdir(parents=True, exist_ok=True)

    event_file_name = (
        f"seq_{seq_id:03d}_{sequences[seq_id][0]*5 + first_frameid}_{sequences[seq_id][1]*5 + first_frameid}.h5"
    )

    event_file_path = new_cam_path / event_file_name

    return event_file_path


def get_sequential_frames(dataframe: pd.DataFrame, metadata_path: Path):
    """Return consecutive frame ids where position is not null.

    Args:
        dataframe (pd.DataFrame): Pandas dataframe
        metadata_path (Path): CSV path to triangulated 3D points

    Returns:
        list, list, float: consecutive frame ids, sequences longer than length 10, first frame ids
    """
    # Find the consecutive frames where 'frame_id' is consecutive and 'position_x' is not null
    succesful_triangulations_path = metadata_path / "points_3d.csv"
    succesful_triangulations_df = dataframe[
        (dataframe["position_x"].notnull()) & (dataframe["position_y"].notnull()) & (dataframe["position_z"].notnull())
    ]
    succesful_triangulations_df.to_csv(succesful_triangulations_path, index=False)

    first_frameid = dataframe.frame_id.values[0]

    # Extract the trigger IDs to iterate throught the raw files
    gt_frameid = succesful_triangulations_df.frame_id.values.tolist()
    frame_ids = [int((i - first_frameid) / 5) for i in gt_frameid if not np.isnan(i)]

    # Initialize variables
    start_frame = None
    end_frame = None
    count = 0
    sequences = []
    # TODO fix -> directly write into sequence and filter out long ones + give long criteria as argument
    # Iterate through the sorted frame IDs
    for i, frame_num in enumerate(frame_ids):
        if start_frame is None:
            # Start of a new sequence
            start_frame = frame_num
            end_frame = frame_num
            count = 1
        else:
            if frame_num == end_frame + 1:
                # Current frame ID is consecutive, update end_frame and count
                end_frame = frame_num
                count += 1
            else:
                # Sequence ended, append to list and reset variables
                sequences.append((start_frame * 5 + first_frameid, end_frame * 5 + first_frameid, count))
                start_frame = frame_num
                end_frame = frame_num
                count = 1

    # Append the last sequence if it wasn't already added
    if start_frame is not None and end_frame is not None:
        sequences.append((start_frame * 5 + first_frameid, end_frame * 5 + first_frameid, count))

    long_sequences = []
    for seq in sequences:
        start_frame = seq[0]
        end_frame = seq[1]
        seq_count = seq[2]

        if seq_count >= 10:
            long_sequences.append(
                (int((start_frame - first_frameid) / 5), int((end_frame - first_frameid) / 5), seq_count)
            )

    return frame_ids, long_sequences, first_frameid


def get_metadata_dict(root_dir, cam_name):
    """
    Retrieve metadata information from a YAML file based on sequence and camera names.

    Args:
        root_dir (str): Root directory containing dataset.
        cam_name (str): The name of the camera.

    Returns:
        dict: A dictionary containing the following metadata information:
            - 'resolution': The resolution of the camera.
            - 'camera_matrix': The camera matrix.
            - 'distortion_coeffs': The distortion coefficients.
    """
    yaml_files = os.listdir(root_dir)

    for yaml_file in yaml_files:
        if yaml_file.endswith(".yaml"):
            yaml_file_path = os.path.join(root_dir, yaml_file)

    metadata_dict = {}

    calib_params = calibration.CameraCalibrationParameters()
    calib_params.initialize(yaml_file_path)

    metadata_dict["resolution"] = calib_params.cameras[cam_name].resolution
    metadata_dict["camera_matrix"] = calib_params.cameras[cam_name].camera_matrix
    metadata_dict["distortion_coeffs"] = calib_params.cameras[cam_name].distortion_coeffs
    return metadata_dict


def main():  # NOQA D103
    """Main script"""
    args = parse_args()
    evs_raw_dir = Path(args.evs_raw_dir)
    h5_dir = Path(args.h5_dir)
    metadata_path = Path(args.metadata_path)
    cam_config_path = Path(args.cam_params_path)

    # Check if in sequence folder
    dataframe, _, _ = get_csv_txt(metadata_path)

    # Get triangulation frame ids and uninterrupted sequences
    _, sequences, first_frameid = get_sequential_frames(dataframe, metadata_path)

    for raw_file_path in evs_raw_dir.glob("*.raw"):
        # Get params
        camera_name = raw_file_path.stem
        recording_name = evs_raw_dir.parent.stem

        # Initialize calibration parameter for backprojection
        calib_params = calibration.CameraCalibrationParameters()
        calib_params.initialize(str(cam_config_path))

        source_path_events = raw_file_path
        source_path_event_triggers = metadata_path / "triggers.txt"

        # Get metadata dictionary
        metadata_dict = get_metadata_dict(
            root_dir=metadata_path.parent.parent,
            cam_name=camera_name,
        )

        h5_file_path = h5_dir / raw_file_path.stem
        if h5_file_path.exists():
            continue

        # Initialize event reader
        event_window_iterator = FixedSizeTriggerEventReader(str(source_path_events), source_path_event_triggers)
        seq_id_counter = 0
        with tqdm(total=len(sequences), desc="Processing: {}/{}".format(recording_name, camera_name)) as pbar:
            for idx, event_window in enumerate(event_window_iterator):
                # If index equal to the beginning of a sequence
                if idx == sequences[seq_id_counter][0]:
                    h5_path_events = get_h5_event_path(h5_file_path, sequences, seq_id_counter, first_frameid)

                    # Initialize h5 event writer
                    h5writer = H5Writer(h5_path_events)

                    event_ts = event_window[:, 0]
                    selected_idx = np.where(event_window[:, 0] >= (event_ts[-1] - 0.001))

                    event_t0_us = event_window[:, 0][selected_idx][0] * 1e6
                    normalized_event_t = event_window[:, 0][selected_idx] * 1e6 - event_t0_us

                    events = Events(
                        event_window[:, 1][selected_idx].astype("uint16"),
                        event_window[:, 2][selected_idx].astype("uint16"),
                        event_window[:, 3][selected_idx].astype("uint8"),
                        normalized_event_t.astype("int64"),
                    )
                    h5writer.add_data(events)

                    metadata_dict["t0"] = event_t0_us
                    metadata_dict["frame_id_interval"] = (
                        sequences[seq_id_counter][0],
                        sequences[seq_id_counter][1],
                    )
                    h5writer.add_metadata(metadata_dict)

                # If index is in range of sequence
                elif sequences[seq_id_counter][1] + 1 > idx > sequences[seq_id_counter][0]:
                    normalized_event_t = event_window[:, 0][:] * 1e6 - event_t0_us
                    events = Events(
                        event_window[:, 1][:].astype("uint16"),
                        event_window[:, 2][:].astype("uint16"),
                        event_window[:, 3][:].astype("uint8"),
                        normalized_event_t.astype("int64"),
                    )
                    h5writer.add_data(events)

                # If index equal to end of sequence
                if idx == sequences[seq_id_counter][1]:
                    normalized_event_t = event_window[:, 0][:] * 1e6 - event_t0_us
                    events = Events(
                        event_window[:, 1][:].astype("uint16"),
                        event_window[:, 2][:].astype("uint16"),
                        event_window[:, 3][:].astype("uint8"),
                        normalized_event_t.astype("int64"),
                    )

                    h5writer.add_data(events)
                    seq_id_counter += 1

                    pbar.update(1)

                # Stop early if all sequences have been processed
                if seq_id_counter == len(sequences):
                    break


if __name__ == "__main__":
    main()
