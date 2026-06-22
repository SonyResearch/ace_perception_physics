"""
@brief Main script for labeling proffesional player data.

@file label_pro_player_data.py
@author Asude Aydin (asude.aydin@sony.com)
@date 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from termcolor import cprint
from tqdm import tqdm

from evs.utilities.event_readers import FixedSizeTriggerEventReader

from data_generation.tools.event_data_format import Events
from data_generation.tools.extract_aps_triangulations import extract_first_aps_frameid
from data_generation.tools.extract_triggers import extract_triggers
from data_generation.tools.h5_writer import H5Writer


def parse_args():
    """
    Parse input arguments.

    @return: Parsed argument struct
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/media/chaydina/T71/asude_overhead_evs_recordings/tmp/20240528_144758_sakamoto_vs_murano_set_1",
        help="Path to folder containing calibration file and evs, labels, and rosbag folder.",
        # required=True,
    )

    parser.add_argument(
        "--cam_name",
        default="evs00050028",
        type=str,
        help="Name of the EVS camera",
        # required=True,
    )

    args = parser.parse_args()

    return args


def round_to_nearest_increment(value, increment=0.005):
    """Round to nearest increment."""
    scale = int(1 / increment)
    return round(value * scale) / scale


def main():  # NOQA D103
    """Main script."""
    args = parse_args()

    root_dir = Path(args.root_dir)

    rosbag_root_dir = root_dir / "rosbag"
    rosbag_files = list(rosbag_root_dir.glob("*.db3"))

    assert len(rosbag_files) == 1

    label_root_dir = Path(args.root_dir) / "labels"
    label_files = list(label_root_dir.glob("*.pt"))

    assert label_files

    config_files = list(root_dir.glob("*.yaml"))

    assert len(config_files) == 1

    metadata_root_dir = Path(args.root_dir) / "metadata" / args.cam_name
    metadata_root_dir.mkdir(parents=True, exist_ok=True)

    cam_name = args.cam_name

    source_path_event_triggers = Path(args.root_dir) / "evs" / "triggers.txt"
    if source_path_event_triggers.exists():
        cprint(
            f"Trigger file already exists at {source_path_event_triggers}",
            color="green",
        )
    else:
        extract_triggers(args)  # 1
        cprint(f"Trigger file created at {source_path_event_triggers}", color="green")

    # Read start time of first APS frame from txt file.
    start_time_file = Path(args.root_dir) / "start_time.txt"
    if start_time_file.exists():
        with open(start_time_file, "r", encoding="utf-8") as file:
            start_time = float(file.read().strip())
    else:
        cprint(
            f"Start time file not found at {start_time_file}. Extracting from rosbag.",
            color="yellow",
        )
        start_time = extract_first_aps_frameid(str(rosbag_files[0]), root_dir)  # 1
    cprint(f"Start time of first APS frame: {start_time}", color="green")

    source_path_events = Path(args.root_dir) / "evs" / f"{cam_name}.raw"
    assert source_path_events.exists()

    h5_root_dir = Path(args.root_dir) / "h5" / args.cam_name

    ####################################### EVENTS #########################################

    metadata_files = sorted(list(metadata_root_dir.glob("*.csv")))
    pbar = tqdm(metadata_files)

    for metadata_file in pbar:
        pbar.set_description("Processing %s" % metadata_file.name)

        dataframe = pd.read_csv(metadata_file)

        first_label_ts = dataframe["first_label_timestamp"].unique()[0]
        last_label_ts = dataframe["last_label_timestamp"].unique()[0]
        label_name = dataframe["label_name"].unique().item()
        seq_id = dataframe["sequence_id"].unique().item()

        # Get unique frame ids
        label_timestamps = np.arange(first_label_ts, last_label_ts + 0.005, 0.005)

        gt_frameid = [
            int(np.round((i - start_time) * 1e3) / 5)
            for i in label_timestamps
            if not np.isnan(i)
        ]

        assert (np.diff(gt_frameid) == 1).all()

        # Initialize h5 event writer
        h5_path_events = h5_root_dir / f"seq_{int(seq_id):03d}_{label_name}.h5"
        h5_path_label = h5_root_dir / f"seq_{int(seq_id):03d}_{label_name}_label.h5"
        assert h5_path_label.exists()

        if h5_path_events.exists():
            continue

        event_window_iterator = FixedSizeTriggerEventReader(
            str(source_path_events), source_path_event_triggers
        )

        assert gt_frameid[-1] < event_window_iterator.num_trigger_events

        # Iterate through event windows and add to h5 file
        h5writer = H5Writer(h5_path_events)
        metadata_dict = {}
        t0_failed = False
        for idx, event_window in enumerate(event_window_iterator):
            if idx in gt_frameid:
                if idx == gt_frameid[0]:
                    event_ts = event_window[:, 0]

                    selected_idx = np.where(event_ts >= (event_ts[-1] - 0.001))

                    if not selected_idx[0].size:
                        t0_failed = True
                        cprint(
                            f"No events found in the last 1ms of the first frame for {metadata_file.name}.",
                            color="yellow",
                        )
                        continue

                    event_t0_us = event_window[:, 0][selected_idx][0] * 1e6
                    metadata_dict["t0"] = event_t0_us
                    h5writer.add_metadata(metadata_dict)

                    normalized_event_t = event_ts[selected_idx] * 1e6 - event_t0_us
                    events = Events(
                        event_window[:, 1][selected_idx].astype("uint16"),
                        event_window[:, 2][selected_idx].astype("uint16"),
                        event_window[:, 3][selected_idx].astype("uint8"),
                        normalized_event_t.astype("int64"),
                    )
                    h5writer.add_data(events)

                else:
                    if t0_failed:
                        event_t0_us = event_window[:, 0][0] * 1e6
                        metadata_dict["t0"] = event_t0_us
                        h5writer.add_metadata(metadata_dict)
                        t0_failed = False

                    normalized_event_t = event_window[:, 0][:] * 1e6 - event_t0_us
                    events = Events(
                        event_window[:, 1][:].astype("uint16"),
                        event_window[:, 2][:].astype("uint16"),
                        event_window[:, 3][:].astype("uint8"),
                        normalized_event_t.astype("int64"),
                    )
                    h5writer.add_data(events)

                if idx == gt_frameid[-1]:
                    break

            # if idx == gt_frameid[0]:

            #         event_ts = event_window[:, 0]

            #         first_time_diff = round_to_nearest_increment(
            #             (first_aps_ts - first_label_ts), 0.001
            #         )
            #         selected_idx = np.where(
            #             event_ts >= (event_ts[-1] - 0.001)
            #         )

            #         event_t0_us = event_window[:, 0][selected_idx][0] * 1e6
            #         normalized_event_t = event_ts[selected_idx] * 1e6 - event_t0_us

            #         events = Events(
            #             event_window[:, 1][selected_idx].astype("uint16"),
            #             event_window[:, 2][selected_idx].astype("uint16"),
            #             event_window[:, 3][selected_idx].astype("uint8"),
            #             normalized_event_t.astype("int64"),
            #         )
            #         h5writer.add_data(events)

            #         metadata_dict["t0"] = event_t0_us
            #         h5writer.add_metadata(metadata_dict)

            #     # If index is in range of sequence
            #     elif gt_frameid[-1] + 1 > idx > gt_frameid[0]:
            #         normalized_event_t = event_window[:, 0][:] * 1e6 - event_t0_us
            #         events = Events(
            #             event_window[:, 1][:].astype("uint16"),
            #             event_window[:, 2][:].astype("uint16"),
            #             event_window[:, 3][:].astype("uint8"),
            #             normalized_event_t.astype("int64"),
            #         )
            #         h5writer.add_data(events)

            #     # If index equal to end of sequence
            #     if idx == gt_frameid[-1]:
            #         last_time_diff = round_to_nearest_increment(
            #             (last_label_ts - last_aps_ts), 0.001
            #         )

            #         if last_time_diff == 0:
            #             normalized_event_t = event_window[:, 0][:] * 1e6 - event_t0_us

            #             events = Events(
            #                 event_window[:, 1][:].astype("uint16"),
            #                 event_window[:, 2][:].astype("uint16"),
            #                 event_window[:, 3][:].astype("uint8"),
            #                 normalized_event_t.astype("int64"),
            #             )

            #         else:
            #             event_ts = event_window[:, 0]

            #             selected_idx = np.where(
            #                 event_ts <= (event_ts[0] + last_time_diff)
            #             )
            #             normalized_event_t = event_ts[selected_idx] * 1e6 - event_t0_us

            #             events = Events(
            #                 event_window[:, 1][selected_idx].astype("uint16"),
            #                 event_window[:, 2][selected_idx].astype("uint16"),
            #                 event_window[:, 3][selected_idx].astype("uint8"),
            #                 normalized_event_t.astype("int64"),
            #             )

            #         h5writer.add_data(events)

            #         break

    ####################################### EVENTS #########################################


if __name__ == "__main__":
    main()
