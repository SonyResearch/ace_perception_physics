# Confidential, Copyright 2024, Sony AI, All rights reserved.
# pylint: disable = wrong-import-position, unused-variable, consider-using-with, line-too-long
"""Event reader. Iterates over events according to duration or event size."""

import os
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd
from metavision_core.event_io.raw_reader import RawReader

root_dir_path = pathlib.Path(
    __file__
).parent.parent.parent.parent.parent.parent.parent.parent.parent.absolute()
e2vid_path = os.path.join(root_dir_path, "ext/e2vid")
# print(e2vid_path)
sys.path.append(e2vid_path)


def get_ext_trigger_timestamps(path_to_trigger_file):
    """
    Get the timestamps of special events.
    @return: Timestamps for external trigger events
    """

    trigger_file = open(path_to_trigger_file, "r", encoding="utf-8")
    timestamps = [int(ts) for ts in trigger_file.read().splitlines()]
    trigger_file.close()

    return {"t": timestamps}


class FixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, start_index=0):
        print("Will use fixed size event windows with {} events".format(num_events))
        print("Output frame rate: variable")
        self.iterator = pd.read_csv(
            path_to_event_file,
            delim_whitespace=True,
            header=None,
            names=["t", "x", "y", "pol"],
            dtype={"t": np.float64, "x": np.int16, "y": np.int16, "pol": np.int16},
            engine="c",
            skiprows=start_index + 1,
            chunksize=num_events,
            nrows=None,
            memory_map=True,
        )

    def __iter__(self):
        return self

    def __next__(self):
        # with Timer("Reading event window from file"):
        event_window = self.iterator.__next__().values
        return event_window


class FixedDurationEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, duration_ms=50.0, start_index=0):
        print(
            "Will use fixed duration event windows of size {:.2f} ms".format(
                duration_ms
            )
        )
        print("Output frame rate: {:.1f} Hz".format(1000.0 / duration_ms))
        file_extension = os.path.splitext(path_to_event_file)[1]
        assert file_extension in [".txt", ".zip"]
        self.is_zip_file = file_extension == ".zip"

        if self.is_zip_file:  # '.zip'
            self.zip_file = zipfile.ZipFile(path_to_event_file)
            files_in_archive = self.zip_file.namelist()
            assert (
                len(files_in_archive) == 1
            )  # make sure there is only one text file in the archive
            self.event_file = self.zip_file.open(files_in_archive[0], "r")
        else:
            self.event_file = open(path_to_event_file, "r", encoding="utf-8")

        # ignore header + the first start_index lines
        for i in range(1 + start_index):
            self.event_file.readline()

        self.last_stamp = None
        self.duration_s = duration_ms / 1000.0

    def __iter__(self):
        return self

    def __del__(self):
        if self.is_zip_file:
            self.zip_file.close()

        self.event_file.close()

    def __next__(self):
        # with Timer("Reading event window from file"):
        event_list = []
        for line in self.event_file:
            if self.is_zip_file:
                line = line.decode("utf-8")
            t_s, x_px, y_px, pol = line.split(" ")
            t_s, x_px, y_px, pol = float(t_s), int(x_px), int(y_px), int(pol)
            event_list.append([t_s, x_px, y_px, pol])
            if self.last_stamp is None:
                self.last_stamp = t_s
            if t_s > self.last_stamp + self.duration_s:
                self.last_stamp = t_s
                event_window = np.array(event_list)
                return event_window

        raise StopIteration


class FixedSizeTriggerEventReader:
    """
    Reads events from a '.raw' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    """

    def __init__(self, path_to_event_file, path_to_trigger_file):
        print("Will use fixed duration event windows of size  ms")
        file_extension = os.path.splitext(path_to_event_file)[1]
        assert file_extension in [".raw"]

        self.record_raw = RawReader(path_to_event_file, max_events=500000000)
        self.last_stamp = 0
        self.count_trigger = 0

        # acquire the special events
        self.ext_triggers = get_ext_trigger_timestamps(path_to_trigger_file)
        self.num_trigger_events = len(self.ext_triggers["t"])
        print("Number of trigger events: {}".format(len(self.ext_triggers["t"])))

    def __iter__(self):
        return self

    def __del__(self):
        print("Done")

    def __next__(self):
        # with Timer("Reading event window from file"):
        while not self.record_raw.is_done():
            try:
                trigger_ts = self.ext_triggers["t"][self.count_trigger]
            except IndexError as error:
                print(error)
                continue
            events = self.record_raw.load_delta_t(trigger_ts - self.last_stamp)
            self.last_stamp = self.ext_triggers["t"][self.count_trigger]
            self.count_trigger += 1
            event_array = np.array(
                [events["t"] / 1000000, events["x"], events["y"], events["p"]]
            )
            event_window = event_array.transpose()
            # print("Time intervall: {}us".format(self.ext_triggers["t"][self.count_trigger] - self.last_stamp))
            # print("Progress bar: {}%".format(100.0 * self.count_trigger / len(self.ext_triggers["t"])))
            return event_window

        raise StopIteration
