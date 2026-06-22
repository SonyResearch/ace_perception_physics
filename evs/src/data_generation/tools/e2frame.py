"""Event frame generator."""
# pylint: disable = line-too-long
# SPDX-License-Identifier: MIT

import numpy as np


def events_to_frame(events, width=1280, height=720):
    """
    Generates event frames.

    Returns event frames with values corresonding to scaled number of events.
    """
    event_frame = np.zeros((3, height, width), np.uint8)
    x_events = events[:, 1].astype(np.int32)
    y_events = events[:, 2].astype(np.int32)
    pols = events[:, 3]
    # Polarity should be +1 / -1

    for i, x_event in enumerate(x_events):
        if pols[i] == 1:
            event_frame[0, y_events[i], x_event] += 1
        else:
            event_frame[1, y_events[i], x_event] += 1

    return event_frame


def events_to_timesurface(events, width=1280, height=720):
    """
    Generates event frames.

    Returns event frames with values corresonding to scaled number of events.
    """
    event_frame = np.zeros((3, height, width))
    x_events = events[:, 1].astype(np.int32)
    y_events = events[:, 2].astype(np.int32)
    pols = events[:, 3]
    timestamps = events[:, 0]

    # Normalize timestamps to 1
    normalized_timestamps = (timestamps - timestamps[0]) / (max(timestamps) - timestamps[0])

    for i, x_event in enumerate(x_events):
        if pols[i] == 1:
            event_frame[0, y_events[i], x_event] = normalized_timestamps[i]
        else:
            event_frame[1, y_events[i], x_event] = normalized_timestamps[i]

    return event_frame
