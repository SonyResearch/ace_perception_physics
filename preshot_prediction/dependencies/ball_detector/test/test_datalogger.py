#!/usr/bin/env python3
"""
Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import os
import pickle
import sys

import numpy
import pytest
from ball_detector import datalogger_utils
from rosbag_dataloaders import dataloaders

UPDATE_DATA = False


def compare(a_var, b_var):
    """
    Recursive equality comparison between two variables
    """
    if isinstance(a_var, str):
        return a_var == b_var
    if isinstance(a_var, dict):
        if len(a_var) != len(b_var):
            return False
        for key in a_var.keys():
            if not compare(a_var[key], b_var.get(key)):
                return False
        return True
    if hasattr(a_var, "__iter__"):
        if len(a_var) != len(b_var):
            return False
        for a_item, b_item in zip(a_var, b_var):
            if not compare(a_item, b_item):
                return False
        return True
    if isinstance(a_var, (float, numpy.float32, numpy.float64)):
        return numpy.isclose(a_var, b_var, equal_nan=True)
    return a_var == b_var


@pytest.mark.parametrize(
    "log_file_path",
    [
        os.path.join(os.path.realpath(os.path.dirname(__file__)), "data", "ball_detector.ace"),
    ],
)
def test_log_reading(log_file_path):
    """
    Test loading a complete log file
    """
    pickle_file_path = log_file_path.replace(".ace", ".pkl", 1)

    data_list = list(datalogger_utils.read_log_file(log_file_path))

    if UPDATE_DATA:
        with open(pickle_file_path, "wb") as file:
            pickle.dump(data_list, file)
    else:
        with open(pickle_file_path, "rb") as file:
            data_list_ref = pickle.load(file)
            assert compare(data_list_ref, data_list)


@pytest.mark.parametrize(
    "log_file_path",
    [
        os.path.join(os.path.realpath(os.path.dirname(__file__)), "data", "ball_detector.ace"),
    ],
)
def test_ros_conversion(log_file_path):
    """
    Test converting a log file into ROS2 messages
    """
    data_list = list(datalogger_utils.read_log_file(log_file_path))

    parser = dataloaders.get_parser(log_file_path)
    topic_names = list(parser.get_topics())
    assert len(topic_names) > 0
    for topic_name in topic_names:
        messages = list(parser.get_messages(topic_name))
        assert len(data_list) == len(messages)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
