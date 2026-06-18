#!/usr/bin/env python3
"""
Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import contextlib
import os
import pickle
import sys

import numpy
import pytest
from rosbag_dataloaders import dataloaders
from triangulation import datalogger_pybind as datalogger
from triangulation import datalogger_utils
from triangulation import python_module as triangulation

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


def test_triangulated_point():
    """
    Test writing and reading TriangulatedPoint structure
    """
    file_name = "test.ace"
    data_struct = triangulation.TriangulatedPoint()
    data_struct.position = numpy.random.rand(3)
    data_struct.covariance = numpy.random.rand(3, 3)

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name, "", "") as writer:
        writer.write(data_struct)

    with datalogger.DataReader(file_name) as reader:
        loaded_struct = triangulation.TriangulatedPoint()
        reader.read(loaded_struct)
        assert numpy.all(data_struct.position == loaded_struct.position)
        assert numpy.all(data_struct.covariance == loaded_struct.covariance)


@pytest.mark.parametrize(
    "log_file_path",
    [
        os.path.join(os.path.realpath(os.path.dirname(__file__)), "data", "triangulation.ace"),
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
        os.path.join(os.path.realpath(os.path.dirname(__file__)), "data", "triangulation.ace"),
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
