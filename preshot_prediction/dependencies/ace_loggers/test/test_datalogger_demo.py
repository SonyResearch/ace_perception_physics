#!/usr/bin/env python3
"""
Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import contextlib
import os
import sys

import numpy
import pytest
from ace_loggers import datalogger_demo_pybind as datalogger


def test_ineritance():
    file_name = "test_ineritance.ace"
    data_str = "some text string!"

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)

    with datalogger.DataWriter(file_name) as writer:
        writer.write(data_str)

    with datalogger.DataReader(file_name) as reader:
        assert data_str == reader.read_str()


def test_primitive():
    file_name = "test_primitive.ace"
    data_struct = datalogger.HasPrimitivesStruct(numpy.random.randint(0, 1000), numpy.random.rand())

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name) as writer:
        writer.write(data_struct)

    with datalogger.DataReader(file_name) as reader:
        loaded_struct = datalogger.HasPrimitivesStruct()
        reader.read(loaded_struct)
        assert data_struct.var_int == loaded_struct.var_int
        assert data_struct.var_float == pytest.approx(loaded_struct.var_float)


def test_overriding():
    file_name = "test_overriding.ace"
    data_struct = datalogger.HasPointerStruct(numpy.random.randint(0, 1000), numpy.random.rand(), "random text!")

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name) as writer:
        writer.write(data_struct)

    with datalogger.DataReader(file_name) as reader:
        loaded_struct = datalogger.HasPointerStruct()
        reader.read(loaded_struct)
        assert data_struct.var_int == loaded_struct.var_int
        assert data_struct.var_float == pytest.approx(loaded_struct.var_float)
        assert data_struct.var_str == loaded_struct.var_str


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
