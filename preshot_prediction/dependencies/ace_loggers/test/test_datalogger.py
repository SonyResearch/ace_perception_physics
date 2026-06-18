#!/usr/bin/env python3
"""
Confidential, Copyright 2025, Sony AI, All rights reserved.
"""

import contextlib
import os
import re
import sys
import zoneinfo
from datetime import datetime

import ace_loggers
import numpy
import pytest
from ace_loggers import datalogger_pybind as datalogger


def _write_mat_3d(writer, data_mat3d):
    for data_mat2d in data_mat3d:
        writer.write(data_mat2d)


def test_metadata():
    metadata = {
        "module_name": "ace_loggers",
        "log_version": "0.0.0",
        "test_none": None,
        "test_bool1": True,
        "test_bool2": False,
        "test_int": 1,
        "test_float": 1.0,
        "test_list": ["1", 1, 1.0],
        "test_dict": {"key": "value", "another": 1.0, "third": False},
    }
    file_name = "test_metadata.ace"

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name, metadata) as writer:
        pass

    with datalogger.DataReader(file_name) as reader:
        assert reader.module_name == metadata["module_name"]
        assert reader.log_version == metadata["log_version"]
        reader_metadata = reader.get_metadata()
        for key, value in metadata.items():
            assert reader_metadata[key] == value


# warning: the writer would be default to int32 regardless which dtype is specified for numpy scalers
def test_int32():
    metadata = {"module_name": "ace_loggers", "log_version": "0.0.0"}
    file_name = "test_int32.ace"

    data_vec = numpy.random.randint(0, 1000, 16)

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name, metadata) as writer:
        for data_int32 in data_vec:
            writer.write(data_int32.item())

    with datalogger.DataReader(file_name) as reader:
        assert reader.module_name == metadata["module_name"]
        assert reader.log_version == metadata["log_version"]
        for data_int32 in data_vec:
            read_int32 = reader.read_int32()
            assert data_int32 == pytest.approx(read_int32)
            assert isinstance(read_int32, int)


def test_float64():
    metadata = {"module_name": "ace_loggers", "log_version": "0.0.0"}
    file_name = "test_float64.ace"

    data_vec = numpy.random.rand(16)
    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name, metadata) as writer:
        for data_float64 in data_vec:
            writer.write(data_float64.item())

    with datalogger.DataReader(file_name) as reader:
        assert reader.module_name == metadata["module_name"]
        assert reader.log_version == metadata["log_version"]
        for data_float64 in data_vec:
            read_float64 = reader.read_float64()
            assert data_float64 == pytest.approx(read_float64)
            assert isinstance(read_float64, float)


def test_str():
    metadata = {"module_name": "ace_loggers", "log_version": "0.0.0"}
    file_name = "test_str.ace"

    data_str = "some text string!"

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name, metadata) as writer:
        writer.write(data_str)

    with datalogger.DataReader(file_name) as reader:
        assert reader.module_name == metadata["module_name"]
        assert reader.log_version == metadata["log_version"]
        assert data_str == reader.read_str()


def test_time():
    metadata = {"module_name": "ace_loggers", "log_version": "0.0.0"}
    file_name = "test_time.ace"

    data_time = datetime.now()

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name, metadata) as writer:
        writer.write(data_time)

    with datalogger.DataReader(file_name) as reader:
        assert reader.module_name == metadata["module_name"]
        assert reader.log_version == metadata["log_version"]
        assert data_time == reader.read_time()


def test_mat2d_float64():
    metadata = {"module_name": "ace_loggers", "log_version": "0.0.0"}
    file_name = "test_mat2d_float64.ace"

    data_mat3d = numpy.random.random(size=(3, 10, 5))

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name, metadata) as writer:
        _write_mat_3d(writer, data_mat3d)

    with datalogger.DataReader(file_name) as reader:
        assert reader.module_name == metadata["module_name"]
        assert reader.log_version == metadata["log_version"]
        for data_mat2d in data_mat3d:
            read_data = reader.read_mat2d_float64()
            assert data_mat2d.shape == read_data.shape
            assert data_mat2d.dtype == read_data.dtype
            assert numpy.allclose(data_mat2d, read_data)


def test_mat2d_float32():
    metadata = {"module_name": "ace_loggers", "log_version": "0.0.0"}
    file_name = "test_mat2d_float32.ace"

    data_mat3d = numpy.random.random(size=(3, 10, 5)).astype(numpy.float32)

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name, metadata) as writer:
        _write_mat_3d(writer, data_mat3d)

    with datalogger.DataReader(file_name) as reader:
        assert reader.module_name == metadata["module_name"]
        assert reader.log_version == metadata["log_version"]
        for data_mat2d in data_mat3d:
            read_data = reader.read_mat2d_float32()
            assert data_mat2d.shape == read_data.shape
            assert data_mat2d.dtype == read_data.dtype
            assert numpy.allclose(data_mat2d, read_data)


def test_ace_logger_disabled():
    os.environ["ACE_DATALOGGER_DISABLE"] = "1"

    metadata = {"module_name": "ace_loggers", "log_version": "0.0.0"}
    file_name = "test_ace_logger_disabled.ace"

    data_str = "some text string!"

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name, metadata) as writer:
        writer.write(data_str)

    del os.environ["ACE_DATALOGGER_DISABLE"]  # revert back so not to affect other tests

    assert not os.path.exists(file_name)


def test_log_path_change():
    for path_prefix in ["/tmp/ace_logs/", "/var/tmp/ace_logs/"]:
        os.environ["ACE_DATALOGGER_PATH"] = path_prefix
        file_name = ace_loggers.construct_log_path("test_log_path_change")
        del os.environ["ACE_DATALOGGER_PATH"]  # revert back so not to affect other tests

        assert file_name.startswith(path_prefix)


def test_log_timezone_change():
    file_suffix = "test_log_timezone_change"
    for timezone in ["UTC", "Asia/Tokyo", "Europe/Zurich"]:
        os.environ["ACE_DATALOGGER_TIMEZONE"] = timezone
        file_name = ace_loggers.construct_log_path(file_suffix)
        del os.environ["ACE_DATALOGGER_TIMEZONE"]  # revert back so not to affect other tests

        time_str = re.search(f"/log_(\\d+\\-\\d+\\-\\d+T\\d+\\-\\d+\\-\\d+)_{file_suffix}", file_name).group(1)
        tzinfo = zoneinfo.ZoneInfo(timezone)
        log_time = datetime.strptime(time_str, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=tzinfo)
        assert abs((datetime.now(tzinfo) - log_time).total_seconds()) < 3


def test_empty_log_filename():
    metadata = {"module_name": "ace_loggers", "log_version": "0.0.0"}
    file_name = ""

    data_str = "some text string!"

    with contextlib.suppress(FileNotFoundError):
        os.remove(file_name)
    with datalogger.DataWriter(file_name, metadata) as writer:
        writer.write(data_str)

    assert not os.path.exists(file_name)


def test_typed_log_name():
    main_type = "test"
    sub_type = "type"
    base_name = ace_loggers.construct_log_path(main_type)
    full_path = ace_loggers.construct_sub_log_path(base_name, sub_type)

    assert full_path == ace_loggers.construct_log_path(f"{main_type}_{sub_type}")


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
