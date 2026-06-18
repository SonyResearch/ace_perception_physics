#!/usr/bin/env python3
"""
Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import os
import sys

import ace_yaml as yaml
import pytest


def test_ace_yaml_hierarchy():
    file_path = os.path.join(os.path.dirname(__file__), "test_params", "level03_01.yaml")
    with open(file_path, "r", encoding="UTF-8") as fp:
        params = yaml.full_load(fp)

    assert yaml.INHERITANCE_NODE_NAME not in params
    assert params["param1"] == "level03_01"
    assert params["param2"] == "level03_01"
    assert params["param3"] == "level02_01"
    assert params["param4"] == "level01_02"
    assert params["param5"]["x"] == "level01_02"
    assert params["param5"]["y"] == "level01_01"
    assert params["param5"]["z"] == "level01_02"


def test_final_yaml():
    file_path = os.path.join(os.path.dirname(__file__), "test_params", "final_override.yaml")
    try:  # this should throw a ValueError exception
        with open(file_path, "r", encoding="UTF-8") as fp:
            yaml.full_load(fp)
        assert False, "Final didn't throw an exception!"
    except ValueError as e:
        assert True


def test_external_yaml():
    file_path = os.path.join(os.path.dirname(__file__), "test_params", "external_01.yaml")
    try:
        with open(file_path, "r", encoding="UTF-8") as fp:
            params = yaml.full_load(fp)
        assert "gravity" in params
    except (RuntimeError, IndexError):
        # expected to fail as it requires an external dependency which may not always exist on the CI side..
        pass


def test_ace_yaml_types():
    file_path = os.path.join(os.path.dirname(__file__), "test_params", "types.yaml")
    with open(file_path, "r", encoding="UTF-8") as fp:
        params = yaml.full_load(fp)

    for key in ["true_value", "True_value"]:
        assert isinstance(params[key], bool)
        assert params[key] == True

    for key in ["false_value", "False_value"]:
        assert isinstance(params[key], bool)
        assert params[key] == False

    assert isinstance(params["int_value"], int)
    assert params["int_value"] == 1

    assert isinstance(params["float_value"], float)
    assert params["float_value"] == 1.0

    for key in ["y", "Y", "yes", "Yes", "YES", "n", "N", "no", "No", "NO"]:
        assert isinstance(params[key], str)
        assert params[key] == key


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
