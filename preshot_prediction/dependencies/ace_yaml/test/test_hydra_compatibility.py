#!/usr/bin/env python3
"""
Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import os
import sys

import ace_yaml as yaml
import pytest
from hydra import compose, initialize


def test_ace_yaml_hydra():
    file_path = os.path.join(os.path.dirname(__file__), "test_params", "level03_01.yaml")
    with open(file_path, "r", encoding="UTF-8") as fp:
        ace_params = yaml.full_load(fp)

    initialize(config_path="test_params", job_name="test_hydra")
    hydra_cfg = compose(config_name="level03_01")

    print(ace_params)
    print(hydra_cfg)
    assert ace_params["param1"] == hydra_cfg["param1"]
    assert ace_params["param2"] == hydra_cfg["param2"]
    assert ace_params["param3"] == hydra_cfg["param3"]


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
