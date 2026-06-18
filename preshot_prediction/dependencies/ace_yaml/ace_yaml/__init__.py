# pylint: skip-file
# type: ignore
# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""
Provides common YAML functionalities
"""

import ace_setuptools
from yaml import *

# Must call load_dependencies() before importing python_module
ace_setuptools.load_dependencies()

from . import python_module

INHERITANCE_NODE_NAME = python_module.INHERITANCE_NODE_NAME

full_load = python_module.full_load
full_load_all = python_module.full_load
safe_load = python_module.full_load
unsafe_load = python_module.full_load
unsafe_load_all = python_module.full_load
load_from_string = python_module.load_string
to_python = python_module.to_python


def load(fp, *args, **kwargs):
    return full_load(fp)


def load_all(fp, *args, **kwargs):
    return full_load(fp)


def load_string(fp, *args, **kwargs):
    return load_from_string(fp)


def load_dict(d, *args, **kwargs):
    return load_string(dump(d))
