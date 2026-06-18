# Confidential, Copyright 2025, Sony AI, All rights reserved
"""
A centralized place for controlling real time profiles
"""

from functools import wraps
import ace_setuptools

ace_setuptools.load_dependencies()

# pylint: disable=wrong-import-position
from ace_rt_profiles.bindings import *


def profile_guard(func):
    """Function decorator version of ProfileGuard"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with ProfileGuard():  # pylint: disable=undefined-variable
            return func(*args, **kwargs)

    return wrapper
