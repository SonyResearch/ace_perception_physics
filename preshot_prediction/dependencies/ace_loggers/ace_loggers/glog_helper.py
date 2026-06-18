"""Converting from ros logging to glog logging settings

@author Thomas van de Wiel (thomas.vandewiel@sony.com)
@date 2021
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import sys

import colored_glog as glog


def roslog_to_glog_level(log_level) -> int:
    """
    Converts of logging level from ros format to glog format
    @param log_level        The ros format logging level
    @returns glog_level     The glog format logging level
    """

    # Glog level from ROS log level
    if log_level.lower() == "fatal":
        glog_level = 3
    elif log_level.lower() == "error":
        glog_level = 2
    elif log_level.lower() == "warn":
        glog_level = 1
    elif log_level.lower() == "info" or log_level.lower() == "debug":
        glog_level = 0
    else:
        raise ValueError(f"The logging level must be either fatal, error, warn, info or debug but got {log_level}")

    return glog_level


def set_glog_level_from_gflag(default_level="INFO") -> int:
    """
    Hacky solution to parse and set the C++ glog level to PyPI glog.
    Might be improved later.
    Looks up for the gflag "minloglevel" and parses the C++ log level. Then transforms it to the python glog format.
    In case the gflag is not found, `INFO` level will be defaulted.
    @return: The assigned glog level as an integer
    """

    try:
        glog_level = [(int(i.split("=")[1])) for i in sys.argv if i.startswith("--minloglevel")][0]
        assert glog_level in range(4), f"Invalid glog_level={glog_level}"
        glog.setLevel((glog_level + 2) * 10)
    except IndexError:
        glog_level = 0
        glog.setLevel(default_level)
        glog.warning(f"Could not read the default logging level from the gflags. Set to {default_level} by default")
    return glog_level
