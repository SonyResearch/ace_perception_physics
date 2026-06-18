# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""
Contains shared utilities for logging
"""

import ace_setuptools

ace_setuptools.load_dependencies()  # pylint: disable= no-member

# should be imported after dependencies were loaded
import ace_loggers.datalogger_pybind as datalogger  # pylint: disable= wrong-import-position

construct_log_path = datalogger.construct_log_path  # argument: log_name (not path!)
construct_sub_log_path = (
    datalogger.construct_sub_log_path
)  # argument: log_path (string, path: base_path.ext), type (string, not path), return "base_path_type.ext"
