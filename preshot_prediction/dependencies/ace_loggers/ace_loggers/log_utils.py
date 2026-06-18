"""
Scripts to assist with log file handling.
"""

__copyright__ = "Confidential, Copyright 2025, Sony AI, All rights reserved."
__author__ = "Etienne Walther"
__maintainer__ = "Etienne Walther"
__email__ = "etienne.walther@sony.com"

import os
import re
from datetime import datetime
from pathlib import Path


def get_log_identifier(filepath: str, strict: bool = True) -> str:
    """Extracts the unique log identifier from the provided filepath. (eg. `log_2024-10-30T16-05-11` from
    `logs/log_2024-10-30T16-05-11_pose_estimator.ace`)

    Args:
        filepath (str): Path to any log file.

    Returns:
        (str): Returns unique log identifier.
    """
    log_identifier = os.path.basename(filepath)[:23]

    # Define the regex pattern for validation
    pattern = r"^log_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$"

    # Check if the prefix matches the expected format
    if strict and not re.match(pattern, log_identifier):
        raise ValueError(f"No valid log identifier found in provided filepath: {filepath} -> {log_identifier}")

    if not strict and not re.match(pattern, log_identifier):
        log_identifier = ""

    return log_identifier


def get_location_till_date(filepath: str, strict: bool = True) -> str:
    """
    Return the directory path up to and including the YYYYMMDD folder.
    Always ends with a trailing '/'.
    """
    parts = Path(filepath).parts  # preserves leading "/" if present

    date_index = -1
    for i, k in enumerate(parts):
        if re.fullmatch(r"\d{8}", k):
            try:
                datetime.strptime(k, "%Y%m%d")  # validate real date
                date_index = i  # keep the last valid date dir
            except ValueError:
                pass

    if date_index == -1:
        if strict:
            raise ValueError(f"No valid YYYYMMDD directory found in: {filepath}")
        return ""

    wanted = Path(*parts[: date_index + 1])
    return str(wanted) + os.sep  # ensure trailing '/'
