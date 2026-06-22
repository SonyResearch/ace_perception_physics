"""
@brief module contains functions for various file and folder operations.

@file utils.py
@author Asude Aydin (asude.aydin@sony.com)
@date 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import os


def check_file_extension_exists(root: str, extension: str):
    """
    Check if a file with a .mat extension exists in the list of files.

    Args:
    - files (str): Path of directory to check file existence containing a type of extension
    - extension (str): Extension type to check

    Returns:
    - True if a .mat file exists, False otherwise.
    """
    files = os.listdir(root)
    return any(file.endswith(extension) for file in files)


def read_txt_file(file_path):
    """
    Read a text file of single integers line by line.

    Args:
    - file_path (str): The path to the text file.

    Returns:
    - values (list): A list containing the values read from the text file.
    """
    values = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Remove leading and trailing whitespace and add the value to the list
            values.append(int(line.strip()))
    return values


def check_value_exists_df(dataframe, column_name, value):
    """
    Check if a value exists in a pandas DataFrame column.

    Args:
    - dataframe (DataFrame): The pandas DataFrame.
    - column_name (str): The name of the column to check.
    - value: The value to check for in the specified column.

    Returns:
    - True if the value exists in the column, False otherwise.
    """
    return value in dataframe[column_name].values


def check_value_in_range_df(dataframe, column_name, value, lower_bound=None, upper_bound=None):
    """
    Check if a value exists in a pandas DataFrame column and optionally if it falls within a specified range.

    Args:
    - dataframe (DataFrame): The pandas DataFrame.
    - column_name (str): The name of the column to check.
    - value: The value to check for in the specified column.
    - lower_bound: The lower bound of the range (inclusive). Default is None.
    - upper_bound: The upper bound of the range (inclusive). Default is None.

    Returns:
    - True if the value exists in the column and, optionally, falls within the specified range, False otherwise.
    """
    if lower_bound is not None and upper_bound is not None:
        in_range = (dataframe[column_name] >= lower_bound) & (dataframe[column_name] <= upper_bound)
        return (value in dataframe[column_name].values) and in_range.any()
    return value in dataframe[column_name].values
