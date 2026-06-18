"""
Helper functions for logging data into json files.

@author Guillem Torrente (guillem.torrente@sony.com)
@date 2023
@version 0.0
@copyright Confidential, Copyright 2025, Sony AI, All rights reserved.
"""
import errno
import os
import shutil
import glob
import threading
from typing import Any, Union

import colored_glog as glog
import numpy as np
import orjson

from ace_rt_profiles import ProfileManager


def safe_mkdir_recursive(directory: str, overwrite: bool = False):
    """Checks whether a directory exists, and if not, creates it.
    Args:
        directory: the directory to create
        overwrite: if True and the directory exists, clears its contents. Defaults to False.
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except OSError as err:
                glog.error(f"[ERROR] Error while removing directory {directory}: {err}")


def safe_mknode_recursive(destiny_dir: str, node_name: str, overwrite: bool):
    """Checks whether a directory and node within it exist, and if not, creates them.
    Args:
        destiny_dir: the directory to create the node in
        node_name: the name of the node to create within the destiny_dir
        overwrite: if True and the directory or file exist, clears their contents. Defaults to False.
    """
    safe_mkdir_recursive(destiny_dir)
    if overwrite and os.path.exists(os.path.join(destiny_dir, node_name)):
        os.remove(os.path.join(destiny_dir, node_name))
    if not os.path.exists(os.path.join(destiny_dir, node_name)):
        os.mknod(os.path.join(destiny_dir, node_name))
        return False
    return True


def clean_dir(node_dir: str):
    """Removes all files in directory

    Args:
        node_dir (str): the directory to clean
    """
    if os.path.exists(node_dir):
        for elem in glob.glob(os.path.join(node_dir, "*")):
            os.remove(elem)


def get_files_in_dir(node_dir: str, pattern: str) -> list:
    """
    Returns a lost of all existing files matching a regex pattern in a given directory.
    Args:
        node_dir (str): the directory where to check
        pattern (str): regex pattern to filter files

    Returns:
        list: The list of files matching the pattern. Empty list otherwise
    """
    return sorted(glob.glob(os.path.join(node_dir, pattern)))


def does_node_exist(node_dir: str, node_name: str) -> bool:
    """Checks whether node_name within node_dir exists.

    Args:
        node_dir (str): the directory where to check
        node_name (str): the name of the node to check

    Returns:
        bool: True if the node exists, False otherwise
    """
    return os.path.exists(os.path.join(node_dir, node_name))


def save_data_as_json(save_dir: str, save_name: str, data: dict, overwrite=False, rt_profile: str = ""):
    """Saves data as a json file with name save_name within directory save_dir.

    Args:
        save_dir (str): The directory where to save the file
        save_name (str): The name of the json file, including the .json extension.
        data (dict): The data to save as a serializable dictionary.
        overwrite (bool, optional): if True the file will be replaced if it already exists. Defaults to False.
    """

    if rt_profile:
        ProfileManager.get_instance().apply(rt_profile)

    assert ".json" in save_name, f"Invalid save_name: {save_name}. Must be a json file."

    if safe_mknode_recursive(destiny_dir=save_dir, node_name=save_name, overwrite=overwrite):
        glog.warning("[WARN] Log file not saved since the file already exists!")
        return

    with open(f"{os.path.join(save_dir, save_name)}", "wb") as data_file:
        data_file.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY))


def np_to_list(my_data: Any) -> Any:
    """Transforms the input data such that it can be serialized with json by transforming np.arrays to lists.
    Currently this handles dicts, lists and np.arrays to a certain depth.

    Args:
        my_data (Any): The input data

    Returns:
        Any: The transformed data
    """

    if isinstance(my_data, dict):
        return dict_serialize(my_data)
    if isinstance(my_data, (list, np.ndarray)):
        return array_serialize(my_data)
    return my_data


# pylint: disable=broad-except
def dict_serialize(my_dict: dict) -> dict:
    """Transforms the np arrays within a dictionary such that the dictionary can be serialized with json

    Args:
        my_dict (dict): the target dictioanry

    Returns:
        dict: The transformed dictionary
    """

    for key in list(my_dict.keys()):
        if not isinstance(key, str):
            try:
                my_dict[str(key)] = my_dict[key]
            except Exception as err_1:
                try:
                    my_dict[repr(key)] = my_dict[key]
                    glog.info(f"[INFO] {err_1} could not convert the key with str()")
                except Exception as err_2:
                    glog.warning(f"[INFO] {err_2} could not convert the key with str() or repr()")
            del my_dict[key]

    for key, val in my_dict.items():
        if isinstance(val, dict):
            my_dict[key] = dict_serialize(val)
        elif isinstance(val, np.ndarray) and len(val.shape) == 1:
            my_dict[key] = val.tolist()
        elif isinstance(val, np.ndarray) and len(val.shape) == 2:
            my_dict[key] = [elem.tolist() for elem in val]
        elif isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], np.ndarray):
                my_dict[key] = [val_i.tolist() for val_i in val if isinstance(val_i, np.ndarray)]
            else:
                my_dict[key] = val

    return my_dict


def array_serialize(my_arr: Union[list, np.ndarray]) -> list:
    """Transforms an array or array of arrays into a list or list of lists

    Args:
        my_arr (Union[list, np.ndarray]): input array, non serializable

    Returns:
        list: output list, serializable
    """
    if isinstance(len(my_arr) > 0 and my_arr[0], np.ndarray):
        return [aux.tolist() for aux in my_arr]

    if isinstance(my_arr, np.ndarray):
        return my_arr.tolist()

    return my_arr


def log_in_separate_thread(log_dir: str, file_name: str, data: dict, rt_profile: str = ""):
    """Starts a thread that calls the function `save_data_as_json`

    Args:
        log_dir (str): The directory where to save the file
        file_name (str): The name of the json file, including the .json extension.
        data (dict): The data to save as a serializable dictionary.
        rt_profile (str): Real time profile for running the function.
    """
    kwargs = {"save_dir": log_dir, "save_name": file_name, "data": data, "rt_profile": rt_profile}
    dump_process = threading.Thread(target=save_data_as_json, kwargs=kwargs, daemon=False)
    dump_process.start()
