"""
@brief Two H5 Writer classes for iteratively saving events and labels into corresponding h5 files.

@file h5_writer
@author Asude Aydin (asude.aydin@sony.com)
@date 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import weakref
from pathlib import Path

import h5py
import numpy as np


class H5Writer:
    """
    This class provides functionality to create and write event data to an HDF5 file.

    Attributes:
        outfile (Path): The path to the HDF5 file to be created.

    Methods:
        close_callback: Static method to perform cleanup operations when closing the HDF5 file.
        add_data: Adds data to the HDF5 file.
        add_metadata: Adds metadata to the HDF5 file.

    """

    def __init__(self, outfile: Path):
        """Initialize the H5Writer object.

        Args:
            outfile (Path): The path to the HDF5 file to be created.

        Raises:
            AssertionError: If the output file already exists.
        """
        # Ensure that the output file does not already exist
        assert not Path(outfile).exists(), Path(outfile)

        # Create the HDF5 file
        self.h5f = h5py.File(str(outfile), "w")
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        # create hdf5 datasets
        shape = (2**16,)
        maxshape = (None,)
        compression = "lzf"
        self.h5f.create_dataset(
            "events/x",
            shape=shape,
            dtype="u2",
            chunks=shape,
            maxshape=maxshape,
            compression=compression,
        )
        self.h5f.create_dataset(
            "events/y",
            shape=shape,
            dtype="u2",
            chunks=shape,
            maxshape=maxshape,
            compression=compression,
        )
        self.h5f.create_dataset(
            "events/p",
            shape=shape,
            dtype="u1",
            chunks=shape,
            maxshape=maxshape,
            compression=compression,
        )
        self.h5f.create_dataset(
            "events/t",
            shape=shape,
            dtype="i8",
            chunks=shape,
            maxshape=maxshape,
            compression=compression,
        )

        # Initialize previous size
        self.prev_size = 0

    @staticmethod
    def close_callback(h5f: h5py.File):
        """Perform cleanup operations when closing the HDF5 file.

        Args:
            h5f (h5py.File): The HDF5 file object.
        """
        last_ms = int(h5f["events/t"][-1] / 1e3)
        ms_array = np.array(range(0, last_ms + 1))
        ms_to_idx = np.searchsorted(h5f["events/t"][:] / 1e3, ms_array, side="left")
        h5f.create_dataset("ms_to_idx", len(ms_to_idx), dtype="u4")
        h5f["ms_to_idx"][:] = ms_to_idx

        # Close h5 file
        h5f.close()

    def add_data(self, events: np.array):
        """Add data to the HDF5 file.

        Args:
            events (np.array): The array of events to be added.

        """
        current_size = events.size
        new_size = self.prev_size + current_size

        # Resize dataset to accomdate data size
        self.h5f["events/x"].resize(new_size, axis=0)
        self.h5f["events/y"].resize(new_size, axis=0)
        self.h5f["events/p"].resize(new_size, axis=0)
        self.h5f["events/t"].resize(new_size, axis=0)

        # Write data to datasets
        self.h5f["events/x"][self.prev_size : new_size] = events.x
        self.h5f["events/y"][self.prev_size : new_size] = events.y
        self.h5f["events/p"][self.prev_size : new_size] = events.p
        self.h5f["events/t"][self.prev_size : new_size] = events.t

        self.prev_size = new_size

    def add_metadata(self, metadata_dict: dict):
        """Add metadata to the HDF5 file.

        Args:
            metadata_dict (dict): A dictionary containing metadata information.

        """
        for key, value in metadata_dict.items():
            if key in ["resolution", "camera_matrix", "distortion_coeffs"]:
                dataset_name = f"calibration/{key}"
                self.h5f.create_dataset(dataset_name, data=value)
            else:
                dataset_name = f"{key}"
                self.h5f.create_dataset(dataset_name, shape=(1,), dtype="i8", data=value)


class H5WriterLabel:
    """
    This class provides functionality to create and write label data to an HDF5 file.

    Attributes:
        outfile (Path): The path to the HDF5 file to be created.

    Methods:
        close_callback: Static method to perform cleanup operations when closing the HDF5 file.
        add_data: Adds data to the HDF5 file.

    """

    def __init__(self, outfile: Path, default_shape: int = 0):
        """Initialize the H5WriterLabel object.

        Args:
            outfile (Path): The path to the HDF5 file to be created.
            default_shape: Set to value when you want to write all labels at once.
            Otherwise set to 0 for, 1 by 1 writing into h5 file.
        Raises:
            AssertionError: If the output file already exists.
        """
        # Ensure that the output file does not already exist
        assert not Path(outfile).exists(), Path(outfile)

        self.default_shape = default_shape

        # Create the HDF5 file
        self.h5f = h5py.File(str(outfile), "w")
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        # create hdf5 datasets
        compression = "lzf"
        self.h5f.create_dataset("points", shape=(default_shape, 2), maxshape=(None, 2), compression=compression)
        self.h5f.create_dataset("radius", shape=(default_shape,), maxshape=(None,), compression=compression)
        self.h5f.create_dataset("points_orig", shape=(default_shape, 2), maxshape=(None, 2), compression=compression)
        self.h5f.create_dataset("radius_orig", shape=(default_shape,), maxshape=(None,), compression=compression)
        self.h5f.create_dataset("velocities", shape=(default_shape, 2), maxshape=(None, 2), compression=compression)
        self.h5f.create_dataset("timestamps", shape=(default_shape,), maxshape=(None,), compression=compression)

        # Initialize previous size
        self.prev_size = 0

    @staticmethod
    def close_callback(h5f: h5py.File):
        """Perform cleanup operations when closing the HDF5 file.

        Args:
            h5f (h5py.File): The HDF5 file object.
        """
        # Close h5 file
        h5f.close()

    def add_data(self, label_dict: dict):
        """Add label data to the HDF5 file.

        Args:
            label_dict (dict): A dictionary containing label data.

        """

        if self.default_shape == 0:
            current_size = 1
            new_size = self.prev_size + current_size

            # Resize dataset to accomdate data size
            self.h5f["points"].resize(new_size, axis=0)
            self.h5f["radius"].resize(new_size, axis=0)
            self.h5f["points_orig"].resize(new_size, axis=0)
            self.h5f["radius_orig"].resize(new_size, axis=0)
            self.h5f["velocities"].resize(new_size, axis=0)
            self.h5f["timestamps"].resize(new_size, axis=0)

            # Write data to datasets
            self.h5f["points"][self.prev_size : new_size] = [label_dict["points"]]
            self.h5f["radius"][self.prev_size : new_size] = [label_dict["radius"]]
            self.h5f["points_orig"][self.prev_size : new_size] = [label_dict["points_orig"]]
            self.h5f["radius_orig"][self.prev_size : new_size] = [label_dict["radius_orig"]]
            self.h5f["velocities"][self.prev_size : new_size] = [label_dict["velocities"]]
            self.h5f["timestamps"][self.prev_size : new_size] = [label_dict["timestamps"]]

            self.prev_size = new_size

        else:
            # Write data to datasets
            self.h5f["points"][:] = [label_dict["points"]]
            self.h5f["radius"][:] = [label_dict["radius"]]
            self.h5f["points_orig"][:] = [label_dict["points_orig"]]
            self.h5f["radius_orig"][:] = [label_dict["radius_orig"]]
            self.h5f["velocities"][:] = [label_dict["velocities"]]
            self.h5f["timestamps"][:] = [label_dict["timestamps"]]
