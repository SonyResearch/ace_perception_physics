# pylint: disable=invalid-name
# TODO(asude): clean pylint

"""
@brief Data class for events.

@file event_data_format.py
@author Asude Aydin (asude.aydin@sony.com)
@date 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""


from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Events:
    """Data format for events."""

    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    t: np.ndarray

    size: int = field(init=False)

    def __post_init__(self):
        """Initialize data types."""
        assert self.x.dtype == np.uint16
        assert self.y.dtype == np.uint16
        assert self.p.dtype == np.uint8
        assert self.t.dtype == np.int64

        assert self.x.shape == self.y.shape == self.p.shape == self.t.shape
        assert self.x.ndim == 1

        # Without the frozen option, we could just do: self.size = self.x.size
        super().__setattr__("size", self.x.size)

        if self.size > 0:
            assert np.max(self.p) <= 1
