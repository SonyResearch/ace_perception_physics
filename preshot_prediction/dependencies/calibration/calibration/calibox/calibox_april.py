# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""
Camera calibration toolbox - April grid detector.
"""
import contextlib

import numpy

from ace_apriltags import python_module as apriltags


class Detector(apriltags.AprilGridDetector):
    """
    A wrapper around the AprilGridDetector from the apriltags library to customize its interface.
    """

    def __init__(self, args):
        """
        Constructs the underlying AprilGridDetector instance.

        @param args (argparse.Namespace): The command-line options containing parameters for the detector.
        """
        super().__init__(args.tag_rows, args.tag_cols, args.tag_size, args.tag_spacing)

    def get_object_points(self):
        """
        Retrieves the object points from the underlying AprilGridDetector instance.

        @return numpy.ndarray: A 3D array of object points as defined by the AprilGridDetector.
        """
        return super().points()

    def detect(self, img_bgr):
        """
        Detects feature points in the provided image.

        @param img_bgr (numpy.ndarray): Input image in BGR format as a NumPy array.

        @return tuple: A tuple containing:
            - retval (bool): Indicates whether the detection was successful.
            - image_points (numpy.ndarray): An array of detected 2D points,
              with NaN values for points that were not detected.
        """
        with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
            retval, image_points, visibility_mask = super().detect(img_bgr)

        if retval:
            visibility_mask = numpy.array(visibility_mask, dtype=bool)
            image_points[~visibility_mask] = numpy.nan

        return retval, image_points


def add_parser(subparsers):
    """
    Adds a sub-parser to the main argument parser for AprilTag detection.

    @param subparsers (argparse._SubParsersAction): The subparsers action to which the new parser will be added.
    """
    apriltag_parser = subparsers.add_parser("april", help="Using AprilTag for grid detection")
    apriltag_parser.add_argument("--tag-rows", type=int, default=6, help="Number of rows in the AprilTag grid")
    apriltag_parser.add_argument("--tag-cols", type=int, default=6, help="Number of columns in the AprilTag grid")
    apriltag_parser.add_argument("--tag-size", type=float, default=0.088, help="Size of each AprilTag [in meters]")
    apriltag_parser.add_argument(
        "--tag-spacing", type=float, default=0.3, help="Spacing between AprilTags [relative to tag size]"
    )
